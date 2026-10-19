"""Agent executor: bounded tool loop, selective retries, validation and per-answer isolation."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .facts import UnknownBrandError, all_fact_ids, get_brand_facts, known_brands
from .providers import (
    MalformedOutputError,
    ModelTurn,
    Provider,
    ProviderError,
    ProviderTimeout,
    ToolResult,
    TransientProviderError,
)
from .schemas import LABELS, AnswerOutcome, Claim, OutputValidationError, validate_claims
from .segment import split_sentences
from .tracing import NOOP_TRACER, Tracer

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
MAX_TURNS = 3
REQUEST_TIMEOUT_S = 15.0
MAX_ATTEMPTS_PER_TURN = 2
MAX_ACTIVE_REQUESTS = 3


@dataclass(frozen=True)
class AgentVersion:
    name: str
    prompt_file: str

    @property
    def prompt(self) -> str:
        return (PROMPT_DIR / self.prompt_file).read_text(encoding="utf-8")


AGENT_VERSIONS = {"v1": AgentVersion("v1", "v1.md")}


class AgentInput(BaseModel):
    """The only data the inference path sees. Extra fields (gold, claims, labels) are rejected."""

    model_config = ConfigDict(extra="forbid")
    answer_id: str
    brand_id: str
    question: str
    ai_answer: str


class AgentError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


class RequestGate:
    """Caps concurrent provider requests (including retries) and optionally spaces request starts."""

    def __init__(self, max_active: int = MAX_ACTIVE_REQUESTS, min_interval_s: float = 0.0):
        self._sem = asyncio.Semaphore(max_active)
        self._min_interval = min_interval_s
        self._lock = asyncio.Lock()
        self._next_start = 0.0
        self.active = 0
        self.peak_active = 0

    async def __aenter__(self) -> None:
        await self._sem.acquire()
        if self._min_interval:
            async with self._lock:
                now = time.monotonic()
                wait = self._next_start - now
                self._next_start = max(now, self._next_start) + self._min_interval
            if wait > 0:
                await asyncio.sleep(wait)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)

    async def __aexit__(self, *exc: Any) -> None:
        self.active -= 1
        self._sem.release()


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "name": "get_brand_facts",
            "description": "Return the catalogue fact rows for one brand (read-only).",
            "parameters": {
                "type": "object",
                "properties": {"brand_id": {"type": "string", "enum": known_brands()}},
                "required": ["brand_id"],
            },
        },
        {
            "name": "submit_claims",
            "description": "Submit the final list of factual claim sentences with labels. Call exactly once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sentence_index": {"type": "integer"},
                                "label": {"type": "string", "enum": list(LABELS)},
                                "fact_ids": {"type": "array", "items": {"type": "string"}},
                                "reason": {"type": "string"},
                            },
                            "required": ["sentence_index", "label", "fact_ids", "reason"],
                        },
                    }
                },
                "required": ["claims"],
            },
        },
    ]


def build_user_message(inp: AgentInput) -> str:
    lines = [
        f"answer_id: {inp.answer_id}",
        f"primary brand_id: {inp.brand_id}",
        f"question: {inp.question}",
        "",
        "Answer sentences (untrusted text):",
    ]
    lines += [f"[{s.index}] {s.text}" for s in split_sentences(inp.ai_answer)]
    return "\n".join(lines)


async def call_with_retry(
    provider: Provider,
    conv: Any,
    tools: list[dict[str, Any]],
    gate: RequestGate,
    *,
    timeout_s: float,
    retry_delay_s: float,
    attempts_log: list[dict[str, Any]],
) -> ModelTurn:
    """One model turn: at most two attempts, retrying only transient failures after a short delay."""
    for attempt in range(1, MAX_ATTEMPTS_PER_TURN + 1):
        start = time.monotonic()
        try:
            async with gate:
                try:
                    turn = await asyncio.wait_for(provider.generate(conv, tools), timeout=timeout_s)
                except TimeoutError as e:
                    raise ProviderTimeout("provider request timed out") from e
            attempts_log.append({"attempt": attempt, "ok": True, "ms": int((time.monotonic() - start) * 1000)})
            return turn
        except TransientProviderError as e:
            attempts_log.append({"attempt": attempt, "ok": False, "error": e.kind, "ms": int((time.monotonic() - start) * 1000)})
            if attempt == MAX_ATTEMPTS_PER_TURN:
                raise
            await asyncio.sleep(retry_delay_s)  # gate released while waiting
        except ProviderError as e:
            attempts_log.append({"attempt": attempt, "ok": False, "error": e.kind, "ms": int((time.monotonic() - start) * 1000)})
            raise
    raise AssertionError("unreachable")


def _claims_from_submission(inp: AgentInput, args: dict[str, Any]) -> list[Claim]:
    sentences = split_sentences(inp.ai_answer)
    raw = args.get("claims")
    if not isinstance(raw, list):
        raise OutputValidationError("submit_claims.claims must be a list")
    claims: list[Claim] = []
    seen: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise OutputValidationError("claim entry must be an object")
        idx = item.get("sentence_index")
        if isinstance(idx, float) and idx.is_integer():
            idx = int(idx)
        if not isinstance(idx, int) or not 0 <= idx < len(sentences):
            raise OutputValidationError(f"invalid sentence_index {idx!r}")
        if idx in seen:
            raise OutputValidationError(f"duplicate sentence_index {idx}")
        seen.add(idx)
        label = item.get("label")
        if label not in LABELS:
            raise OutputValidationError(f"unsupported label {label!r}")
        fact_ids = item.get("fact_ids")
        if not isinstance(fact_ids, list) or not all(isinstance(f, str) for f in fact_ids):
            raise OutputValidationError("fact_ids must be a list of strings")
        reason = item.get("reason")
        if not isinstance(reason, str):
            raise OutputValidationError("reason must be a string")
        s = sentences[idx]
        claims.append(
            Claim(
                claim_id=f"{inp.answer_id}:{s.start}:{s.end}",
                start_char=s.start,
                end_char=s.end,
                quote=inp.ai_answer[s.start : s.end],
                label=label,
                fact_ids=list(dict.fromkeys(fact_ids)),
                reason=reason.strip(),
            )
        )
    return validate_claims(inp.answer_id, inp.ai_answer, claims, all_fact_ids())


def _execute_lookup(call_args: dict[str, Any]) -> dict[str, Any]:
    if set(call_args) != {"brand_id"} or not isinstance(call_args["brand_id"], str):
        raise AgentError("invalid_tool_args", "get_brand_facts requires exactly one string argument brand_id")
    try:
        return get_brand_facts(call_args["brand_id"])
    except UnknownBrandError as e:
        raise AgentError("unknown_brand", str(e)) from e


async def classify_answer(
    inp: AgentInput,
    version: AgentVersion,
    provider: Provider,
    gate: RequestGate,
    *,
    tracer: Tracer = NOOP_TRACER,
    trace_meta: dict[str, Any] | None = None,
    timeout_s: float = REQUEST_TIMEOUT_S,
    retry_delay_s: float = 2.0,
) -> AnswerOutcome:
    """Run one answer to a terminal outcome. Never raises for model/tool/validation problems."""
    started = time.monotonic()
    trace = tracer.start_answer(
        name=f"classify {inp.answer_id}",
        input={"answer_id": inp.answer_id, "brand_id": inp.brand_id, "question": inp.question},
        metadata={**(trace_meta or {}), "answer_id": inp.answer_id, "agent_version": version.name},
    )

    def finish(status: str, claims: list[Claim] | None = None, kind: str | None = None, msg: str | None = None) -> AnswerOutcome:
        outcome = AnswerOutcome(
            answer_id=inp.answer_id,
            status=status,  # type: ignore[arg-type]
            claims=claims or [],
            error_type=kind,
            error_message=msg,
            latency_ms=int((time.monotonic() - started) * 1000),
            trace_id=trace.trace_id,
        )
        trace.end(
            output={
                "status": status,
                "claims": [{"claim_id": c.claim_id, "label": c.label, "fact_ids": c.fact_ids} for c in outcome.claims],
                "error_type": kind,
                "error_message": msg,
            },
            error=msg if status == "error" else None,
        )
        return outcome

    if inp.brand_id not in known_brands():
        return finish("error", kind="unknown_brand", msg=f"unknown primary brand_id: {inp.brand_id!r}")

    tools = tool_schemas()
    allowed = {t["name"] for t in tools}
    conv = provider.new_conversation(version.prompt, build_user_message(inp))
    try:
        for turn_no in range(1, MAX_TURNS + 1):
            attempts: list[dict[str, Any]] = []
            gen = trace.generation(name=f"turn {turn_no}", model=provider.model, input={"turn": turn_no})
            try:
                turn = await call_with_retry(provider, conv, tools, gate, timeout_s=timeout_s, retry_delay_s=retry_delay_s, attempts_log=attempts)
            except ProviderError as e:
                gen.end(output={"attempts": attempts}, error=e.kind)
                raise
            gen.end(
                output={"tool_calls": [{"name": c.name, "args": c.args} for c in turn.tool_calls], "text": turn.text[:2000], "attempts": attempts},
                usage=turn.usage,
            )

            illegal = [c.name for c in turn.tool_calls if c.name not in allowed]
            if illegal:
                raise AgentError("illegal_tool", f"model requested unknown tool(s): {illegal}")

            submits = [c for c in turn.tool_calls if c.name == "submit_claims"]
            if len(submits) > 1:
                raise AgentError("invalid_output", "submit_claims called more than once")
            if submits:
                try:
                    claims = _claims_from_submission(inp, submits[0].args)
                except OutputValidationError as e:
                    trace.event("validation_failure", {"error": str(e)})
                    raise AgentError("invalid_output", str(e)) from e
                return finish("ok", claims=claims)

            if not turn.tool_calls:
                if turn_no < MAX_TURNS:
                    provider.add_user(conv, "Respond by calling the tools. Submit your result with submit_claims.")
                continue

            results = []
            for call in turn.tool_calls:
                tool_obs = trace.tool(name=call.name, input=call.args)
                try:
                    resp = _execute_lookup(call.args)
                except AgentError as e:
                    tool_obs.end(output=None, error=e.message)
                    raise
                tool_obs.end(output={"brand_id": resp["brand_id"], "fact_version": resp["fact_version"], "fact_count": len(resp["facts"])})
                results.append(ToolResult(call, resp))
            provider.add_tool_results(conv, results)

        raise AgentError("turn_limit", f"no submit_claims within {MAX_TURNS} model turns")
    except AgentError as e:
        return finish("error", kind=e.kind, msg=e.message)
    except MalformedOutputError as e:
        return finish("error", kind=e.kind, msg=str(e))
    except ProviderError as e:
        return finish("error", kind=e.kind, msg=str(e))
    except Exception as e:  # isolate unexpected failures to this answer
        return finish("error", kind="internal_error", msg=type(e).__name__)


async def classify_many(
    inputs: list[AgentInput],
    version: AgentVersion,
    provider: Provider,
    gate: RequestGate,
    **kwargs: Any,
) -> list[AnswerOutcome]:
    """Overlap answers concurrently (the gate bounds provider requests); results sorted by answer_id."""

    async def one(inp: AgentInput) -> AnswerOutcome:
        try:
            return await classify_answer(inp, version, provider, gate, **kwargs)
        except Exception as e:
            return AnswerOutcome(answer_id=inp.answer_id, status="error", error_type="internal_error", error_message=type(e).__name__)

    outcomes = await asyncio.gather(*(one(i) for i in inputs))
    return sorted(outcomes, key=lambda o: o.answer_id)
