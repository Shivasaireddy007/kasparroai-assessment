"""Provider adapters. The agent talks to a small neutral interface so a fake provider can drive tests.

A provider owns its native conversation history (Gemini needs its raw model content, including
thought signatures, sent back unchanged), and exposes generate / add_tool_results / add_user.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol


class ProviderError(Exception):
    """Base class. `kind` is a sanitized error type used in answer outcomes and API responses."""

    kind = "provider_error"


class TransientProviderError(ProviderError):
    """Timeout, rate limit or explicit server-side transient failure. Retried at most once."""

    kind = "provider_transient"


class ProviderTimeout(TransientProviderError):
    kind = "provider_timeout"


class PermanentProviderError(ProviderError):
    """Authentication, bad request and other non-retryable failures."""

    kind = "provider_error"


class MalformedOutputError(ProviderError):
    """The provider replied, but the reply is unusable. Never retried."""

    kind = "invalid_output"


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str | None = None


@dataclass
class ModelTurn:
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class ToolResult:
    call: ToolCall
    response: dict[str, Any]




class Provider(Protocol):
    name: str
    model: str

    def new_conversation(self, system: str, user: str) -> Any: ...

    async def generate(self, conv: Any, tools: list[dict[str, Any]]) -> ModelTurn: ...

    def add_tool_results(self, conv: Any, results: list[ToolResult]) -> None: ...

    def add_user(self, conv: Any, text: str) -> None: ...

    def settings(self) -> dict[str, Any]: ...


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, timeout_s: float = 15.0, thinking_level: str | None = None):
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model
        self.timeout_s = timeout_s
        self.thinking_level = thinking_level
        # SDK-level retries are disabled; the executor owns retry policy.
        self._client = genai.Client(
            http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000), retry_options=types.HttpRetryOptions(attempts=1)
            )
        )

    def settings(self) -> dict[str, Any]:
        return {"thinking_level": self.thinking_level, "sdk_timeout_s": self.timeout_s, "temperature": "provider default"}

    def new_conversation(self, system: str, user: str) -> dict[str, Any]:
        t = self._types
        return {"system": system, "contents": [t.Content(role="user", parts=[t.Part(text=user)])]}

    def add_user(self, conv: dict[str, Any], text: str) -> None:
        t = self._types
        conv["contents"].append(t.Content(role="user", parts=[t.Part(text=text)]))

    def add_tool_results(self, conv: dict[str, Any], results: list[ToolResult]) -> None:
        t = self._types
        parts = [
            t.Part(function_response=t.FunctionResponse(id=r.call.id, name=r.call.name, response=r.response))
            for r in results
        ]
        conv["contents"].append(t.Content(role="user", parts=parts))

    async def generate(self, conv: dict[str, Any], tools: list[dict[str, Any]]) -> ModelTurn:
        from google.genai import errors

        t = self._types
        decls = [
            t.FunctionDeclaration(name=x["name"], description=x["description"], parameters_json_schema=x["parameters"])
            for x in tools
        ]
        config = t.GenerateContentConfig(
            system_instruction=conv["system"],
            tools=[t.Tool(function_declarations=decls)],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=t.ThinkingConfig(thinking_level=self.thinking_level) if self.thinking_level else None,
        )
        try:
            resp = await self._client.aio.models.generate_content(model=self.model, contents=conv["contents"], config=config)
        except errors.APIError as e:
            code = getattr(e, "code", None) or 0
            msg = f"HTTP {code} {getattr(e, 'status', '')}".strip()
            quota_ids = [
                v.get("quotaId")
                for d in ((e.details or {}).get("error", {}).get("details", []) if isinstance(e.details, dict) else [])
                for v in d.get("violations", [])
                if isinstance(v, dict) and v.get("quotaId")
            ]
            if quota_ids:
                msg += f" quota={','.join(quota_ids)}"
            if code == 429 or code >= 500:
                raise TransientProviderError(msg) from e
            raise PermanentProviderError(msg) from e
        except TimeoutError as e:
            raise ProviderTimeout("provider request timed out") from e
        except Exception as e:  # httpx timeouts and transport errors
            if "timeout" in type(e).__name__.lower():
                raise ProviderTimeout("provider request timed out") from e
            raise PermanentProviderError(type(e).__name__) from e

        if not resp.candidates or resp.candidates[0].content is None:
            raise MalformedOutputError("response had no candidate content")
        conv["contents"].append(resp.candidates[0].content)
        u = resp.usage_metadata
        usage = {}
        if u is not None:
            usage = {
                k: v
                for k, v in {
                    "input": u.prompt_token_count,
                    "output": u.candidates_token_count,
                    "thinking": getattr(u, "thoughts_token_count", None),
                    "total": u.total_token_count,
                }.items()
                if v is not None
            }
        calls = [ToolCall(name=fc.name or "", args=dict(fc.args or {}), id=fc.id) for fc in (resp.function_calls or [])]
        text = "".join(p.text for p in resp.candidates[0].content.parts or [] if p.text and not getattr(p, "thought", False))
        return ModelTurn(tool_calls=calls, text=text, usage=usage)


def provider_from_env() -> Provider:
    name = os.environ.get("LLM_PROVIDER", "")
    model = os.environ.get("LLM_MODEL", "")
    if name == "gemini":
        return GeminiProvider(model, thinking_level=os.environ.get("LLM_THINKING_LEVEL") or None)
    raise ValueError(f"unsupported LLM_PROVIDER {name!r}")
