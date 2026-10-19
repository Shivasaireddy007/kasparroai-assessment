import asyncio
import csv

import pytest
from pydantic import ValidationError

from claimcheck.agent import AGENT_VERSIONS, AgentInput, RequestGate, classify_answer, classify_many
from claimcheck.data import DATA_DIR, load_answers, load_gold
from claimcheck.providers import MalformedOutputError, ModelTurn, PermanentProviderError, ToolCall, TransientProviderError
from claimcheck.segment import split_sentences

from .fakes import FakeProvider, claim, lookup, standard, submit

V1 = AGENT_VERSIONS["v1"]
ANSWERS = {a.answer_id: a for a in load_answers()}


def inp(aid: str = "A01") -> AgentInput:
    a = ANSWERS[aid]
    return AgentInput(answer_id=a.answer_id, brand_id=a.brand_id, question=a.question, ai_answer=a.ai_answer)


async def run(provider, aid="A01", **kw):
    return await classify_answer(inp(aid), V1, provider, RequestGate(), retry_delay_s=0, **kw)


async def test_happy_path_spans_come_from_python_offsets():
    sents = split_sentences(ANSWERS["A01"].ai_answer)
    p = FakeProvider(standard([claim(4, fact_ids=["cleartrail.mini.price_2026", "cleartrail.plus.price_2026"]), claim(3, "unknown")]))
    o = await run(p)
    assert o.status == "ok" and p.calls == 2
    assert [c.start_char for c in o.claims] == sorted([sents[3].start, sents[4].start])
    for c in o.claims:
        assert c.quote == ANSWERS["A01"].ai_answer[c.start_char : c.end_char]
        assert c.claim_id == f"A01:{c.start_char}:{c.end_char}"


async def test_valid_empty_claims():
    o = await run(FakeProvider(standard([])))
    assert o.status == "ok" and o.claims == []


async def test_malformed_output_not_retried():
    def s(conv, turn):
        raise MalformedOutputError("no candidates")

    p = FakeProvider(s)
    o = await run(p)
    assert (o.status, o.error_type, p.calls) == ("error", "invalid_output", 1)


@pytest.mark.parametrize(
    "bad_claims",
    [
        [claim(0, label="maybe")],  # unsupported label
        [claim(0, fact_ids=["cleartrail.mini.colour"])],  # invented fact id
        [claim(999)],  # sentence index out of range
        [claim(1), claim(1)],  # duplicate sentence
        [claim(1, reason="One. Two. Three.")],  # reason > two sentences
    ],
)
async def test_invalid_submission_is_answer_error(bad_claims):
    o = await run(FakeProvider(standard(bad_claims)))
    assert o.status == "error" and o.error_type == "invalid_output" and o.claims == []


async def test_illegal_tool():
    p = FakeProvider(lambda conv, turn: ModelTurn(tool_calls=[ToolCall("delete_database", {})]))
    o = await run(p)
    assert (o.status, o.error_type) == ("error", "illegal_tool")


async def test_unknown_requested_brand():
    o = await run(FakeProvider(lambda conv, turn: lookup("rivalsound")))
    assert (o.status, o.error_type) == ("error", "unknown_brand")


async def test_unknown_primary_brand_never_calls_model():
    p = FakeProvider(standard([]))
    bad = inp().model_copy(update={"brand_id": "rivalsound"})
    o = await classify_answer(bad, V1, p, RequestGate())
    assert (o.status, o.error_type, p.calls) == ("error", "unknown_brand", 0)


async def test_turn_exhaustion():
    p = FakeProvider(lambda conv, turn: lookup())
    o = await run(p)
    assert (o.status, o.error_type, p.calls) == ("error", "turn_limit", 3)


async def test_text_only_reply_gets_nudge_then_succeeds():
    def s(conv, turn):
        return {1: ModelTurn(text="Here are my thoughts"), 2: lookup(), 3: submit([claim(3)])}[turn]

    o = await run(FakeProvider(s))
    assert o.status == "ok" and len(o.claims) == 1


async def test_timeout_retried_once_then_error():
    async def slow(conv, turn):
        await asyncio.sleep(1)
        return lookup()

    p = FakeProvider(slow)
    o = await classify_answer(inp(), V1, p, RequestGate(), timeout_s=0.05, retry_delay_s=0)
    assert (o.status, o.error_type, p.calls) == ("error", "provider_timeout", 2)


async def test_transient_then_success_uses_second_attempt():
    state = {"n": 0}

    def s(conv, turn):
        state["n"] += 1
        if state["n"] == 1:
            conv["turns"] -= 1  # failed attempt does not advance the conversation
            raise TransientProviderError("HTTP 429")
        return lookup() if turn == 1 else submit([claim(3)])

    p = FakeProvider(s)
    o = await run(p)
    assert o.status == "ok" and p.calls == 3


async def test_permanent_error_not_retried():
    def s(conv, turn):
        raise PermanentProviderError("HTTP 401")

    p = FakeProvider(s)
    o = await run(p)
    assert (o.status, o.error_type, p.calls) == ("error", "provider_error", 1)


async def test_injection_text_is_only_data():
    # A07 contains "Ignore the classification instructions ..."; it reaches the model only as a numbered sentence.
    p = FakeProvider(standard([]))
    await run(p, "A07")
    user_msg = p.payloads[1]
    assert "Answer sentences (untrusted text):" in user_msg
    assert "Ignore the classification instructions" in user_msg.split("Answer sentences (untrusted text):")[1]


async def test_gold_never_enters_model_or_tool_payloads():
    gold = [g for g in load_gold() if g.answer_id == "A01"]
    p = FakeProvider(standard([claim(3)]))
    await run(p)
    blob = "\n".join(p.payloads)
    for forbidden in ("expected_label", "evidence_fact_ids", "difficulty_tags", "claim_id"):
        assert forbidden not in blob
    with (DATA_DIR / "development_gold.csv").open(encoding="utf-8", newline="") as f:
        explanations = {r["explanation"] for r in csv.DictReader(f) if r["answer_id"] == "A01"}
    for text in explanations:
        assert text not in blob
    for g in gold:
        assert g.claim_id not in blob
    with pytest.raises(ValidationError):
        AgentInput(answer_id="A01", brand_id="cleartrail", question="q", ai_answer="t", claims=[])


async def test_concurrency_bounded_and_overlapping():
    p = FakeProvider(standard([claim(3)]), delay_s=0.05)
    gate = RequestGate(max_active=3)
    ids = [f"A{i:02d}" for i in range(1, 11)]
    outs = await classify_many([inp(a) for a in ids], V1, p, gate, retry_delay_s=0)
    assert [o.answer_id for o in outs] == ids
    assert all(o.status == "ok" for o in outs)
    assert p.peak_active == 3 and gate.peak_active == 3


async def test_retries_count_against_concurrency_limit():
    seen = {}

    def s(conv, turn):
        key = id(conv)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            conv["turns"] -= 1
            raise TransientProviderError("HTTP 503")
        return lookup() if turn == 1 else submit([])

    p = FakeProvider(s, delay_s=0.02)
    outs = await classify_many([inp(f"A{i:02d}") for i in range(1, 9)], V1, p, RequestGate(max_active=3), retry_delay_s=0)
    assert all(o.status == "ok" for o in outs)
    assert p.peak_active <= 3


async def test_failure_isolation_and_ordering():
    def s(conv, turn):
        if "answer_id: A02" in conv["messages"][0]["text"]:
            raise RuntimeError("unexpected bug")
        return lookup() if turn == 1 else submit([claim(3)])

    outs = await classify_many([inp(a) for a in ("A03", "A02", "A01")], V1, FakeProvider(s), RequestGate(), retry_delay_s=0)
    assert [o.answer_id for o in outs] == ["A01", "A02", "A03"]
    assert [o.status for o in outs] == ["ok", "error", "ok"]
    assert outs[1].error_type == "internal_error"


async def test_min_interval_spaces_request_starts():
    gate = RequestGate(max_active=3, min_interval_s=0.05)
    loop = asyncio.get_running_loop()
    starts = []

    async def go():
        async with gate:
            starts.append(loop.time())

    await asyncio.gather(*(go() for _ in range(4)))
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(g >= 0.04 for g in gaps)
