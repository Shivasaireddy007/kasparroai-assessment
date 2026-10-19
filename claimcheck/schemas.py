"""Output models and validation of predicted claims against the source text."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Label = Literal["supported", "contradicted", "unknown"]
LABELS: tuple[str, ...] = ("supported", "contradicted", "unknown")


class Claim(BaseModel):
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: Label
    fact_ids: list[str]
    reason: str


class AnswerOutcome(BaseModel):
    """Terminal outcome for one scheduled answer. status=error means no accepted predictions."""

    answer_id: str
    status: Literal["ok", "error"]
    claims: list[Claim] = []
    error_type: str | None = None
    error_message: str | None = None
    latency_ms: int | None = None
    trace_id: str | None = None


class OutputValidationError(ValueError):
    """Agent output failed validation; becomes an answer-level error."""


def _sentence_count(reason: str) -> int:
    from .segment import split_sentences

    return len(split_sentences(reason))


def validate_claims(answer_id: str, text: str, claims: list[Claim], fact_ids: frozenset[str]) -> list[Claim]:
    """Check spans, quotes, IDs, labels, fact IDs and overlap. Returns claims sorted by start_char."""
    seen: set[str] = set()
    for c in claims:
        if not (0 <= c.start_char < c.end_char <= len(text)):
            raise OutputValidationError(f"span out of bounds: {c.start_char}:{c.end_char}")
        if text[c.start_char : c.end_char] != c.quote:
            raise OutputValidationError(f"quote does not equal source slice at {c.start_char}:{c.end_char}")
        if c.quote != c.quote.strip():
            raise OutputValidationError("quote has leading or trailing whitespace")
        expected_id = f"{answer_id}:{c.start_char}:{c.end_char}"
        if c.claim_id != expected_id:
            raise OutputValidationError(f"claim_id {c.claim_id!r} != {expected_id!r}")
        if c.claim_id in seen:
            raise OutputValidationError(f"duplicate claim_id {c.claim_id}")
        seen.add(c.claim_id)
        if c.label not in LABELS:
            raise OutputValidationError(f"unsupported label {c.label!r}")
        bad = [f for f in c.fact_ids if f not in fact_ids]
        if bad:
            raise OutputValidationError(f"unknown fact_ids: {bad}")
        if _sentence_count(c.reason) > 2:
            raise OutputValidationError("reason longer than two sentences")
    ordered = sorted(claims, key=lambda c: c.start_char)
    for a, b in zip(ordered, ordered[1:]):
        if b.start_char < a.end_char:
            raise OutputValidationError(f"overlapping spans {a.claim_id} and {b.claim_id}")
    return ordered
