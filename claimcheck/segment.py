"""Deterministic sentence segmentation with exact code-point offsets into the original text."""

from __future__ import annotations

import re
from dataclasses import dataclass

# A sentence ends at . ! or ? followed by whitespace (or end of text). Decimal points such as
# "0.35" or "5.3" are never followed by whitespace, so they do not end a sentence.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Sentence:
    index: int
    start: int
    end: int
    text: str


def split_sentences(text: str) -> list[Sentence]:
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    tail_end = len(text.rstrip())
    if start < tail_end:
        spans.append((start, tail_end))
    out = []
    for s, e in spans:
        # Trim any leading whitespace (e.g. text beginning with a newline).
        while s < e and text[s].isspace():
            s += 1
        if s < e:
            out.append(Sentence(len(out), s, e, text[s:e]))
    return out
