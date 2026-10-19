"""CSV loading. Files are read with the csv module and UTF-8; ai_answer is never altered."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASET_VERSION = "long-brand-answers-50-v4"
FACT_VERSION = "speaker-catalogue-v4"
EVALUATOR_VERSION = "sentence-spans-and-labels-v4"


@dataclass(frozen=True)
class Answer:
    answer_id: str
    split: str
    brand_id: str
    question: str
    ai_answer: str


@dataclass(frozen=True)
class Fact:
    fact_id: str
    brand_id: str
    model_id: str
    field: str
    value: Any
    value_type: str
    unit: str | None
    condition: str | None
    valid_from: date
    valid_to: date


@dataclass(frozen=True)
class GoldClaim:
    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    expected_label: str
    evidence_fact_ids: tuple[str, ...]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_value(raw: str, value_type: str) -> Any:
    if value_type == "number":
        return int(raw) if raw.lstrip("-").isdigit() else float(raw)
    if value_type == "boolean":
        if raw == "true":
            return True
        if raw == "false":
            return False
        raise ValueError(f"invalid boolean text: {raw!r}")
    if value_type == "string":
        return raw
    raise ValueError(f"unknown value_type: {value_type!r}")


def load_answers(path: Path | None = None) -> list[Answer]:
    rows = _read(path or DATA_DIR / "answers.csv")
    return [Answer(r["answer_id"], r["split"], r["brand_id"], r["question"], r["ai_answer"]) for r in rows]


def load_facts(path: Path | None = None) -> list[Fact]:
    out = []
    for r in _read(path or DATA_DIR / "brand_facts.csv"):
        out.append(
            Fact(
                fact_id=r["fact_id"],
                brand_id=r["brand_id"],
                model_id=r["model_id"],
                field=r["field"],
                value=parse_value(r["value"], r["value_type"]),
                value_type=r["value_type"],
                unit=r["unit"] or None,
                condition=r["condition"] or None,
                valid_from=date.fromisoformat(r["valid_from"]),
                valid_to=date.fromisoformat(r["valid_to"]),
            )
        )
    return out


def load_gold(path: Path | None = None) -> list[GoldClaim]:
    out = []
    for r in _read(path or DATA_DIR / "development_gold.csv"):
        ids = tuple(x for x in r["evidence_fact_ids"].split("|") if x)
        out.append(
            GoldClaim(r["answer_id"], r["claim_id"], int(r["start_char"]), int(r["end_char"]), r["quote"], r["expected_label"], ids)
        )
    return out


def file_hashes(data_dir: Path | None = None) -> dict[str, str]:
    d = data_dir or DATA_DIR
    return {
        name: hashlib.sha256((d / name).read_bytes()).hexdigest()
        for name in ("answers.csv", "brand_facts.csv", "development_gold.csv")
    }
