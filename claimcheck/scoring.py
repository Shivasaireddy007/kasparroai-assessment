"""Deterministic scorer: exact span matching, extraction and joint P/R/F1, per-label and macro F1,
confusion matrix and missing/extra/wrong-label accounting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .data import GoldClaim
from .schemas import LABELS, AnswerOutcome

ROWS = ("supported", "contradicted", "unknown", "no_gold")
COLS = ("supported", "contradicted", "unknown", "missing_or_error")


def _div(n: int, d: int) -> float:
    return n / d if d else 0.0


def _prf(tp: int, pred: int, actual: int) -> dict[str, float]:
    return {"precision": _div(tp, pred), "recall": _div(tp, actual), "f1": _div(2 * tp, pred + actual)}


def check_outcomes(scheduled: Iterable[str], outcomes: Iterable[AnswerOutcome]) -> dict[str, AnswerOutcome]:
    """Exactly one terminal outcome per scheduled answer, nothing else."""
    scheduled = list(scheduled)
    by_id: dict[str, AnswerOutcome] = {}
    for o in outcomes:
        if o.answer_id in by_id:
            raise ValueError(f"duplicate outcome for {o.answer_id}")
        by_id[o.answer_id] = o
    missing = [a for a in scheduled if a not in by_id]
    extra = [a for a in by_id if a not in set(scheduled)]
    if missing or extra:
        raise ValueError(f"outcome mismatch: missing={missing} unscheduled={extra}")
    return by_id


def score(
    scheduled: Iterable[str],
    outcomes: Iterable[AnswerOutcome],
    gold: Iterable[GoldClaim],
    texts: Mapping[str, str],
) -> dict[str, Any]:
    scheduled = sorted(scheduled)
    by_id = check_outcomes(scheduled, outcomes)
    sched = set(scheduled)
    gold = [g for g in gold if g.answer_id in sched]

    gold_by_key = {(g.answer_id, g.start_char, g.end_char): g for g in gold}
    if len(gold_by_key) != len(gold):
        raise ValueError("duplicate gold spans")

    preds = []
    for aid in scheduled:
        o = by_id[aid]
        if o.status != "ok":
            continue
        for c in o.claims:
            if texts[aid][c.start_char : c.end_char] != c.quote:
                raise ValueError(f"prediction quote mismatch: {c.claim_id}")
            preds.append((aid, c))

    confusion = {r: {c: 0 for c in COLS} for r in ROWS}
    matched: set[tuple[str, int, int]] = set()
    M = C = 0
    unmatched_pred_ids: list[str] = []
    wrong_label: list[dict[str, str]] = []
    per_answer = {aid: {"G": 0, "P": 0, "M": 0, "C": 0} for aid in scheduled}

    for aid, c in preds:
        per_answer[aid]["P"] += 1
        key = (aid, c.start_char, c.end_char)
        g = gold_by_key.get(key)
        if g is None or key in matched:
            unmatched_pred_ids.append(c.claim_id)
            confusion["no_gold"][c.label] += 1
            continue
        matched.add(key)
        M += 1
        per_answer[aid]["M"] += 1
        confusion[g.expected_label][c.label] += 1
        if c.label == g.expected_label:
            C += 1
            per_answer[aid]["C"] += 1
        else:
            wrong_label.append({"claim_id": g.claim_id, "expected": g.expected_label, "predicted": c.label})

    missed_gold_ids = []
    for g in gold:
        per_answer[g.answer_id]["G"] += 1
        if (g.answer_id, g.start_char, g.end_char) not in matched:
            missed_gold_ids.append(g.claim_id)
            confusion[g.expected_label]["missing_or_error"] += 1

    G, P = len(gold), len(preds)
    per_label = {}
    for lab in LABELS:
        tp = confusion[lab][lab]
        pred_pos = sum(confusion[r][lab] for r in ROWS)
        actual = sum(confusion[lab][c] for c in COLS)
        per_label[lab] = _prf(tp, pred_pos, actual) | {"tp": tp, "predicted": pred_pos, "actual": actual}

    per_answer_scores = {
        aid: {
            **v,
            "status": by_id[aid].status,
            "extraction_f1": _div(2 * v["M"], v["P"] + v["G"]),
            "joint_f1": _div(2 * v["C"], v["P"] + v["G"]),
        }
        for aid, v in per_answer.items()
    }

    return {
        "counts": {
            "answers": len(scheduled),
            "answer_errors": sum(1 for o in by_id.values() if o.status != "ok"),
            "G": G,
            "P": P,
            "M": M,
            "C": C,
        },
        "extraction": _prf(M, P, G),
        "joint": _prf(C, P, G),
        "gold_coverage": _div(C, G),
        "per_label": per_label,
        "macro_f1": sum(per_label[l]["f1"] for l in LABELS) / len(LABELS),
        "confusion": confusion,
        "missed_gold_ids": missed_gold_ids,
        "unmatched_prediction_ids": unmatched_pred_ids,
        "wrong_label": wrong_label,
        "per_answer": per_answer_scores,
    }


def check_run_summary(scheduled: Iterable[str], outcomes: Iterable[AnswerOutcome]) -> dict[str, Any]:
    """Check split: counts only; all quality fields stay null."""
    scheduled = sorted(scheduled)
    by_id = check_outcomes(scheduled, outcomes)
    P = sum(len(o.claims) for o in by_id.values() if o.status == "ok")
    return {
        "counts": {
            "answers": len(scheduled),
            "answer_errors": sum(1 for o in by_id.values() if o.status != "ok"),
            "G": None,
            "P": P,
            "M": None,
            "C": None,
        },
        "extraction": None,
        "joint": None,
        "gold_coverage": None,
        "per_label": None,
        "macro_f1": None,
        "confusion": None,
    }
