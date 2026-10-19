"""Evaluation runs: run one agent version over a fixed CSV split, persist, score (development only),
write a JSON report and attach Langfuse scores.

CLI: uv run python -m claimcheck.runner --version v1 --split development
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import db
from .agent import (
    AGENT_VERSIONS,
    MAX_ACTIVE_REQUESTS,
    MAX_ATTEMPTS_PER_TURN,
    MAX_TURNS,
    REQUEST_TIMEOUT_S,
    AgentInput,
    RequestGate,
    classify_many,
)
from .data import DATASET_VERSION, EVALUATOR_VERSION, FACT_VERSION, file_hashes, load_answers, load_gold
from .providers import Provider
from .schemas import AnswerOutcome
from .scoring import check_run_summary, score
from .tracing import NOOP_TRACER, Tracer

REPORT_DIR = Path(__file__).resolve().parent.parent / "runs"
RETRY_DELAY_S = 2.0


def source_revision() -> str | None:
    root = Path(__file__).resolve().parent.parent
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "claimcheck"], cwd=root, capture_output=True, text=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return None


def run_settings(provider: Provider, min_interval_s: float) -> dict[str, Any]:
    return {
        "max_turns": MAX_TURNS,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "max_attempts_per_turn": MAX_ATTEMPTS_PER_TURN,
        "retry_delay_s": RETRY_DELAY_S,
        "max_active_requests": MAX_ACTIVE_REQUESTS,
        "min_interval_s": min_interval_s,
        "provider": provider.settings(),
    }


def claim_matches(outcomes: list[AnswerOutcome], gold: list[Any]) -> dict[str, dict[str, Any]]:
    by_span = {(g.answer_id, g.start_char, g.end_char): g for g in gold}
    out = {}
    for o in outcomes:
        for c in o.claims:
            g = by_span.get((o.answer_id, c.start_char, c.end_char))
            out[c.claim_id] = {
                "matched_gold": g is not None,
                "expected_label": g.expected_label if g else None,
                "label_correct": (g.expected_label == c.label) if g else None,
            }
    return out


def post_langfuse_scores(tracer: Tracer, run_id: str, outcomes: list[AnswerOutcome], metrics: dict[str, Any], matches: dict[str, dict[str, Any]]) -> None:
    missed_by_answer: dict[str, list[str]] = {}
    for gid in metrics["missed_gold_ids"]:
        missed_by_answer.setdefault(gid.split(":")[0], []).append(gid)
    extra = set(metrics["unmatched_prediction_ids"])
    for o in outcomes:
        if not o.trace_id:
            continue
        pa = metrics["per_answer"][o.answer_id]
        meta = {
            "missed_gold_ids": missed_by_answer.get(o.answer_id, []),
            "extra_prediction_ids": [c.claim_id for c in o.claims if c.claim_id in extra],
            "G": pa["G"], "P": pa["P"], "M": pa["M"], "C": pa["C"], "status": o.status,
        }
        tracer.score(o.trace_id, "extraction_f1", pa["extraction_f1"], metadata=meta)
        tracer.score(o.trace_id, "joint_f1", pa["joint_f1"], metadata=meta)
        for c in o.claims:
            m = matches[c.claim_id]
            if m["matched_gold"]:
                tracer.score(
                    o.trace_id, "label_correct", 1.0 if m["label_correct"] else 0.0, comment=c.claim_id,
                    metadata={"span_id": c.claim_id, "expected": m["expected_label"], "predicted": c.label},
                )
    for name in ("extraction", "joint"):
        tracer.score_run(run_id, f"run_{name}_f1", metrics[name]["f1"])
    tracer.score_run(run_id, "run_macro_f1", metrics["macro_f1"])


async def run_evaluation(
    agent_version: str,
    split: str,
    provider: Provider,
    *,
    tracer: Tracer = NOOP_TRACER,
    database_url: str | None = None,
    report_dir: Path | None = REPORT_DIR,
    min_interval_s: float | None = None,
    retry_delay_s: float = RETRY_DELAY_S,
    answer_ids: list[str] | None = None,
) -> dict[str, Any]:
    if agent_version not in AGENT_VERSIONS:
        raise ValueError(f"unknown agent_version {agent_version!r}")
    if split not in ("development", "check"):
        raise ValueError(f"unknown split {split!r}")
    version = AGENT_VERSIONS[agent_version]
    if min_interval_s is None:
        min_interval_s = float(os.environ.get("LLM_MIN_INTERVAL_S") or 0)

    answers = [a for a in load_answers() if a.split == split and (answer_ids is None or a.answer_id in answer_ids)]
    inputs = [AgentInput(answer_id=a.answer_id, brand_id=a.brand_id, question=a.question, ai_answer=a.ai_answer) for a in answers]
    meta = {
        "split": split,
        "agent_version": agent_version,
        "prompt_version": f"{agent_version}-{hashlib.sha256(version.prompt.encode()).hexdigest()[:12]}",
        "dataset_version": DATASET_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "fact_version": FACT_VERSION,
        "provider": provider.name,
        "model": provider.model,
        "settings": run_settings(provider, min_interval_s),
        "file_hashes": file_hashes(),
        "source_revision": source_revision(),
    }

    conn = await db.connect(database_url)
    try:
        await db.init_schema(conn)
        run_id = await db.create_run(conn, meta)
        started_at = datetime.now(UTC)
        t0 = time.monotonic()
        report: dict[str, Any] = {"run_id": str(run_id), "status": "running", **meta, "started_at": started_at.isoformat()}
        outcomes: list[AnswerOutcome] = []
        matches: dict[str, dict[str, Any]] = {}
        try:
            gate = RequestGate(MAX_ACTIVE_REQUESTS, min_interval_s)
            trace_meta = {k: meta[k] for k in ("split", "agent_version", "prompt_version", "dataset_version", "evaluator_version", "fact_version", "provider", "model")}
            trace_meta["run_id"] = str(run_id)
            outcomes = await classify_many(inputs, version, provider, gate, tracer=tracer, trace_meta=trace_meta, retry_delay_s=retry_delay_s)
            texts = {a.answer_id: a.ai_answer for a in answers}
            scheduled = [a.answer_id for a in answers]

            if split == "development":
                gold = [g for g in load_gold() if g.answer_id in set(scheduled)]
                metrics = score(scheduled, outcomes, gold, texts)
                matches = claim_matches(outcomes, gold)
                answer_scores = metrics["per_answer"]
                counts = metrics["counts"]
                stored_metrics = {k: v for k, v in metrics.items() if k not in ("counts", "per_answer")}
            else:
                summary = check_run_summary(scheduled, outcomes)
                answer_scores, counts = None, summary["counts"]
                stored_metrics = None

            await db.save_outcomes(conn, run_id, outcomes, answer_scores, matches)
            await db.finish_run(conn, run_id, "completed", counts, stored_metrics)
            if split == "development":
                post_langfuse_scores(tracer, str(run_id), outcomes, metrics, matches)
            report.update(
                status="completed",
                counts=counts,
                metrics=stored_metrics,
                per_answer_scores=answer_scores,
                gate_peak_active_requests=gate.peak_active,
            )
        except Exception as e:
            await db.finish_run(conn, run_id, "failed", None, None, error=type(e).__name__)
            report.update(status="failed", error=f"{type(e).__name__}: {e}")
            raise
        finally:
            tracer.flush()
            report["duration_s"] = round(time.monotonic() - t0, 1)
            report["finished_at"] = datetime.now(UTC).isoformat()
            report["answers"] = [
                {**o.model_dump(), "claims": [{**c.model_dump(), **matches.get(c.claim_id, {})} for c in o.claims]}
                for o in outcomes
            ]
            if report_dir is not None:
                report_dir.mkdir(parents=True, exist_ok=True)
                path = report_dir / f"{split}_{agent_version}_{started_at:%Y%m%dT%H%M%SZ}_{str(run_id)[:8]}.json"
                path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
                report["report_path"] = str(path)
        return report
    finally:
        await conn.close()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    from .providers import provider_from_env
    from .tracing import tracer_from_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=sorted(AGENT_VERSIONS))
    ap.add_argument("--split", required=True, choices=["development", "check"])
    ap.add_argument("--answers", nargs="*", help="optional subset of answer_ids (smoke tests only)")
    args = ap.parse_args()
    report = asyncio.run(
        run_evaluation(args.version, args.split, provider_from_env(), tracer=tracer_from_env(), answer_ids=args.answers)
    )
    print(json.dumps({k: report.get(k) for k in ("run_id", "status", "counts", "duration_s", "report_path")}, indent=2))
    m = report.get("metrics")
    if m:
        print(
            f"extraction P/R/F1 {m['extraction']['precision']:.3f}/{m['extraction']['recall']:.3f}/{m['extraction']['f1']:.3f} | "
            f"joint P/R/F1 {m['joint']['precision']:.3f}/{m['joint']['recall']:.3f}/{m['joint']['f1']:.3f} | macro F1 {m['macro_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
