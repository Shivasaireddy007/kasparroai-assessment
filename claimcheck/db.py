"""PostgreSQL persistence with parameterized SQL (psycopg 3, async)."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .schemas import AnswerOutcome

SCHEMA = (Path(__file__).resolve().parent / "schema.sql").read_text(encoding="utf-8")


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


async def connect(url: str | None = None) -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(url or database_url(), row_factory=dict_row, autocommit=True)


async def init_schema(conn: psycopg.AsyncConnection) -> None:
    await conn.execute(SCHEMA)


async def create_run(conn: psycopg.AsyncConnection, meta: dict[str, Any]) -> uuid.UUID:
    run_id = uuid.uuid4()
    await conn.execute(
        """INSERT INTO runs (run_id, split, agent_version, prompt_version, dataset_version, evaluator_version,
               fact_version, provider, model, settings, file_hashes, source_revision, status)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'running')""",
        (
            run_id,
            meta["split"],
            meta["agent_version"],
            meta["prompt_version"],
            meta["dataset_version"],
            meta["evaluator_version"],
            meta["fact_version"],
            meta["provider"],
            meta["model"],
            Jsonb(meta["settings"]),
            Jsonb(meta["file_hashes"]),
            meta.get("source_revision"),
        ),
    )
    return run_id


async def save_outcomes(
    conn: psycopg.AsyncConnection,
    run_id: uuid.UUID,
    outcomes: list[AnswerOutcome],
    answer_scores: dict[str, Any] | None = None,
    claim_matches: dict[str, dict[str, Any]] | None = None,
) -> None:
    """One terminal row per answer and one row per accepted prediction, in a single transaction."""
    async with conn.transaction():
        async with conn.cursor() as cur:
            for o in outcomes:
                await cur.execute(
                    """INSERT INTO answer_results (run_id, answer_id, status, error_type, error_message, latency_ms,
                           trace_id, claim_count, scores)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        run_id,
                        o.answer_id,
                        o.status,
                        o.error_type,
                        o.error_message,
                        o.latency_ms,
                        o.trace_id,
                        len(o.claims),
                        Jsonb(answer_scores[o.answer_id]) if answer_scores else None,
                    ),
                )
                for c in o.claims:
                    m = (claim_matches or {}).get(c.claim_id, {})
                    await cur.execute(
                        """INSERT INTO claim_results (run_id, answer_id, claim_id, start_char, end_char, quote, label,
                               fact_ids, reason, matched_gold, expected_label, label_correct)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (
                            run_id,
                            o.answer_id,
                            c.claim_id,
                            c.start_char,
                            c.end_char,
                            c.quote,
                            c.label,
                            Jsonb(c.fact_ids),
                            c.reason,
                            m.get("matched_gold"),
                            m.get("expected_label"),
                            m.get("label_correct"),
                        ),
                    )


async def finish_run(
    conn: psycopg.AsyncConnection,
    run_id: uuid.UUID,
    status: str,
    counts: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    error: str | None = None,
) -> None:
    await conn.execute(
        """UPDATE runs SET status = %s, counts = %s, metrics = %s, error = %s, finished_at = now()
           WHERE run_id = %s""",
        (status, Jsonb(counts) if counts is not None else None, Jsonb(metrics) if metrics is not None else None, error, run_id),
    )


async def get_run(conn: psycopg.AsyncConnection, run_id: uuid.UUID) -> dict[str, Any] | None:
    async with conn.cursor() as cur:
        await cur.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,))
        run = await cur.fetchone()
        if run is None:
            return None
        await cur.execute(
            """SELECT a.answer_id, a.status, a.error_type, a.error_message, a.latency_ms, a.trace_id,
                      a.claim_count, a.scores,
                      COALESCE(
                        jsonb_agg(
                          jsonb_build_object(
                            'claim_id', c.claim_id, 'start_char', c.start_char, 'end_char', c.end_char,
                            'quote', c.quote, 'label', c.label, 'fact_ids', c.fact_ids, 'reason', c.reason,
                            'matched_gold', c.matched_gold, 'expected_label', c.expected_label,
                            'label_correct', c.label_correct)
                          ORDER BY c.start_char, c.end_char
                        ) FILTER (WHERE c.claim_id IS NOT NULL),
                        '[]'::jsonb) AS claims
               FROM answer_results a
               LEFT JOIN claim_results c ON c.run_id = a.run_id AND c.answer_id = a.answer_id
               WHERE a.run_id = %s
               GROUP BY a.run_id, a.answer_id
               ORDER BY a.answer_id""",
            (run_id,),
        )
        answers = await cur.fetchall()
    run["answers"] = answers
    return run
