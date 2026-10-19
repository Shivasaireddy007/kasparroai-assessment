"""Thin FastAPI layer over the agent, runner and database.

Run: uv run uvicorn claimcheck.api:app
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, field_validator

from . import db
from .agent import AGENT_VERSIONS, AgentInput, RequestGate, classify_answer
from .facts import known_brands
from .providers import Provider
from .runner import run_evaluation
from .schemas import Claim
from .tracing import NOOP_TRACER, Tracer

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # gold / claim-list fields are rejected with 422
    answer_id: str
    brand_id: str
    question: str
    ai_answer: str
    agent_version: str

    @field_validator("agent_version")
    @classmethod
    def _known_version(cls, v: str) -> str:
        if v not in AGENT_VERSIONS:
            raise ValueError(f"agent_version must be one of {sorted(AGENT_VERSIONS)}")
        return v


class ClassifyResponse(BaseModel):
    answer_id: str
    agent_version: str
    claims: list[Claim]


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_version: str
    split: Literal["development", "check"]

    @field_validator("agent_version")
    @classmethod
    def _known_version(cls, v: str) -> str:
        return ClassifyRequest._known_version(v)


def _error(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"type": kind, "message": message}})


def _default_provider() -> Provider:
    from .providers import provider_from_env

    return provider_from_env()


def _default_tracer() -> Tracer:
    from .tracing import tracer_from_env

    return tracer_from_env()


def create_app(
    provider_factory: Callable[[], Provider] = _default_provider,
    tracer_factory: Callable[[], Tracer] = _default_tracer,
    database_url: str | None = None,
    classify_kwargs: dict[str, Any] | None = None,
    run_kwargs: dict[str, Any] | None = None,
) -> FastAPI:
    """Factories and kwargs are injection points for tests (fake provider, test database, no delays)."""
    app = FastAPI(title="Claim extraction and classification")

    @app.post("/classify", response_model=ClassifyResponse)
    async def classify(req: ClassifyRequest) -> Any:
        if req.brand_id not in known_brands():
            return _error(404, "unknown_brand", f"unknown primary brand_id {req.brand_id!r}")
        try:
            provider = provider_factory()
        except Exception:
            return _error(502, "provider_error", "provider is not configured")
        inp = AgentInput(answer_id=req.answer_id, brand_id=req.brand_id, question=req.question, ai_answer=req.ai_answer)
        outcome = await classify_answer(
            inp, AGENT_VERSIONS[req.agent_version], provider, RequestGate(), tracer=tracer_factory(),
            trace_meta={"split": "api", "provider": provider.name, "model": provider.model},
            **(classify_kwargs or {}),
        )
        if outcome.status == "error":
            status = 504 if outcome.error_type == "provider_timeout" else 502
            return _error(status, outcome.error_type or "provider_error", outcome.error_message or "")
        return ClassifyResponse(answer_id=req.answer_id, agent_version=req.agent_version, claims=outcome.claims)

    @app.post("/evaluation-runs", status_code=201)
    async def create_run(req: RunRequest) -> Any:
        try:
            report = await run_evaluation(
                req.agent_version, req.split, provider_factory(), tracer=tracer_factory(), database_url=database_url,
                **(run_kwargs or {}),
            )
        except Exception as e:
            return _error(500, "run_failed", type(e).__name__)
        return {
            "run_id": report["run_id"],
            "status": report["status"],
            "split": report["split"],
            "agent_version": report["agent_version"],
            "counts": report.get("counts"),
            "metrics": report.get("metrics"),
            "report_path": report.get("report_path"),
        }

    @app.get("/evaluation-runs/{run_id}")
    async def get_run(run_id: uuid.UUID) -> Any:
        conn = await db.connect(database_url)
        try:
            run = await db.get_run(conn, run_id)
        finally:
            await conn.close()
        if run is None:
            return _error(404, "not_found", f"run {run_id} not found")
        return run

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        return _error(500, "internal_error", type(exc).__name__)

    return app


app = create_app()
