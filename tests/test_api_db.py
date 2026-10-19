"""API validation, PostgreSQL persistence and joined retrieval, using a fake provider and a test database."""

import asyncio
import os
import uuid

import httpx
import psycopg
import pytest

from claimcheck import db
from claimcheck.api import create_app
from claimcheck.providers import MalformedOutputError
from claimcheck.tracing import NOOP_TRACER

from .fakes import FakeProvider, claim, lookup, submit

TEST_DB = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost:5432/claimcheck_test")


def _db_available() -> bool:
    try:
        psycopg.connect(TEST_DB, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason=f"test database unavailable: {TEST_DB}")


def scripted(conv, turn):
    """A02 -> invalid output; A03 -> valid empty; A04 -> timeout; others -> two claims."""
    user = conv["messages"][0]["text"]
    if "answer_id: A02" in user:
        raise MalformedOutputError("no candidates")
    if turn == 1:
        return lookup()
    if "answer_id: A03" in user:
        return submit([])
    return submit([claim(4, "contradicted", ["cleartrail.mini.warranty_india"]), claim(3, "supported")])


async def timeout_script(conv, turn):
    await asyncio.sleep(1)
    return lookup()


@pytest.fixture(autouse=True)
async def clean_db():
    conn = await db.connect(TEST_DB)
    await conn.execute("DROP TABLE IF EXISTS claim_results, answer_results, runs CASCADE")
    await db.init_schema(conn)
    await conn.close()


def client(script=scripted, tmp_path=None, timeout_s=15.0):
    app = create_app(
        provider_factory=lambda: FakeProvider(script),
        tracer_factory=lambda: NOOP_TRACER,  # never send test traces to the real Langfuse project
        database_url=TEST_DB,
        classify_kwargs={"retry_delay_s": 0, "timeout_s": timeout_s},
        run_kwargs={"retry_delay_s": 0, "min_interval_s": 0, "report_dir": tmp_path},
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


A01_BODY = {
    "answer_id": "A01",
    "brand_id": "cleartrail",
    "question": "q",
    "ai_answer": "Advice first. ClearTrail Mini costs INR 1,299. Plus costs more. Mini has no Wi-Fi. Fourth claim here.",
    "agent_version": "v1",
}


async def test_classify_200_with_validated_claims():
    async with client() as c:
        r = await c.post("/classify", json=A01_BODY)
    assert r.status_code == 200
    claims = r.json()["claims"]
    assert [x["start_char"] for x in claims] == sorted(x["start_char"] for x in claims)
    for x in claims:
        assert A01_BODY["ai_answer"][x["start_char"] : x["end_char"]] == x["quote"]


@pytest.mark.parametrize(
    "patch",
    [
        {"expected_label": "supported"},
        {"claims": []},
        {"gold": []},
        {"agent_version": "v9"},
        {"ai_answer": None},
    ],
)
async def test_classify_422(patch):
    async with client() as c:
        r = await c.post("/classify", json={**A01_BODY, **patch})
    assert r.status_code == 422


async def test_classify_404_unknown_primary_brand():
    async with client() as c:
        r = await c.post("/classify", json={**A01_BODY, "brand_id": "rivalsound"})
    assert r.status_code == 404 and r.json()["error"]["type"] == "unknown_brand"


async def test_classify_502_invalid_output():
    async with client(script=lambda conv, turn: (_ for _ in ()).throw(MalformedOutputError("bad"))) as c:
        r = await c.post("/classify", json=A01_BODY)
    assert r.status_code == 502 and r.json()["error"]["type"] == "invalid_output"


async def test_classify_504_timeout():
    async with client(script=timeout_script, timeout_s=0.05) as c:
        r = await c.post("/classify", json=A01_BODY)
    assert r.status_code == 504 and r.json()["error"]["type"] == "provider_timeout"


async def test_development_run_persists_and_joins(tmp_path):
    async with client(tmp_path=tmp_path) as c:
        r = await c.post("/evaluation-runs", json={"agent_version": "v1", "split": "development"})
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "completed"
        assert body["counts"]["answers"] == 40 and body["counts"]["G"] == 376
        assert body["counts"]["answer_errors"] == 1
        assert body["metrics"]["joint"]["f1"] is not None

        g = await c.get(f"/evaluation-runs/{body['run_id']}")
    assert g.status_code == 200
    run = g.json()
    answers = run["answers"]
    assert [a["answer_id"] for a in answers] == [f"A{i:02d}" for i in range(1, 41)]
    by = {a["answer_id"]: a for a in answers}
    assert by["A02"]["status"] == "error" and by["A02"]["error_type"] == "invalid_output" and by["A02"]["claims"] == []
    assert by["A03"]["status"] == "ok" and by["A03"]["claims"] == [] and by["A03"]["claim_count"] == 0
    a01 = by["A01"]["claims"]
    assert [x["start_char"] for x in a01] == sorted(x["start_char"] for x in a01)
    # A01 sentences 3 and 4 are gold claims: 3 is supported (correct), 4 is contradicted (correct).
    assert all(x["matched_gold"] for x in a01) and all(x["label_correct"] for x in a01)
    assert run["file_hashes"]["answers.csv"] and run["settings"]["max_turns"] == 3
    assert list(tmp_path.glob("development_v1_*.json"))


async def test_check_run_keeps_quality_null(tmp_path):
    async with client(tmp_path=tmp_path) as c:
        r = await c.post("/evaluation-runs", json={"agent_version": "v1", "split": "check"})
        body = r.json()
        assert r.status_code == 201 and body["metrics"] is None
        assert body["counts"]["answers"] == 10 and body["counts"]["G"] is None
        g = (await c.get(f"/evaluation-runs/{body['run_id']}")).json()
    assert g["metrics"] is None
    for a in g["answers"]:
        assert a["scores"] is None
        for x in a["claims"]:
            assert x["matched_gold"] is None and x["label_correct"] is None


async def test_run_request_validation():
    async with client() as c:
        assert (await c.post("/evaluation-runs", json={"agent_version": "v1", "split": "final"})).status_code == 422
        assert (await c.post("/evaluation-runs", json={"agent_version": "v1", "split": "check", "gold": 1})).status_code == 422


async def test_get_run_404_and_422():
    async with client() as c:
        assert (await c.get(f"/evaluation-runs/{uuid.uuid4()}")).status_code == 404
        assert (await c.get("/evaluation-runs/not-a-uuid")).status_code == 422


async def test_unique_constraints():
    conn = await db.connect(TEST_DB)
    meta = {k: "x" for k in ("agent_version", "prompt_version", "dataset_version", "evaluator_version", "fact_version", "provider", "model")}
    run_id = await db.create_run(conn, {**meta, "split": "check", "settings": {}, "file_hashes": {}})
    await conn.execute("INSERT INTO answer_results (run_id, answer_id, status) VALUES (%s, 'A01', 'ok')", (run_id,))
    with pytest.raises(psycopg.errors.UniqueViolation):
        await conn.execute("INSERT INTO answer_results (run_id, answer_id, status) VALUES (%s, 'A01', 'ok')", (run_id,))
    ins = "INSERT INTO claim_results (run_id, answer_id, claim_id, start_char, end_char, quote, label, fact_ids, reason) VALUES (%s, 'A01', %s, 0, 5, 'q', 'unknown', '[]', 'r')"
    await conn.execute(ins, (run_id, "A01:0:5"))
    with pytest.raises(psycopg.errors.UniqueViolation):
        await conn.execute(ins, (run_id, "A01:0:5b"))
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        await conn.execute(
            "INSERT INTO claim_results (run_id, answer_id, claim_id, start_char, end_char, quote, label, fact_ids, reason) VALUES (%s, 'A99', 'x', 0, 5, 'q', 'unknown', '[]', 'r')",
            (run_id,),
        )
    await conn.close()
