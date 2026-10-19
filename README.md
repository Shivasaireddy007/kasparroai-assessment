# Claim extraction and classification agent

Kasparro FSE I assessment. An agent reads a long AI-written answer about speaker products, finds its
factual claim sentences, and labels each one `supported`, `contradicted` or `unknown` against the supplied
catalogue (`brand_facts.csv`). A deterministic harness scores extraction and labels; runs are stored in
PostgreSQL, traced in Langfuse, and exposed through a thin FastAPI service.

Results, the v1 → v2 change and the write-up are in [RESULTS.md](RESULTS.md).

## Requirements

- macOS/Linux, [uv](https://docs.astral.sh/uv/) (installs Python 3.12 and pinned dependencies from `uv.lock`)
- PostgreSQL 14+ (developed on Postgres.app 18)
- A Gemini API key (free tier is enough: `gemini-3.5-flash-lite` allows 15 RPM / 500 requests per day)
- A Langfuse project (free cloud tier)

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv is not installed
uv sync                                             # Python 3.12 + dependencies (see uv.lock)

createdb claimcheck                                 # application database
createdb claimcheck_test                            # used only by tests (tables are dropped/recreated)

cp .env.example .env                                # then fill in keys; .env is git-ignored
```

`.env` settings (see `.env.example`):

| Variable | Meaning |
|---|---|
| `DATABASE_URL` | e.g. `postgresql://localhost:5432/claimcheck` |
| `TEST_DATABASE_URL` | defaults to `postgresql://localhost:5432/claimcheck_test` |
| `LLM_PROVIDER`, `LLM_MODEL` | `gemini`, `gemini-3.5-flash-lite` |
| `GEMINI_API_KEY` | your key |
| `LLM_THINKING_LEVEL` | empty = provider default |
| `LLM_MIN_INTERVAL_S` | minimum seconds between request starts (free-tier pacing), default 6 |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | Langfuse project keys and region host |

The schema is created automatically (`claimcheck/schema.sql`) on the first run.

## Run

```bash
# Evaluation runs (writes runs/<split>_<version>_<timestamp>_<id>.json, stores rows in Postgres, traces in Langfuse)
uv run python -m claimcheck.runner --version v1 --split development
uv run python -m claimcheck.runner --version v2 --split development
uv run python -m claimcheck.runner --version v1 --split check
uv run python -m claimcheck.runner --version v2 --split check

# API
uv run uvicorn claimcheck.api:app
#   POST /classify           {answer_id, brand_id, question, ai_answer, agent_version}
#   POST /evaluation-runs    {agent_version, split}
#   GET  /evaluation-runs/{run_id}
```

## Test

```bash
uv run pytest -q
```

Tests use a scripted fake provider and the no-op tracer; no API key or network is needed. API/DB tests
need the `claimcheck_test` database and are skipped (and reported as skipped) if it is unreachable.

## Design

```
claimcheck/
  data.py       CSV loading (csv module, UTF-8, typed values, booleans parsed as text), file hashes
  segment.py    deterministic sentence splitter with exact code-point offsets
  facts.py      read-only get_brand_facts(brand_id) tool
  providers.py  Gemini adapter + error classification (transient / permanent / malformed)
  agent.py      tool loop, request gate, retries, validation, per-answer isolation
  prompts/      v1.md, v2.md (versioned prompts)
  schemas.py    output models and claim validation
  scoring.py    exact-span scorer and metrics
  runner.py     evaluation runs: agent → scorer → Postgres → JSON report → Langfuse scores
  db.py, schema.sql   PostgreSQL persistence (psycopg 3, parameterized SQL)
  tracing.py    Langfuse tracer (one trace per answer) and no-op tracer
  api.py        FastAPI routes
tests/          unit, fake-provider, API and DB tests
data/           issued CSVs, unchanged, with SHA256SUMS
runs/           saved live-run reports
```

**Extraction by sentence index.** Python splits `ai_answer` into numbered sentences with exact
offsets (the splitter reproduces all 376 development gold spans; this is checked by a test). The model
sees only the numbered sentences and returns sentence indices; Python derives `start_char`, `end_char`,
`quote` and `claim_id`. The model never counts characters, and the output stays short enough for the
15-second request timeout. The model still decides which sentences are claims. Gold spans are never used
on the inference path.

**Tool loop.** The model gets two tools: `get_brand_facts(brand_id)` (brand restricted to the catalogue
brands) and `submit_claims(claims)`. Turn 1 is normally a fact lookup, turn 2 the submission; at most
three model turns per answer. The executor validates tool names and arguments and executes lookups
itself. An unknown tool, an unknown brand, invalid output (bad index, label, fact ID, reason > 2
sentences, duplicates, overlap), a provider failure or turn exhaustion is an answer-level error with no
accepted predictions. Errors are kept, never converted to `unknown`.

**Reliability.** Answers run concurrently; a shared gate allows at most 3 active provider requests
(retries included) and spaces request starts for the free-tier quota. Each request has a 15 s timeout.
Timeouts, 429 and 5xx are retried once after a 2 s delay; malformed output and other 4xx are not
retried. Every scheduled answer gets one terminal outcome; outputs are sorted by answer and span.

**Scoring.** A prediction matches gold only on identical `(answer_id, start_char, end_char)`. The
scorer reports extraction and joint P/R/F1, gold coverage C/G, per-label P/R/F1 and macro F1, the 4×4
confusion matrix, and missed / unmatched / wrong-label IDs. Answer errors keep their gold claims in G.
Check runs store predictions and counts only; all quality fields are null.

**Langfuse.** One trace per answer (session = run ID) with a generation observation per model call
(tokens when available) and a tool observation per lookup; validation failures are logged as events.
Development traces receive `extraction_f1`, `joint_f1` and per-claim `label_correct` scores, with
missing and extra claims in score metadata; run-level F1 scores attach to the session. Check traces
get no quality scores.
