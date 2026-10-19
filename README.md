# Claim extraction and classification agent (Kasparro FSE I assessment)

Work in progress. Full install/run/test instructions, results and write-up will be added here.

## Setup (so far)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv; provides Python 3.12
uv sync
uv run pytest -q
```

Issued CSVs are stored unchanged in `data/`; hashes in `data/SHA256SUMS`.
