"""Export Langfuse evidence for saved runs: per-answer trace IDs and URLs, observation summaries and
scores. Output contains no credentials. Usage: uv run python scripts/export_langfuse.py runs/*.json"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from langfuse import get_client  # noqa: E402


def _all(fetch, **kw):
    out, cursor = [], None
    while True:
        res = fetch(limit=100, cursor=cursor, **kw)
        out += res.data
        cursor = getattr(getattr(res, "meta", None), "cursor", None)
        if not cursor or not res.data:
            return out


def main(paths: list[str]) -> None:
    lf = get_client()
    out_dir = ROOT / "evidence" / "langfuse"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        report = json.loads(Path(p).read_text(encoding="utf-8"))
        run_id = report["run_id"]
        answers = []
        for a in report["answers"]:
            tid = a.get("trace_id")
            if not tid:
                continue
            obs = _all(lf.api.observations.get_many, trace_id=tid, fields="core,basic,usage")
            scores = _all(lf.api.scores.get_many, trace_id=tid)
            answers.append(
                {
                    "answer_id": a["answer_id"],
                    "status": a["status"],
                    "trace_id": tid,
                    "trace_url": lf.get_trace_url(trace_id=tid),
                    "observations": sorted(
                        [
                            {
                                "type": o.type,
                                "name": o.name,
                                "level": str(o.level) if o.level else None,
                                "status_message": o.status_message,
                                "usage": getattr(o, "usage_details", None),
                                "latency_s": getattr(o, "latency", None),
                            }
                            for o in obs
                        ],
                        key=lambda x: (x["type"] or "", x["name"] or ""),
                    ),
                    "scores": sorted(
                        [{"name": s.name, "value": s.value, "comment": s.comment} for s in scores],
                        key=lambda x: (x["name"], x["comment"] or ""),
                    ),
                }
            )
        session_scores = [{"name": s.name, "value": s.value} for s in _all(lf.api.scores.get_many, session_id=run_id)]
        summary = {
            "run_id": run_id,
            "split": report["split"],
            "agent_version": report["agent_version"],
            "report": Path(p).name,
            "traces": len(answers),
            "generation_observations": sum(o["type"] == "GENERATION" for a in answers for o in a["observations"]),
            "tool_observations": sum(o["type"] == "TOOL" for a in answers for o in a["observations"]),
            "trace_scores": sum(len(a["scores"]) for a in answers),
            "session_scores": session_scores,
            "answers": answers,
        }
        dest = out_dir / Path(p).name.replace(".json", "_langfuse.json")
        dest.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"{dest.name}: traces={summary['traces']} generations={summary['generation_observations']} "
              f"tools={summary['tool_observations']} trace_scores={summary['trace_scores']} session_scores={len(session_scores)}")


if __name__ == "__main__":
    main(sys.argv[1:])
