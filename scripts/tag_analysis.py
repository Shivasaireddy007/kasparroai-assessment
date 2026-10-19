"""Per-difficulty-tag breakdown of development runs (no API calls).
Usage: python3 scripts/tag_analysis.py runs/development_v1_*.json runs/development_v2_2*.json"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(paths: list[str]) -> None:
    gold = list(csv.DictReader((ROOT / "data" / "development_gold.csv").open(encoding="utf-8", newline="")))
    runs = []
    for p in paths:
        r = json.loads(Path(p).read_text(encoding="utf-8"))
        preds = {c["claim_id"]: c["label"] for a in r["answers"] if a["status"] == "ok" for c in a["claims"]}
        errors = {a["answer_id"] for a in r["answers"] if a["status"] != "ok"}
        runs.append((r["agent_version"], preds, errors))
    stats: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "wrong": 0, "missed": 0, "error": 0}))
    totals: dict[str, int] = defaultdict(int)
    for g in gold:
        for tag in g["difficulty_tags"].split("|"):
            totals[tag] += 1
            for name, preds, errors in runs:
                s = stats[tag][name]
                if g["answer_id"] in errors:
                    s["error"] += 1
                elif g["claim_id"] not in preds:
                    s["missed"] += 1
                elif preds[g["claim_id"]] == g["expected_label"]:
                    s["correct"] += 1
                else:
                    s["wrong"] += 1
    names = [n for n, _, _ in runs]
    head = "| Tag | Gold | " + " | ".join(f"{n} correct" for n in names) + " | " + " | ".join(f"{n} missed / wrong / answer-error" for n in names) + " |"
    print(head)
    print("|" + "---|" * (2 + 2 * len(names)))
    for tag in sorted(totals, key=lambda t: (-totals[t], t)):
        cells = [f"{stats[tag][n]['correct']}/{totals[tag]} ({stats[tag][n]['correct'] / totals[tag]:.0%})" for n in names]
        cells += [f"{stats[tag][n]['missed']} / {stats[tag][n]['wrong']} / {stats[tag][n]['error']}" for n in names]
        print(f"| {tag} | {totals[tag]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
