"""Re-score saved predictions and compare with the report that produced them.

    python -m gutcheck.rescore results/runs/*/run-transfer-v4

Reads predictions.jsonl (gutcheck's, or Kev's `kev.benchmark` output, whose lines carry the same per-question rows)
and prints gutcheck's headline metrics next to the saved report.json. Used to check that gutcheck and Kev's scorer
agree to the last digit on every run in this repository.
"""
import json, sys
from pathlib import Path

from . import metrics

FIELDS = ("n", "acc", "ece", "brier", "confident_error_rate", "coverage_at_5pct_error")


def rescore(run_dir):
    run_dir = Path(run_dir)
    rows, latencies = [], []
    for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        rows += rec["rows"]
        latencies.append(rec.get("latency_ms", (rec.get("prediction") or {}).get("latency_ms", 0.0)))
    ours = metrics.summarize(rows, latencies)["clean"]
    saved = json.loads((run_dir / "report.json").read_text())["clean"]
    return ours, saved


def main(argv=None):
    worst = 0.0
    for d in (argv or sys.argv[1:]):
        if not (Path(d) / "report.json").exists() or not (Path(d) / "predictions.jsonl").stat().st_size:
            print(f"skip {d}: no completed run"); continue
        ours, saved = rescore(d)
        diffs = {k: abs(float(ours[k]) - float(saved[k])) for k in FIELDS}
        worst = max(worst, max(diffs.values()))
        flag = "OK " if max(diffs.values()) < 1e-9 else "DIFF"
        print(f"{flag} {d}: " + "  ".join(f"{k}={ours[k]:.4f}" + ("" if diffs[k] < 1e-9 else f"(saved {saved[k]:.4f})") for k in FIELDS))
    print(f"largest difference across all runs: {worst:.2e}")
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
