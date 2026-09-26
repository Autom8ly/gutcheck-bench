"""Collect every results/runs/<setup>/<suite>/report.json into results/summary.json and results/tasks.json,
and print a markdown table per suite.

    python harness/summarize.py [results dir]
"""
import json, sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"


def peak(run_dir):
    for p in (run_dir / "peak_vram_mib", run_dir.parent / "peak_vram_mib"):
        if p.exists():
            return int(p.read_text().strip())
    return None


rows, tasks = [], {}
for rep in sorted((ROOT / "runs").glob("*/*/report.json")):
    setup, suite = rep.parent.parent.name, rep.parent.name
    r = json.loads(rep.read_text()); c = r["clean"]
    rows.append({"setup": setup, "suite": suite, "n": c["n"], "acc": c["acc"], "ece": c["ece"], "brier": c["brier"],
                 "confident_err": c["confident_error_rate"], "auto_at_5pct": c["coverage_at_5pct_error"],
                 "p50_ms": r["latency_ms"]["p50"], "p95_ms": r["latency_ms"]["p95"],
                 "peak_vram_mib": None if setup == "jev-hosted" else peak(rep.parent)})
    for t, m in r["tasks"].items():
        tasks.setdefault(suite, {}).setdefault(t, {})[setup] = {"acc": m["acc"], "n": m["n"]}

(ROOT / "summary.json").write_text(json.dumps(rows, indent=1))
(ROOT / "tasks.json").write_text(json.dumps(tasks, indent=1))
fmt = lambda v, f: "—" if v is None else format(v, f)
for suite in sorted({r["suite"] for r in rows}):
    print(f"\n### {suite}\n")
    print("| setup | n | acc | ECE | wrong ≥90% | auto @5% | p50 ms | p95 ms | peak VRAM |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted((r for r in rows if r["suite"] == suite), key=lambda r: -r["acc"]):
        v = r["peak_vram_mib"]
        print(f"| {r['setup']} | {r['n']} | {r['acc']:.3f} | {r['ece']:.3f} | {r['confident_err']*100:.1f}% | "
              f"{r['auto_at_5pct']*100:.0f}% | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {fmt(v and v/1024, '.1f')} GB |")
