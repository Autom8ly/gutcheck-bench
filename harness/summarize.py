"""Summarize every ~/jev-eval/results/<name>/run-*/ into one table (markdown + JSON)."""
import json, statistics, sys
from pathlib import Path

ROOT = Path.home() / "jev-eval" / "results"


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


rows = []
for rep in sorted(ROOT.glob("*/run*/report.json")):
    name, run = rep.parent.parent.name, rep.parent.name
    r = json.load(open(rep))
    c = r.get("clean") or {}
    lat = []
    for line in open(rep.parent / "predictions.jsonl"):
        p = json.loads(line).get("prediction") or {}
        if p.get("latency_ms") is not None:
            lat.append(p["latency_ms"])
    vram = rep.parent.parent / "peak_vram_mib"
    pf = r.get("paired_flip") or {}
    rows.append({
        "model": name, "suite": run.replace("run-", "").replace("run", "transfer-v4"),
        "n": c.get("n"), "acc": c.get("acc"), "ece": c.get("ece"), "brier": c.get("brier"),
        "confident_err": c.get("confident_error_rate"), "auto_at_5pct": c.get("coverage_at_5pct_error"),
        "order_invariance": pf.get("invariance_rate"),
        "p50_ms": pct(lat, 0.5), "p95_ms": pct(lat, 0.95),
        "peak_vram_mib": int(vram.read_text()) if vram.exists() else None,
        "rejected": (r.get("coverage") or {}).get("rejected_records"),
    })

json.dump(rows, open(ROOT / "summary.json", "w"), indent=1)
f = lambda v, fmt: "—" if v is None else format(v, fmt)
print("| model | suite | n | acc | ECE | Brier | conf.err | auto@5% | p50 ms | p95 ms | peak VRAM | rejected |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for x in sorted(rows, key=lambda x: (x["suite"], -(x["acc"] or 0))):
    print(f"| {x['model']} | {x['suite']} | {x['n']} | {f(x['acc'], '.3f')} | {f(x['ece'], '.3f')} | {f(x['brier'], '.3f')} | "
          f"{f(x['confident_err'], '.3f')} | {f(x['auto_at_5pct'], '.2f')} | {f(x['p50_ms'], '.0f')} | {f(x['p95_ms'], '.0f')} | "
          f"{f(x['peak_vram_mib'], 'd')} MiB | {x['rejected']} |")

# per-task accuracy matrix (clean variants) for strengths/weaknesses
tasks = {}
for rep in sorted(ROOT.glob("*/run*/report.json")):
    name = rep.parent.parent.name
    for t, m in (json.load(open(rep)).get("tasks") or {}).items():
        if isinstance(m, dict) and m.get("acc") is not None:
            tasks.setdefault(t, {})[name] = {"acc": m["acc"], "n": m.get("n")}
json.dump(tasks, open(ROOT / "tasks.json", "w"), indent=1)
if "--tasks" in sys.argv:
    models = sorted({m for v in tasks.values() for m in v})
    print("\n| task | n | " + " | ".join(models) + " |")
    print("|---|---|" + "---|" * len(models))
    for t in sorted(tasks):
        n = next(iter(tasks[t].values()))["n"]
        print(f"| {t} | {n} | " + " | ".join(f"{tasks[t][m]['acc']:.2f}" if m in tasks[t] else "—" for m in models) + " |")
