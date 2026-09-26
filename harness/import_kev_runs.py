"""One-off: convert our first round of transfer-v4 runs (recorded with Kev's benchmark tool before gutcheck existed)
into gutcheck's format. Reads each old run's predictions.jsonl, keeps only the per-question rows and latency, and
re-scores them with gutcheck.metrics (identical to the original scores; see gutcheck.rescore).

    python harness/import_kev_runs.py <old results dir> <new results dir>
"""
import json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gutcheck import metrics  # noqa: E402

KEEP = ("id", "question", "task", "type", "variant", "keys", "label", "p")
old_root, new_root = Path(sys.argv[1]), Path(sys.argv[2])
for old in sorted(old_root.glob("*/run*/predictions.jsonl")):
    run = old.parent; name = run.parent.name
    if not old.stat().st_size or not (run / "report.json").exists():
        continue
    out = new_root / name / "transfer-v4"; out.mkdir(parents=True, exist_ok=True)
    rows, lat = [], []
    with (out / "predictions.jsonl").open("w") as f:
        for line in old.read_text().splitlines():
            rec = json.loads(line)
            ms = (rec.get("prediction") or {}).get("latency_ms")
            new = [{k: r[k] for k in KEEP} for r in rec["rows"]]
            rows += new; lat.append(ms)
            f.write(json.dumps({"id": rec["id"], "latency_ms": ms, "rows": new}) + "\n")
    old_report = json.loads((run / "report.json").read_text())
    report = metrics.summarize(rows, lat)
    remote = old_report.get("remote") or {}
    report.update(suite="kev/evals/v4/transfer-v4 (development split)", endpoint=remote.get("base_url"),
                  requested_model=remote.get("requested_model"), served_model=remote.get("served_model"),
                  records=len(lat), questions=len(rows),
                  note="recorded with kev.benchmark before gutcheck existed; re-scored with gutcheck.metrics")
    (out / "report.json").write_text(json.dumps(report, indent=1))
    peak = run.parent / "peak_vram_mib"
    if peak.exists():
        shutil.copy(peak, out / "peak_vram_mib")
    print(f"{name}: acc={report['clean']['acc']:.4f} n={report['clean']['n']}")
