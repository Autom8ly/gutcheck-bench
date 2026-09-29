"""Stability and uncertainty analysis for saved gutcheck runs.

    python stability/analyze.py                      # the published Kev-4B 4-bit analysis
    python stability/analyze.py --reference results/runs/kev-4b-4bit/jabr-v2 \
        --runs stability/runs/kev-4b-4bit/* --jev results/runs/jev-hosted/jabr-v2

For each run against the reference run: questions whose probabilities changed, the largest change, top-answer flips,
0.9-threshold crossings, flips among near-threshold questions and, for concurrency runs, changes by the size of the
batch each request actually ran in. Then 95% bootstrap intervals (accuracy, the paired gap to Jev) and coverage at a
5% error budget with the threshold chosen on held-out halves.
"""
import argparse, collections, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gutcheck.metrics import coverage_at_error

NEAR = ((0.5, 0.45, 0.55), (0.6, 0.55, 0.65), (0.9, 0.85, 0.95))


def load(run):
    """-> {(record id, question): (p, label)} and {record id: (burst, server_ms)} for concurrency runs."""
    rows, meta = {}, {}
    for line in open(Path(run) / "predictions.jsonl"):
        j = json.loads(line)
        for r in j["rows"]:
            rows[(r["id"], r["question"])] = (np.array(r["p"]), r["label"])
        if "burst" in j:
            meta[j["id"]] = (j["burst"], j.get("server_ms"))
    return rows, meta


def compare(ref, run):
    rows, meta = load(run)
    keys = sorted(set(ref) & set(rows))
    dp = np.array([np.abs(rows[k][0] - ref[k][0]).max() for k in keys])
    flips = sum(rows[k][0].argmax() != ref[k][0].argmax() for k in keys)
    cross = sum((rows[k][0].max() >= 0.9) != (ref[k][0].max() >= 0.9) for k in keys)
    acc = np.mean([rows[k][0].argmax() == rows[k][1] for k in keys])
    print(f"{Path(run).name:16} n={len(keys)} acc={acc:.4f} changed={int((dp > 0).sum())} max|dp|={dp.max():.4f} "
          f"top-answer flips={flips} crossed 0.9={cross}")
    for cut, lo, hi in NEAR:
        near = [k for k in keys if lo <= ref[k][0].max() < hi]
        n = sum(((rows[k][0].max() >= cut) != (ref[k][0].max() >= cut)) or rows[k][0].argmax() != ref[k][0].argmax()
                for k in near)
        print(f"    near {cut}: {n}/{len(near)} changed decision")
    if meta:   # batch size = requests in the same burst reporting the same server model time
        groups = collections.Counter(meta.values())
        by = collections.defaultdict(lambda: [0, 0])
        for k, d in zip(keys, dp):
            size = groups[meta[k[0]]]
            by[size][0] += 1; by[size][1] += int(d > 0)
        print("    by batch size (requests, changed):", dict(sorted((s, tuple(v)) for s, v in by.items())))


def intervals(ref, jev, n_boot=10000, n_split=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = sorted(ref)
    ok = np.array([ref[k][0].argmax() == ref[k][1] for k in keys])
    conf = np.array([ref[k][0].max() for k in keys])
    jok = np.array([jev[k][0].argmax() == jev[k][1] for k in keys]) if jev else None
    acc, gap = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(keys), len(keys))
        acc.append(ok[i].mean())
        if jok is not None:
            gap.append(jok[i].mean() - ok[i].mean())
    print(f"\naccuracy {ok.mean():.3f}, 95% CI {np.percentile(acc, 2.5):.3f}-{np.percentile(acc, 97.5):.3f} (n={len(keys)})")
    if gap:
        print(f"gap to Jev {jok.mean() - ok.mean():.3f}, 95% CI {np.percentile(gap, 2.5):.3f}-{np.percentile(gap, 97.5):.3f}")
    print(f"confidently wrong (>=0.9 and wrong): {int((~ok & (conf >= 0.9)).sum())} of {len(keys)}, "
          f"{int((conf >= 0.9).sum())} confident answers")
    cov, err = [], []
    for _ in range(n_split):
        p = rng.permutation(len(keys)); a, b = p[: len(keys) // 2], p[len(keys) // 2:]
        n = int(round(coverage_at_error(conf[a], ok[a], 0.05) * len(a)))
        if n == 0:
            cov.append(0.0); err.append(0.0); continue
        take = conf[b] >= np.sort(conf[a])[::-1][n - 1]
        cov.append(take.mean()); err.append((~ok[b][take]).mean() if take.any() else 0.0)
    q = lambda v: f"{np.median(v):.3f} ({np.percentile(v, 2.5):.3f}-{np.percentile(v, 97.5):.3f})"
    print(f"coverage at 5% error: in-sample {coverage_at_error(conf, ok, 0.05):.3f}; held-out {q(cov)}, "
          f"realised error {q(err)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default=str(ROOT / "results/runs/kev-4b-4bit/jabr-v2"))
    ap.add_argument("--runs", nargs="*", default=sorted(str(p) for p in (ROOT / "stability/runs/kev-4b-4bit").glob("*")))
    ap.add_argument("--jev", default=str(ROOT / "results/runs/jev-hosted/jabr-v2"))
    a = ap.parse_args()
    ref, _ = load(a.reference)
    for run in a.runs:
        compare(ref, run)
    intervals(ref, load(a.jev)[0] if a.jev else None)


if __name__ == "__main__":
    main()
