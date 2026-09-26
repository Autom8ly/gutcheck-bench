"""Scoring for typed decisions: accuracy, calibration and selective automation.

Each scored question is a row {"p": [probabilities in option order], "label": index of the right option, "task": str,
"type": "choice"|"noul"|"score", "variant": "clean"|...}.

Attribution: these metric definitions (binned ECE, Brier, floored NLL, confident-error rate, and coverage at an error
budget with whole confidence ties) are based on Kev's `kev/metrics.py` and `kev/benchmark.py`
(https://github.com/jaredpalmer/kev, Apache-2.0, commit 105c76958d4c). This module is a smaller reimplementation of
the subset we report, written so the scorer does not depend on any one tool under test. `gutcheck.rescore` checks that
it reproduces Kev's scorer exactly on every saved run.
"""
import math

import numpy as np

EPSILON = 1e-9   # floor for log-probabilities of options a model gives exactly zero


def ece(confidence, correct, bins=10):
    """Expected calibration error: equal-width confidence bins, weighted |accuracy - confidence|."""
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
        if in_bin.any():
            total += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(total)


def _risk_curve(confidence, correct):
    """Accept decisions in descending confidence, whole ties at a time: (accepted counts, error counts) per threshold."""
    order = np.argsort(-confidence, kind="stable")
    sorted_conf, errors = confidence[order], np.cumsum(~correct[order])
    ends = np.r_[np.flatnonzero(sorted_conf[1:] != sorted_conf[:-1]), len(order) - 1]
    return ends + 1, errors[ends]


def coverage_at_error(confidence, correct, budget):
    """Largest share of decisions you can accept automatically, most confident first, while errors among the
    accepted stay within `budget`. Rewards honest confidence, not just accuracy."""
    accepted, errors = _risk_curve(np.asarray(confidence, float), np.asarray(correct, bool))
    ok = np.flatnonzero(errors <= budget * accepted)
    return float(accepted[ok[-1]] / accepted[-1]) if len(ok) else 0.0


def score(rows):
    """Metrics over a population of rows."""
    if not rows:
        raise ValueError("no rows to score")
    acc, conf, brier, nll, mae = [], [], [], [], []
    for r in rows:
        p, y = np.asarray(r["p"], float), r["label"]
        acc.append(int(p.argmax() == y)); conf.append(float(p.max()))
        brier.append(float(((p - np.eye(len(p))[y]) ** 2).sum()))
        nll.append(-math.log(max(float(p[y]), EPSILON)))
        if r["type"] == "score":
            mae.append(abs(float(p @ np.arange(len(p))) - y))
    confidence, correct = np.asarray(conf), np.asarray(acc, bool)
    high = confidence >= 0.9
    out = {"n": len(rows), "acc": float(correct.mean()), "ece": ece(confidence, correct), "brier": float(np.mean(brier)),
           "nll": float(np.mean(nll)), "mean_conf": float(confidence.mean()),
           "confident_error_rate": float((high & ~correct).mean()), "coverage_at_0_9": float(high.mean()),
           "coverage_at_5pct_error": coverage_at_error(confidence, correct, 0.05),
           "coverage_at_1pct_error": coverage_at_error(confidence, correct, 0.01)}
    if mae:
        out["score_mae"] = float(np.mean(mae))
    return out


def summarize(rows, latencies_ms):
    """Headline block over clean questions, a per-task breakdown, and latency."""
    clean = [r for r in rows if r.get("variant", "clean") == "clean"]
    tasks = {}
    for r in clean:
        tasks.setdefault(r["task"], []).append(r)
    lat = np.asarray(latencies_ms, float)
    return {"clean": score(clean), "tasks": {t: score(v) for t, v in sorted(tasks.items())},
            "latency_ms": {"p50": float(np.median(lat)), "p95": float(np.quantile(lat, 0.95)), "n": int(len(lat))}}
