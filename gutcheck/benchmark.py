"""Score any /v1/systemone endpoint on a suite.

    python -m gutcheck.benchmark --endpoint http://127.0.0.1:8008 --model kev-latest \
        --suite suites/jabr-v2.jsonl --out runs/my-tool

Hosted endpoints: put the key in GUTCHECK_API_KEY (or JEV_API_KEY for TypeSafe's Jev), or pass --api-key-file.
Writes predictions.jsonl (one line per record) and report.json (headline, per-task, latency).
"""
import argparse, json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

import numpy as np

from . import metrics, suites


def api_key(args):
    if args.api_key_file:
        return Path(args.api_key_file).expanduser().read_text().strip()
    return os.environ.get("GUTCHECK_API_KEY") or os.environ.get("JEV_API_KEY") or ""


def call(endpoint, body, key, timeout=120, retries=3):
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            start = time.perf_counter()
            with urllib.request.urlopen(urllib.request.Request(f"{endpoint.rstrip('/')}/v1/systemone", data=data, headers=headers),
                                        timeout=timeout) as resp:
                answer = json.loads(resp.read())
            return answer, 1000 * (time.perf_counter() - start)
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == retries - 1:
                raise RuntimeError(f"endpoint failed after {retries} attempts: {error}") from error
            time.sleep(2 ** attempt)


def distribution(q, answer):
    """The answer's probabilities in option order, renormalized; a yes/no answer carries only P(yes)."""
    keys = suites.option_keys(q)
    if q["type"] == "noul":
        yes = float(answer["noul"]); p = np.array([1 - yes, yes])
    else:
        raw = {str(k): float(v) for k, v in answer["probabilities"].items()}
        if set(raw) != set(keys):
            raise ValueError(f"returned options {sorted(raw)} differ from requested {keys}")
        p = np.array([raw[k] for k in keys])
    if not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("invalid probabilities")
    return (p / p.sum()).tolist()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True, help="base URL of a /v1/systemone server")
    ap.add_argument("--model", default="default", help="model name to send in each request")
    ap.add_argument("--suite", required=True, help="gutcheck JSONL file or Kev frozen-suite directory")
    ap.add_argument("--out", required=True, help="output directory (must not exist)")
    ap.add_argument("--api-key-file", help="file holding the bearer key for a hosted endpoint")
    ap.add_argument("--limit", type=int, help="score only the first N records (smoke test)")
    a = ap.parse_args(argv)

    records = suites.load(a.suite)[: a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    key, rows, latencies, served = api_key(a), [], [], None
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for i, record in enumerate(records, 1):
            answer, ms = call(a.endpoint, suites.request(record, a.model), key)
            served = answer.get("model", served)
            new = []
            for qid, q in record["questions"].items():
                new.append({"id": record["id"], "question": qid, "task": q["task"], "type": q["type"],
                            "variant": record.get("variant", "clean"), "keys": suites.option_keys(q),
                            "label": suites.label_index(q), "p": distribution(q, answer["answers"][qid])})
            rows += new; latencies.append(ms)
            f.write(json.dumps({"id": record["id"], "latency_ms": ms, "rows": new}) + "\n")
            if i % 100 == 0:
                print(f"scored {i}/{len(records)}", file=sys.stderr, flush=True)
    report = metrics.summarize(rows, latencies)
    report.update(suite=str(a.suite), endpoint=a.endpoint, requested_model=a.model, served_model=served,
                  records=len(records), questions=len(rows))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    c = report["clean"]
    print(json.dumps({"n": c["n"], "acc": round(c["acc"], 4), "ece": round(c["ece"], 4),
                      "confident_error_rate": round(c["confident_error_rate"], 4),
                      "coverage_at_5pct_error": round(c["coverage_at_5pct_error"], 4),
                      "p50_ms": round(report["latency_ms"]["p50"], 1)}, indent=1))


if __name__ == "__main__":
    main()
