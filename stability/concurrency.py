"""Concurrency test for a /v1/systemone server: send the suite in simultaneous bursts of C requests so the server
batches them, and record per-request probabilities (gutcheck predictions format), the server's model time for the
batch each request ran in (`latency_ms` in the response) and the client round trip.

Paced to stay under a gateway rate limit: at most --rate requests per minute on average (default 55).

    python stability/concurrency.py --endpoint https://your-gateway --suite suites/jabr-v2.jsonl \
        --out stability/runs/kev-4b-4bit/concurrency-8 --concurrency 8 --api-key-file ~/.kev_key
"""
import argparse, json, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gutcheck import benchmark, metrics, suites



def post(endpoint, body, key, user_agent=None, retries=8):
    headers = {"content-type": "application/json"}
    if user_agent:
        headers["user-agent"] = user_agent
    if key:
        headers["authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            start = time.perf_counter()
            req = urllib.request.Request(f"{endpoint.rstrip('/')}/v1/systemone", data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                answer = json.loads(resp.read())
            return answer, 1000 * (time.perf_counter() - start)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise RuntimeError(f"HTTP {error.code}") from error
            time.sleep(min(60, 5 * 2 ** attempt))
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="kev-latest")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--api-key-file")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rate", type=float, default=55, help="max requests per minute on average")
    ap.add_argument("--user-agent")
    a = ap.parse_args()
    key = open(a.api_key_file).read().strip() if a.api_key_file else ""
    records = suites.load(a.suite)[: a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    c = a.concurrency
    rows, rtts, server_ms, bursts = [], [], [], []
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f, ThreadPoolExecutor(c) as pool:
        for b in range(0, len(records), c):
            chunk = records[b:b + c]
            t0 = time.perf_counter()
            results = list(pool.map(lambda r: post(a.endpoint, suites.request(r, a.model), key, a.user_agent), chunk))
            wall = 1000 * (time.perf_counter() - t0)
            bursts.append({"n": len(chunk), "wall_ms": wall, "server_ms": [ans.get("latency_ms") for ans, _ in results]})
            for record, (answer, ms) in zip(chunk, results):
                new = [{"id": record["id"], "question": qid, "task": q["task"], "type": q["type"],
                        "variant": record.get("variant", "clean"), "keys": suites.option_keys(q),
                        "label": suites.label_index(q), "p": benchmark.distribution(q, answer["answers"][qid])}
                       for qid, q in record["questions"].items()]
                rows += new; rtts.append(ms); server_ms.append(answer.get("latency_ms"))
                f.write(json.dumps({"id": record["id"], "latency_ms": ms, "server_ms": answer.get("latency_ms"),
                                    "burst": b // c, "rows": new}) + "\n")
            if (b // c) % max(1, 100 // c) == 0:
                print(f"scored {b + len(chunk)}/{len(records)}", file=sys.stderr, flush=True)
            pause = len(chunk) * 60 / a.rate - (time.perf_counter() - t0)
            if pause > 0:
                time.sleep(pause)
    report = metrics.summarize(rows, rtts)
    sm = sorted(s for s in server_ms if s is not None)
    report.update(concurrency=c, records=len(records), bursts=bursts,
                  server_ms={"p50": sm[len(sm) // 2], "p90": sm[int(0.9 * (len(sm) - 1))], "max": sm[-1]} if sm else None)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    cl = report["clean"]
    print(json.dumps({"concurrency": c, "acc": round(cl["acc"], 4), "rtt_p50": round(report["latency_ms"]["p50"], 1),
                      "server_ms": report["server_ms"]}), flush=True)


if __name__ == "__main__":
    main()
