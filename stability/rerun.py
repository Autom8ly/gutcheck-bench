"""Score an endpoint with gutcheck.benchmark, paced for a rate-limited gateway, with an optional User-Agent (some
edge proxies block script user agents). Same arguments as gutcheck.benchmark, plus:

    --rate N          at most N requests per minute (default: unlimited)
    --user-agent UA   User-Agent header to send

    python stability/rerun.py --endpoint https://your-gateway --model kev-latest --suite suites/jabr-v2.jsonl \
        --out stability/runs/kev-4b-4bit/rerun-1 --api-key-file ~/.kev_key --rate 55
"""
import json, sys, time, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gutcheck import benchmark


def paced_call(rate, user_agent):
    gap, last = (60.0 / rate if rate else 0.0), [0.0]

    def call(endpoint, body, key, timeout=120, retries=8):
        headers = {"content-type": "application/json"}
        if user_agent:
            headers["user-agent"] = user_agent
        if key:
            headers["authorization"] = f"Bearer {key}"
        data = json.dumps(body).encode()
        for attempt in range(retries):
            wait = last[0] + gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last[0] = time.monotonic()
            try:
                start = time.perf_counter()
                req = urllib.request.Request(f"{endpoint.rstrip('/')}/v1/systemone", data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    answer = json.loads(resp.read())
                return answer, 1000 * (time.perf_counter() - start)
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                    raise RuntimeError(f"HTTP {error.code}") from error
                time.sleep(min(60, 5 * 2 ** attempt))
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == retries - 1:
                    raise RuntimeError(f"endpoint failed after {retries} attempts: {error}") from error
                time.sleep(2 ** attempt)
    return call


if __name__ == "__main__":
    argv, rate, ua = sys.argv[1:], 0.0, None
    for flag in ("--rate", "--user-agent"):
        if flag in argv:
            i = argv.index(flag); value = argv[i + 1]; del argv[i:i + 2]
            if flag == "--rate":
                rate = float(value)
            else:
                ua = value
    benchmark.call = paced_call(rate, ua)
    benchmark.main(argv)
