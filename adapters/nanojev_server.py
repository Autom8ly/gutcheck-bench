"""POST /v1/systemone proxy in front of NanoJev's /api/evaluate (started as a subprocess). Env: PORT (proxy port)."""
import atexit, json, os, subprocess, sys, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8140"))
INNER = PORT + 1000
proc = subprocess.Popen([sys.executable, "scripts/serve_decisions.py", "--checkpoint-dir", "checkpoints/NanoJev-unified",
                         "--web-root", "web", "--port", str(INNER), "--disable-native-triton"])
atexit.register(proc.terminate)
for _ in range(240):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{INNER}/api/health", timeout=2); break
    except Exception:
        if proc.poll() is not None: sys.exit("NanoJev server exited")
        time.sleep(1)
NAME = "nanojev-0.6b-unified-games"


def translate(questions):
    out = {}
    for qid, q in questions.items():
        t = q["type"]
        if t == "noul":
            out[qid] = {"type": "boolean", "instructions": q.get("instructions") or qid}
        elif t == "choice":
            c = q["criteria"]
            out[qid] = {"type": "choice", "instructions": q.get("instructions") or qid,
                        "criteria": {k: (v or k) for k, v in (c.items() if isinstance(c, dict) else ((x, x) for x in c))}}
        else:
            out[qid] = {"type": "score", "instructions": q.get("instructions") or qid, "criteria": [str(x) for x in q["criteria"]]}
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/v1/models", "/health"):
            return self._send(200, {"data": [{"id": NAME}], "model": NAME})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        payload = {"states": [{"id": "s", "state": req["state"], "questions": translate(req["questions"])}]}
        start = time.perf_counter()
        try:
            r = urllib.request.Request(f"http://127.0.0.1:{INNER}/api/evaluate", data=json.dumps(payload).encode(),
                                       headers={"content-type": "application/json"})
            with urllib.request.urlopen(r, timeout=300) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return self._send(400, {"error": e.read().decode(errors="replace")[:500]})
        answers = {}
        for qid, a in body["states"][0]["answers"].items():
            p = a["probabilities"]
            if a["type"] == "boolean":
                answers[qid] = {"noul": float(a["p_true"])}
            elif a["type"] == "choice":
                answers[qid] = {"choice": a["choice"], "probabilities": p, "confidence": max(p.values())}
            else:
                probs = {str(i): v for i, v in enumerate(p.values())} if not all(k.isdigit() for k in p) else p
                answers[qid] = {"score": a["level"], "probabilities": probs, "confidence": max(probs.values())}
        self._send(200, {"model": NAME, "answers": answers, "latency_ms": 1000 * (time.perf_counter() - start)})


if __name__ == "__main__":
    print(f"serving {NAME} on 127.0.0.1:{PORT} (inner {INNER})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
