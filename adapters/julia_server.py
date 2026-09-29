"""TypeSafe-style POST /v1/systemone server wrapping Supersonic Labs' Julia-1 (official Python runtime, BF16/FP32
checkpoint). Env: JULIA_DIR (checkout of SupersonicLabs/Julia-1), JULIA_DEVICE=cpu|cuda|mps, JULIA_MAX_LENGTH, PORT.
GET /v1/stats reports the device and peak memory (process RSS; CUDA peak allocation when on a GPU)."""
import json, os, resource, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

import torch
from julia import load_model

JULIA_DIR = os.environ.get("JULIA_DIR", "Julia-1")
DEVICE = os.environ.get("JULIA_DEVICE", "cpu")
MAX_LENGTH = int(os.environ.get("JULIA_MAX_LENGTH", "8192"))
PORT = int(os.environ.get("PORT", "8120"))

ENGINE = load_model(JULIA_DIR, device=DEVICE, strict_encoding=True, max_length=MAX_LENGTH, head_length=512)
LOCK = Lock()
NAME = "julia-1"


def state_text(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def to_jev(q, ans):
    """Julia's answer -> Jev's answer shape: noul -> P(true); choice/score -> probabilities keyed like the request."""
    probs = {str(k): float(v) for k, v in ans["probabilities"].items()}
    if q["type"] == "noul":
        return {"noul": probs["true"]}
    return {"probabilities": probs}


def stats():
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mib = peak_rss / (1024 * 1024) if sys.platform == "darwin" else peak_rss / 1024
    out = {"device": DEVICE, "peak_rss_mib": round(peak_rss_mib, 1), "torch": torch.__version__}
    if DEVICE == "cuda":
        out["cuda_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/v1/stats"):
            return self._send(200, stats())
        self._send(200 if self.path.startswith("/v1/health") else 404, {"ok": True, "model": NAME})

    def do_POST(self):
        if not self.path.startswith("/v1/systemone"):
            return self._send(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        questions = body["questions"]
        for q in questions.values():   # Jev accepts choice options without descriptions; Julia needs text: use the ID
            if q.get("type") == "choice" and isinstance(q.get("criteria"), dict):
                q["criteria"] = {k: (v or k) for k, v in q["criteria"].items()}
        try:
            with LOCK:
                t = time.perf_counter()
                result = ENGINE.predict(state=state_text(body["state"]), questions=questions)
                ms = 1000 * (time.perf_counter() - t)
            answers = {qid: to_jev(questions[qid], a) for qid, a in result["answers"].items()}
        except Exception as error:   # report, don't crash the server
            return self._send(400, {"error": f"{type(error).__name__}: {error}"})
        self._send(200, {"model": NAME, "answers": answers, "latency_ms": round(ms, 1)})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"julia-1 on {DEVICE}, port {PORT}", file=sys.stderr, flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
