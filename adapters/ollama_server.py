"""Generative baseline behind POST /v1/systemone: asks an Ollama model for each typed answer as schema-constrained JSON
(temperature 0, thinking off), one question per call, as production prompt-based decisions do today. The chosen option
gets probability 1.0 (a generative answer carries no distribution), so only accuracy and latency are meaningful.
Env: OLLAMA_MODEL (default qwen3:8b), PORT."""
import json, os, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
PORT = int(os.environ.get("PORT", "8120"))
OLLAMA = "http://127.0.0.1:11434/api/chat"


def options(q):
    t = q["type"]
    if t == "noul":
        return {"yes": "yes", "no": "no"}
    if t == "choice":
        c = q["criteria"]
        return dict(c) if isinstance(c, dict) else {x: x for x in c}
    return {str(i): str(level) for i, level in enumerate(q["criteria"])}


def ask(state, q):
    opts = options(q)
    listing = "\n".join(f"- {k}: {v}" for k, v in opts.items())
    prompt = (f"STATE:\n{state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)}\n\n"
              f"QUESTION: {q.get('instructions', '')}\n\nOPTIONS:\n{listing}\n\n"
              'Reply with JSON {"answer": "<option key>"} using exactly one option key.')
    body = {"model": MODEL, "stream": False, "think": False, "keep_alive": "30m",
            "options": {"temperature": 0, "num_ctx": 8192},
            "format": {"type": "object", "properties": {"answer": {"type": "string", "enum": list(opts)}}, "required": ["answer"]},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    key = json.loads(out["message"]["content"]).get("answer")
    if key not in opts:
        key = next(iter(opts))
    return key, list(opts), out.get("prompt_eval_count") or 0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/v1/models", "/health"):
            return self._send(200, {"data": [{"id": f"ollama-{MODEL}"}], "model": f"ollama-{MODEL}"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        start, answers, tokens = time.perf_counter(), {}, 0
        try:
            for qid, q in req["questions"].items():
                key, keys, n = ask(req["state"], q); tokens += n
                probs = {k: (1.0 if k == key else 0.0) for k in keys}
                if q["type"] == "noul":
                    answers[qid] = {"noul": 1.0 if key == "yes" else 0.0}
                elif q["type"] == "choice":
                    answers[qid] = {"choice": key, "probabilities": probs, "confidence": 1.0}
                else:
                    answers[qid] = {"score": int(key), "probabilities": probs, "confidence": 1.0}
        except Exception as e:
            return self._send(502, {"error": str(e)})
        self._send(200, {"model": f"ollama-{MODEL}", "answers": answers, "usage": {"input_tokens": tokens},
                         "latency_ms": 1000 * (time.perf_counter() - start)})


if __name__ == "__main__":
    print(f"serving ollama-{MODEL} on 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
