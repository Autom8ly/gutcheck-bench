"""POST /v1/systemone server wrapping Nimble's CudaCandidateScorer (NIMBLE_QUANT=4bit|8bit for small GPUs). Env: PORT."""
import json, os, re, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from nimble.scoring.cuda_scorer import CudaCandidateScorer

PORT = int(os.environ.get("PORT", "8130"))
CONFIG = json.loads(Path(".cache/nimble-model.json").read_text())
SCORER = CudaCandidateScorer(**CONFIG)
NAME = f"nimble-9b-{os.environ.get('NIMBLE_QUANT', 'bf16')}"
LOCK = Lock()


def field_name(qid):
    return re.sub(r"[^A-Za-z0-9_]", "_", qid) or "q"


def to_schema(questions):
    schema, back = {}, {}
    for qid, q in questions.items():
        name, t = field_name(qid), q["type"]
        desc = q.get("instructions") or qid
        if t == "noul":
            schema[name] = {"type": "boolean", "description": desc}
        else:
            if t == "choice":
                crit = q["criteria"]
                crit = dict(crit) if isinstance(crit, dict) else {c: c for c in crit}
            else:
                crit = {str(i): str(level) for i, level in enumerate(q["criteria"])}
            schema[name] = {"type": "enum", "choices": list(crit), "description": desc,
                            "choice_descriptions": {k: (v or k) for k, v in crit.items()}}
        back[name] = (qid, q)
    return schema, back


def probs_of(field, keys):
    s = field.get("scores")
    if isinstance(s, dict):
        return {str(k): float(v) for k, v in s.items()}
    return {k: float(v) for k, v in zip(keys, s)}


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
        state = req["state"] if isinstance(req["state"], str) else json.dumps(req["state"], ensure_ascii=False, indent=1)
        start = time.perf_counter()
        try:
            schema, back = to_schema(req["questions"])
            with LOCK:
                res = SCORER.score(state, schema)
            answers = {}
            for name, (qid, q) in back.items():
                f = res["fields"][name]
                if q["type"] == "noul":
                    p = probs_of(f, ["true", "false"])
                    yes = p.get("true", p.get("True", p.get("yes")))
                    answers[qid] = {"noul": float(yes)}
                else:
                    keys = schema[name]["choices"]
                    p = probs_of(f, keys)
                    best = max(p, key=p.get)
                    answers[qid] = ({"choice": best} if q["type"] == "choice" else {"score": int(best)}) | \
                                   {"probabilities": p, "confidence": p[best]}
        except Exception as e:
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})
        self._send(200, {"model": NAME, "answers": answers, "latency_ms": 1000 * (time.perf_counter() - start)})


if __name__ == "__main__":
    print(f"serving {NAME} on 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
