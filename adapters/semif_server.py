"""TypeSafe-style POST /v1/systemone server wrapping SemIf's direct scorer (torch), with an optional bitsandbytes
backbone so Qwen3.5-4B fits an 8 GB GPU.  Env: SEMIF_MODEL, SEMIF_REVISION, SEMIF_QUANT=4bit|8bit|none, PORT."""
import json, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

import torch, transformers
from semif_phase1 import direct

MODEL = os.environ.get("SEMIF_MODEL", "Qwen/Qwen3.5-4B")
REVISION = os.environ.get("SEMIF_REVISION", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
QUANT = os.environ.get("SEMIF_QUANT", "8bit")
PORT = int(os.environ.get("PORT", "8110"))


def load():
    config = transformers.AutoConfig.from_pretrained(MODEL, revision=REVISION)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    cls = transformers.AutoModelForCausalLM
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:
        cls, config = transformers.Qwen3_5ForCausalLM, config.get_text_config()
    kw = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if QUANT == "4bit":
        kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    elif QUANT == "8bit":
        kw["quantization_config"] = transformers.BitsAndBytesConfig(load_in_8bit=True)
    model = cls.from_pretrained(MODEL, revision=REVISION, config=config, **kw).eval()
    return model, tok


MODEL_OBJ, TOK = load()
META = {"source": MODEL, "revision": REVISION, "quant": QUANT}
LOCK = Lock()
NAME = f"semif-{MODEL.split('/')[-1]}-{QUANT}"


def state_text(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def to_row(qid, q, state):
    t = q["type"]
    if t == "noul":
        options = [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}]
    elif t == "choice":
        crit = q["criteria"]
        options = [{"id": k, "description": v or k} for k, v in crit.items()] if isinstance(crit, dict) \
            else [{"id": c, "description": c} for c in crit]
    elif t == "score":
        options = [{"id": str(i), "description": str(level)} for i, level in enumerate(q["criteria"])]
    else:
        raise ValueError(f"unsupported question type {t}")
    return {"id": qid, "state": state_text(state), "question": q.get("instructions") or qid, "options": options}


def answer(q, res):
    probs = dict(zip(res["option_ids"], res["probabilities"]))
    t = q["type"]
    if t == "noul":
        return {"noul": probs["yes"]}
    best = max(probs, key=probs.get)
    if t == "choice":
        return {"choice": best, "probabilities": probs, "confidence": probs[best]}
    return {"score": int(best), "probabilities": probs, "confidence": probs[best]}


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
        if self.path != "/v1/systemone":
            return self._send(404, {"error": "not found"})
        req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        start, answers, tokens = time.perf_counter(), {}, 0
        try:
            with LOCK:
                for qid, q in req["questions"].items():
                    res = direct.score(MODEL_OBJ, TOK, to_row(qid, q, req["state"]), META, max_tokens=4096)
                    answers[qid] = answer(q, res); tokens += res.get("input_tokens") or 0
        except Exception as e:
            return self._send(400, {"error": str(e)})
        self._send(200, {"model": NAME, "answers": answers, "usage": {"input_tokens": tokens},
                         "latency_ms": 1000 * (time.perf_counter() - start)})


if __name__ == "__main__":
    print(f"serving {NAME} on 127.0.0.1:{PORT}; VRAM {torch.cuda.memory_allocated() / 2**20:.0f} MiB", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
