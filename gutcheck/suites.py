"""Test suites as a list of records: {"id", "variant", "state", "questions": {qid: {type, instructions, criteria?, label, task}}}.

Two sources are supported:
- a gutcheck JSONL file (one record per line, as above), e.g. suites/jabr-v2.jsonl;
- a Kev frozen-suite directory (e.g. kev/evals/v4/transfer-v4), read from its development split. Kev's suites are
  not redistributed here; point at your own checkout of https://github.com/jaredpalmer/kev.
"""
import json
from pathlib import Path


def option_keys(q):
    """Options in scoring order: choice keys as given, yes/no as [false, true], score levels as their indices."""
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(q["criteria"]))]


def label_index(q):
    if q["type"] == "choice":
        return option_keys(q).index(q["label"])
    return int(q["label"])            # yes/no: False/True -> 0/1; score: level index


def load(path):
    path = Path(path)
    if path.is_dir():
        return _load_kev_suite(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_kev_suite(directory):
    records = []
    for line in (directory / "development.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line); meta = raw["_meta"]
        questions = {qid: {k: q[k] for k in ("type", "instructions", "criteria", "label") if k in q} | {"task": q["src"]}
                     for qid, q in raw["questions"].items()}
        records.append({"id": meta["id"], "variant": meta.get("variant", "clean"), "state": raw["state"], "questions": questions})
    return records


def request(record, model):
    """The /v1/systemone body for a record: state and typed questions only, never labels."""
    return {"model": model, "state": record["state"],
            "questions": {qid: {k: q[k] for k in ("type", "instructions", "criteria") if k in q} for qid, q in record["questions"].items()}}
