"""Convert jabr/classifier-benchmark suites (CC0) to gutcheck JSONL, using that repo's own loader.

    python suites/convert_jabr.py /path/to/classifier-benchmark v2 > suites/jabr-v2.jsonl

Source: https://github.com/jabr/classifier-benchmark (CC0 1.0). Each case becomes one record with a single question.
"""
import json, sys
from dataclasses import asdict
from pathlib import Path

repo, suite = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(repo))
from bench.cases import load_suite, question_payload  # noqa: E402

for task in load_suite(repo / "cases" / f"{suite}.toml"):
    q = question_payload(task.question)
    for i, case in enumerate(task.cases):
        expected = case.expected
        label = bool(expected) if task.type == "noul" else (int(expected) if task.type == "score" else str(expected))
        question = {**q, "label": label, "task": task.id}
        if task.type == "noul" and not question.get("criteria"):
            question.pop("criteria", None)
        print(json.dumps({"id": f"{task.id}/{i}", "variant": "clean", "state": case.state,
                          "questions": {"answer": question}}, ensure_ascii=False))
