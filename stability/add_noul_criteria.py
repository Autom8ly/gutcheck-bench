"""Write a copy of a gutcheck suite whose yes/no (noul) questions carry explicit criteria {"false": "No", "true": "Yes"}.
Used to check whether Julia-1's yes/no results depend on that wording (they don't: see stability/README.md).

    python stability/add_noul_criteria.py suites/jabr-v2.jsonl /tmp/jabr-v2-noul-criteria.jsonl
"""
import json, sys

src, dst = sys.argv[1], sys.argv[2]
with open(dst, "w", encoding="utf-8") as out:
    for line in open(src, encoding="utf-8"):
        record = json.loads(line)
        for q in record["questions"].values():
            if q["type"] == "noul":
                q["criteria"] = {"false": "No", "true": "Yes"}
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
