import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path(".")
QPATH = ROOT / "experiments/devkit/cases/questions_dev.json"
CPATH = ROOT / "experiments/devkit/cases/cohort_dev.json"

with QPATH.open() as f:
    qdata = json.load(f)

with CPATH.open() as f:
    cdata = json.load(f)

questions = qdata["questions"]
cases = {c["case_id"]: c for c in cdata["cases"]}

absence_questions = [q for q in questions if q["type"] == "absence"]
absence_fields = sorted({q["answer_key"] for q in absence_questions})

def citable_map(case):
    cf = case.get("citable_fields", {})
    if isinstance(cf, dict):
        return cf
    return {
        x["field"]: x.get("value")
        for x in cf
        if isinstance(x, dict) and "field" in x
    }

rows = []

for cid, case in cases.items():
    cf = citable_map(case)

    present = []
    absent = []
    unresolved = []

    for field in absence_fields:
        if field not in cf:
            unresolved.append(field)
        elif cf[field] == 0:
            absent.append(field)
        else:
            present.append(field)

    rows.append({
        "case_id": cid,
        "present": present,
        "absent": absent,
        "unresolved": unresolved,
    })

present_counts = [len(r["present"]) for r in rows]
absent_counts = [len(r["absent"]) for r in rows]
unresolved_counts = [len(r["unresolved"]) for r in rows]

print("=" * 72)
print("ABSENCE COMPRESSION ANALYSIS")
print("=" * 72)

print(f"\nAbsence questions: {len(absence_questions)}")
print(f"Distinct absence-answer fields: {len(absence_fields)}")

print("\nPER-CASE COUNTS WITHIN ABSENCE FIELD UNIVERSE")
print("-" * 72)

print(
    f"present:    mean={mean(present_counts):.2f} "
    f"median={median(present_counts):.1f} "
    f"min={min(present_counts)} max={max(present_counts)}"
)

print(
    f"absent:     mean={mean(absent_counts):.2f} "
    f"median={median(absent_counts):.1f} "
    f"min={min(absent_counts)} max={max(absent_counts)}"
)

print(
    f"unresolved: mean={mean(unresolved_counts):.2f} "
    f"median={median(unresolved_counts):.1f} "
    f"min={min(unresolved_counts)} max={max(unresolved_counts)}"
)

print("\nREPRESENTATION COST PROXY")
print("-" * 72)

print(
    "Direct absence listing requires on average "
    f"{mean(absent_counts):.2f} field mentions/case."
)

print(
    "Closed-world present listing requires on average "
    f"{mean(present_counts):.2f} present field mentions/case "
    "+ one rule saying all other established fields in this set are absent."
)

print("\nMOST COMMON PRESENT FIELDS")
print("-" * 72)

present_freq = Counter()
absent_freq = Counter()
unresolved_freq = Counter()

for r in rows:
    present_freq.update(r["present"])
    absent_freq.update(r["absent"])
    unresolved_freq.update(r["unresolved"])

for field, n in present_freq.most_common(20):
    print(f"{field:24s} {n:3d}")

print("\nMOST COMMON ABSENT FIELDS")
print("-" * 72)

for field, n in absent_freq.most_common(20):
    print(f"{field:24s} {n:3d}")

print("\nMOST COMMON UNRESOLVED FIELDS")
print("-" * 72)

for field, n in unresolved_freq.most_common(20):
    print(f"{field:24s} {n:3d}")

print("\nQUESTION-WEIGHTED ANSWER FIELD FREQUENCY")
print("-" * 72)

answer_freq = Counter(q["answer_key"] for q in absence_questions)

for field, n in answer_freq.most_common():
    print(f"{field:24s} {n:3d}")
