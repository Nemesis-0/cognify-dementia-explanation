import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(".")
QPATH = ROOT / "experiments/devkit/cases/questions_dev.json"
CPATH = ROOT / "experiments/devkit/cases/cohort_dev.json"

with QPATH.open() as f:
    qdata = json.load(f)

with CPATH.open() as f:
    cdata = json.load(f)

questions = qdata["questions"]
cases = {c["case_id"]: c for c in cdata["cases"]}

def citable_map(case):
    cf = case.get("citable_fields", {})
    if isinstance(cf, dict):
        return cf
    return {
        x["field"]: x.get("value")
        for x in cf
        if isinstance(x, dict) and "field" in x
    }

def attribution_rank(case, field):
    for i, a in enumerate(case["attributions"], 1):
        if a["field"] == field:
            return i
    return None

counts = Counter(q["type"] for q in questions)

rank_buckets = defaultdict(Counter)
covered = Counter()
uncovered = Counter()
uncovered_fields = defaultdict(Counter)

for q in questions:
    typ = q["type"]
    cid = q["case_id"]
    key = q.get("answer_key")
    case = cases[cid]

    top5 = case["attributions"][:5]
    top5_fields = {a["field"] for a in top5}
    cf = citable_map(case)

    rank = attribution_rank(case, key)

    if rank is None:
        rank_buckets[typ]["not_attributed"] += 1
    elif rank <= 5:
        rank_buckets[typ]["rank_1_5"] += 1
    elif rank <= 8:
        rank_buckets[typ]["rank_6_8"] += 1
    else:
        rank_buckets[typ]["rank_9_20"] += 1

    if typ == "verdict_confidence":
        has_fact = True

    elif typ in {
        "leading_driver",
        "secondary_driver",
        "direction",
        "counter_evidence",
    }:
        has_fact = key in top5_fields

    elif typ == "cited_value":
        has_fact = key in top5_fields and key in cf

    elif typ == "not_established":
        has_fact = key in top5_fields and key not in cf

    elif typ == "absence":
        has_fact = (
            key in top5_fields
            and key in cf
            and cf[key] == 0
        )

    else:
        has_fact = False

    if has_fact:
        covered[typ] += 1
    else:
        uncovered[typ] += 1
        if key is not None:
            uncovered_fields[typ][key] += 1

print("=" * 72)
print("MiaoScope CURRENT-NOTE STRUCTURAL COVERAGE MAP")
print("=" * 72)

print(f"\nTotal questions: {len(questions)}")

print("\nQUESTION TYPES")
print("-" * 72)

for typ in sorted(counts):
    total = counts[typ]
    cov = covered[typ]
    print(
        f"{typ:20s}"
        f" total={total:4d}"
        f"  carried={cov:4d}"
        f"  missing={total-cov:4d}"
        f"  structural_coverage={cov/total:6.1%}"
    )

total_cov = sum(covered.values())

print("\n" + "=" * 72)
print("OVERALL")
print("=" * 72)
print(f"Facts structurally carried by current note: {total_cov}/{len(questions)}")
print(f"Structural coverage: {total_cov/len(questions):.1%}")

print("\n" + "=" * 72)
print("ANSWER-KEY ATTRIBUTION RANK DISTRIBUTION")
print("=" * 72)

for typ in sorted(counts):
    b = rank_buckets[typ]
    print(
        f"{typ:20s}"
        f" rank1-5={b['rank_1_5']:4d}"
        f" rank6-8={b['rank_6_8']:4d}"
        f" rank9-20={b['rank_9_20']:4d}"
        f" not-attributed={b['not_attributed']:4d}"
    )

print("\n" + "=" * 72)
print("TOP MISSING ANSWER FIELDS BY QUESTION TYPE")
print("=" * 72)

for typ in sorted(uncovered_fields):
    if not uncovered_fields[typ]:
        continue

    print(f"\n{typ}")
    for field, n in uncovered_fields[typ].most_common(15):
        print(f"  {field:24s} {n:3d}")

print("\nNOTE:")
print(
    "This is a structural information-availability map, not the official "
    "Claude comprehension score. It asks whether the answer fact is present "
    "in the current verdict+top5 note design."
)
