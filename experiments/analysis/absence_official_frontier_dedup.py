import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(".")
CPATH = ROOT / "experiments/devkit/cases/cohort_dev.json"
QPATH = ROOT / "experiments/devkit/cases/questions_dev.json"

with CPATH.open() as f:
    cdata = json.load(f)

with QPATH.open() as f:
    qdata = json.load(f)

cases = {c["case_id"]: c for c in cdata["cases"]}
questions = qdata["questions"]

TOKEN_RE = re.compile(r"[a-z][a-z\-']+")
SEED = 20260808
N_PAIRS = 20000

by_case = defaultdict(list)
for q in questions:
    by_case[q["case_id"]].append(q)

absence_q = [q for q in questions if q["type"] == "absence"]


def citable_map(case):
    cf = case.get("citable_fields", {})
    if isinstance(cf, dict):
        return cf
    return {
        x["field"]: x.get("value")
        for x in cf
        if isinstance(x, dict) and "field" in x
    }


def top5_fields(case):
    return {a["field"] for a in case["attributions"][:5]}


# ------------------------------------------------------------------
# Marginal comprehension value of each absence question under the
# official per-case baseline-normalized formula.
#
# If an uncovered question is expected to perform at its measured
# baseline and a covered question becomes correct, its contribution is:
#
# (1 - q_baseline) / [n_questions_in_case * (1 - mean_case_baseline)]
# ------------------------------------------------------------------

field_utility = Counter()

for cid, qs in by_case.items():
    mean_base = sum(q["baseline"] for q in qs) / len(qs)
    denom = len(qs) * (1.0 - mean_base)

    if denom <= 0:
        continue

    for q in qs:
        if q["type"] != "absence":
            continue

        case = cases[cid]
        field = q["answer_key"]

        # Already structurally carried by current top-5 note.
        if field in top5_fields(case):
            continue

        gain = (1.0 - q["baseline"]) / denom
        field_utility[field] += gain


ranked_fields = [
    field for field, _ in field_utility.most_common()
]


# Most common natural-language option label for each absence field.
label_votes = defaultdict(Counter)

for q in absence_q:
    field = q["answer_key"]
    label = q["options"][q["answer_index"]]
    label_votes[field][label] += 1

labels = {
    field: votes.most_common(1)[0][0]
    for field, votes in label_votes.items()
}


def base_note(case):
    mo = case["model_output"]
    attrs = case["attributions"][:5]

    parts = [
        f"{mo['verdict']} probability {mo['dementia_probability']} "
        f"confidence {mo['confidence']}."
    ]

    parts += [
        f"{a['field']} {a['direction']}."
        for a in attrs
    ]

    return " ".join(parts)


def targeted_absent_text(case, selected):
    cf = citable_map(case)
    already_described = top5_fields(case)

    return ", ".join(
        labels[field]
        for field in selected
        if field not in already_described
        and field in cf
        and cf[field] == 0
    )


def structurally_covered(q, selected):
    case = cases[q["case_id"]]
    typ = q["type"]
    key = q.get("answer_key")

    top5 = top5_fields(case)
    cf = citable_map(case)

    if typ == "verdict_confidence":
        return True

    if typ in {
        "leading_driver",
        "secondary_driver",
        "direction",
    }:
        return key in top5

    if typ == "counter_evidence":
        return key in top5

    if typ == "cited_value":
        return key in top5 and key in cf

    if typ == "not_established":
        return key in top5 and key not in cf

    if typ == "absence":
        if key in top5 and key in cf and cf[key] == 0:
            return True

        return (
            key in selected
            and key in cf
            and cf[key] == 0
        )

    return False


def comprehension_proxy(selected):
    case_scores = []

    for cid, qs in by_case.items():
        mean_base = sum(q["baseline"] for q in qs) / len(qs)

        expected_acc = sum(
            1.0 if structurally_covered(q, selected) else q["baseline"]
            for q in qs
        ) / len(qs)

        if mean_base >= 1.0:
            score = 0.0
        else:
            score = (
                (expected_acc - mean_base)
                / (1.0 - mean_base)
            )
            score = max(0.0, min(1.0, score))

        case_scores.append(score)

    return sum(case_scores) / len(case_scores)


def tfidf_matrix(texts):
    docs = [Counter(TOKEN_RE.findall(t.lower())) for t in texts]
    vocab = sorted({w for d in docs for w in d})
    idx = {w: i for i, w in enumerate(vocab)}

    n = len(docs)
    df = np.zeros(len(vocab))

    for d in docs:
        for w in d:
            df[idx[w]] += 1

    idf = np.log((1 + n) / (1 + df)) + 1.0
    m = np.zeros((n, len(vocab)))

    for r, d in enumerate(docs):
        for word, count in d.items():
            m[r, idx[word]] = count

    m *= idf

    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return m / norms


def sensitivity(selected):
    ids = sorted(cases)

    notes = {}

    for cid in ids:
        case = cases[cid]
        note = base_note(case)

        extra = targeted_absent_text(case, selected)

        if extra:
            note += " Record reported absent: " + extra + "."

        notes[cid] = note

    m = tfidf_matrix([notes[c] for c in ids])

    ref = {
        cid: top5_fields(cases[cid])
        for cid in ids
    }

    rng = random.Random(SEED)
    pairs = set()

    total = len(ids) * (len(ids) - 1) // 2

    if total <= N_PAIRS:
        pairs = {
            (i, j)
            for i in range(len(ids))
            for j in range(i + 1, len(ids))
        }
    else:
        while len(pairs) < N_PAIRS:
            i = rng.randrange(len(ids))
            j = rng.randrange(len(ids))

            if i != j:
                pairs.add((min(i, j), max(i, j)))

    prose = []
    evidence = []

    for i, j in sorted(pairs):
        prose.append(
            1.0 - float(np.dot(m[i], m[j]))
        )

        a = ref[ids[i]]
        b = ref[ids[j]]
        union = a | b

        evidence.append(
            1.0 - len(a & b) / len(union)
            if union else 0.0
        )

    rho = float(
        spearmanr(prose, evidence).statistic
    )

    return max(0.0, rho)


rows = []

for k in range(len(ranked_fields) + 1):
    selected = set(ranked_fields[:k])

    comp = comprehension_proxy(selected)
    sens = sensitivity(selected)

    # Faithfulness is constant during this architecture probe,
    # so only the variable 60/20 terms matter for choosing k.
    variable_score = 0.60 * comp + 0.20 * sens

    rows.append({
        "k": k,
        "comp": comp,
        "sens": sens,
        "variable": variable_score,
    })


best = max(rows, key=lambda x: x["variable"])

print("=" * 78)
print("OFFICIAL-WEIGHTED ABSENCE FRONTIER")
print("=" * 78)

print("\nTop fields ranked by baseline-normalized comprehension utility:")
for i, field in enumerate(ranked_fields, 1):
    print(
        f"{i:2d}. {field:22s} "
        f"utility={field_utility[field]:.5f} "
        f"label={labels[field]}"
    )

print("\n" + "=" * 78)
print("FRONTIER")
print("=" * 78)

print(
    f"{'k':>3s} "
    f"{'comp_proxy':>12s} "
    f"{'sensitivity':>12s} "
    f"{'0.6C+0.2S':>12s}"
)

for r in rows:
    marker = "  <-- BEST" if r["k"] == best["k"] else ""

    print(
        f"{r['k']:3d} "
        f"{r['comp']:12.4f} "
        f"{r['sens']:12.4f} "
        f"{r['variable']:12.4f}"
        f"{marker}"
    )

print("\n" + "=" * 78)
print("BEST")
print("=" * 78)

print(f"k = {best['k']}")
print(f"comprehension proxy = {best['comp']:.4f}")
print(f"sensitivity         = {best['sens']:.4f}")
print(f"0.6C + 0.2S         = {best['variable']:.4f}")

print("\nSelected fields:")

for i, field in enumerate(ranked_fields[:best["k"]], 1):
    print(
        f"{i:2d}. {field:22s} {labels[field]}"
    )
