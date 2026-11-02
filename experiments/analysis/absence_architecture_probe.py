import json
import random
import re
from collections import Counter
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
absence_q = [q for q in questions if q["type"] == "absence"]

TOKEN_RE = re.compile(r"[a-z][a-z\-']+")
SENSITIVITY_PAIRS = 20000
SEED = 20260808


def citable_map(case):
    cf = case.get("citable_fields", {})
    if isinstance(cf, dict):
        return cf
    return {
        x["field"]: x.get("value")
        for x in cf
        if isinstance(x, dict) and "field" in x
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


def extract_absent_text(case):
    pieces = []

    for para in case["record_text"].split("\n\n"):
        if "Reported absent:" not in para:
            continue

        text = para.split("Reported absent:", 1)[1].strip()
        text = text.rstrip(".")
        pieces.append(text)

    return "; ".join(pieces)


# Natural-language label for each of the 37 answer fields.
label_votes = {}

for q in absence_q:
    field = q["answer_key"]
    label = q["options"][q["answer_index"]]
    label_votes.setdefault(field, Counter())[label] += 1

labels = {
    field: votes.most_common(1)[0][0]
    for field, votes in label_votes.items()
}

answer_freq = Counter(q["answer_key"] for q in absence_q)
ranked_fields = [
    field for field, _ in answer_freq.most_common()
]


def targeted_absent_text(case, selected):
    cf = citable_map(case)

    absent_labels = [
        labels[field]
        for field in selected
        if field in cf and cf[field] == 0
    ]

    return ", ".join(absent_labels)


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
        for w, c in d.items():
            m[r, idx[w]] = c

    m *= idf
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return m / norms


def sensitivity(notes):
    ids = sorted(notes)
    m = tfidf_matrix([notes[c] for c in ids])

    ref = {
        cid: {a["field"] for a in cases[cid]["attributions"][:5]}
        for cid in ids
    }

    rng = random.Random(SEED)
    pairs = set()

    total = len(ids) * (len(ids) - 1) // 2

    if total <= SENSITIVITY_PAIRS:
        pairs = {
            (i, j)
            for i in range(len(ids))
            for j in range(i + 1, len(ids))
        }
    else:
        while len(pairs) < SENSITIVITY_PAIRS:
            i = rng.randrange(len(ids))
            j = rng.randrange(len(ids))

            if i != j:
                pairs.add((min(i, j), max(i, j)))

    prose = []
    evidence = []

    for i, j in sorted(pairs):
        prose.append(1.0 - float(np.dot(m[i], m[j])))

        a = ref[ids[i]]
        b = ref[ids[j]]
        union = a | b

        evidence.append(
            1.0 - len(a & b) / len(union)
            if union else 0.0
        )

    rho = float(spearmanr(prose, evidence).statistic)
    return max(0.0, rho)


def absence_coverage(selected=None, full=False):
    correct = 0

    for q in absence_q:
        case = cases[q["case_id"]]
        field = q["answer_key"]

        top5 = {
            a["field"]
            for a in case["attributions"][:5]
        }

        if field in top5:
            correct += 1
        elif full:
            correct += 1
        elif selected is not None and field in selected:
            correct += 1

    return correct


def stats(name, selected=None, full=False):
    notes = {}
    extra_counts = []
    total_counts = []

    for cid, case in cases.items():
        base = base_note(case)

        if full:
            extra = extract_absent_text(case)
        elif selected:
            extra = targeted_absent_text(case, selected)
        else:
            extra = ""

        if extra:
            note = base + " Record reported absent: " + extra + "."
        else:
            note = base

        notes[cid] = note
        extra_counts.append(len(extra.split()))
        total_counts.append(len(note.split()))

    cov = absence_coverage(selected, full)
    sens = sensitivity(notes)

    print(f"\n{name}")
    print("-" * 72)
    print(
        f"absence structural coverage: "
        f"{cov}/{len(absence_q)} = {cov/len(absence_q):.1%}"
    )
    print(f"sensitivity proxy:           {sens:.4f}")
    print(
        f"extra words mean/p50/p90/max: "
        f"{np.mean(extra_counts):.1f} / "
        f"{np.median(extra_counts):.0f} / "
        f"{np.percentile(extra_counts,90):.0f} / "
        f"{max(extra_counts)}"
    )
    print(
        f"total words mean/p50/p90/max: "
        f"{np.mean(total_counts):.1f} / "
        f"{np.median(total_counts):.0f} / "
        f"{np.percentile(total_counts,90):.0f} / "
        f"{max(total_counts)}"
    )


print("=" * 72)
print("ABSENCE NOTE ARCHITECTURE PROBE")
print("=" * 72)

stats("BASE")

for k in [5, 10, 15, 20, 25]:
    stats(
        f"TOP-{k} ABSENCE FIELDS",
        selected=set(ranked_fields[:k])
    )

stats("FULL REPORTED-ABSENT TEXT", full=True)

print("\n" + "=" * 72)
print("TOP ABSENCE FIELDS")
print("=" * 72)

for i, field in enumerate(ranked_fields[:25], 1):
    print(
        f"{i:2d}. {field:22s} "
        f"questions={answer_freq[field]:2d} "
        f"label={labels[field]}"
    )
