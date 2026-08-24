# Interpretable AI for Dementia Screening

An evidence-grounded LLM explanation system for translating machine-learning predictions and feature attributions into concise, patient-specific explanations.

Developed for **CognifyChallenge 2026**, a multidisciplinary AI competition hosted in conjunction with the Midwest Healthcare Conference at the University of Illinois Urbana-Champaign.

---

## Overview

The underlying model was a LightGBM classifier for dementia screening. Rather than retraining or replacing the classifier, this project focused on a different question:

> **How can a language model faithfully communicate why a predictive model made a decision, while remaining grounded in the available patient record?**

For each case, the explanation agent received:

- a patient record,
- the classifier's verdict and calibrated dementia probability,
- a confidence category,
- record-supported field values, and
- ranked feature attributions indicating whether each feature increased or decreased the model estimate.

The agent then generated:

1. a short explanation for a reader who had not seen the original model inputs, and
2. a structured list of the model's most important drivers.

The final system used **Qwen2.5-7B-Instruct**, 4-bit inference, deterministic decoding, two few-shot examples, and the model's top five feature attributions.

---

## Why This Was a Challenging Explanation Problem

A good explanation had to satisfy several objectives simultaneously.

It needed to be:

- **grounded** in evidence actually present in the patient record,
- **faithful** to the fitted model's feature attributions,
- sufficiently informative for an independent reader to understand the case, and
- different across patients in ways that reflected differences in the model's underlying evidence.

The official evaluation combined these objectives as:

\[
\text{Composite}
=
\text{Grounding}
\times
\left(
0.20\,\text{Faithfulness}
+
0.60\,\text{Comprehension}
+
0.20\,\text{Sensitivity}
\right).
\]

This created a genuine multi-objective design problem.

For example, adding more patient information could improve reader comprehension while simultaneously introducing irrelevant lexical variation that reduced sensitivity. The goal was therefore not simply to produce longer or more fluent explanations, but to determine **which information should be transmitted and how it should be represented**.

---

## Final Architecture

The final system retained a simple and conservative core:

- exact top-five attribution drivers,
- exact attribution directions,
- record-supported values only,
- calibrated prediction and confidence information,
- concise natural-language explanation, and
- a compact channel for explicitly absent findings.

Conceptually:

```text
Patient record
      +
Model prediction
      +
Top feature attributions
      │
      ▼
Qwen2.5-7B-Instruct
      │
      ├── Prediction summary
      ├── Five attributed drivers
      ├── Record-grounded values
      └── Compact explicit-absence information
      │
      ▼
Evidence-grounded explanation
```

The submitted configuration is available at:

[`configs/final_submission.yaml`](configs/final_submission.yaml)

---

## Validation Strategy

A major focus of the project was preventing development results from becoming a substitute for genuine generalization.

The workflow therefore used several safeguards:

- **Development-only model selection:** competition test outcomes were not used for prompt or architecture selection.
- **Frozen development subsets:** important comparisons were performed on deterministic subsets selected before generation.
- **Paired evaluation:** competing representations were compared on the same cases whenever possible.
- **Failure-mode analysis:** question-level errors were examined rather than relying only on aggregate scores.
- **Hard stopping rules:** architectures that damaged grounding, exceeded the deployment constraint, or became operationally unstable were terminated instead of repeatedly tuned.
- **Real-generator confirmation:** promising synthetic representations were not accepted until the actual constrained LLM could reproduce them reliably.

Once a development subset had been inspected and used to motivate a design change, it was treated as development data rather than as fresh validation.

---

## From Error Analysis to the Final Representation

### 1. Stable baseline

The initial strong baseline achieved excellent grounding and attribution faithfulness but materially weaker reader comprehension.

| Metric | Baseline development result |
|---|---:|
| Grounding | 1.000 |
| Faithfulness | 0.969 |
| Comprehension | 0.758 |
| Sensitivity | 0.618 |
| Composite | 0.772 |

### 2. Identifying the missing information

Question-level analysis showed that many comprehension errors involved **findings explicitly reported as absent in the record**.

These findings often mattered to the reader but were not among the classifier's top five attributed drivers, so a purely attribution-centered explanation systematically omitted them.

### 3. Naïve solution: transmit all absence information

Adding broad absence information substantially improved comprehension, but the additional patient-specific vocabulary caused sensitivity to deteriorate.

| Metric | Full-absence synthetic screen |
|---|---:|
| Grounding | 1.000 |
| Faithfulness | 0.969 |
| Comprehension | 0.909 |
| Sensitivity | 0.408 |
| Composite | 0.821 |

This demonstrated that the **information itself was useful**, but its representation was not.

### 4. Representation redesign

Several structured alternatives were tested to separate semantic information from unnecessary lexical variation.

A structured state representation reached a synthetic composite of approximately **0.874**, but proved too difficult for the constrained generator to execute reliably.

The final solution therefore used a simpler representation: explicit absent findings were transmitted in a compact hyphen-connected sequence while the original attribution explanation remained unchanged.

This preserved most of the comprehension benefit without sacrificing the stability of the core generator.

---

## Real-Generator Development Result

The final architecture was evaluated using actual Qwen2.5-7B generation on a 50-case development benchmark.

| Metric | Final development run |
|---|---:|
| Coverage | **1.000** |
| Grounding | **1.000** |
| Faithfulness | **0.976** |
| Comprehension | **0.844** |
| Sensitivity | **0.624** |
| Composite | **0.826** |

All **50/50 cases produced usable outputs**.

Relative to the baseline, the final representation substantially improved comprehension while preserving grounding, attribution fidelity, and sensitivity.

---

## Organizer-Held-Out Test Result

The submitted system was subsequently evaluated by the competition organizers on held-out test cases that were not used for model or representation selection.

| Metric | Organizer-held-out test |
|---|---:|
| **Composite** | **0.8091** |
| Grounding | **0.9988** |
| Faithfulness | **0.9673** |
| Comprehension | **0.8195** |
| Sensitivity | **0.6248** |
| Coverage | **0.9850** |

The submission ranked **12th among 33 valid entries**.

The similarity between development and organizer-held-out results was particularly informative:

| Metric | Development | Held-out test |
|---|---:|---:|
| Grounding | 1.000 | 0.999 |
| Faithfulness | 0.976 | 0.967 |
| Comprehension | 0.844 | 0.820 |
| Sensitivity | 0.624 | 0.625 |
| Composite | 0.826 | 0.809 |

The relatively small shift between development and held-out performance provided external evidence that the final validation strategy and representation generalized beyond the development cases.

---

## Base-Model Stress Test

A late-stage model comparison tested **Qwen3-8B** using the same frozen explanation architecture.

The goal was deliberately narrow:

> Could a newer base model execute the already-developed system more reliably without model-specific prompt tuning?

Qwen3-8B successfully generated structured outputs, but stress testing revealed systematic false-absence errors. In several cases, findings that were unrecorded — and occasionally findings that were explicitly present — were incorrectly placed in the absence channel.

Because the task strongly rewarded conservative evidence fidelity, the challenger was rejected under a pre-specified stopping rule rather than retuned.

This reinforced an important lesson:

> **Greater general language-model capability does not necessarily imply better performance on a constrained evidence-transmission task.**

For this application, predictable extraction and exact evidence handling were more important than additional generative flexibility.

---

## Key Methodological Lessons

### Explanation quality is multidimensional

A model explanation can be perfectly grounded and faithful while still failing to transmit enough information for a downstream reader.

### Error analysis can matter more than aggregate optimization

The largest improvement came from identifying a specific information bottleneck — omitted absent findings — rather than repeatedly rewriting the prompt.

### Information representation matters

Adding useful information did not automatically improve the overall system. The lexical form used to encode that information changed another evaluation dimension.

### Synthetic improvements must survive deployment

Several offline representations achieved stronger synthetic scores than the final system but could not be generated reliably under the 16 GB inference constraint.

Operational reliability was therefore treated as part of model quality.

### Simpler systems can generalize better

The final system was substantially simpler than several intermediate architectures.

It preserved the stable attribution-explanation core and added only the additional information channel required to address the dominant comprehension failure mode.

---

## Repository Structure

```text
cognify-dementia-explanation/
│
├── README.md
│
├── configs/
│   └── final_submission.yaml
│
└── docs/
    └── methodology_and_ablations.md
```

### `configs/final_submission.yaml`

The final Qwen2.5-7B configuration submitted to the competition, including inference settings, prompt instructions, and few-shot examples.

### `docs/methodology_and_ablations.md`

A more detailed account of the validation strategy, failure analysis, intermediate representations, deployment constraints, ablations, and final held-out evaluation.

---

## Reproducibility and Data Availability

This repository is intended to document the **system design, validation methodology, and final configuration** developed for the project.

Competition test cases are not included.

The repository also does not attempt to redistribute the complete organizer-provided evaluation environment. Instead, it focuses on the components authored and selected during development.

The reported held-out metrics are the results returned by the competition's organizer-side evaluation.

---

## Technical Summary

| Component | Final choice |
|---|---|
| Base model | Qwen2.5-7B-Instruct |
| Quantization | 4-bit |
| Decoding | Deterministic |
| Few-shot examples | 2 |
| Model evidence | Top 5 feature attributions |
| Explanation strategy | Grounded attribution explanation + compact explicit-absence channel |
| GPU constraint | <16 GB VRAM |
| Official held-out composite | **0.8091** |

---

## Scope and Disclaimer

This project was developed for an educational AI competition using competition-provided data and evaluation procedures.

The system explains the reasoning of an existing dementia-screening classifier; it does **not** independently diagnose dementia and is **not intended for clinical use**.

The project should be interpreted as a study in:

- interpretable machine learning,
- evidence-grounded language generation,
- model evaluation,
- validation design, and
- information representation under computational constraints.

---

## Author

**Miao Chen**  
B.S. Statistics, Minor in Data Science  
University of Illinois Urbana-Champaign

Solo team: **MiaoScope**