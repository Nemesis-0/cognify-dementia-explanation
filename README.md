# Interpretable AI for Dementia Screening

An evidence-grounded LLM explanation system for translating machine-learning predictions and feature attributions into concise, patient-specific explanations.

Developed for **CognifyChallenge 2026**, a multidisciplinary AI competition hosted in conjunction with the Midwest Healthcare Conference at the University of Illinois Urbana-Champaign.

---

## Overview

The underlying model was a LightGBM classifier for dementia screening. Rather than retraining or replacing the classifier, this project focused on a different question:

> **How can a language model faithfully communicate why a predictive model made a decision while remaining grounded in the available patient record?**

For each case, the explanation agent received:

- a patient record,
- the classifier's verdict and model probability,
- a confidence category,
- record-supported field values, and
- ranked feature attributions indicating whether each feature increased or decreased the model estimate.

The agent then generated:

1. a short explanation for a reader who had not seen the original model inputs, and
2. a structured list of the model's most important drivers.

The final system used **Qwen2.5-7B-Instruct**, 4-bit inference, deterministic decoding, two few-shot examples, and the model's top five feature attributions.

---

## Why This Was a Challenging Explanation Problem

A useful explanation had to satisfy several objectives simultaneously.

It needed to be:

- **grounded** in evidence actually present in the patient record,
- **faithful** to the fitted model's feature attributions,
- sufficiently informative for an independent reader to understand the case, and
- different across patients in ways that reflected differences in the model's underlying evidence.

The official evaluation combined these objectives as:

**Composite = Grounding × (0.20 × Faithfulness + 0.60 × Comprehension + 0.20 × Sensitivity)**

This created a genuine multi-objective design problem.

For example, adding more patient information could improve reader comprehension while simultaneously introducing lexical variation that reduced sensitivity. The goal was therefore not simply to produce longer or more fluent explanations, but to determine **which information should be transmitted and how it should be represented**.

---

## Final Architecture

The final system retained a simple and conservative core:

- exact top-five attribution drivers,
- exact attribution directions,
- record-supported values only,
- model probability and confidence information,
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

Model, prompt, few-shot, and representation decisions were made using development data rather than competition test outcomes.

The workflow used several safeguards:

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

Question-level error analysis showed that an important source of comprehension failure involved **findings explicitly reported as absent in the record**.

Many of these findings were available in the record but were not among the classifier's top five attributed drivers. A purely attribution-centered explanation therefore omitted information that could still matter to a downstream reader.

### 3. Naïve solution: transmit broad absence information

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

A more compact representation was then tested in which explicit absent findings were transmitted as a hyphen-connected sequence while the original attribution explanation remained unchanged.

One synthetic screen of the compact representation reached approximately:

| Metric | Compact representation screen |
|---|---:|
| Grounding | 1.000 |
| Faithfulness | 0.969 |
| Comprehension | 0.917 |
| Sensitivity | 0.677 |
| Composite | 0.880 |

These synthetic screens were used to study information representation. They were not treated as substitutes for real-generator validation.

The final solution retained the simpler compact absence channel without the additional structured components that made earlier variants more difficult to generate reliably.

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

Relative to the baseline:

- grounding remained essentially perfect,
- faithfulness did not decline,
- sensitivity remained near the baseline level, and
- comprehension increased materially.

This represented a favorable multi-objective trade-off: the primary comprehension weakness improved without materially sacrificing the system's strongest dimensions.

The architecture was then frozen for final submission.

---

## Base-Model Stress Test

A late-stage model comparison tested **Qwen3-8B** using the same frozen explanation architecture.

The goal was deliberately narrow:

> Could a newer base model execute the already-developed system more reliably without model-specific prompt tuning?

Qwen3-8B successfully loaded under the available hardware configuration and produced structured outputs.

However, stress testing revealed a systematic failure mode: findings that were **not recorded**, and in some cases findings that were explicitly **present**, were incorrectly placed in the absence channel.

At least five of eight stress cases exhibited false-absence behavior, and one case also contained a driver-grounding contradiction.

Because this comparison was intended to test the base model rather than reopen model-specific prompt optimization, the challenger was terminated according to the pre-specified stopping rule.

This reinforced an important lesson:

> **Greater general language-model capability does not necessarily imply better performance on a constrained evidence-transmission task.**

For this application, predictable extraction and conservative evidence handling were more important than additional generative flexibility.

---

## Organizer-Held-Out Test Result

The frozen system was subsequently evaluated by the competition organizers on held-out test cases that were not used for model or representation selection.

| Metric | Organizer-held-out test |
|---|---:|
| **Composite** | **0.8091** |
| Grounding | **0.9988** |
| Faithfulness | **0.9673** |
| Comprehension | **0.8195** |
| Sensitivity | **0.6248** |
| Coverage | **0.9850** |

The similarity between the final development benchmark and organizer-held-out evaluation was particularly informative:

| Metric | Development | Held-out test |
|---|---:|---:|
| Grounding | 1.000 | 0.999 |
| Faithfulness | 0.976 | 0.967 |
| Comprehension | 0.844 | 0.820 |
| Sensitivity | 0.624 | 0.625 |
| Composite | 0.826 | 0.809 |

Grounding, faithfulness, and sensitivity remained close to their development values, while comprehension showed a moderate decline.

The relatively small overall shift provided external evidence that the selected representation was not narrowly specific to the development benchmark.

---

## Key Methodological Lessons

### Explanation quality is multidimensional

A model explanation can be highly grounded and faithful while still failing to transmit enough information for a downstream reader.

### Error analysis can matter more than aggregate optimization

The largest improvement came from identifying a specific information bottleneck — omitted absent findings — rather than repeatedly rewriting the prompt.

### Information representation matters

Adding useful information did not automatically improve the overall system. The lexical form used to encode that information changed another evaluation dimension.

### Synthetic improvements must survive deployment

Several offline transformations achieved stronger synthetic scores than the final system but required operations that the constrained generator could not reliably perform.

Operational reliability was therefore treated as part of model quality.

### The final architecture favored simplicity and deployment reliability

Development explored several increasingly structured representations, but the final system returned to a comparatively simple design.

It preserved the stable attribution-explanation core, added only the additional information channel required to address the dominant comprehension weakness, and avoided formatting tasks that increased generator failure risk.

The organizer-held-out evaluation subsequently showed performance close to the final development benchmark.

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

A detailed account of the validation strategy, failure analysis, intermediate representations, deployment constraints, ablations, base-model challenge, and organizer-held-out evaluation.

---

## Reproducibility and Data Availability

This repository is intended to document the **system design, validation methodology, final configuration, and major development findings** of the project.

Competition test cases are not included.

The repository does not attempt to redistribute the complete organizer-provided evaluation environment. Instead, it focuses on the components authored and selected during development.

Synthetic-screen results are explicitly distinguished from real-generator development results and organizer-held-out evaluation.

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
| Development composite | **0.826** |
| Organizer-held-out composite | **0.8091** |

---

## Scope and Disclaimer

This project was developed for an educational AI competition using competition-provided data and evaluation procedures.

The system explains the reasoning of an existing dementia-screening classifier; it does **not** independently diagnose dementia and is **not intended for clinical use**.

The project should be interpreted as a study in:

- interpretable machine learning,
- evidence-grounded language generation,
- model evaluation,
- validation design,
- information representation, and
- reliable LLM inference under computational constraints.

---

## Author

**Miao Chen**  
B.S. Statistics, Minor in Data Science  
University of Illinois Urbana-Champaign

Solo team: **MiaoScope**