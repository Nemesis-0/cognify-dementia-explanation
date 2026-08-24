# Methodology and Ablation Study

## Overview

This project developed an evidence-grounded LLM explanation system for a LightGBM dementia-screening classifier in CognifyChallenge 2026.

The objective was not to improve the underlying classifier or diagnose patients. Instead, the system translated patient-record evidence, model predictions, and feature attributions into concise explanations while balancing four evaluation objectives:

- grounding to the patient record,
- faithfulness to the model's attributed drivers,
- comprehension by a reader who sees only the generated explanation, and
- sensitivity of explanations to differences in model evidence.

Development followed a clean-room, development-only workflow. Model, prompt, few-shot, and representation decisions were made using development data rather than competition test outcomes. Development subsets were frozen before targeted comparisons; once a subset had been inspected or used to guide design, it was treated as development data rather than as an untouched validation set.

The final system used Qwen2.5-7B-Instruct with 4-bit inference, two few-shot examples, and five model attributions per case.

---

## Problem Setting

The underlying LightGBM model produced a dementia-screening prediction together with ranked feature attributions. For each case, the explanation system received:

- a short patient record,
- record-supported fields and values,
- the classifier's verdict and calibrated probability,
- a confidence category, and
- ranked model attributions with directions indicating whether each feature increased or decreased the model estimate.

The generated output contained two components:

1. a short explanatory note intended for a reader who had not seen the original model inputs, and
2. a structured list of model drivers with field names, values when supported by the record, and attribution directions.

The task therefore required more than producing fluent clinical prose. A useful explanation had to communicate sufficient information for downstream understanding while remaining exactly aligned with the fitted model and with evidence present in the patient record.

---

## Evaluation Framework

The official evaluation combined four complementary criteria:

$$
\mathrm{Composite}
=
\mathrm{Grounding}
\times
\left(
0.20 \cdot \mathrm{Faithfulness}
+
0.60 \cdot \mathrm{Comprehension}
+
0.20 \cdot \mathrm{Sensitivity}
\right)
$$

Each component measured a different failure mode:

- **Grounding** checked whether named drivers and quoted values were supported by the patient record. Because grounding acted as a multiplicative factor, unsupported evidence could substantially reduce the overall score.
- **Faithfulness** measured agreement between the generated explanation and the model's highest-ranked feature attributions, including driver identity and direction.
- **Comprehension** measured whether an independent reader, given only the generated note, could correctly answer questions about the patient's model explanation and record-supported evidence.
- **Sensitivity** measured whether differences between explanations across patients tracked differences in the underlying model evidence.

This multi-objective structure created an important methodological constraint: transmitting more patient information could improve comprehension while simultaneously introducing lexical variation that reduced sensitivity.

The project therefore treated explanation design as a **constrained multi-objective optimization problem**, rather than optimizing any single metric in isolation.

---

## Development and Validation Design

The development workflow was designed to reduce leakage and avoid selecting a system from a single favorable sample.

Several principles were used throughout the project.

### Development-only optimization

Prompt design, few-shot selection, attribution count, representation design, and model comparisons were performed using development cases only. Competition test outcomes were not used for model or prompt selection.

### Frozen comparison subsets

For major architecture comparisons, a deterministic subset of development cases was selected before generation. Important case characteristics included:

- confidence category,
- record length,
- number of observed and unobserved variables,
- attribution composition,
- amount of explicitly absent information, and
- question type.

Once a subset had been inspected or used to motivate a new design, it was no longer treated as fresh validation.

### Paired comparisons

Competing systems were evaluated on the same cases whenever possible. This reduced case-level variability and made changes in grounding, faithfulness, comprehension, and sensitivity easier to attribute to the representation itself.

### Hard stopping rules

Several unsuccessful branches were terminated rather than repeatedly tuned. Examples included:

- architectures that exceeded the 16 GB deployment constraint,
- variants that damaged exact driver copying,
- representations that produced large sensitivity losses, and
- model challengers that systematically inferred absent findings not explicitly supported by the record.

This was important because the final system had to run reliably under the organizer's fixed inference environment rather than only produce high offline scores.

---

## Baseline Architecture

An early stable system, referred to during development as the **v10 baseline**, established the core architecture that was retained in the final submission.

Its main design choices were:

- Qwen2.5-7B-Instruct,
- 4-bit inference,
- deterministic decoding,
- two few-shot examples,
- the top five model attributions,
- exact copying of driver identity and direction,
- record-supported values only, and
- short prose describing the verdict, probability, confidence category, and five model drivers.

On a 50-case development benchmark, this system achieved approximately:

| Metric | Baseline |
|---|---:|
| Grounding | 1.000 |
| Faithfulness | 0.969 |
| Comprehension | 0.758 |
| Sensitivity | 0.618 |
| Composite | 0.772 |

The baseline was highly reliable on grounding and attribution fidelity, but comprehension was materially lower than the other components.

This motivated a focused analysis of what information the generated note failed to transmit.

---

## Identifying the Comprehension Bottleneck

Question-level analysis showed that the largest weakness involved **explicitly absent findings**.

For many absence-related questions, the relevant information was available in the patient record but was not among the model's top five attributions. Because the baseline note concentrated almost entirely on the top model drivers, this broader record information was usually omitted.

A development audit found that nearly all absence-question answers were supported somewhere in the record or citable field set, while essentially none of them appeared among the top model attributions.

This revealed a structural mismatch:

> The top-five attribution channel was sufficient for model faithfulness, but not for transmitting all record information needed for reader comprehension.

The resulting hypothesis was that the explanation required a second, compact **record-state channel** in addition to the attribution channel.

---

## Ablation 1: Adding Full Absence Information

The first experiment appended all explicit absent or zero-valued findings to the baseline note.

This substantially improved reader comprehension:

| Metric | Baseline | Full-absence synthetic screen |
|---|---:|---:|
| Grounding | 1.000 | 1.000 |
| Faithfulness | 0.969 | 0.969 |
| Comprehension | 0.758 | 0.909 |
| Sensitivity | 0.618 | 0.408 |
| Composite | 0.772 | 0.821 |

The experiment confirmed that absence information carried substantial value for comprehension.

However, sensitivity deteriorated sharply.

The reason was representational rather than informational. A conventional English list introduced many patient-specific words into each note. Under the lexical similarity component of the sensitivity scorer, those additional terms created large amounts of variation unrelated to differences in the model's top evidence.

This experiment established the central design problem:

> The system needed to transmit broad patient-state information to the reader without allowing that information to dominate the lexical geometry used for sensitivity scoring.

---

## Ablation 2: Structured State Representation

A more structured representation encoded a fixed vocabulary of patient states with compact numeric indicators and, in one variant, additional lexical anchors for the top model drivers.

Offline screening showed that this representation could achieve a strong balance between comprehension and sensitivity.

One synthetic development screen achieved approximately:

| Metric | Structured representation |
|---|---:|
| Grounding | 1.000 |
| Faithfulness | 0.969 |
| Comprehension | 0.920 |
| Sensitivity | 0.642 |
| Composite | 0.874 |

This provided evidence that comprehension and sensitivity were not inherently incompatible.

However, the representation required the generator to perform a large multi-field lookup, recoding, and formatting task. Under the 16 GB deployment constraint, the resulting prompts were longer and real Qwen generation became unstable or exceeded memory limits in some configurations.

This distinction became important throughout the project:

> A strong offline representation is not useful if the required generator cannot execute it reliably under the deployment constraint.

The fixed-state architecture was therefore treated as an informative upper-bound experiment rather than as a deployable final system.

---

## Ablation 3: Compact Mega-Token Representation

The next representation compressed explicit absent findings into a single hyphen-connected sequence, for example:

```text
Absent:angioplasty--stroke--diabetes--antipsychotic--...
```

The design had two goals:

1. preserve enough semantic information for the reader model to recover the underlying absent findings, and
2. reduce the number of independent lexical dimensions introduced into the sensitivity representation.

Offline screening showed that the compact representation largely restored sensitivity while preserving most of the comprehension gain.

A synthetic screen with additional top-driver anchors achieved approximately:

| Metric | Compact representation screen |
|---|---:|
| Grounding | 1.000 |
| Faithfulness | 0.969 |
| Comprehension | 0.917 |
| Sensitivity | 0.677 |
| Composite | 0.880 |

Additional experiments showed that the extra top-driver anchors were not necessary for most of the improvement. Removing them reduced complexity and decreased the risk of corrupting exact field names.

The final design therefore retained only:

- the original stable top-five explanation architecture, and
- one compact absence channel derived from explicitly reported absent findings.

This was substantially simpler than the earlier fixed-state representation.

---

## Real-Generator Validation

Synthetic transformation experiments were useful for identifying promising representations, but they did not establish whether the actual constrained generator could execute the design.

The final architecture was therefore tested using real Qwen2.5-7B generation on 50 development cases.

The result was:

| Metric | Final real-Qwen development run |
|---|---:|
| Coverage | 1.000 |
| Grounding | 1.000 |
| Faithfulness | 0.976 |
| Comprehension | 0.844 |
| Sensitivity | 0.624 |
| Composite | 0.826 |

All 50 cases produced usable outputs.

The key result was the shape of the improvement. Relative to the baseline:

- grounding remained essentially perfect,
- faithfulness did not decline,
- sensitivity remained near the baseline level, and
- comprehension increased materially.

This was the desired Pareto improvement.

The final architecture was therefore frozen as:

> **Qwen2.5-7B + 4-bit inference + two few-shot examples + five model attributions + compact explicit-absence channel**

No additional scorer-specific structures were added after this point.

---

## Base-Model Challenger

A late-stage concern was whether the system had become dependent on an early default choice of Qwen2.5-7B.

To test this without reopening the full optimization process, a single modern challenger, Qwen3-8B, was evaluated using the same frozen architecture.

The challenger could:

- load within the available hardware configuration,
- produce valid JSON, and
- often copy the main attribution drivers correctly.

However, on a small stress-oriented development screen, it systematically made a type of error that was especially harmful for this task: it placed findings that were **not recorded**, and in some cases findings that were explicitly **present**, into the absence channel.

At least five of eight stress cases exhibited false-absence behavior, and one case also contained a true driver-grounding contradiction.

Because the comparison was intended to test only the base model rather than retune the architecture for the challenger, the Qwen3 branch was terminated according to the pre-specified stopping rule.

This experiment reinforced an important property of the task:

> General language-model capability was less important than conservative, mechanical fidelity to the distinction between observed, absent, and unrecorded evidence.

Qwen2.5-7B was therefore retained for the final submission.

---

## Organizer Held-Out Results

After submission, the organizers evaluated the frozen system on the competition test data.

The official result was:

| Metric | Organizer-held-out test |
|---|---:|
| **Composite** | **0.8091** |
| Grounding | 0.9988 |
| Faithfulness | 0.9673 |
| Comprehension | 0.8195 |
| Sensitivity | 0.6248 |
| Coverage | 0.9850 |

The submission ranked **12th among 33 valid entries**.

The most informative result was not the rank alone, but the consistency between development and organizer-held-out performance.

| Metric | Final development run | Held-out test |
|---|---:|---:|
| Grounding | 1.000 | 0.999 |
| Faithfulness | 0.976 | 0.967 |
| Comprehension | 0.844 | 0.820 |
| Sensitivity | 0.624 | 0.625 |
| Composite | 0.826 | 0.809 |

Grounding, faithfulness, and sensitivity remained close to their development values, while comprehension showed a moderate decline.

The held-out result therefore suggested that the final representation generalized substantially better than would be expected from a purely development-specific scoring artifact.

---

## Main Findings

The project produced several methodological lessons.

### 1. Explanation quality is multi-dimensional

A note can be perfectly grounded and faithful to a predictive model while still failing to transmit enough information for a downstream reader.

### 2. More information is not automatically better

Adding clinically relevant record information improved comprehension, but a naïve representation damaged another evaluation objective.

The representation of information mattered almost as much as the information itself.

### 3. Error analysis was more informative than aggregate optimization

The largest improvement came from identifying a specific failure mode — omitted absent findings — rather than broadly modifying the prompt.

### 4. Synthetic upper bounds and deployable systems must be separated

Several offline transformations achieved higher scores than the final system, but required operations that the constrained generator could not reliably perform.

Deployment reliability was therefore treated as part of model quality rather than as a separate engineering concern.

### 5. A simpler final architecture generalized better

Although development explored increasingly structured representations, the final system returned to a relatively simple design:

- preserve the stable attribution explanation,
- add only the information channel that addressed the identified comprehension bottleneck, and
- avoid additional formatting tasks that increased generator failure risk.

---

## Reproducibility Notes

The repository contains the author's final configuration and methodological analysis.

The public repository intentionally focuses on:

- the final submitted configuration,
- validation and diagnostic code written for the project, and
- the methodology and ablation history.

Competition test cases are not included.

The final configuration is available in:

```text
configs/final_submission.yaml
```

The repository is intended to document the reasoning, validation strategy, and system design rather than redistribute the full competition environment.

---

## Final System

The final submission can be summarized as:

```text
Base model:          Qwen2.5-7B-Instruct
Quantization:        4-bit
Decoding:            deterministic
Few-shot examples:   2
Model attributions:  top 5
Explanation:         verdict + calibrated probability + confidence +
                     five attributed drivers + compact explicit-absence channel
```

The final system achieved a **0.8091 composite score on organizer-held-out test data**, with **0.9988 grounding**, **0.9673 attribution faithfulness**, **0.8195 comprehension**, and **0.6248 sensitivity**.