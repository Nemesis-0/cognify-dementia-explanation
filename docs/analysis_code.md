# Historical DEV-analysis code and provenance

This directory publishes **four unmodified Python scripts** recovered from the original CognifyChallenge 2026 development workspace. They illustrate how the MiaoScope project investigated **information coverage and representation trade-offs** while developing its explanation configuration. They are **research probes**, not the final Qwen generator, the official competition scoring implementation, or a software package for clinical use.

## What each original script does

| Script | Historical purpose | Important limitation |
|---|---|---|
| [`comprehension_coverage.py`](../experiments/analysis/comprehension_coverage.py) | Classifies DEV question-answer facts as structurally present or missing from a top-five-attribution explanation, grouped by question type and attribution rank. | Structural presence is **not** empirical reader comprehension. |
| [`absence_compression.py`](../experiments/analysis/absence_compression.py) | Counts observed nonzero, explicitly zero, and unresolved values within the DEV question-derived absence-field universe, comparing representation costs. | `present` means `field != 0` in the citable-field mapping, not a clinical determination; unrecorded is **not** absent. |
| [`absence_architecture_probe.py`](../experiments/analysis/absence_architecture_probe.py) | Compares basic, targeted, and full-absence toy notes with a structural-coverage measure, word counts, and a sampled lexical sensitivity proxy. | Its full-absence coverage assumption can be optimistic; no constrained LLM is run. |
| [`absence_official_frontier_dedup.py`](../experiments/analysis/absence_official_frontier_dedup.py) | Ranks extra absence fields by DEV baseline-weighted potential gain and compares a structural comprehension proxy with a lexical similarity proxy, omitting redundant top-five fields. | Uses the **same DEV questions** for ranking and assessment; `0.60*C + 0.20*S` uses official *weights*, but **C/S are heuristic proxies**, not official metrics. |

The earlier near-duplicate `absence_official_frontier.py`, historical stress-case selector, the unchanged organizer harness, and experiment-specific GPU package copies remain in the private historical workspace and are intentionally omitted from this concise public release.

## Inputs, execution, and data boundary

These are **original historical command-line scripts**, which read input files immediately when executed. To run them as originally written, the caller must work from the repository root and separately have access to the organizer-authorized **development-only** files at:

```text
experiments/devkit/cases/cohort_dev.json
experiments/devkit/cases/questions_dev.json
```

These data files are **not in this public repository**, and no downloader or synthetic substitute is provided. Do **not** supply any competition held-out test file. The scripts do not support `--help` or CLI parameters; the relative paths are part of the preserved originals.

The standard-library-only scripts are `comprehension_coverage.py` and `absence_compression.py`. The other two additionally import `numpy` and `scipy` (`scipy.stats.spearmanr`). The exact original development environment was not frozen in this public repository; dependencies are documented without pretending a tested pinned environment exists.

*Only in an authorized local DEV workspace with the input files present*, commands from the repo root would be:

```bash
python3 experiments/analysis/comprehension_coverage.py
python3 experiments/analysis/absence_compression.py
python3 experiments/analysis/absence_architecture_probe.py
python3 experiments/analysis/absence_official_frontier_dedup.py
```

Outputs are printed diagnostics. The scripts do not generate a final submission YAML or call the Qwen generator. Some fields are ranked using DEV question answer keys, so the frontier is **in-sample exploratory** and must not be presented as evaluation on fresh cases.

## Distinguishing evidence levels

1. **Source-level structural/proxy probes** (these scripts): informational coverage, question-based field selection, lexical distance correlations. They can motivate hypotheses but are not verified reader-model scores.
2. **Synthetic transformation/ablation screens** (described in [`methodology_and_ablations.md`](methodology_and_ablations.md)): separate recorded experiments and approximate composite values. These four scripts alone do **not** reproduce all of those reported screens.
3. **Real Qwen development generation**: separately recorded 50-case development run; composite 0.826 with coverage 1.000.
4. **Organizer-held-out evaluation**: organizer-reported composite 0.8091 with coverage 0.9850, not reproducible from public files alone.

The original final submission configuration is [`configs/final_submission.yaml`](../configs/final_submission.yaml). Inference and organizer scoring depended on the separate, unmodified competition harness. Neither the original underlying classifier nor the official runner is claimed as an author-created implementation in this repository.

## Byte-level source provenance

The following file sizes and SHA-256 hashes describe the **unmodified scripts** extracted from the original project backup; this release changes only surrounding navigation/documentation, not their contents.

| Script | Bytes | SHA-256 |
|---|---:|---|
| `comprehension_coverage.py` | 3769 | `1134a82911171c9744ca8d03a265836eda7b8cbbf9ae3c3b16f5659b4327c822` |
| `absence_compression.py` | 3404 | `c677a0f09df101e4d50da9742a0d147d40cb1cfe9982f1b171ea35569a27a22e` |
| `absence_architecture_probe.py` | 6134 | `fb811cd4d2fcadb4951cc1a77ddeb8b3d437c6bb83515429fa6a4cf1a4e7752e` |
| `absence_official_frontier_dedup.py` | 7995 | `01e49a10ec439e08e455c32953b514041be0022752323a6f25ad5d79659b9ed2` |

**Scientific scope:** This project analyzes explanations of a competition-provided dementia-screening classifier, not diagnoses. Findings are not intended for clinical use. The repo publishes no original cohort cases or question records.
