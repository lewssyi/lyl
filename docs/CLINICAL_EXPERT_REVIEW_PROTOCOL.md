# Blinded Clinical Expert Review Protocol

Status: **clinical expert validation pending**. No expert ratings have been collected or inferred by code.

## Objective and claim boundary

The review evaluates whether offline Sequence-LIME maps and versioned concept evidence are physiologically plausible and clinically useful for sleep staging. It does not establish causal model reasoning, diagnostic efficacy, or independent clinical concept ground truth.

## Sampling

Build the pack from the fixed ISRUC test split only after the canonical parity guard passes. Sample separately by true stage (W/N1/N2/N3/REM), prediction correctness, and transition/stable status. The default target is at least two items per non-empty cell. Retain population weights so stage-balanced agreement is not reported as a population estimate. N1 is always reported separately.

## Blinding

`tools/build_clinical_expert_review_pack.py` randomizes item order and creates anonymous IDs. The reviewer directory contains the three signals, an offline local-attribution heatmap, versioned raw-physiology concept evidence, and a blank rating sheet. It does not contain model prediction, confidence, correctness, subject/recording identity, or unblinded epoch index. These fields are stored in a separate `private_key.csv` for the analysis custodian. Reviewers must not access the key until ratings are locked.

## Rating form

Each reviewer independently scores every item from 1 (poor) to 5 (excellent):

1. `physiological_plausibility`: compatibility of highlighted EEG/EOG/EMG regions with known physiology.
2. `temporal_localization`: whether the highlighted intervals are locally defensible.
3. `concept_evidence_coherence`: coherence between the signal and `raw_physiology_v1` evidence.
4. `clinical_usefulness`: potential usefulness as a review aid, without assuming diagnostic validity.

Reviewers also provide `reviewer_stage` (W/N1/N2/N3/REM/uncertain), `artifact_flag` (yes/no), and optional comments. They must not be told the model stage until the database is locked.

## Analysis

Run `tools/analyze_clinical_expert_ratings.py` only on the locked combined form. Report item/reviewer counts, missingness, per-domain mean/median/IQR, two-way random-effects absolute-agreement ICC(2,1), stage agreement, agreement stratified by true stage/correctness/transition status, and sensitivity excluding artifact-flagged items. Small or degenerate strata are marked not estimable rather than pooled silently.

## Acceptance and reporting

This protocol does not preregister a clinical-success threshold. Results are descriptive until the clinical team approves thresholds and reviewer qualifications. Manuscript language must remain “clinical expert validation pending” until named human experts complete the blinded protocol. Automated concept parity, randomization sanity, and deletion/insertion tests are not substitutes for expert validation.

## Data governance

The generated pack may contain physiological waveforms and must stay in the approved secure project environment. Do not commit the pack, images, rating data, or private key. Commit only this protocol, the pack builder, the analysis script, and manifests/hashes for an authorized completed review.
