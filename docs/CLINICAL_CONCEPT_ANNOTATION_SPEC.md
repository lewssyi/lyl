# Clinical Concept Annotation Layer Specification

## Status and claim boundary

The Clinical Concept Annotation Layer (CCA) is `IMPLEMENTATION_VALID / EXPERT_VALIDATION_PENDING`. It generates **derived clinical annotation** after model attribution. It is not a concept bottleneck, model reasoning ground truth, causal contribution, diagnosis, or clinical-validity result.

## Execution order and isolation

The only allowed order is:

`immutable MF-SeqLIME attribution → select Top-K regions → extract raw physiological concepts → compute annotations`.

CCA must not modify the attribution vector, ranking, logits, prediction, or query result. It cannot add a concept score back into MF-SeqLIME. The public input is `ModelAttributionResult`; the public output is the distinct `ClinicalAnnotationResult`. An exact array-equality guard runs before return.

True stage is never an input. Predicted stage may appear only as display context and cannot select a concept, mapping, or evidence weight.

## Concept source

Concepts use the canonical `raw_physiology_v1` definitions from `src/clinical_concepts.py` and `src/lime_explanation.py`:

- EEG Delta, Theta, Alpha, Sigma, Beta from EEG channel 0.
- EOG Slow Eye and Rapid Eye from EOG channel 1.
- EMG Low Tone and Muscle Activation from EMG channel 2.

The raw-waveform definition is explicitly different from `segment_proxy_v1`; they cannot be described as mathematically interchangeable.

## Mapping formula

Let `a_f` be the immutable MF attribution for feature `f`, `S` the Top-10 features by stable descending `|a_f|`, `c` a concept, and `m_fc` the binary feature-to-concept channel mapping. The activation `u_c` is the mean raw physiological activation over selected segments on the concept's channel; when no selected feature belongs to that channel, the mean over all 30 segments is used. The annotation evidence is:

`annotation_c = u_c × sum(f in S) |a_f| × |m_fc|`.

This formula is stage-independent. It describes physiological evidence overlapping model-important regions; it does not claim the model internally represents the named concept.

## Output fields

The sample-level table keeps three separate JSON vectors:

- `model_attribution_score` — immutable model sensitivity.
- `clinical_concept_activation` — raw stage-independent signal activation.
- `clinical_annotation_score` — derived overlap evidence.

It also records selected feature indices, concept source/version, predicted stage as display-only context, exact ranking-unchanged status, coverage, region overlap, and expert review status.

## Current implementation validation

The fixed 36 evaluation has 108 sample-seed rows. Attribution is exactly unchanged in every row; `ranking_modified=false`, `predicted_stage_used_for_strength=false`, and `true_stage_used=false`. Mean concept coverage is 0.9969, mean model-region overlap is 1.0, and three-seed concept stability is 1.0. These are software and repeatability checks only.

## Human validation requirement

Qualified blinded reviewers must evaluate concept presence, region relevance, polarity, confidence, and clinical plausibility under `docs/CLINICAL_EXPERT_REVIEW_PROTOCOL.md`. Until inter-rater reliability and adjudication are complete, all outputs retain `EXPERT_VALIDATION_PENDING`; no clinical-validity PASS is permitted.

## Reproducibility artifacts

- Configuration: `configs/xai/clinical_annotation.yaml`.
- Implementation: `src/xai/clinical_concept_annotation.py`, `src/xai/attribution_types.py`.
- Results: `outputs/sequence_lime_two_versions/isruc_seed42/clinical_annotation_sample_level.csv` and `clinical_annotation_summary.csv`.
- Tests: `tests/test_xai_two_versions.py`.
