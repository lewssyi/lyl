> **STATUS: SUPERSEDED HISTORICAL CANDIDATE**
>
> The candidate did not establish model faithfulness under the authoritative
> protocol. The specification below is retained as provenance and does not
> define the current paper method.

# Context-Preserving Model-Faithful Sequence-LIME Specification

## Status and scope

The implemented candidate is named **Context-Preserving Model-Faithful Sequence-LIME (MF-SeqLIME)**. Its frozen configuration is `V_ZERO + Y2 logit margin`, selected on the ISRUC validation split only. Its scientific status is `MF_SEQLIME_PARTIAL_PASS`, not a paper-level pass.

MF-SeqLIME estimates local model sensitivity under a specified input replacement. It does not estimate global or causal feature contribution.

## Input and context

The classifier receives a 75-epoch overlapping context window. For a target epoch, every model query reconstructs all canonical windows that cover that epoch, replaces only the configured target regions, runs the current evaluated model with `training=False`, and averages the target-position response over covering windows. The explanation unit is 3 channels × 30 semantic segments = 90 features.

The canonical checkpoint and default classifier path are read-only. MF-SeqLIME is an evaluator; it is not an inference gate or training head.

## Frozen model response

The target is Y2 logit margin:

`z(original predicted class) - max(z(other classes))`.

T0 keeps the original model class fixed and is the pre-specified primary randomization protocol. T1 explains the randomized model's current class and is always reported separately. T0 and T1 must never be pooled.

## Frozen attribution composition

The selected V_ZERO ranking is the zero-intervention single-feature marginal computed from the evaluated model. Every channel×segment feature is replaced by zero within the context-preserving wrapper and the model response change supplies its score. The implementation may compute the dynamic local surrogate for fidelity and equal-budget accounting, but no predicted-stage, true-stage, or fixed stage-conditioned channel prior may change the final ranking.

The validation search included pure classic, pure zero, pure physiological, equal zero/physiological, and 25% blends. The winning configuration and Y2 target were serialized before the fixed test evaluation. Test data cannot enter selection or weighting.

## Query and cache contract

The fixed test uses 128 train masks, 32 local validation masks, and 1,530 total recorded queries for every compared method. Validation selection uses 2,106 queries per candidate row. Cache identity includes at least:

`sample_id + evaluated_model_hash + randomization_level + randomization_seed + method + target_protocol + perturbation + mask_seed + query_budget + evaluator_version`.

Legacy or mismatched identities are rejected. The model fingerprint is computed from every in-memory weight, not from a caller-supplied label.

## Randomization contract

R0 is the canonical checkpoint. R1–R6 independently reload that checkpoint and randomize the registered layer group plus downstream layers. R7 creates a fresh full-model initialization and does not load the checkpoint. Seeds are 3101, 3102, and 3103.

Every eager/fixed-graph perturbation, marginal, surrogate response, and deletion/insertion query must use the supplied randomized model. A compiled runner is stored on that model object and cannot capture a different model.

The primary pass rule uses T0: R7 similarity must approach the 1,000-permutation null, the R0 faithfulness advantage must decay at R7, and at least 2/3 seeds must agree. T1 is a separate sensitivity analysis and any failure remains a reported limitation.

## Validation and test policy

Validation selection jointly considers external/local surrogate fidelity, deletion/insertion faithfulness, mask-seed stability, randomization sensitivity, and equal budget. It cannot optimize only deletion AUC. The frozen test set consists of the unchanged v1.1.3 36 samples and is evaluated once.

Paper-level PASS additionally requires validation surrogate fidelity to improve over classic, at least one faithfulness endpoint to improve without material harm to the other, non-inferior stability, T0 full-model randomization pass, no stage prior, equal budget, and truthful correctness/transition strata. The current candidate fails the first condition and has a T1 limitation, hence PARTIAL PASS.

## Public result type

`ModelAttributionResult` contains sample ID, method ID, immutable attribution vector, target class/protocol, query count, and provenance metadata. Concept-only M5/M6 scores are rejected by this type. Clinical annotations cannot be written back to `model_attribution_score`.

## Reproducibility artifacts

- Configuration: `configs/xai/mf_seqlime_validation.yaml`, `configs/xai/mf_seqlime_test.yaml`.
- Selection: `outputs/sequence_lime_two_versions/isruc_seed42/validation_method_selection.csv` and `frozen_validation_selection.json`.
- Fixed test: `paper_comparison_table.csv`, `test_model_faithfulness.csv`, `test_stability.csv`, `test_randomization_sanity.csv`.
- Guards: `final_validation/final_guard_summary.json` and `fixed_graph_memory_1200.json`.
- Implementation: `src/xai/model_faithful_sequence_lime.py`, `tools/sequence_lime_two_version_evaluation.py`.

## Protocol-consistency boundary

The 2026-07-23 audit found the historical A/B/C and newer validation randomization
results protocol-incompatible. The future entry is `classic_sequence_lime_consistency_v1`:
1,200/306 Bernoulli masks, Y2, 1,530 queries, T0/T1, R0/R3/R5/R7, seeds 3101-3103,
a 1,000-permutation null and protocol-versioned cache identity. Both independent
12-sample groups pass 0/3 T0 and 0/3 T1. The former `MF_SEQLIME_PARTIAL_PASS`
randomization support is superseded, and feedback training remains forbidden.
