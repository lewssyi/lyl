# Sequence-LIME Full Evaluation Results

## Current result

- Dataset/checkpoint: ISRUC A1-zero, seed 42.
- Formal evaluator shards: `sequence_lime_v1.1.2`, Git `96c4fcb6926a32ca540d6370e7427c1d0f228b11`.
- Guarded merger: `sequence_lime_v1.1.3`, Git `f241202e8deeb49d0ba524ac4d9f3d7ef6976f5f`.
- Current run ID: `full_evaluation_v113_merged_guarded`.
- Result directory: `outputs/sequence_lime_paper_evaluation/isruc_seed42/runs/full_evaluation_v113_merged_guarded`.
- Current index: `outputs/sequence_lime_paper_evaluation/isruc_seed42/CURRENT_RESULTS_MANIFEST.json`.
- Merge manifest: `full_evaluation_shard_merge_manifest.json` reports 36 samples, zero overlap, eight completed source shards, and clean source provenance.

The configured maximum was 40. The formal selector produced 36 unique epochs after the population, correctness, stage, transition, and stable strata were combined and deduplicated. The set contains 24 correct and 12 incorrect predictions, 17 transition-boundary and 19 stable epochs, and all five true and predicted stages.

## Validation status

- Every required formal table exists, is larger than one byte, and contains data.
- `sample_level_metrics.csv`: 36 rows.
- `faithfulness_curves.csv`: 720 rows.
- `clinical_alignment_sample_level.csv`: 144 rows.
- `stability_sample_level.csv`: 108 rows.
- `model_randomization_sanity.csv`: 108 rows.
- `perturbation_robustness.csv`: 8,640 rows.
- The nine paper/result files in `CURRENT_RESULTS_MANIFEST.json` have recorded sizes and SHA256 hashes.
- Final maintained suite: 24 tests passed.
- Canonical batch-16 parity: maximum absolute probability difference `0.0`, argmax agreement `1.0`, and Accuracy/Macro-F1/Kappa differences `0.0`.
- Sequence-wrapper maximum absolute difference: `5.960464477539063e-08`.

## Scientific result

The full evaluator's audit status is `PARTIAL PASS`, but its paper-level decision is `PAPER_LEVEL_INTERPRETABILITY_FAIL`. This is an evidence boundary, not an implementation failure.

- Local surrogate fidelity was estimable for 34 samples. Median validation weighted R2 was `0.63894` and P10 was `0.46350`, below the project-prespecified fidelity threshold.
- Within-checkpoint perturbation stability passed: mean absolute coefficient Spearman `0.81039`, mean top-10 Jaccard `0.74083`, dominant-channel agreement `0.90741`, and mean top-3 concept Jaccard `0.93827` across 108 comparisons.
- Faithfulness showed a directional advantage over the random baseline with paired statistical support under the evaluated replacement rules.
- Mean raw positive concept alignment was `0.62967` across 36 samples.
- Model randomization was measured across 108 sample/level rows but did not pass the project sanity rule.
- The requested primary surrogate budget was 1,100 queries per sample, with four separately reported replacement endpoints and canonical window-inference batch 16.

Therefore the allowed conclusion is limited to: the overlapping-window Sequence-LIME implementation is valid and its local fidelity, perturbation faithfulness, stability, randomization sensitivity, and concept alignment are quantitatively measurable. The current evidence does **not** support explanation reliability, universal clinical validity, or superiority over baselines.

## Failure and recovery provenance

The first formal attempt (`sequence_lime_v1.1.1`) retained a small GPU allocation on every eager perturbation forward and exhausted 24 GiB devices after thousands of calls. Its partial caches were preserved as diagnostic history and were not merged. Version 1.1.2 moved only repeated perturbation inference to one fixed-signature TensorFlow graph while retaining the canonical batch-16 endpoint. A 1,200-forward benchmark stayed at 5.02 MiB of TensorFlow current allocations; the former eager path increased from 5.75 MiB to 865.01 MiB. All eight v1.1.2 shards then completed.

Version 1.1.3 fixes CLI routing so `--merge-shard-run-ids` enters the guarded merge before any model/data loading. The guarded merger rejects incomplete shards, nonempty failure tables, missing/header-only raw outputs, and overlapping sample IDs.

## Open requirements

- Cross-seed model stability still requires additional frozen checkpoints.
- Deployment-safe normalization performance requires separately trained and evaluated checkpoints.
- Clinical expert validation remains pending and requires qualified human reviewers.

## Adjunct A/B/C diagnosis and two-version evaluation

The v1.1.3 formal run above remains the sole current historical evaluation and none of its result hashes changed. A subsequent isolated root-cause audit is indexed under `outputs/sequence_lime_abc_diagnosis/isruc_seed42/`, with its validation-frozen two-version evaluation under `outputs/sequence_lime_two_versions/isruc_seed42/`.

The audit found `MIXED_A_PLUS_C`: the old randomization route changed zero variables and its cache/resume identity omitted the full evaluated-model hash; after those defects were fixed, classic Sequence-LIME passed full-model randomization while the historical predicted-stage-prior fusion failed. The validation-only selection froze `V_ZERO + Y2` before the fixed-36 test.

On the frozen test, the selected candidate reduced deletion AUC from 0.46233 to 0.40862 and increased insertion AUC from 0.74277 to 0.76950, with mask-seed Spearman increasing from 0.41846 to approximately 1.0. External validation weighted R2 remained exactly 0.66766 for both classic and the candidate. T0 full-model randomization passed 2/3 seeds, whereas T1 passed only 1/3. The resulting scientific status is `MF_SEQLIME_PARTIAL_PASS`, not a paper-level superiority result.

Clinical Concept Annotation is a separate post-attribution layer. It has `IMPLEMENTATION_VALID / EXPERT_VALIDATION_PENDING` status and is excluded from all model-faithfulness scores. Final isolated guards reproduced exact canonical batch-16 probabilities, sequence-wrapper maximum difference `5.960464477539063e-08`, zero current-memory growth across 1,200 fixed-graph forwards, and 58/58 maintained tests.
