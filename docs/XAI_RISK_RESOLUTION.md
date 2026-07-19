# XAI Risk Resolution Register

This register is code-evidence driven. “Resolved” means the repository has an enforceable control and a passing test or audited artifact. Empirical claims that need additional checkpoints or human review remain explicitly open.

## 1. Git reproducibility

- Risk: the evaluator, configuration, and provenance could not be reconstructed from Git.
- Code evidence: `tools/sequence_lime_paper_evaluation.py` now has a version, deterministic checkpoint loader, strict parity guard, and per-run provenance. `docs/A1_ZERO_MAINLINE_MANIFEST.md` records artifact hashes and the executable parity command.
- Measure: version the evaluator/docs/configs/tests; keep large artifacts out of Git and refer to paths plus SHA256.
- Tests: clean historical commit `e5fdba0` and the current implementation both reproduce the frozen probability cache exactly.
- Status: resolved for the A1-zero/XAI mainline. Output cleanup and the final `CURRENT_RESULTS_MANIFEST.json` are completed in the guarded-evaluation commit.

## 2. Branch/mainline conflict

- Risk: the former branch name implied a channel-residual experiment although A1-zero is the actual classifier.
- Evidence/measure: branch renamed to `codex/a1-zero-mainline`; recovery tag preserved; the generic launcher default is `A1_concept_film_zero`; the manifest defines A1-zero as the only mainline.
- Tests: launcher default/config static test and final README/reference scan.
- Status: resolved in code; README/tool cleanup is verified in the cleanup commit.

## 3. Stale reports versus current parity

- Risk: old parity-fail and later pass reports could be treated as equally current.
- Evidence/measure: every formal run records UTC generation time, Git state, checkpoint/split/cache hashes, evaluator version, resolved configuration, and result files. Runs use independent IDs/manifests. `CURRENT_RESULTS_MANIFEST.json` is the sole current-output index.
- Tests: provenance fields and manifest isolation tests.
- Status: resolved by the guarded evaluator; obsolete output files are backed up externally before removal.

## 4. Incomplete full pipeline

- Risk: empty one-byte CSVs were previously mistaken for completion.
- Evidence: pre-audit files `sample_level_metrics.csv`, `surrogate_fidelity.csv`, `stability_stage_summary.csv`, `model_randomization_sanity.csv`, and `paper_main_table.csv` were one byte because the general pipeline had never completed; specialized confirmation files did not populate them.
- Measure: empty required outputs are fatal; smoke, functional, confirmation, and full profiles have isolated manifests and resume fingerprints. Required outputs include fidelity, stability, randomization, deletion/insertion, stage/correctness summaries, runtime/query accounting, and the paper table.
- Tests: 2-sample smoke, 5-sample functional run, stage-stratified confirmation, then full evaluation.
- Status: code-resolved; completed empirical runs are listed in `CURRENT_RESULTS_MANIFEST.json`.

## 5. Sample-selection bias

- Risk: the previous correct-only stage-balanced N=40 set could be read as population performance.
- Measure: retain that specialist result, but formal sampling distinguishes random population, correct, incorrect, true-stage, predicted-stage, transition/stable, and N1 strata. Outputs label `faithfulness on correctly predicted cases`, `faithfulness on misclassified cases`, `population-weighted estimate`, and `stage-balanced estimate` separately.
- Tests: manifest stratum counts and weighting tests.
- Status: resolved for reporting. Small/empty strata remain marked not estimable rather than silently pooled.

## 6. Estimand boundary

- Risk: physiological-replacement gains were generalized to all LIME perturbations.
- Measure: zero, physiological, Gaussian, and local-mean endpoints are computed and reported separately. Paper output includes: “PC-SeqLIME superiority is specific to the physiological-perturbation estimand.”
- Tests: required-endpoint and no-selective-merging checks.
- Status: resolved. No universal-superiority claim is permitted.

## 7. Predicted-stage prior circularity

- Risk: a predicted-stage concept prior can make explanations agree with the prediction by construction.
- Measure: formal A–G ablation: classic Sequence-LIME; zero marginal; physiological marginal; both marginals; both plus predicted-stage prior; both plus stage-independent prior; shuffled prior. D versus E is the primary circularity comparison. Training targets exclude predicted/true-stage priors by default.
- Tests: target manifests reject stage-conditioned priors unless an explicit ablation flag is present.
- Status: implementation resolved; empirical D–E interpretation is reported without treating E as independent evidence.

## 8. Inconsistent concept definitions

- Risk: the model used 30x28 engineered segment features while dashboards recomputed raw-waveform formulas under the same name.
- Evidence/measure: `src/clinical_concepts.py` is the only names/index/formula source. The two estimands are explicitly `segment_proxy_v1` and `raw_physiology_v1`; they are not mathematically equivalent. Model, dashboard, and evaluator import the canonical module.
- Tests/artifacts: `tests/test_clinical_concepts.py`; `outputs/concept_definition_parity/concept_definition_parity.{json,csv}` on real ISRUC samples.
- Result: 10-sample audit, 90 concept rows, mean available Pearson 0.8586, mean absolute difference 1.5104; `mathematically_equivalent=false`.
- Status: resolved.

## 9. Label leakage in concept targets

- Risk: true labels could condition concept targets and be misdescribed as independent clinical annotations.
- Evidence/measure: `concept_prior_weight` defaults to 0. A positive value requires `LFTS_LABEL_CONDITIONED_CONCEPT_TARGETS=1`, emits a strong warning, is train-only, and must set `label_conditioned=true` in the target manifest. Validation/test target construction forces the prior weight to zero. Canonical A1-zero produces no concept loss and never feeds labels into forward inference.
- Tests: label-policy unit tests and lambda-zero/model parity.
- Status: resolved. Label-conditioned variants remain explicit ablations only.

## 10. Transductive recording normalization

- Risk: canonical per-recording normalization uses unlabeled statistics from the complete recording, including future epochs.
- Evidence/measure: canonical behavior is preserved as `recording_robust_zscore_canonical` for checkpoint parity. Added default-off `prefix_causal_robust_zscore`, `train_fitted_global_scaler`, and `epoch_local_normalization` modes. Train-fitted state is accepted only when marked `fit_scope=train_only`.
- Tests: future-epoch mutation leaves prefix outputs unchanged; epoch-local outputs are independent; non-train scaler states are rejected.
- Status: code risk resolved and canonical limitation documented. Deployment-safe performance requires a separate checkpoint/experiment and is not substituted into the frozen mainline.

## 11. Uncontrolled LIME randomness

- Risk: global `np.random` made masks and coefficients irreproducible.
- Evidence/measure: `run_multimodal_lime` accepts a seed or `np.random.Generator`; all stochastic perturbation helpers receive the local generator; outputs record seed and mask SHA256.
- Tests: identical sample/checkpoint/seed yields identical mask hash and exactly equal coefficients; different seeds change the mask hash.
- Status: resolved.

## 12. Attribution is not causality

- Risk: local sensitivity was described as global causal reasoning.
- Measure: code/manifests/docs use `local attribution`, `perturbation sensitivity`, `derived concept evidence`, and `interventional faithfulness under a specified replacement rule`. They forbid `global causal contribution`, `causal channel importance`, and `model reasoning ground truth`. Random/head randomization, reverse/random/energy ranking, and replacement sensitivity are reported as sanity/faithfulness checks only.
- Status: terminological/code control resolved; causal validity is not claimed.

## 13. HMM wording

- Risk: optional HMM candidates could be mistaken for the final canonical decoder.
- Evidence/measure: all manifests state: HMM is an optional validation-selected candidate; A1-zero selected temperature argmax; LIME explains pre-HMM neural probabilities.
- Tests: provenance/static wording checks.
- Status: resolved.

## 14. Test-side oracle analysis

- Risk: a best test candidate could leak into final metrics or the paper main table.
- Evidence/measure: validation-selected configuration is the sole final endpoint. Test-side best is diagnostic/oracle only and disabled from formal tables. `assert_no_test_oracle_fields` rejects any paper-main-table column containing `oracle`.
- Tests: `tests/test_xai_safety_guards.py`.
- Status: resolved.

## 15. Missing clinical validation

- Risk: automated evidence could be misrepresented as expert validation.
- Measure: `docs/CLINICAL_EXPERT_REVIEW_PROTOCOL.md`, `tools/build_clinical_expert_review_pack.py`, and `tools/analyze_clinical_expert_ratings.py` implement stratified blinded review, a private unblinding key, structured ratings, and ICC analysis.
- Status: code support resolved; **clinical expert validation pending**. This part cannot be completed by Codex and remains a human requirement.

## Residual external requirements

1. Cross-seed model stability needs additional frozen seed checkpoints; current results are `within-checkpoint perturbation stability` only.
2. Deployment-safe normalization performance needs separately trained/evaluated checkpoints.
3. Clinical expert validation requires qualified human reviewers and an approved secure review environment.
