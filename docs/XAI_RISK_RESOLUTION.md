# XAI Risk Resolution Register

This register is code-evidence driven. “Resolved” means the repository has an enforceable control and a passing test or audited artifact. Empirical claims that need additional checkpoints or human review remain explicitly open.

## 1. Git reproducibility

- Risk: the evaluator, configuration, and provenance could not be reconstructed from Git.
- Code evidence: `tools/sequence_lime_paper_evaluation.py` now has a version, deterministic checkpoint loader, strict parity guard, and per-run provenance. `docs/A1_ZERO_MAINLINE_MANIFEST.md` records artifact hashes and the executable parity command.
- Measure: version the evaluator/docs/configs/tests; keep large artifacts out of Git and refer to paths plus SHA256.
- Tests: clean historical commit `e5fdba0` and the current implementation both reproduce the frozen probability cache exactly.
- Status: resolved for the A1-zero/XAI mainline. The current guarded result is indexed by `outputs/sequence_lime_paper_evaluation/isruc_seed42/CURRENT_RESULTS_MANIFEST.json`; full provenance and outcome are in `docs/XAI_FULL_EVALUATION_RESULTS.md`.

## 2. Branch/mainline conflict

- Risk: the former branch name implied a channel-residual experiment although A1-zero is the actual classifier.
- Evidence/measure: branch renamed to `codex/a1-zero-mainline`; recovery tag preserved; the generic launcher default is `A1_concept_film_zero`; the manifest defines A1-zero as the only mainline.
- Tests: launcher default/config static test and final README/reference scan.
- Status: resolved in code; README/tool cleanup is verified in the cleanup commit.

## 3. Stale reports versus current parity

- Risk: old parity-fail and later pass reports could be treated as equally current.
- Evidence/measure: every formal run records UTC generation time, Git state, checkpoint/split/cache hashes, evaluator version, resolved configuration, and result files. Runs use independent IDs/manifests. `CURRENT_RESULTS_MANIFEST.json` is the sole current-output index.
- Tests: provenance fields and manifest isolation tests.
- Status: resolved by the guarded evaluator. The sole current index points to run `full_evaluation_v113_merged_guarded`; failed v1.1.1 partial runs are retained only as diagnostic history and are not merged.

## 4. Incomplete full pipeline

- Risk: empty one-byte CSVs were previously mistaken for completion.
- Evidence: pre-audit files `sample_level_metrics.csv`, `surrogate_fidelity.csv`, `stability_stage_summary.csv`, `model_randomization_sanity.csv`, and `paper_main_table.csv` were one byte because the general pipeline had never completed; specialized confirmation files did not populate them.
- Measure: empty required outputs are fatal; smoke, functional, confirmation, and full profiles have isolated manifests and resume fingerprints. Required outputs include fidelity, stability, randomization, deletion/insertion, stage/correctness summaries, runtime/query accounting, and the paper table.
- Tests: 2-sample smoke, 5-sample functional run, stage-stratified confirmation, and an eight-shard formal evaluation followed by guarded merge.
- Result: all 36 unique selected samples completed. All nine required formal result files and seven raw merged tables are nonempty; the guarded merge reports zero sample overlap. The paper-level conclusion is `PAPER_LEVEL_INTERPRETABILITY_FAIL`, because local surrogate fidelity and model-randomization sanity did not both pass the project thresholds.
- Status: pipeline completion is resolved. Explanation reliability is not supported by the current empirical result and is not claimed.

## 5. Sample-selection bias

- Risk: the previous correct-only stage-balanced N=40 set could be read as population performance.
- Measure: retain that specialist result, but formal sampling distinguishes random population, correct, incorrect, true-stage, predicted-stage, transition/stable, and N1 strata. Outputs label `faithfulness on correctly predicted cases`, `faithfulness on misclassified cases`, `population-weighted estimate`, and `stage-balanced estimate` separately.
- Tests: manifest stratum counts and weighting tests.
- Result: the deduplicated formal set contains 36 epochs: 24 correct, 12 incorrect, 17 transition-boundary, and 19 stable; all five true and predicted stages are represented.
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
- Result: within-checkpoint stability passed, but model-randomization sanity failed the project rule. This prevents a paper-level explanation-reliability claim even though faithfulness was directionally better than random under the evaluated rules.
- Status: terminological/code control resolved; causal validity and explanation reliability are not claimed.

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

## Current formal evaluation

The authoritative empirical summary is `docs/XAI_FULL_EVALUATION_RESULTS.md`. Run `full_evaluation_v113_merged_guarded` contains 36 unique samples and was produced from eight completed v1.1.2 shards using the v1.1.3 guarded merger. Final validation passed 24 maintained tests and exact canonical batch-16 probability parity (`max_abs_diff=0.0`, `argmax_agreement=1.0`).

## 16. Randomized-model routing and cache identity

- Risk: a nominal randomized evaluation could route through an absent head name, retain the original model in a wrapper/graph, or restore an attribution generated by a different model.
- Evidence: the pre-fix `final_layer` route changed 0 variables and produced output difference 0; cache keys omitted the complete in-memory weight fingerprint and sample resume did not validate it.
- Measure: R1 now targets the actual `main_output`; R2-R6 use audited cumulative layer groups; R7 is fresh initialization. Every perturbation and metric receives the evaluated model explicitly. Cache identity includes the full model fingerprint, randomization level/seed, method, target protocol, perturbation, sample, mask seed, budget, and evaluator version.
- Tests/results: R1 changes 2 variables and output by up to 0.96757; R7 changes 159 variables and output by up to 0.96023. Model-local compiled runners and cache mismatch rejection are covered by `tests/test_xai_randomization_route.py`.
- Status: resolved. CASE C remains in provenance because it explains the historical failure.

## 17. Predicted-stage prior contamination

- Risk: the historical PC score can remain stable after model randomization because 20% of its ranking is determined by the predicted-stage concept prior.
- Evidence: after implementation fixes, M0 R7 signed Spearman is 0.0067 versus a mean null upper bound of 0.2076, while M4 is 0.3111 and M5 T0 is exactly 1.0. M0 passes 3/3 diagnostic seeds; M4 and M5 pass 0/3.
- Measure: the formal MF candidate excludes predicted-stage, true-stage, and fixed stage-conditioned channel priors. Concept interpretation is moved into the separate CCA result type.
- Status: historical-method risk confirmed (`CASE_A_PC_FUSION_OR_PRIOR_DOMINATED`) and controlled in the new candidate. Historical M4 remains available only as a labeled comparator.

## 18. Validation/test selection separation

- Risk: test faithfulness could select a favorable target or fusion weight.
- Measure: 12 validation samples were selected by a deterministic stage/correctness and transition/stable rule before attribution inspection. Six candidate compositions and Y0/Y1/Y2 were compared on validation; `V_ZERO + Y2` was serialized with `test_data_used=false` before one fixed-36 test evaluation.
- Evidence: `validation_method_selection.csv`, `frozen_validation_selection.json`, shard manifests, and `final_scientific_decision.json` all record validation-only selection and equal budgets.
- Status: resolved. Test results did not modify the frozen configuration.

## 19. Two-version claim separation

- Risk: automated clinical features could be fed back into attribution and then misreported as model faithfulness or clinical validity.
- Measure: immutable `ModelAttributionResult` and distinct `ClinicalAnnotationResult`; annotation executes only after Top-K attribution selection and exact equality is checked before return. Predicted stage is display-only and true stage is not an input.
- Result: all 108 sample-seed annotations retained exact attribution and `ranking_modified=false`. CCA is excluded from the model-faithfulness table.
- Status: software separation resolved; clinical expert validation remains pending.

## Updated current boundary

The historical v1.1.3 result remains `PAPER_LEVEL_INTERPRETABILITY_FAIL`. The isolated two-version candidate is `MF_SEQLIME_PARTIAL_PASS`: deletion/insertion faithfulness and mask-seed stability improve, but validation surrogate fidelity does not, and T1 randomization passes only 1/3 seeds. Explanation-guided training remains prohibited.

## 20. Surrogate-neighborhood optimization and validation stopping

- Risk: better pointwise surrogate error or deterministic zero stability could be mistaken for a jointly better, model-sensitive explanation method.
- Measure: an isolated validation-only study compared Classic Bernoulli Ridge with structured retention/contiguous sampling, boundary-adaptive sampling, and budget-deducted weak-zero calibration. Every candidate used 1,530 requested queries, a unique 1,506-mask neighborhood, disjoint local train/validation masks, Y2, T0/T1, and three mask/randomization seeds. R7 was compared with a 1,000-permutation null.
- Evidence: all 144 small-validation shards completed. C1-70 reduced MAE/RMSE to 0.1407/0.2048, but every candidate reduced weighted R² below Classic 0.7685 and worsened both deletion and insertion AUC. Every candidate passed 0/3 T0 and 0/3 T1 seeds; R7 similarity stayed above its null q95. In C3, zero-component stability was 1.0 while LIME-component stability was 0.6171, and increasing alpha increased both final stability and undesirable R7 similarity.
- Status: `VALIDATION_NO_QUALIFIED_CANDIDATE`. Final-30 validation and fixed-36 test were not run. The prior `MF_SEQLIME_PARTIAL_PASS` boundary remains; feedback training remains prohibited.
- Guards: final canonical batch-16 parity and the 1,200-forward fixed-graph memory guard passed; the maintained suite passed 79/79 tests.

## 21. Classic Sequence-LIME protocol consistency

- Risk: historical M0 3/3 and newer C0 0/3 outcomes could be compared as one protocol.
- Measure: recover exact samples and rerun both groups through one versioned 1,530-query evaluator.
- Evidence: all 288 A/C shards completed; both groups pass 0/3 T0 and 0/3 T1, and A-C bootstrap intervals include zero. Six A and four C samples are context-dominated, but sample attributes do not explain the historical gap.
- Status: `CASE_PROTOCOL_DRIFT`. Historical 3/3 evidence and `MF_SEQLIME_PARTIAL_PASS` randomization support are `superseded_by_protocol_consistency_audit`; B was not triggered and feedback training remains prohibited.
- Guards: canonical probability max difference 0, argmax agreement 1.0, metric differences 0, sequence-wrapper maximum difference `5.960464477539063e-08`.
