> **STATUS: SUPERSEDED HISTORICAL CANDIDATE**
>
> The candidate did not establish model faithfulness under the authoritative
> protocol. Original negative and diagnostic results remain preserved below.

# MF-SeqLIME Surrogate Improvement Report

## Scientific decision

**Decision: `VALIDATION_NO_QUALIFIED_CANDIDATE`.** The previously established overall boundary remains `MF_SEQLIME_PARTIAL_PASS`; no new candidate is promoted. The fixed 30-sample final-validation stage and fixed-36 test stage were not run because the 12-sample validation hard gate admitted zero candidates. This is the preregistered stopping rule, not a computational failure.

This study changed only local perturbation sampling and surrogate attribution construction. It did not train a model, change the canonical checkpoint, modify normal classification inference, use a stage-conditioned prior, alter Clinical Concept Annotation, or inspect test explanation endpoints for selection.

## Frozen protocol and completed computation

- Branch: `codex/a1-zero-mainline`; frozen anchor: `9978a3b5c3f75eb8d56a24b583894a0658b4f6ce`.
- Science/aggregation HEAD: `70c88f9`; implementation lineage: `4277c53`, `94f312c`, `8b1d34d`, `d234188`, `70c88f9`.
- Data boundary: 30 validation samples and the previously fixed 36 test samples were serialized before candidate evaluation; physical source-file overlap is zero.
- Smoke: 4 validation samples, R0/R7, 8/8 corrected `disjoint_v2` shards passed output, identity, mask, budget, and window-accounting guards.
- Small validation: 12 validation samples × R0/R3/R5/R7 × seeds 3101/3102/3103 = 144/144 complete shards, 0 failures.
- Candidate families: C0 Classic Bernoulli Ridge; C1 structured retention/contiguous mixes 30%, 50%, 70%; C2 70/30 base/boundary-adaptive sampling; C3 C2-style model-driven sampling plus budget-deducted current-model zero calibration at alpha 0, 0.1, 0.2, 0.3.
- Target: Y2 target-class logit margin under both T0 fixed-original-class and T1 randomized-model-class protocols.
- Budget: every candidate records 1,506 neighborhood requests plus 24 faithfulness requests, total 1,530. All 1,506 neighborhood masks are unique and train/local-validation masks are disjoint.
- Null: 1,000 deterministic attribution permutations for every candidate × protocol × randomization seed group.
- Test use: `false`; test evaluation count: `0`.

## Validation results

| Candidate | weighted R² | MAE | RMSE | deletion AUC ↓ | insertion AUC ↑ | mask-seed ρ | R7 | null q95 | T0 pass | T1 pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 Classic | 0.7685 | 0.2021 | 0.2622 | **0.4451** | **0.8403** | **0.8605** | 0.2380 | 0.1161 | 0/3 | 0/3 |
| C1 structured 30% | 0.7371 | 0.1518 | 0.2208 | 0.4902 | 0.8287 | 0.6989 | 0.1867 | 0.1156 | 0/3 | 0/3 |
| C1 structured 50% | 0.7393 | 0.1421 | 0.2102 | 0.4980 | 0.8246 | 0.7070 | 0.1962 | 0.1162 | 0/3 | 0/3 |
| C1 structured 70% | 0.7355 | **0.1407** | **0.2048** | 0.5003 | 0.8249 | 0.7159 | 0.2084 | 0.1155 | 0/3 | 0/3 |
| C2 boundary 50% | 0.7192 | 0.1508 | 0.2153 | 0.5004 | 0.8229 | 0.6420 | **0.1732** | 0.1156 | 0/3 | 0/3 |
| C3 alpha 0 | 0.7080 | 0.1600 | 0.2354 | 0.4991 | 0.8205 | 0.6171 | 0.1758 | 0.1157 | 0/3 | 0/3 |
| C3 alpha 0.1 | 0.7080 | 0.1600 | 0.2354 | 0.4944 | 0.8270 | 0.7448 | 0.1905 | 0.1156 | 0/3 | 0/3 |
| C3 alpha 0.2 | 0.7080 | 0.1600 | 0.2354 | 0.4968 | 0.8263 | 0.7779 | 0.2061 | 0.1162 | 0/3 | 0/3 |
| C3 alpha 0.3 | 0.7080 | 0.1600 | 0.2354 | 0.5000 | 0.8256 | 0.8062 | 0.2207 | 0.1168 | 0/3 | 0/3 |

All eight non-baseline candidates reduced MAE and RMSE, but every one reduced weighted R². All also worsened both faithfulness endpoints relative to C0: deletion AUC increased and insertion AUC decreased. No candidate or C0 reached the permutation-null region for any T0 or T1 seed. Thus the better pointwise error did not constitute a qualified fidelity improvement.

For C3, the model-driven LIME component stability was 0.6171 while zero-component stability was 1.0000. Final stability rose monotonically from 0.6171 at alpha 0 to 0.8062 at alpha 0.3, while R7 similarity worsened from 0.1758 to 0.2207. This directly shows that the apparent stability gain was increasingly inherited from the deterministic zero component and did not improve model-randomization sensitivity.

![Validation screening](../outputs/mf_seqlime_surrogate_improvement/isruc_seed42/figures/fig_validation_screening.png)

## Hard-gate outcome

Every non-baseline candidate failed at least these gates:

1. weighted R² was not at least C0;
2. neither deletion nor insertion improved while the other remained non-inferior;
3. T0 R7 pass count was 0/3 rather than at least 2/3;
4. T1 R7 pass count was 0/3 rather than at least 2/3;
5. observed R7 attribution similarity remained above the 1,000-permutation q95 null threshold.

Equal-budget, split-isolation, no-label adaptive sampling, current-randomized-model zero calibration, and randomization faithfulness-decay bookkeeping passed. The hard thresholds were not relaxed.

## Canonical and software guards

The final isolated guard passed: canonical batch-16 maximum probability difference 0.0; argmax agreement 1.0; Accuracy, Macro-F1, and Kappa differences 0.0; sequence-wrapper maximum difference 5.960464477539063e-08. The 1,200-forward fixed-graph guard had zero current-memory growth and zero late-half span; checkpoint SHA256 remained `e86a84790fd7351c1864158eef79fbf28437a7a8a5f26662b93bcf1aef14b6ed`. The complete maintained test suite passed 79/79.

## Required questions

1. **Did stratified-retention sampling improve surrogate R²?** No. Every C1 mix was below C0 (best C1: 0.7393 versus 0.7685).
2. **Did contiguous-block sampling improve surrogate R²?** No. None of the 30/50/70% structured mixtures exceeded C0, and increasing the contiguous fraction did not reverse the deficit.
3. **Did boundary-adaptive refinement improve surrogate R²?** No. C2 reached 0.7192, below both C0 and all C1 variants.
4. **Which sampling improved MAE/RMSE most?** C1 with 70% contiguous masks: MAE 0.1407 and RMSE 0.2048. This improvement coexisted with worse R².
5. **Which sampling improved deletion/insertion most?** None improved either endpoint over C0. Among new candidates, C1-30 was least degraded (deletion 0.4902, insertion 0.8287).
6. **Which sampling had the best T0 randomization result?** C3 alpha 0 had the lowest mean T0 R7 similarity (0.1850), closely followed by C2 (0.1876), but both passed 0/3 seeds.
7. **Which sampling had the best T1 randomization result?** C2 was lowest at 0.1588, but passed 0/3 seeds.
8. **Was pure V_ZERO stability mainly deterministic zero marginal?** Yes. In the new C3 audit the zero component had stability 1.0000 versus 0.6171 for its LIME component, consistent with the historical diagnosis.
9. **Did weak-zero calibration preserve faithfulness while improving model sensitivity?** No. Alpha increased final stability but also increased R7 similarity and did not restore either faithfulness endpoint.
10. **Did validation select a qualified candidate?** No; qualified count was zero.
11. **Was test run only once?** Test was not run at all, as required by the failed validation gate.
12. **Was a new candidate significantly better than Classic Sequence-LIME?** No. The direction of every required fidelity/faithfulness/randomization gate rules out that claim; no significance claim is made.
13. **Did surrogate fidelity genuinely improve?** No. Lower MAE/RMSE was offset by lower weighted R² for every candidate.
14. **Did T0 reach at least 2/3 seeds?** No; every candidate was 0/3.
15. **Did T1 reach at least 2/3 seeds?** No; every candidate was 0/3.
16. **What is the final decision?** `VALIDATION_NO_QUALIFIED_CANDIDATE`; retain the prior `MF_SEQLIME_PARTIAL_PASS` boundary.
17. **Is the method qualified for a feedback pilot?** No: `NOT_QUALIFIED_FOR_FEEDBACK_TRAINING`.
18. **What still blocks a paper-level claim?** Lower validation R², worse deletion/insertion endpoints, 0/3 T0 and T1 passes, and R7 similarity above permutation null.
19. **Was canonical parity preserved?** Yes; every final parity and fixed-graph memory guard passed.
20. **What are branch, HEAD, worktree, and commits?** Branch `codex/a1-zero-mainline`; science HEAD `70c88f9`; final worktree is required to be clean. Relevant commits are `4277c53`, `94f312c`, `8b1d34d`, `d234188`, and `70c88f9`; the final manifest records the closing documentation commit.

## 中文总结（不超过 500 字）

候选方法：C1 分层保留率/连续块、C2 边界自适应、C3 弱零校准。邻域采样改进：所有候选均采用唯一掩码、训练/局部验证隔离及相同 1530 查询预算。Surrogate fidelity：C1-70 的 MAE/RMSE 最低，但所有候选 weighted R² 均低于 Classic。Faithfulness：所有候选的 deletion AUC 与 insertion AUC 均劣于 Classic。Stability：弱零权重越大越稳定，但主要来自稳定度 1.0 的确定性 zero 分量。T0 randomization：全部 0/3。T1 randomization：全部 0/3。相比 Classic LIME：无候选通过综合硬门槛。论文级判定：`VALIDATION_NO_QUALIFIED_CANDIDATE`，保留 `MF_SEQLIME_PARTIAL_PASS`。是否允许 feedback pilot：否。下一步唯一建议：停止本阶段，不运行测试或训练；若开展新研究，应重新设计模型驱动的局部响应，而非继续增加 zero 权重。

## Protocol-consistency update

A subsequent audit confirmed that newer validation C0 0/3 is not an isolated sample
anomaly. The historical 12 diagnostic samples also pass 0/3 T0 and 0/3 T1 under the
same current 1,530-query protocol; A-C bootstrap intervals include zero. Historical
Classic M0 3/3 evidence used an incompatible protocol and is superseded. The phrase
"retain the prior `MF_SEQLIME_PARTIAL_PASS` boundary" above is historical; the current
boundary is `MF_SEQLIME_PARTIAL_PASS_SUPERSEDED_BY_PROTOCOL_CONSISTENCY_AUDIT`.
