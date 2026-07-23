# Classic Sequence-LIME Protocol-Consistency Audit

## Scope and decision

The sole scientific decision is `CASE_PROTOCOL_DRIFT`. The historical A/B/C
Classic Sequence-LIME 3/3 randomization sanity and the newer validation 0/3
result were not produced by the same protocol. Under one current,
protocol-versioned evaluator, both the historical 12-sample diagnostic group A
and the newer 12-sample validation group C pass 0/3 T0 seeds and 0/3 T1 seeds.
Their between-group differences are statistically compatible with sampling
variation. The historical 3/3 conclusion is therefore superseded; it cannot be
used as evidence that Classic Sequence-LIME randomization sensitivity
generalizes.

This audit did not alter LIME attribution, MF-SeqLIME, zero calibration,
Clinical Concept Annotation, the classifier, the checkpoint, or training. It
did not use test results for method selection.

## Frozen inputs and unified protocol

- Branch: `codex/a1-zero-mainline`.
- Frozen A1-zero anchor: `9978a3b5c3f75eb8d56a24b583894a0658b4f6ce`.
- Audit pre-documentation HEAD: `dee3d30`.
- Checkpoint SHA256: `e86a84790fd7351c1864158eef79fbf28437a7a8a5f26662b93bcf1aef14b6ed`.
- Evaluator: `sequence_lime_v1.1.3`, wrapped by
  `classic_sequence_lime_consistency_v1`.
- Method: Classic Sequence-LIME only, zero replacement, no prior.
- Target: Y2 log-probability margin.
- Protocols: T0 fixes the original predicted class; T1 uses the randomized
  model's current predicted class.
- Models: R0/R3/R5/R7, seeds 3101/3102/3103.
- Masks: 1,200 train + 306 local validation, unique Bernoulli masks, shared
  seed formula, original and zero masks included.
- Faithfulness: 24 deletion/insertion queries; total budget exactly 1,530.
- Null: 1,000 attribution permutations per sample/protocol/seed, followed by
  frozen group aggregation.
- Pass: R7 absolute Spearman must be null-compatible, R0 faithfulness advantage
  must decay at R7, and at least 2/3 seeds must agree.

The 36 old samples and 30 serialized validation samples were recovered without
resampling. Old/new physical source overlap and sample-ID overlap are both
zero. Group A is the historical fixed 12-sample diagnostic subset. Group C is
the 12 validation samples actually used in the surrogate study. Group B was
preregistered as conditional and was not run because A and C were statistically
compatible and had identical pass counts.

## Material protocol drift

The historical A/B/C run used 256 train and 64 validation masks,
`local_stratified` deletion-heavy sampling, the seed formula `seed + sample`,
multiple surrogate targets, 2,154 queries, signed-Spearman-oriented null
aggregation, a different faithfulness comparison and a different pass
aggregation. The newer validation run used 1,200 + 306 Bernoulli masks, a
different seed formula, Y2 only, 1,530 queries, absolute Spearman and a newer
candidate/protocol/seed rule. Cache identity did not previously expose one
shared protocol version. These fields can materially change randomization
conclusions.

R7 model construction itself did not drift after the v1.1.3 route repair.
Both audits use the current randomized model for all responses. The substantive
inconsistency was in neighborhood configuration, target multiplicity, metric
sign handling, null/pass aggregation, faithfulness endpoint and cache/resume
identity.

## Unified rerun results

All 288 A/C shards completed under the same evaluator and query budget. The
first launcher exposed a GPU-slot scheduling defect; failed attempts were
preserved, the launcher was stopped, and a one-serial-worker-per-GPU retry
completed with zero failures without restarting completed shards.

For group A, T0 mean R7 absolute Spearman by seed was 0.1725, 0.2339 and
0.3233, compared with null q95 values 0.1140, 0.1178 and 0.1169. T1 values were
0.2670, 0.1813 and 0.2462, all above null. Thus T0 and T1 are both 0/3.

For group C, T0 values were 0.2601, 0.2057 and 0.2893, and T1 values were
0.2535, 0.1751 and 0.2441. All exceeded null q95 near 0.115-0.118; T0 and T1
are both 0/3. Faithfulness advantage decayed strongly at R7 in every seed, but
that necessary condition could not override failed null compatibility.

The A-minus-C bootstrap interval for R7 absolute Spearman was
[-0.0901, 0.0738] for T0 and [-0.0719, 0.0902] for T1. Both include zero, and
both groups have the same seed-pass counts. Therefore group B was not triggered.

## Sample-dependency analysis

The sample-level diagnostic produced 18/48 sample-protocol passes even though
no group/protocol reached 2/3 seeds. Six of 12 A samples and four of 12 C
samples had context-dependence ratio >=0.5. This shows a context-dominated
subpopulation exists, but its frequency and the A/C randomization endpoints do
not explain the historical 3/3 versus newer 0/3 discrepancy.

Confidence, Y2 margin, entropy, correctness, transition status, local response
variance, effective sample size and context ratio all had bootstrap intervals
covering zero for association with individual pass. Surrogate weighted R2 had
Spearman -0.336 (95% bootstrap CI [-0.574, -0.067]); target-epoch occlusion
drop had -0.323 ([-0.595, -0.010]). These are unadjusted descriptive findings
from a small, repeated sample-protocol dataset and are not treated as causal or
used for selection.

## Direct answers

1. **Did old A/B/C and new validation use the same protocol?** No.
2. **Which configuration fields changed?** Mask counts, mask distribution,
   seed formula, target multiplicity, total queries, null aggregation,
   similarity sign handling, pass rule, faithfulness endpoint, cache identity
   and resume validation.
3. **Which code paths changed?** The two runs shared response, sequence-wrapper,
   current randomization and Ridge primitives, but entered them through
   different mask/config and aggregation routes. The audit now provides one
   protocol-versioned entry and one pass aggregation.
4. **Do the old 12 samples still pass under the unified protocol?** No; T0 and
   T1 are both 0/3.
5. **Do the old fixed 36 samples pass?** Not re-evaluated. The preregistered
   condition for B was not met; its historical 3/3 claim is superseded rather
   than reused.
6. **Why was new validation 0/3?** R7 absolute Spearman stayed above its
   permutation q95 for every seed, although faithfulness decayed.
7. **Is the discrepancy protocol drift, sample dependence, or both?** Primary
   and only supported main decision: `CASE_PROTOCOL_DRIFT`. Sample heterogeneity
   exists but does not explain the historical discrepancy.
8. **Does Classic Sequence-LIME randomization generalize?** No such claim is
   supported; both independent 12-sample groups fail the unified rule.
9. **Do margin, confidence or correctness explain the discrepancy?** No; their
   descriptive bootstrap intervals cover zero and A/C remain compatible.
10. **Does transition/stable status explain it?** No.
11. **Does target-epoch occlusion drop explain it?** It is associated with
    individual pass status descriptively, but cannot explain old 3/3 versus new
    0/3 because unified A and C outcomes are the same.
12. **Does context-dependence ratio explain it?** No; it identifies
    heterogeneous samples but not the historical group discrepancy.
13. **Is there a context-dominated sample class?** Yes: 6/12 A and 4/12 C under
    the diagnostic threshold, reported without an eligibility gate.
14. **Must the `MF_SEQLIME_PARTIAL_PASS` boundary change?** Yes. It is now
    `MF_SEQLIME_PARTIAL_PASS_SUPERSEDED_BY_PROTOCOL_CONSISTENCY_AUDIT`; its old
    randomization support is not current evidence.
15. **Can feedback training start?** No.
16. **Should hierarchical context attribution be the next stage?** Not within
    this audit. The only immediate action is consistent use of the single
    evaluator and corrected reporting; a context method would require a new
    preregistered study.
17. **Was canonical parity preserved?** Yes: probability max difference 0,
    argmax agreement 1.0, metric differences 0, and sequence-wrapper maximum
    difference `5.960464477539063e-08`.
18. **Branch, HEAD, worktree and commits?** Branch
    `codex/a1-zero-mainline`; audit commits are `f168c5d`, `31b8dab`,
    `dee3d30` and the closing documentation commit. The final manifest records
    the delivery HEAD; the worktree is required to be clean.

## Validation

The maintained suite passed 95/95 checks: 74 `unittest` tests plus all 21
pytest-style surrogate-improvement tests. Because the frozen `l_venv` does not
contain the `pytest` package, those 21 plain test functions were executed with
a one-shot compatibility shim limited to the two APIs they use (`raises` and
`approx`); no package was installed and no repository or environment file was
changed. The isolated 1,200-forward fixed-graph guard also passed with zero
measured growth, while checkpoint and model-weight hashes remained unchanged.

## Reproducibility outputs

The authoritative root is
`outputs/sequence_lime_protocol_consistency_audit/isruc_seed42/`. It contains
the resolved config diff, code-path audit, recovered sample manifests,
sample/group results, 1,000-permutation nulls, occlusion/dependency analyses,
final decision, figure source and rendered PDF/PNG. No protected formal output
root was modified.

## 中文总结（不超过 500 字）

主判定：`CASE_PROTOCOL_DRIFT`。是否存在协议漂移：是；旧 A/B/C 与新 validation 在 mask 数量/分布、seed、query budget、null、Spearman 符号、faithfulness 与 pass 聚合上不一致。旧样本复现：统一协议下旧 12 样本 T0/T1 均为 0/3，历史 3/3 不可复现；按预注册条件未运行旧 36。新 validation：T0/T1 均为 0/3。主要差异因素：协议而非样本；A/C bootstrap 区间跨 0。Classic LIME 泛化性：不支持。当前方法边界：原 `MF_SEQLIME_PARTIAL_PASS` 已被本审计取代。是否允许 feedback training：否。下一步唯一建议：所有后续报告只使用统一、版本化 evaluator，先修正文献结论，不开发新方法或启动训练。
