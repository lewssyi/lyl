> **STATUS: VALIDATION FAILED HISTORICAL CANDIDATE**
>
> Validation decision: `VALIDATION_NO_QUALIFIED_HIERARCHICAL_CANDIDATE`.
> Paper-level decision: `HC_SEQLIME_FAIL`. The fixed 36-sample test was not
> executed. Results below are preserved as a negative feasibility study.

# Hierarchical Context Attribution Feasibility Study

## Executive decision

The strictly post-hoc HC-SeqLIME feasibility study stopped at validation with
`VALIDATION_NO_QUALIFIED_HIERARCHICAL_CANDIDATE`. Neither retained
configuration passed all ten preregistered validation gates. The frozen
36-sample test was therefore not run, Clinical Concept Annotation was not
rerun, the scientific status remains `MODEL_FAITHFULNESS_NOT_ESTABLISHED`, and
feedback training remains `BLOCKED`.

The negative decision is not caused by an incomplete run. All 1,080 validation
jobs completed with nonempty required outputs and equal 1,530-query budgets.
H3-G9-F45 improved context-only surrogate R², but both hierarchical candidates
were inferior to Classic-90 on the joint faithfulness endpoint and seed
stability, passed 0/3 T0 and 0/3 T1 randomization seeds, and retained R7
similarity above the 1,000-permutation null boundary.

## Frozen boundary and protocol

- Branch: `codex/a1-zero-mainline`.
- Classifier: frozen A1-zero checkpoint, SHA256
  `e86a84790fd7351c1864158eef79fbf28437a7a8a5f26662b93bcf1aef14b6ed`.
- Checkpoint policy: read-only; normal forward inference unchanged.
- Attribution target: model output only; no true-stage or predicted-stage
  prior, no stage channel prior, no explanation loss, no gradient feedback.
- Randomization: R0/R3/R5/R7, seeds 3101/3102/3103, T0 and T1 reported
  separately.
- Null: 1,000 deterministic permutations, absolute Spearman.
- Budget: 1,530 logical explanation queries for Classic-90 and every
  hierarchical candidate.
- Primary replacement: preregistered zero replacement.
- Data: 16 development, 30 validation, and frozen 36 test samples from
  physically disjoint source recordings.
- Selection: development could only exclude obvious failures; validation alone
  froze the decision; test data were never used for selection.

The local-mean and recording-neutral replacement implementations and isolation
tests exist, but no post-selection robustness experiment was started after the
zero-primary validation stop condition fired. They are therefore not pooled
with, or presented as confirmation of, the primary result.

## Execution record

Smoke completed 8/8 final jobs. One initial cache-write race produced a
preserved corrupt-cache diagnostic; atomic publication fixed the issue and
only the failed job was rerun.

Development completed 384/384 jobs and 3,072 candidate/protocol rows. It
evaluated G15/G9/G25, 30/45 local features, H0/H1/H2/H3, and R0/R3/R5/R7.
H1/H2 showed weak faithfulness, 45 features improved H3 local R² over 30 for
every group scheme, and development retained H3-G25-F45 and H3-G9-F45.
All 121 undefined R² values caused by constant randomized-model responses were
preserved and reported as reliability diagnostics rather than deleted.

Validation completed 1,080/1,080 jobs:

- 30 samples × 4 randomization levels × 3 seeds × one Classic-90 baseline;
- 30 samples × 4 levels × 3 seeds × two retained H3 candidates;
- both T0 and T1 emitted by every job;
- 0 final failures, 0 bad manifests, 0 missing required files;
- frozen-test execution count: 0.

## Validation results

### Surrogate fidelity

| Method | Local R² | Context R² | Hierarchical R² |
|---|---:|---:|---:|
| Classic-90 | 0.810626 | n/a | 0.810626 |
| H3-G25-F45 | 0.750169 | 0.736808 | 0.743489 |
| H3-G9-F45 | 0.699711 | **0.863927** | 0.781819 |

G9 passed the literal “context or hierarchical R² above Classic” gate through
its context-only R², but its combined hierarchical R² remained below Classic.
G25 did not pass the fidelity gate.

### Faithfulness endpoints

Lower deletion AUC and higher insertion AUC are favorable.

| Method | Deletion AUC | Insertion AUC | Insertion − deletion |
|---|---:|---:|---:|
| Classic-90 | -0.292432 | **2.640766** | **2.933198** |
| H3-G25-F45 | **-0.307761** | 2.352117 | 2.659878 |
| H3-G9-F45 | -0.189773 | 2.445273 | 2.635045 |

G25 improved deletion slightly but worsened insertion materially. G9 worsened
both endpoints. Neither passed the joint endpoint gate.

### Seed stability and randomization

| Method | Mask-seed stability | T0 seed passes | T1 seed passes |
|---|---:|---:|---:|
| Classic-90 | **0.895459** | 0/3 | 0/3 |
| H3-G25-F45 | 0.813499 | 0/3 | 0/3 |
| H3-G9-F45 | 0.860199 | 0/3 | 0/3 |

For H3-G25-F45, mean R7 similarity ranged from 0.1304 to 0.1728 while
per-seed null q95 ranged from 0.0913 to 0.0926. For H3-G9-F45, observed values
ranged from 0.1414 to 0.1838 versus null q95 from 0.0985 to 0.1000. R7
faithfulness decayed strongly for every seed and protocol, but attribution
similarity did not reach the null boundary. Consequently both T0 and T1 passed
0/3 seeds for both candidates.

### Context dependence

Across validation samples, the mean absolute target contribution was 2.882.
For G25, mean neighbor plus distant context contribution was 2.621 and mean
context-dependence ratio was 0.512; 56.7% of samples exceeded the fixed 0.5
context-dominated threshold. For G9, corresponding context contribution was
1.009, the ratio was 0.296, and 20.0% exceeded 0.5.

Thus the classifier shows mixed, granularity-sensitive target/context
dependence. The evidence does not support a universal claim that the model
mainly uses context, nor does it support the claim that within-target failure
is explained solely by context use.

### Reliability and coverage

Validation-frozen reliability thresholds were the 10th percentile of response
variance, 25th percentiles of hierarchical R² and seed stability, a
context-dependence threshold of 0.5, and method/protocol-specific
1,000-permutation R7 null q95 values. All samples were assigned exactly one
state; unreliable rows were never dropped.

| Method | T0 reliable coverage | T1 reliable coverage |
|---|---:|---:|
| Classic-90 | 3.3% | 10.0% |
| H3-G25-F45 | **16.7%** | **16.7%** |
| H3-G9-F45 | 10.0% | 6.7% |

`RANDOMIZATION_UNRELIABLE` was the dominant rejection reason. For
H3-G25-F45 it covered 15/30 samples under each protocol; for H3-G9-F45 it
covered 19/30 T0 and 20/30 T1. Reliable coverage was zero for several N2/N3
strata. REM was the easiest stratum for G9, while N2/N3 were generally the
hardest. Transition samples were harder for G25 (6.7% reliable versus 26.7%
for stable samples). Incorrect samples were not consistently harder: the
direction changed by candidate and T0/T1, so no general correctness effect is
claimed.

## Ten validation hard gates

| Gate | H3-G25-F45 | H3-G9-F45 |
|---|:---:|:---:|
| 1. Context/hierarchical R² above Classic | FAIL | PASS |
| 2. One endpoint improves, other noninferior | FAIL | FAIL |
| 3. Stability noninferior to Classic | FAIL | FAIL |
| 4. T0 at least 2/3 seeds | FAIL (0/3) | FAIL (0/3) |
| 5. T1 at least 2/3 seeds | FAIL (0/3) | FAIL (0/3) |
| 6. R7 similarity reaches permutation null | FAIL | FAIL |
| 7. Faithfulness decays after randomization | PASS | PASS |
| 8. Equal query budget | PASS | PASS |
| 9. No stage prior | PASS | PASS |
| 10. Complete coverage/rejection reporting | PASS | PASS |

All ten gates were mandatory. No candidate qualified; the fixed test remained
blocked.

## Guards and maintained tests

- Exact canonical artifact probability difference: 0.
- Scientific prediction parity: probability max difference 0, argmax
  agreement 1.0, Accuracy/Macro-F1/Kappa differences 0.
- Sequence-wrapper maximum difference:
  `5.960464477539063e-08`; argmax agreement 1.0.
- Fixed-graph guard: 1,200 forwards, measured current-memory growth 0 bytes.
- Checkpoint hash and in-memory model-weight fingerprint unchanged.
- Maintained tests: 101 `unittest` checks plus 21 pytest-style plain test
  functions, 122/122 passed. The frozen environment lacks the pytest package,
  so the latter used the same one-shot `raises`/`approx` compatibility shim;
  no package, environment, or repository dependency was changed.

## Provenance note

During the initial status/guard phase, invoking the legacy
`tools/final_xai_guards.py --help` unexpectedly executed that script because
it does not honor argparse help. This rewrote only legacy guard/parity
artifacts under `outputs/sequence_lime_two_versions/isruc_seed42/final_validation/`.
No classifier checkpoint, model weight, method result, sample selection, or
scientific decision changed. The incident is retained as provenance; all
formal guards for this study were subsequently rerun under the isolated
hierarchical-context output root.

## Required final answers

1. **Target epoch or context?** Mixed and granularity-sensitive. G25 makes
   target and context contributions comparable; G9 remains target-dominant.
2. **Context-dominated share?** 56.7% for G25 and 20.0% for G9 at the frozen
   ratio ≥0.5 definition.
3. **Contribution distribution?** Mean absolute target contribution 2.882;
   mean neighbor+distant context contribution 2.621 (G25) or 1.009 (G9).
4. **Best G15/G9/G25?** No scheme is validated as best. G9 had the best
   context R²; G25 had lower R7 similarity; G15 was excluded in development.
5. **Best 30/45/90 granularity?** H3-45 beat H3-30 in development, but
   Classic-90 retained better validation fidelity and stability. No HC
   granularity qualified.
6. **Did hierarchical surrogate improve R²?** No. The best combined H3 R² was
   0.781819 versus Classic 0.810626.
7. **Did deletion/insertion improve?** No jointly. G25 improved deletion only;
   G9 improved neither.
8. **Did stability improve?** No; 0.813499/0.860199 versus Classic 0.895459.
9. **Did T0 reach 2/3?** No, both candidates were 0/3.
10. **Did T1 reach 2/3?** No, both candidates were 0/3.
11. **Was R7 near null?** No; every candidate/seed/protocol mean exceeded its
    permutation-null q95.
12. **Reliable coverage?** Best was 16.7% for G25 under T0 and T1; all
    candidate/protocol values were 6.7–16.7%.
13. **Easiest/hardest stages?** REM was easiest for G9; N2/N3 were generally
    hardest. Small strata preclude a broad physiological claim.
14. **Were incorrect/transition samples harder?** Transition was harder for
    G25. Incorrect was not consistently harder across candidate/protocol.
15. **Qualified validation candidate?** No.
16. **Fixed test run?** No; execution count is 0.
17. **Paper level reached?** No: `HC_SEQLIME_FAIL`.
18. **Feedback training prohibited?** Yes, `BLOCKED`.
19. **Canonical parity preserved?** Yes, including sequence-wrapper and memory
    guards.
20. **Branch/HEAD/worktree/commits?** Branch
    `codex/a1-zero-mainline`; staged implementation/analysis commits include
    `09e6e14`, `1856f0f`, `6e52445`, `124330d`, `639c667`, `6c2f61e`,
    `9edebb4`, `1fdd3fb`, `d53d76c`, and `a445967`, followed by the closing
    documentation commit. The final manifest records the delivery HEAD and a
    clean worktree.

## 中文总结（不超过 500 字）

预测依赖结构：目标 epoch 与上下文混合且依赖分组粒度；G25 中上下文贡献与目标接近，G9 仍以目标为主。层级解释候选：H3-G25-F45、H3-G9-F45 均未通过十项 validation 硬门。Surrogate fidelity：G9 context R² 高于 Classic，但 combined hierarchical R² 仍低。Faithfulness：G25 仅改善 deletion，insertion 变差；G9 两端均未改善。Stability：两候选均低于 Classic。Randomization：T0/T1 均为 0/3，R7 similarity 高于 1000 次置换 null。可靠解释覆盖率：6.7%–16.7%。相比 Classic：未证明整体优势。论文级判定：`HC_SEQLIME_FAIL`，固定 36 test 未运行。Feedback 状态：`BLOCKED`。下一步：若继续研究，须预注册新的结构假设与可靠性目标，并使用新的独立 validation/test，不能据本次 test（未运行）或当前结果调参。
