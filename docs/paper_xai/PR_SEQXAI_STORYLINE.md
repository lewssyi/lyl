# PR-SeqXAI Paper Storyline

## One-sentence positioning

**PR-SeqXAI is a strictly post-hoc explanation framework for overlapping
sleep-sequence inference that separates context-preserving perturbation
sensitivity, physiology-oriented annotation, and explicit reliability
disclosure.**

**PR-SeqXAI 是面向重叠睡眠序列推理的严格事后解释框架；它将上下文保持的扰动敏感性、
生理语义注释和可靠性披露分成三个彼此独立的层。**

PR-SeqXAI is not a new attribution algorithm. Its frozen definition is:

```text
PR-SeqXAI
= Context-Preserving Classic Sequence-LIME
+ independent Clinical Concept Annotation
+ Reliability-Aware Visualization
```

The underlying attribution remains **Context-Preserving Classic
Sequence-LIME**, with the restricted scientific role of exploratory Local
Perturbation Sensitivity.

## Core narrative

### Problem 1: epoch-wise perturbation does not reproduce sequence inference

The classifier predicts through overlapping 75-epoch windows. Perturbing one
epoch as an isolated sample would change the query semantics and would not
reproduce the canonical sequence-inference path.

PR-SeqXAI perturbs the target epoch, reconstructs every covering 75-epoch
window, queries the frozen classifier, and aggregates the target-position
pre-HMM neural response across overlapping windows. This establishes
**context-preserving, overlapping-window-aware sequence inference**. It does
not establish model faithfulness.

### Problem 2: a coefficient heatmap has no direct physiological vocabulary

Channel-by-segment sensitivity coefficients do not themselves identify EEG
frequency bands, EOG eye-movement patterns, or EMG tone. The independent
Clinical Concept Annotation layer computes nine fixed
`raw_physiology_v1` concepts and maps those derived measurements to the
high-sensitivity regions.

The annotation layer cannot change the attribution array, ranking,
prediction, or model. True stage is joined only after annotation for
reporting. Unsupported evidence and opposing-concept conflicts are retained
and reported. These constraints support a physiology-oriented representation,
not clinical correctness or causal model reasoning.

### Problem 3: a heatmap alone hides explanation failure modes

The reliability-aware presentation reports local surrogate fit,
insertion/deletion response, mask-seed stability, full-model randomization,
reliability state, and conflict warnings beside the unchanged sensitivity
map. Negative results remain visible: canonical randomization passed 0/3
seeds, only 1/30 validation cases met the complete reliability rule, and the
authoritative status remains `MODEL_FAITHFULNESS_NOT_ESTABLISHED`.

The contribution is therefore **explicit uncertainty and failure
disclosure**, not validated reliability or improved human interpretation.

## Contributions for the paper

### English

1. A dual-view epoch representation framework combining raw multichannel
   waveforms and dynamic statistical-spectral segments.
2. A TCN-Transformer sequence model capturing local transition patterns and
   global dependencies across overlapping sleep sequences.
3. PR-SeqXAI, a context-preserving post-hoc explanation framework that
   separates model-derived perturbation sensitivity, physiological concept
   annotation, and explanation reliability.

### 中文

1. 提出融合原始多通道波形与动态统计—频谱片段的双视图 epoch 表征框架。
2. 构建 TCN–Transformer 序列模型，以建模重叠睡眠序列中的局部转移模式和全局依赖。
3. 提出 PR-SeqXAI 事后解释框架，将模型扰动敏感性、生理概念注释和解释可靠性严格分离，
   并保持重叠窗口推理语义一致。

The third contribution must not be described with `outperforms LIME`,
`superior attribution`, `model-faithful`, `causal`, or `clinically
validated`.

## Reusable manuscript language

### Abstract XAI sentence

**English.** We further introduce PR-SeqXAI, a strictly post-hoc framework
that preserves overlapping sequence context while separating local
perturbation sensitivity, physiology-oriented annotation, and reliability
disclosure; the current evidence validates the implementation and automatic
annotation audit, but does not establish model faithfulness, human
interpretability, or clinical utility.

**中文。** 我们进一步提出 PR-SeqXAI：一个保持重叠序列上下文、并将局部扰动敏感性、
生理语义注释与可靠性披露严格分离的事后解释框架。现有证据支持其工程实现与自动注释审计，
但尚未建立模型忠实性、人类可理解性或临床效用。

### Method positioning

**English.** PR-SeqXAI is a framework-level composition rather than a new
attribution estimator. Context-Preserving Classic Sequence-LIME produces the
only model-derived sensitivity map; Clinical Concept Annotation and
reliability visualization are independent downstream layers.

**中文。** PR-SeqXAI 是框架级组合，而非新的归因估计器。唯一由模型查询产生的敏感性图来自
Context-Preserving Classic Sequence-LIME；临床概念注释和可靠性可视化均为独立下游层。

### Results positioning

**English.** Frozen-artifact audits established exact attribution, ranking,
and color-scale identity across the base and enhanced presentations. Automatic
physiology checks showed 100.0% concept coverage, 0.0% unsupported
annotations, 10.0% conflicts, 96.7% channel-concept consistency, and 0.973
cross-seed concept stability. These endpoints measure coverage, consistency,
and stability, not clinical validity or model faithfulness.

**中文。** 冻结产物审计确认基础展示与增强展示的 attribution、ranking 和 color scale
完全一致。自动生理审计得到 100.0% 概念覆盖、0.0% unsupported、10.0% conflict、
96.7% 通道—概念一致性和 0.973 跨种子概念稳定性；这些指标衡量覆盖性、一致性和稳定性，
不代表临床有效性或模型忠实性。

### Limitations

**English.** Human interpretability and clinical utility were not evaluated
in the current study. A frozen blinded evaluation package has been prepared
for future assessment. Consequently, we make no claim that users found the
display easier to interpret, preferred it, completed tasks more accurately,
or were less likely to misinterpret an explanation.

**中文。** 当前研究尚未评价人类可理解性和临床效用；已冻结盲法评价包供未来研究使用。
因此，本文不声称使用者更易理解、更偏好该展示、任务正确率更高或误读率更低。

### Figure-caption boundary

**English.** The panels report model-derived local perturbation sensitivity,
derived physiology-oriented annotation, and reliability/conflict disclosure
as separate information layers. Identical sensitivity maps are retained
across display variants. The annotations are not causal explanations or
evidence of clinical validation; human evaluation was not performed.

**中文。** 各面板分别呈现模型查询得到的局部扰动敏感性、派生生理语义注释和可靠性/冲突
披露；不同展示版本使用完全相同的敏感性图。注释不构成因果解释或临床验证证据，且本研究
未开展人类评价。

## Main-text and supplement allocation

The main text should contain:

- Figure A: guarded post-hoc architecture;
- Figure B: reliability audit;
- Figure C: a manuscript composite of the frozen reliability-coverage panel
  and the existing automatic physiology-metric summary;
- Figure D: fixed-rule representative cases;
- one table of automatic physiological consistency metrics.

The supplement should retain protocol drift, MF failure, structured-
neighborhood failure, weak-zero failure, HC failure, complete randomization
tables, all 30 blinded pairs, and the frozen expert package. Supplementary
placement does not demote negative evidence; it preserves protocol detail
without obscuring the central framework narrative.
