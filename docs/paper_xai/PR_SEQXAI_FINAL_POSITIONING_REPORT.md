# PR-SeqXAI Final Positioning Report

## Final position

PR-SeqXAI is the **Physiology-Aware and Reliability-Controlled Post-hoc
Sequence Explanation Framework**. It is a framework-level composition:

```text
Context-Preserving Classic Sequence-LIME
+ independent Clinical Concept Annotation
+ Reliability-Aware Visualization
```

It is not a new attribution algorithm and does not replace the canonical
attribution name.

## Required decisions

1. **没有评价者是否阻止论文继续？**
   不阻止以“序列适配、自动生理审计和可靠性披露框架”为第三项贡献继续写作；它阻止任何
   人类可理解性、偏好、任务正确率、误读减少和临床效用结论。
   Evaluator count is zero.

2. **PR-SeqXAI 的创新对象是什么？**
   是严格分离模型扰动敏感性、派生生理概念和可靠性/冲突证据的框架及表示协议。

3. **时间序列适配证据是什么？**
   目标 epoch 扰动后重建全部覆盖的 75-epoch 窗口，聚合 target-position pre-HMM
   response；canonical parity、wrapper parity、缓存/模型 provenance 和 memory guard
   均通过。

4. **生理特性证据是什么？**
   30-case 自动审计得到 coverage 100.0%、unsupported 0.0%、conflict 10.0%、
   channel-concept consistency 96.7% 和 cross-seed stability 0.973。

5. **可视化创新如何定义？**
   定义为三层解释表示：模型衍生的 channel-temporal sensitivity、信号衍生的
   physiological annotation、审计衍生的 reliability/conflict disclosure。

6. **哪些证据来自自动评价？**
   Attribution/ranking/color-scale identity，以及 coverage、unsupported、
   conflict、channel-concept consistency、top-region evidence、cross-seed
   stability、stage agreement 和 concept sparsity。

7. **哪些人类相关结论不能声明？**
   不能声称更容易理解、专家偏好、可靠性判断更准确、完成更快、错误分析更有用、
   误读更少或具有临床效用。

8. **能否作为论文第三项核心贡献？**
   可以，但必须限定为 post-hoc framework、sequence adaptation、automatic audit
   和 uncertainty disclosure；不得写成 attribution superiority。

9. **能否作为独立 XAI 论文唯一贡献？**
   当前证据不足。若作为独立 XAI 论文，缺失的人类评价、临床评价和未建立的
   model faithfulness 会构成核心证据缺口。

10. **Model faithfulness 状态是什么？**
    `NOT_ESTABLISHED`，并继续保留权威总状态
    `MODEL_FAITHFULNESS_NOT_ESTABLISHED`。

11. **Clinical validation 状态是什么？**
    `NOT_VALIDATED`。CCA 仅为自动派生注释，专家评价未进行。

12. **Feedback 状态是什么？**
    `PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD`；不参与训练或正常推理。
    Normal inference does not read PR-SeqXAI artifacts.

## Multidimensional status

| Dimension | Status |
| --- | --- |
| Engineering | `IMPLEMENTATION_AND_AUDIT_PIPELINE_VALID` |
| Sequence adaptation | `CONTEXT_PRESERVING_SEQUENCE_INFERENCE_ESTABLISHED` |
| Physiology annotation | `AUTOMATIC_PHYSIOLOGY_CONSISTENCY_SUPPORTED` |
| Visualization | `PHYSIOLOGY_AND_RELIABILITY_ENHANCED_PRESENTATION_IMPLEMENTED` |
| Human interpretability | `NOT_EVALUATED` |
| Clinical validation | `NOT_VALIDATED` |
| Model faithfulness | `NOT_ESTABLISHED` |
| Feedback training | `PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD` |

`PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED` is retained. It means that
genuine human evaluation has not yet been completed; it does not mean that the
framework implementation, automatic physiological audit, or display
construction failed.

## Paper-ready bilingual text

### Contribution

**English.** We introduce PR-SeqXAI, a strictly post-hoc framework for
overlapping sleep-sequence inference that separates context-preserving local
perturbation sensitivity, physiology-oriented annotation, and explicit
reliability disclosure.

**中文。** 我们提出 PR-SeqXAI：一个面向重叠睡眠序列推理的严格事后解释框架，将上下文
保持的局部扰动敏感性、生理语义注释和可靠性披露彼此分离。

### Abstract

**English.** Frozen-artifact audits confirm sequence-query consistency and
exact isolation of the attribution from the downstream display, while
automatic physiology metrics quantify annotation coverage, consistency,
stability, and conflicts. Human interpretability, clinical utility, and model
faithfulness remain unestablished.

**中文。** 冻结产物审计确认序列查询一致性以及 attribution 与下游展示的严格隔离；
自动生理指标量化了注释覆盖、一致性、稳定性和冲突。人类可理解性、临床效用和模型忠实性
仍未建立。

### Method

**English.** The only model-derived map is produced by Context-Preserving
Classic Sequence-LIME. Clinical Concept Annotation and reliability
visualization operate downstream and cannot modify attribution, ranking,
prediction, normal inference, or training.

**中文。** 唯一由模型查询生成的图来自 Context-Preserving Classic Sequence-LIME；
临床概念注释和可靠性可视化均位于下游，不能修改 attribution、ranking、prediction、
正常推理或训练。

### Results

**English.** Across 30 frozen validation cases, the display variants retained
bitwise-identical attributions and rankings. Automatic annotation achieved
100.0% coverage, 0.0% unsupported output, 10.0% conflict, 96.7%
channel-concept consistency, and 0.973 cross-seed stability.

**中文。** 在 30 个冻结 validation 案例中，两种展示保持 bitwise-identical 的
attribution 和 ranking；自动注释得到 100.0% coverage、0.0% unsupported、
10.0% conflict、96.7% 通道—概念一致性和 0.973 跨种子稳定性。

### Limitations

**English.** Human interpretability and clinical utility were not evaluated
in the current study. A frozen blinded evaluation package has been prepared
for future assessment. The automatic metrics do not establish causal
physiology or model faithfulness.

**中文。** 当前研究未评价人类可理解性或临床效用，已冻结盲法评价包供未来评估使用。
自动指标不建立因果生理机制或模型忠实性。

### Figure-caption boundary

**English.** Derived annotation and reliability information are shown beside,
but never fused into, the unchanged Local Perturbation Sensitivity map. Human
and clinical validation were not performed.

**中文。** 派生注释和可靠性信息仅与未改变的 Local Perturbation Sensitivity 图并列
呈现，绝不融合回 attribution；本研究未进行人类或临床验证。

## Reconstruction verification

- No LIME experiment, model inference, training, attribution recomputation,
  metric recomputation, or A/B reallocation was performed.
- The four frozen automatic-result hashes checked by the guard are unchanged.
- The expert package remains 92 files; its Git tree is
  `2c9603065c7758fd45f2cc825d0a76fce32feb0c` and its package-manifest
  SHA-256 remains
  `26f3900ae498d8a3d463d3bc9e0efaaf7b63c0955931b91716037f5891c55fe9`.
- The repository-wide wording audit classified 107 matches as 64
  `CURRENT_ALLOWED`, 43 `HISTORICAL_WITH_WARNING`, and 0
  `PROHIBITED_CURRENT_CLAIM`.
- The narrative-only static guard suite passes 10/10 checks.
