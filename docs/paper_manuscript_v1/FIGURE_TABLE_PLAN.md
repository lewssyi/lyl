# 正文图表冻结计划 V1

## 总体规则

所有主图只展示当前生效计算路径。模型训练路径、Canonical LIME、Reliability、Context Influence 与 Physiological Annotation 使用不同颜色和单向箭头；后三者均从冻结输出向外流，不允许画回 loss、attention、FiLM 或 classifier。图中文字统一使用 **Context-Preserving Classic Sequence-LIME**；Context Influence 标注为 independent optional diagnostic；生理层标注为 derived annotation, expert validation pending。

## Fig. 1 总体模型与 post-hoc 边界

内容固定为 raw waveform 与 segment-level features 两个输入，分别进入 epoch encoder，在 epoch 级融合后进入 Local TCN–Global Transformer 和五分类头。分类头之后分出三条独立只读路径：Canonical LIME、Reliability Evaluation、Context Influence；Physiological Annotation 由冻结 attribution 与 raw physiology 派生。九个概念不得画成 A1-zero 训练输入。图中显式标注 “Frozen model for XAI”“Pre-HMM response”“No training feedback”。结构来源为 `src/model_training.py`、`outputs/final_evidence_synthesis/isruc/canonical_mainline.json` 和 `docs/paper_xai/XAI_METHOD_SECTION.md`。

## Fig. 2 Raw/Segment 双视图编码

左侧展示 `(3000,3)` raw 输入及 kernel 5/11/21 多尺度卷积，右侧展示 `(30,28)` segment tensor 及 kernel 3/5/9 多尺度卷积；中间强调同一 physical epoch 和 cross-view alignment。不要为 28 个 feature 维度编造未核实标签。来源为 `_time_distributed_raw_multiscale_features` 与 `_time_distributed_segment_multiscale_features`。

## Fig. 3 TCN–Transformer 细节

展示 75-epoch token sequence、local/wide convolution、dilation 1/2/4/8/1/2 的六个 residual TCN blocks、四头 Transformer、逐 epoch 五类输出和 stride-5 overlapping-window aggregation。用 separate callout 说明正式分类采用 validation-selected temperature argmax，HMM 非 canonical。来源为 `src/model_training.py` 与 `CURRENT_RESULTS_MANIFEST.json`。

## Fig. 4 Context-Preserving Classic Sequence-LIME

展示 target epoch 的 3×30 mask、target-only channel–segment perturbation、raw/segment 同步更新、全部 target-covering 75-epoch 窗口重建、target-position Pre-HMM probability aggregation 和 weighted local Ridge。图注必须把输出称为 Local Perturbation Sensitivity，并注明 `MODEL_FAITHFULNESS_NOT_ESTABLISHED`。不得在图中加入 MF/HC、confidence reweighting、physiological reweighting 或 batch-16 路线。

## Fig. 5 Reliability、Context Influence 与代表案例

建议使用三至四个 panel。Panel A 使用现有 overall reliability diagram 和 equal-mass ECE；Panel B 使用 risk-coverage curve；Panel C 用简化示意和两样本表说明 independent Context Influence batch=1 smoke，而不画群体统计；Panel D 使用冻结代表案例，叠加 reliability state 与 derived physiological annotation，并保留冲突/不可靠标记。Reliability 图源位于 `outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/figures/`；案例来源为 `docs/paper_xai/XAI_CASE_STUDY_SELECTION.csv`。若版面不足，Context Influence smoke 移至补充材料。

## Table I 数据集

ISRUC 行报告 100 subjects、100 recordings、87,187 epochs 和 70/15/15 subject split；SleepEDF 行可报告 78 subjects、153 recordings、187,311 epochs，但 performance split/result 必须标 `TO_VERIFY`，直到与实际正式 run 对齐。来源为两个 `dataset_protocols/*/protocol_report.json`。

## Table II 主结果

当前只允许放 ISRUC A1-zero seed42：Accuracy 0.830860、Macro-F1 0.819873、kappa 0.782580 及五类 F1。SleepEDF 同构 A1-zero 行保持 `TO_VERIFY`，不得放入 SCR-Lite 数值或实际使用 ISRUC split 的 `edf_c0` 数值。若目标期刊要求双数据集主验证，Table II 在投稿前仍不完整。

## Table III 消融

正文优先保留 A1-zero 与同 seed SCR-Lite 负比较；Epoch-UTF-Lite、Pair SupCon 与 local contrast 可放补充表，并注明 seed/protocol。Base encoder 和 isolated sequence backbone 两行保持 `MISSING_REQUIRED_ABLATION`，不得从不配对历史 run 补齐。XAI 的 MF/HC 失败候选不混入模型架构消融表。

## Table IV XAI 评估

分成三个清晰分区：prediction reliability（ECE/NLL/Brier/AURC）；canonical LIME reliability（local R²、seed stability、reliable coverage、randomization 0/3）；auxiliary analyses（Context Influence 10/10 engineering checks、physiology automatic audit、0 evaluator）。每行写清 split、N、seed 和证据路径，避免把 validation 13,000、selected-30、diagnostic 12 或 development smoke 2 混为同一分母。

## 图表提交前检查

- 确认 Fig.1 没有有效 concept-training arrow，也没有 XAI-to-training feedback。
- 确认 Fig.4 的 response 是 Pre-HMM probability，且 attribution 维度为 90。
- 确认 Fig.5 的 coverage cutoff 不是 clinical/deployment threshold。
- 确认 Table II 不混 checkpoint、split、seed、decoder 或模型 variant。
- 确认 Table IV 将 Context Influence 和 Physiological Annotation 标为辅助分析。
- 所有图注加入有限声明，避免 faithful、causal、clinically validated 或 superior LIME。
