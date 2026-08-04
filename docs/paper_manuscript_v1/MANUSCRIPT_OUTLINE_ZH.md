# 中文论文完整大纲 V1

## 题目与摘要

建议题目围绕“面向睡眠分期的双视图时序建模与证据受限的 post-hoc 解释”展开，不在题目中使用“faithful”“clinical validation”或“superior LIME”。摘要按背景、方法、结果、结论四段逻辑组织：方法概括 raw/segment 双视图、Local TCN–Global Transformer、Context-Preserving Classic Sequence-LIME、Reliability Evaluation，以及两个独立辅助层；结果只给 ISRUC 冻结主指标与 validation reliability 指标；Context Influence 只写工程 smoke，生理标注只写自动审计；结论强调 post-hoc 分层隔离和透明的失败边界。数值来源为 `outputs/final_evidence_synthesis/isruc/canonical_mainline.json`、`outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/overall_metrics.json` 与 `docs/paper_xai/XAI_RESULTS_SECTION.md`。

## 1. 引言

### 1.1 研究背景

- 说明自动睡眠分期既需要 epoch 内形态信息，也需要跨 epoch 阶段转换上下文。
- 说明性能、概率可靠性与解释可读性是不同问题，不能由单一 attribution 分数替代。
- 外部文献与临床背景引用目前为 **TO_VERIFY**，投稿前需建立正式参考文献表。

### 1.2 现有工作局限

- 讨论仅使用 raw waveform 或仅使用工程特征的局限，以及忽略重叠序列上下文的 epoch-wise 解释问题。
- 区分 attribution、reliability、context influence 与 physiological annotation，避免把下游展示当作模型机制证据。
- 相关工作证据来自外部文献，当前项目内没有完成的引用索引，统一标记 **TO_VERIFY**。

### 1.3 本文贡献

- 双视图 epoch 表征与 75-epoch TCN–Transformer 序列分类主线，证据为 `src/model_training.py` 与 `canonical_mainline.json`。
- Context-Preserving Classic Sequence-LIME：在不破坏序列上下文的前提下输出 90 维 Local Perturbation Sensitivity，证据为 `docs/paper_xai/XAI_METHOD_SECTION.md`。
- 独立 Reliability Evaluation：报告 calibration 与 selective risk，证据为 `SLEEPXVIT_RELIABILITY_AUDIT.md`。
- Context Influence 和 Physiological Annotation 作为独立辅助分析，而非 attribution 重加权或训练分支，证据为两份冻结报告。

## 2. 相关工作

### 2.1 深度睡眠分期

组织 raw-signal encoder、handcrafted/segment features、TCN 与 Transformer 的文献脉络。所有模型比较与引用均为 **TO_VERIFY**，不得从项目文件推断 SOTA 排名。

### 2.2 时间序列局部解释

介绍 LIME 类扰动解释、时间分段和序列上下文保持。本文只声称对 classic LIME 查询路径进行了 sequence-aware adaptation，不声称提出新的已验证优越 attribution estimator。

### 2.3 可靠性与生理语义展示

区分预测 calibration、解释稳定性、模型随机化检查与专家可解释性评估。强调本项目无真实专家评分，因此相关的人因结论保留为未来工作。

## 3. 方法

### 3.1 Overall framework

描述双视图输入、epoch encoder、TCN–Transformer、五类输出，以及冻结模型之后的三个独立 post-hoc 分支。图 1 不得把九个临床概念画成 A1-zero 的有效训练输入。

### 3.2 Raw waveform encoder

写明 `(3000,3)` 输入、多尺度卷积核 5/11/21、融合与 160 维 epoch 表征。来源为 `src/model_training.py`。

### 3.3 Segment-level feature encoder

写明 `(30,28)` 输入、多尺度卷积核 3/5/9、全局池化与 96 维表征。只称为 segment-level engineered features，不在没有字段字典证据时逐一命名 28 维。

### 3.4 Local TCN–Global Transformer

写明 75-epoch 输入、局部/宽卷积、六个 dilated residual TCN blocks 与四头 Transformer。来源为 `_sequence_state_tcn_decoder`。

### 3.5 Sequence prediction and decoding

说明重叠窗口预测与 target-position aggregation；正式分类使用 validation-selected temperature argmax，解释使用 Pre-HMM raw probability，HMM 非 canonical。

### 3.6 Context-Preserving Classic Sequence-LIME

写明 3×30 维空间、target-epoch 扰动、raw/segment 同步更新、所有覆盖窗口重建、距离加权 Ridge 与无训练反馈。统一协议的 randomization 评估结果放结果章节，不在方法中包装成通过。

### 3.7 Reliability Evaluation

定义 equal-mass/equal-width ECE、NLL、multiclass Brier、risk-coverage、AURC 与 subject bootstrap；阈值只作为 validation descriptive operating points。

### 3.8 Independent Context Influence

将其描述为独立的 context epoch controlled replacement audit，固定 batch=1、float32、`training=False`、Pre-HMM response，并与 90 维 LIME 权重严格分离。

### 3.9 Independent Physiological Annotation

说明九个 `raw_physiology_v1` 概念只对冻结 attribution 的高敏感区域做派生标注；不重排、不重新加权、不反馈训练。注明 expert validation pending。

## 4. 实验设置

### 4.1 数据集与划分

主文可完整报告 ISRUC 固定 subject split。SleepEDF 可在 Table I 报告数据资产规模，但同构 A1-zero 性能结果为 **TO_VERIFY**；若 Table II 设计成双数据集主表，则这是投稿前必须解决的证据缺口。

### 4.2 训练与模型选择

写 seed42、学习率、训练阶段、global batch 与 validation selector。优化器为 **TO_VERIFY**。不得用 test 选择模型或解释方法。

### 4.3 分类指标

使用 Accuracy、Macro-F1、Cohen's kappa 与 per-class F1。所有数值引用 `canonical_mainline.json`。

### 4.4 XAI 与可靠性协议

明确 validation-30、三 seed 重复测量、unified T0/T1 randomization 与 13,000-epoch reliability audit 的不同样本范围，禁止混合分母。

## 5. 结果

### 5.1 ISRUC 主结果

报告 Accuracy 0.830860、Macro-F1 0.819873、kappa 0.782580，并讨论 N1 F1 0.594013 是主要困难类别。来源为 `outputs/final_evidence_synthesis/isruc/canonical_mainline.json`。

### 5.2 SleepEDF 主结果

当前只写 **TO_VERIFY**。可在附注中解释现有 SCR-Lite 结果属于不同模型，不能与 A1-zero 主表合并。

### 5.3 消融与稳定性

主文优先放同 split、同 selector 的 A1-zero 与 SCR seed42 负比较；基础 encoder 与 isolated sequence backbone 消融缺失，必须透明报告。两 seed 稳定性仅作描述性结果。

### 5.4 Prediction reliability

报告 primary equal-mass ECE、NLL、Brier、AURC 与 risk-coverage 曲线；equal-width ECE 为 sensitivity。说明这些是 validation descriptive evidence，不是 deployment threshold。

### 5.5 Canonical XAI 结果

报告 local R²、seed stability、reliable coverage 与 0/3 randomization passes；结论必须保持 `MODEL_FAITHFULNESS_NOT_ESTABLISHED`。

### 5.6 辅助 Context Influence 与生理标注

Context Influence 只报告两个 smoke 与十项 PASS，不能写定量优越性。生理标注报告自动覆盖、一致性和冲突率，同时明确真实盲评未完成。

### 5.7 代表性案例

依据 `docs/paper_xai/XAI_CASE_STUDY_SELECTION.csv` 使用固定案例，展示可靠与不可靠状态及冲突，不把案例图当作总体有效性证据。

## 6. 讨论

### 6.1 方法价值

讨论双视图和 context-preserving 查询路径如何使模型、概率可靠性、局部敏感性与生理标注的责任边界清晰。

### 6.2 负结果的意义

直接解释 0/3 randomization passes、3.3% reliable coverage、HC/MF 未晋级与无专家评价。负结果用于限制结论，而非改写为成功。

### 6.3 局限性

列出 SleepEDF 同构主结果、基础消融、外部引用和专家评价缺口；Context Influence 仅 n=2 development smoke；selected-30 存在 selection bias；不声称临床有效性或跨 batch 等价。

### 6.4 未来工作

只建议补全论文必要证据，不扩展新的 LIME 变体。任何新实验需另行授权，且不得由本写作包自动触发。

## 7. 结论

结论聚焦于双视图睡眠序列模型、context-preserving post-hoc sensitivity 与独立可靠性/上下文/生理审计的可复现组织方式。重申 model faithfulness 与 clinical validity 尚未建立。
