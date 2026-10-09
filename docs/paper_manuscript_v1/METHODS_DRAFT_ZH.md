# 方法章节中文草稿 V1

## Overall framework

本文采用双视图睡眠序列建模框架。对每个 30 秒物理 epoch，模型同时接收三通道原始波形与由相同波形构造的 segment-level feature tensor。前者保留局部形态与跨通道动态，后者提供按时间段组织的工程描述。两个视图分别完成 epoch 内编码后在 epoch 级融合，再由局部 TCN 与全局 Transformer 建模 75 个连续 epoch 的序列关系，并为序列中的每个位置输出 W、N1、N2、N3 和 REM 五类预测。所有解释分析均发生在冻结分类模型之后，不向分类损失、参数更新或正常推理返回任何信号。

正式 A1-zero 配置保留 Concept-FiLM 的零输入控制结构，但九个临床概念没有在该 checkpoint 中形成已验证的有效训练贡献。因此，论文主架构只呈现 raw waveform、segment features、TCN–Transformer 与分类头。九个生理概念被安排在完全独立的 post-hoc annotation 层，而不是训练输入分支。

## Raw waveform encoder

Raw 分支的单个 epoch 输入形状为 `(3000, 3)`。模型首先通过核大小为 9、步长为 2 的一维卷积和池化形成下采样 stem，随后并行使用核大小为 5、11 和 21 的多尺度卷积分支。各分支的输出拼接后，再经 160 通道卷积、全局平均池化、全连接映射和层归一化形成 160 维 epoch-level raw representation。该实现对 75 个序列位置使用共享参数的 time-distributed 计算，从而使每个 epoch 的波形编码规则一致。

## Segment encoder

Segment 分支的单个 epoch 输入形状为 `(30, 28)`，其中 30 表示与解释空间一致的时间分段，28 表示已有工程特征维度。输入先经层归一化，再并行通过核大小为 3、5 和 9 的一维卷积分支；拼接后的多尺度表示经全局平均池化、96 维全连接映射和层归一化得到 segment-level epoch representation。现有冻结证据没有提供可直接用于论文的完整 28 维字段字典，因此本稿不为每个维度补写未经核实的生理名称。

## TCN–Transformer sequence model

Raw 与 segment 两个 epoch representation 拼接后，经 160 维全连接层与层归一化形成统一的 epoch token。序列骨干首先并联核大小为 3 和 7 的局部与宽感受野卷积，再将其与输入特征融合。随后六个残差 TCN block 按 1、2、4、8、1、2 的膨胀率建模局部到中程的阶段依赖。每个 block 使用两层核大小为 3 的卷积、层归一化、非线性激活、dropout 与残差连接。启用的四头 Transformer block 在 TCN 之后建模 75-epoch 范围内的全局依赖，最终经逐位置全连接层输出五类 logits 与概率。

## Sequence prediction

推理以长度 75、prediction stride 5 的重叠窗口覆盖整条 recording。一个物理 epoch 可能出现在多个窗口和不同 target position 中；模型对这些位置输出的五类神经概率进行既定聚合，得到 epoch-level raw probability。正式分类结果采用 validation-selected temperature argmax。HMM 只在历史 validation 候选集中存在，不是 canonical 最终解码。所有解释响应均使用聚合后的 Pre-HMM raw neural probability，不能与 Post-HMM 或其他平滑标签混用。

## Context-Preserving Classic Sequence-LIME

Context-Preserving Classic Sequence-LIME 是本文唯一的 canonical model-derived attribution。对目标 epoch，将三条通道各划分为 30 个时间段，得到 `3 × 30 = 90` 维二值可解释空间。每个 mask 只控制目标物理 epoch 中相应 channel–segment 的保留或替换，周围 epoch 保持原状。为保证两个模型输入视图一致，波形被扰动后，相应 segment features 必须同步重算或更新；随后重建所有覆盖该目标 epoch 的 75-epoch 重叠窗口，并聚合各窗口 target position 的 Pre-HMM 响应。

局部代理模型以二值 mask 为自变量、冻结分类器的聚合响应为因变量，并按照 mask 与原始全保留样本之间的距离赋权，拟合 local Ridge。得到的 90 个系数只解释预先定义替换规则邻域中的 Local Perturbation Sensitivity，不代表全局因果贡献。统一 randomization consistency audit 从相同 Pre-HMM 概率构造固定目标的 Y2 log-probability margin 作为检查端点，但该审计不改变 canonical attribution 的定义或排序。当前随机化结果未建立 model faithfulness，因此所有论文表述保留 `MODEL_FAITHFULNESS_NOT_ESTABLISHED`。

## Reliability Evaluation

可靠性评估与 LIME 完全分离。Primary 数据固定为 ISRUC validation 的 13,000 个唯一物理 epoch、15 个 subjects，以及未经重新校准的 `val_probs_raw`。置信度定义为五类 Pre-HMM 概率的最大值。Primary ECE 使用 15 个 equal-mass bins，sensitivity ECE 使用 15 个 equal-width bins；两种方案均保存 bin 边界、样本量、accuracy 与 mean confidence。NLL 直接评价真实类别概率，multiclass Brier score 对五类 one-hot 概率误差平方和按 epoch 求均值。

Risk-coverage 将 epoch 按 confidence 从高到低排序，coverage 表示保留比例，risk 定义为一减 selective accuracy。AURC 使用从 1 到 n 的全部 top-k coverage 风险的离散均值。置信区间采用 subject-level bootstrap：有放回抽取 15 个 subjects，并保留每个被抽 subject 的全部 epoch。100%、90%、80%、70%、50% 与 30% coverage 只是预先固定的 validation descriptive operating points，不是 deployment 或 clinical thresholds。可靠性指标不会重新加权、筛选或修改 LIME attribution。

## Independent Context Influence

Context Influence 是与 90 维 LIME 权重平行的独立诊断。对固定 target epoch，审计在同一 recording 内选择相邻 context offset，并用预定义的 normalized-zero waveform 或确定性真实 epoch baseline 替换该 physical context epoch。替换同时作用于 raw waveform 和由其重算的 segment features；仅共同覆盖 target 与 context 的窗口被修改，其余覆盖 target 的窗口保持不变。响应类别由未扰动聚合概率的 argmax 固定，signed influence 定义为原始 target-class Pre-HMM probability 减去 counterfactual probability。

正式论文实现固定 inference batch size 为 1、float32、`training=False`、75-epoch context 与 stride-5 重叠窗口，不运行 HMM、梯度或权重更新，也不声称 batch-size invariance。现有证据仅包含两个冻结 development smoke 样本，用于证明程序完整运行、baseline 恢复、no-op、单 epoch 隔离、cross-view update、窗口覆盖、target-position mapping、有限输出、重复稳定性和文件可读性。由于没有更大规模定量结果，Context Influence 只能作为 optional diagnostic 或 supplementary analysis。

## Independent Physiological Annotation

Physiological Annotation 从冻结的 `raw_physiology_v1` 测量中派生九个概念：EEG Delta、EEG Theta、EEG Alpha、EEG Sigma、EEG Beta、EOG Slow Eye、EOG Rapid Eye、EMG Low Tone 与 EMG Muscle Activation。标注层把这些独立信号证据与冻结 attribution 的高敏感区域并列展示，并显式记录 unsupported evidence 与 opposing-concept conflict。真实阶段不用于生成标注，预测阶段只作为显示上下文。

该层不改变 90 维 sensitivity、排序、颜色尺度、分类概率或模型参数。现有自动审计确认基础与增强显示中的 attribution bitwise identical，但真实盲评人数为 0，所以它只能被描述为 implementation-valid、expert-validation-pending 的派生标注和评价工具，不能被描述为更忠实、更易解释或经过临床验证的 attribution 方法。
