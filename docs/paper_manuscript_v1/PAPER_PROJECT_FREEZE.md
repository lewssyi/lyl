# 论文项目冻结清单 V1

## 冻结范围

本写作包以远端分支 `codex/a1-zero-mainline`、提交 `57cc1e8e454867232bd89e147cdbbe8ac94ba070` 为内容冻结点。本文唯一的正式分类主线是 ISRUC 上的 A1-zero、训练 seed 42、划分 seed 42；唯一 canonical XAI 名称是 **Context-Preserving Classic Sequence-LIME**。本目录只索引既有代码、日志和结果，不产生新的训练、推理、参数搜索或数值审计。

## 正式模型结构

正式模型接收两个同步视图：每个 epoch 的三通道 raw waveform，张量形状为 `(3000, 3)`；以及相同 epoch 的 30 段、每段 28 维工程特征，张量形状为 `(30, 28)`。序列输入长度固定为 75 个 epoch。Raw 分支使用 `v3_multiscale` 的多尺度一维卷积编码，segment 分支使用核大小 3、5、9 的多尺度一维卷积编码；两个分支在 epoch 级拼接并映射到 160 维。序列建模采用 `v3_tcn`：局部/宽卷积入口、膨胀率为 1、2、4、8、1、2 的残差 TCN，以及启用的四头 Transformer 块，最后逐 epoch 输出 W、N1、N2、N3、REM 五类概率。

A1-zero 的配置名是 `A1_concept_film_zero`。代码中保留的 Concept-FiLM 结构接收零概念输入，当前资产不能证明九个临床概念对 A1-zero 训练产生了有效贡献。因此，论文主架构不得把九个概念画成有效训练输入；九个概念只出现在独立的 post-hoc Physiological Annotation 层。模型实现来源为 `src/model_training.py` 中的 `create_sequence_sleep_model`、`_time_distributed_raw_multiscale_features`、`_time_distributed_segment_multiscale_features` 与 `_sequence_state_tcn_decoder`。

## 数据集与划分

ISRUC 协议资产包含 100 个 subjects、100 个 recording 和 87,187 个 30 秒 epoch；固定 subject-level 划分为 70/15/15 subjects。A1-zero 预测资产中 train/validation/test 分别为 61,115、13,000、13,072 个物理 epoch。类别顺序固定为 W、N1、N2、N3、REM。数据经过 `clinical_canonical` 预处理与 recording-level robust z-score normalization，不拟合跨 train/validation/test 的总体 scaler。依据为 `dataset_protocols/ISRUC/protocol_report.json`、`dataset_protocols/ISRUC/split.json` 与 `outputs/final_evidence_synthesis/isruc/canonical_mainline.json`。

SleepEDF 的当前数据协议资产记录 78 个 subjects、153 个 recordings、187,311 个 epoch；但现有可定位的正式数值并非与 ISRUC A1-zero 完全同构的 checkpoint/protocol。`result_20260710_070633_edf_A1_scr_lite_seed42` 启用了 stage concept router，不能冒充 A1-zero；`result_20260627_180622_edf_c0_no_concept` 日志虽含 `edf` 名称，却实际列出 ISRUC 的 Sxxx 划分。故 SleepEDF 的同构主结果状态为 **TO_VERIFY**，不得进入跨数据集主性能表，直至既有资产中找到或另行授权生成严格同构证据。

## Checkpoint、训练配置与解码口径

ISRUC 正式 checkpoint 为 `outputs/checkpoints/result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42.weights.h5`，SHA-256 为 `e86a84790fd7351c1864158eef79fbf28437a7a8a5f26662b93bcf1aef14b6ed`。正式 split prediction 为 `outputs/predictions/result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42_split_predictions.npz`，SHA-256 为 `53cd6b51c008e0dd47dd187654c8eddacd5d0a69037c48aeecafe5efa9594e39`。初始学习率为 0.001；训练包括 5 个 micro warm-start epoch、4 个 macro stabilization epoch，以及最多到第 31 epoch 的 joint fine-tuning。全局训练 batch 为 32（2 replicas，每 replica 16）。优化器在现有冻结总表中记为 `UNKNOWN_FROM_LOG`，论文中保持 **TO_VERIFY**，不得猜测。

正式分类结果使用 validation-selected temperature argmax；HMM 只属于验证阶段候选解码器，不是 canonical 最终解码。所有解释模块的响应均来自重叠窗口聚合后的五类 **Pre-HMM raw neural probability**，不得与 Post-HMM 标签或最终平滑标签混用。

## 正式 ISRUC 指标

在 13,072 个 test epoch 上，A1-zero seed42 的 Accuracy 为 0.830860，Macro-F1 为 0.819873，Cohen's kappa 为 0.782580。五类 F1 分别为 W 0.918388、N1 0.594013、N2 0.805556、N3 0.882239、REM 0.899167。唯一数值来源为 `outputs/final_evidence_synthesis/isruc/canonical_mainline.json` 与 `outputs/final_evidence_synthesis/isruc/paired_isruc_result_table.csv`。

## Canonical XAI 与独立模块

Context-Preserving Classic Sequence-LIME 将目标 epoch 的三通道波形划分为 30 个时间段，形成 `3 × 30 = 90` 维二值扰动空间。扰动目标 epoch 时，同步更新 raw waveform 与相应 segment features，并重建所有覆盖该物理 epoch 的 75-epoch 重叠窗口；目标位置的 Pre-HMM 响应经窗口聚合后，由距离加权 local Ridge 拟合。它只能称为 exploratory Local Perturbation Sensitivity，当前权威状态仍为 `MODEL_FAITHFULNESS_NOT_ESTABLISHED`。

Reliability Evaluation 是独立的 validation 描述性评估，使用未重新校准的 `val_probs_raw`，不改变 LIME。Independent Context Influence 是 LIME 之外的诊断工具，正式解释推理固定为 batch=1、float32、`training=False`、75-epoch context、Pre-HMM response；它不声称 batch-size invariance，两个 development smoke 只证明程序和关键科学操作可用。Independent Physiological Annotation 从 `raw_physiology_v1` 生成九类派生证据，不重新加权或重排 LIME，不反馈训练；其当前状态为 `PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED`，专家评价人数为 0。

## 允许的论文声明

- Raw waveform 与 segment-level feature 的双视图编码。
- Local TCN 与 global Transformer 的 75-epoch 序列建模，以及五分类睡眠阶段预测。
- Context-preserving overlapping-window inference 与 90 维 target-epoch channel–segment perturbation。
- Raw/segment cross-view consistent update、weighted local Ridge 和 Pre-HMM response。
- 冻结 validation 上的 ECE、NLL、Brier、risk-coverage 与 AURC 描述性可靠性结果。
- Context Influence 的 batch=1 工程实现通过两个 development smoke；其角色是可选诊断或补充分析。
- 独立生理标注层已完成自动覆盖/一致性审计，且 attribution bitwise unchanged。
- 所有 XAI 模块均为 post-hoc，且不参与 training feedback。

## 禁止的论文声明

- 不得声称本方法已证明优于传统 LIME、已建立 model faithfulness 或提供 causal attribution。
- 不得声称生理标注提高 attribution 真实性、分类性能或临床有效性。
- 不得声称 enhanced display 提高可解释性、可靠性判断或专家偏好；真实盲评尚未完成。
- 不得声称 batch=1 与 batch=16 数值等价或具有 batch-size invariance。
- 不得把 HC-SeqLIME、MF-SeqLIME、未验证的生理重加权、concept modulation 或 batch-16 路线写成当前主方法。
- 不得把 SleepEDF 的不同模型/不同划分结果与 ISRUC A1-zero 混为同一 checkpoint 主表。
