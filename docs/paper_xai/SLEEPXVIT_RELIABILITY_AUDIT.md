# SleepXViT-inspired Offline Reliability Audit

## 1. 审计范围与结论摘要

本报告记录 ISRUC A1-zero、seed 42 的一次完全离线可靠性审计。Primary 集合固定为 `val_probs_raw`、`val_y_true`、`val_recording_id` 与 `val_epoch_index`，共 13,000 个唯一物理 epoch、15 个 validation subjects。概率来自 75-epoch 重叠序列窗口完成 canonical target-position 聚合后的五类 Pre-HMM 原始概率；本轮没有重新 softmax、没有概率校准拟合、没有模型前向推理，也没有访问固定 test-36 XAI 样本。

核心结论是：validation 上没有明显的整体过度置信或欠置信。平均置信度为 0.833711，accuracy 为 0.830154，二者差为 +0.003558，即仅有约 0.36 个百分点的整体过度置信；各 equal-mass bin 同时存在轻微过度置信和欠置信，不能用单一方向概括全部区间。Primary ECE 为 0.010686，NLL 为 0.424695，multiclass Brier score 为 0.241856，AURC 为 0.050646。低置信度与更高错误率具有清晰的描述性对应，六个预先指定的 coverage 工作点呈严格单调的风险变化。

这些结果提供了独立于 LIME 的论文信息，因此建议进入下一阶段 Sequence Context Influence Audit 的 development 2-sample smoke；该建议不改变 canonical LIME，也不改变任何权威科学状态。

## 2. 不可突破边界与权威状态

本轮保持以下状态不变：

- `MODEL_FAITHFULNESS_NOT_ESTABLISHED`
- `PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED`
- `PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD`

本轮没有修改 A1-zero 模型、checkpoint、训练代码、canonical 90 维 attribution、样本集合或 attribution 排序；没有训练、微调、温度缩放或其他校准拟合；没有以置信度重加权 LIME；没有启用 HC/MF 历史候选；没有运行 Sequence Context Influence Audit；没有运行解释聚类、PCA 或 K-means；没有读取或运行固定 test-36 XAI 样本。

## 3. 资产、split 与 lineage

正式输入为：

- 预测资产：`outputs/predictions/result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42_split_predictions.npz`
- split 定义：`dataset_protocols/ISRUC/split.json`
- checkpoint：`outputs/checkpoints/result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42.weights.h5`
- class order：`W, N1, N2, N3, REM`

静态规划审计的 source HEAD 是 `a834bf9`；规划报告提交节点是 `ad0dc77`；二者承担不同的 provenance 角色，不构成 lineage 失败。正式审计代码提交为 `4283935be4f0449c04ce138594c275fdb2667084`。

预测资产 SHA-256 为 `53cd6b51c008e0dd47dd187654c8eddacd5d0a69037c48aeecafe5efa9594e39`，checkpoint SHA-256 为 `e86a84790fd7351c1864158eef79fbf28437a7a8a5f26662b93bcf1aef14b6ed`，split SHA-256 为 `8363546a86b81676fe6171f2e50405d2d7447cbef1f48dbf397cd53bf3f037d9`。

`train_probs_raw` 的 shape 为 `(61115, 5)`，明确属于 train split，包含 61,115 个唯一物理 epoch 和 70 个 subjects；它仅用于输入资产与 split 完整性核验，不用于选择 reliability threshold。`val_probs_raw` 的 shape 为 `(13000, 5)`，包含 13,000 个唯一物理 epoch和 15 个 subjects，是全部 primary 统计的唯一集合。

Subject ID 由 `recording_id` 与相应 split 中有序的 `split_files`/`split_subjects` 确定性连接得到，没有猜测。物化得到的 validation subjects 为 `S016, S017, S024, S040, S050, S056, S061, S062, S063, S066, S078, S079, S087, S088, S090`，与 split 定义中的 15 个 subjects 完全一致。train 与 validation 不存在 subject 重叠或物理 epoch 重叠。

独立运行时检查确认 `*_probs_raw` 全部有限、取值位于概率范围内、行和近似为 1，且 argmax 与已存储的 raw argmax 标签一致。train 与 validation 的最大行和绝对误差分别为 `2.2854e-7` 与 `2.1150e-7`。审计直接使用这些五类概率，未再次 softmax，未混用 `*_pred`、`*_y_pred_final` 或 post-HMM 输出。`val_probs_calibrated` 未参与 primary 或 sensitivity 计算，也未被描述为独立 held-out calibration evidence。

## 4. 统计协议

Confidence 定义为每个 epoch 的最大预测类别概率。Primary ECE 使用 15 个 equal-mass/equal-frequency bins；sensitivity ECE 使用 15 个 equal-width bins。两种协议均保存每个 bin 的完整边界、样本量、accuracy 与 mean confidence，未根据结果选择 ECE 版本。MCE 仅作为补充指标。

Multiclass Brier score 定义为每个 epoch 的五类概率与 one-hot 标签平方差之和，再对 epoch 取均值。Risk-coverage 按 confidence 从高到低保留 epoch，risk 定义为 `1 - selective accuracy`。本报告的离散 AURC 定义为：

`AURC = (1/n) * sum(k=1..n) [1 - accuracy(最高置信度的前 k 个 epoch)]`。

置信区间使用 subject-level bootstrap：固定 seed `20260802`，有放回抽取 15 个 subjects，并保留每个被抽 subject 的全部 epoch。共 2,000 次有效迭代，失败 0 次；95% CI 为 percentile interval。没有退化为 epoch bootstrap。

## 5. Overall reliability 结果

| 指标 | 点估计 | Subject-bootstrap 95% CI |
|---|---:|---:|
| Accuracy | 0.830154 | — |
| Macro-F1 | 0.816125 | — |
| NLL | 0.424695 | [0.382625, 0.463887] |
| Multiclass Brier score | 0.241856 | [0.215918, 0.266965] |
| ECE，15 equal-mass，primary | 0.010686 | [0.007074, 0.021069] |
| ECE，15 equal-width，sensitivity | 0.009938 | — |
| MCE，15 equal-mass | 0.022990 | — |
| MCE，15 equal-width | 0.670828 | — |
| AURC，discrete mean risk | 0.050646 | [0.039866, 0.063314] |

Equal-width MCE 的 0.670828 来自最低有数据区间中的一个极稀疏 bin：边界 `[0.266667, 0.333333)`、`n=1`、mean confidence 0.329172、accuracy 1.0。它说明 equal-width MCE 对稀疏 bin 很敏感，不代表总体 calibration 极差；因此 primary 解读以预先指定、每 bin 约 866–867 个 epoch 的 equal-mass ECE/MCE 为准，同时完整保留 equal-width 结果作为敏感性分析。

总体上不支持“明显过度置信”或“明显欠置信”。聚合差异仅为 +0.003558，primary 最大 bin gap 为 0.022990，而且 bin-level gap 正负混合。该判断仅针对当前 validation 上的未再校准 Pre-HMM 概率，不是跨数据集或部署环境的 calibration 保证。

## 6. Risk-coverage 与描述性工作点

| 目标 coverage | 实际 coverage | n | subjects | Selective accuracy | Error risk | Confidence cutoff |
|---:|---:|---:|---:|---:|---:|---:|
| 100% | 1.00 | 13,000 | 15 | 0.830154 | 0.169846 | 0.329172 |
| 90% | 0.90 | 11,700 | 15 | 0.864615 | 0.135385 | 0.557031 |
| 80% | 0.80 | 10,400 | 15 | 0.895481 | 0.104519 | 0.644231 |
| 70% | 0.70 | 9,100 | 15 | 0.925165 | 0.074835 | 0.739733 |
| 50% | 0.50 | 6,500 | 15 | 0.969231 | 0.030769 | 0.904028 |
| 30% | 0.30 | 3,900 | 15 | 0.991538 | 0.008462 | 0.979105 |

六个预先指定工作点的相邻 risk 变化方向全部一致；coverage 与 risk 的 Spearman 相关为 1.0。换言之，在这些 validation 描述性工作点上，排除较低 confidence epoch 后 risk 单调下降。正确预测的平均 confidence 为 0.867946，中位数为 0.942457；错误预测的平均 confidence 为 0.666386，中位数为 0.638229。因此，低 confidence 确实描述性地对应更高错误率。

这些 cutoff 不是经优化选择的阈值，只是 validation descriptive operating points；不得称为 deployment threshold 或 clinical threshold。

## 7. 分层结果与 N1 审计

五类 one-vs-rest primary ECE 分别为：W 0.008331、N1 0.028200、N2 0.038859、N3 0.019939、REM 0.002612。因此，按 one-vs-rest equal-mass ECE，N2 是 calibration 最差的类别；按真实阶段 accuracy 和 error burden，N1 最困难。二者必须分开表述。

| 真实阶段 | n | Accuracy | Mean confidence | NLL | Multiclass Brier |
|---|---:|---:|---:|---:|---:|
| W | 3,088 | 0.907383 | 0.895449 | 0.265935 | 0.133340 |
| N1 | 1,660 | 0.601205 | 0.696724 | 0.890863 | 0.554336 |
| N2 | 4,199 | 0.779233 | 0.773748 | 0.543458 | 0.315978 |
| N3 | 2,470 | 0.937247 | 0.922005 | 0.183109 | 0.095840 |
| REM | 1,583 | 0.887555 | 0.878218 | 0.307480 | 0.157080 |

N1 同时表现为最低 accuracy、最低 mean confidence、最高 NLL 与最高 Brier。N1 共 662 个错误，其中 84 个为 confidence ≥ 0.90 的高置信错误：占全部 N1 epoch 的 5.06%，占 N1 错误的 12.69%。所以 N1 的主问题是低 confidence 与高错误并存，但不能忽略其非零且有意义的高置信错误尾部。

Subject-level 指标显示明显个体异质性：accuracy 范围为 0.740560–0.896000，中位数 0.828125；primary ECE 范围为 0.016389–0.063031，中位数 0.031873；AURC 范围为 0.024529–0.101950，中位数 0.046023。最低 accuracy 和最高 AURC 均出现在 S017；最高 accuracy 和最低 AURC 均出现在 S050；最高 subject-level ECE 出现在 S062。论文中应报告该异质性，不能只呈现 pooled validation 点估计。

## 8. Selected-30 探索性解释一致性

该部分严格标记为 `selected-30`、`exploratory`、`n=30`，存在 selection bias，不能外推到 13,000 个 validation epoch。输入仅为冻结的 `B0_CLASSIC_90`、T0、R0，30 个独立样本，每个样本的 seeds 3101/3102/3103 是重复测量而不是 90 个独立样本。共核验 90 个 attribution 文件，并与 primary validation 的 confidence、预测阶段和真实阶段精确对齐。

样本内三对 seed 的平均 Spearman 为 0.895459，平均 cosine similarity 为 0.976508，平均 top-10 overlap 为 0.854444。Confidence 与解释诊断量的样本级 Spearman 描述性关系为：local R² `rho=-0.0198`、response variance `rho=0.4332`、ESS `rho=0.0042`、mask-seed stability `rho=0.1933`、R7 similarity `rho=0.0714`、seed-pair Spearman `rho=0.1933`、cosine `rho=0.3188`、top-10 overlap `rho=0.2654`。其中仅 response variance 的未校正双侧 p 值为 0.0168；这些 p 值未做多重比较校正，只能作为描述性探索，不构成确认性推断。

本轮没有执行 PCA、K-means、五类解释聚类或任何其他 explanation clustering，也没有利用 selected-30 修改或重排 90 维 attribution。

## 9. 图表与产物

正式输出根目录为 `outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/`。其中包含 37 个非空文件：6 个 JSON、12 个 CSV、README，以及 6 张图的 PDF、PNG、SVG 三种格式。六张图为：

1. overall reliability diagram；
2. five-class one-vs-rest reliability diagram；
3. risk-coverage curve；
4. confidence distribution：correct vs incorrect；
5. true-stage confidence 与 error rate；
6. validation coverage operating points。

所有图采用白色背景，并标注 ISRUC validation、13,000 unique epochs、15 subjects、Pre-HMM、uncalibrated 与 A1-zero seed42 checkpoint lineage。完整 bin、risk curve、subject-level、selected-30 与 provenance 数据均保存在同一隔离目录，未覆盖历史输出。

## 10. 可写入论文的结论与限制

可写入论文主文或 reliability 模块的内容包括：在预先固定的 validation split 上，未再校准的 Pre-HMM 概率具有 ECE、NLL、Brier 与 AURC 所描述的内部可靠性；低 confidence 对应更高错误率；固定 coverage 工作点呈单调风险下降；类别和 subject 间存在异质性。写作必须明确这是 validation descriptive evaluation，而不是用结果选择出的阈值。

适合放入补充材料或明确标记为探索性的内容包括：equal-width sensitivity ECE/MCE、所有 validation coverage cutoff、subject 极值、selected-30 的 seed 一致性和 confidence–diagnostic 相关。Selected-30 必须保留 n=30、重复测量、selection bias 与不可外推的说明。

当前证据不能支持：独立 held-out calibration、外部校准、临床阈值、部署效用、临床安全性、置信度可提高或重加权 LIME、解释与模型机制一致、canonical attribution 已建立模型忠实度、生理语义可视化已验证，或 feedback/explanation-guided training 可以启用。

## 11. 是否进入 Context Audit pilot

唯一建议是：进入下一阶段，但仅启动既定顺序的第一步 development 2-sample smoke。依据是输入 lineage 完整、13,000 个 unique validation epoch 与 15 个 subject 均通过验证、六个固定工作点的 risk-coverage 趋势合理、Reliability 模块提供了独立论文信息，而且不需要修改 canonical LIME 或任何权威状态。

未来顺序必须固定为：development 2-sample smoke → frozen development 16 → protocol freeze → frozen validation 30。当前报告不授权直接运行 frozen development 16 或 validation 30，更不授权访问 test-36。

## 12. 完成性与可复现性检查

审计实现通过 9 个 synthetic/unit checks，覆盖概率语义与禁止二次 softmax、logit 拒绝、两种 ECE binning、离散 AURC、确定性 subject 映射、重复物理 epoch 拒绝、multiclass Brier 定义及 selected-30 的 cosine/top-10 计算。Subject bootstrap 为 2,000/2,000 有效迭代。全部 JSON 可解析，全部 CSV 与图文件非空。正式运行前后 checkpoint、预测资产和 split 的 SHA-256 保持一致；未生成模型推理日志，也没有 test-36 访问记录。

以上结果不改变三项权威科学状态，并将 Reliability Evaluation 与 canonical 90 维 Sequence-LIME 保持为完全独立的论文模块。
