# SleepXViT 借鉴决策：Canonical Sequence-LIME 的可靠性评估与独立上下文审计规划

> 规划日期：2026-08-02
>
> 远端项目：`/home/b502/wssyi/LIME-for-Time-Series`
>
> 审计分支 / HEAD：`codex/a1-zero-mainline` / `a834bf9`
>
> 本文性质：静态资产与代码路径检查后的论文增强规划；本轮未实现方法、未启动推理、未运行固定 test-36。

## 1. 决策摘要

**唯一推荐方案：方案 C——保留当前唯一 canonical attribution，同时增加“预测可靠性评估”和“独立序列上下文影响审计”。**

推荐的论文结构是三条互不回流的证据线：

1. **Context-Preserving Classic Sequence-LIME**：仍是唯一的 90 维 model-derived attribution；
2. **Reliability-aware Explanation Evaluation**：用已经保存的 Pre-HMM 概率评估预测可靠性，并将可靠性与现有解释诊断并列报告；
3. **Sequence Context Influence Audit**：通过邻近 epoch 的独立反事实输入审计 Pre-HMM 响应，不生成、融合或重排任何 LIME 权重。

方案 C 的价值不在于宣称一种新的归因算法，而在于把“单 epoch 内的局部扰动敏感性”“预测是否可靠”和“相邻 epoch 是否改变目标响应”拆成三个可审计对象。这比只增加置信度图更符合序列睡眠分期的论文问题，同时不触碰现有 canonical LIME。

下列状态必须原样保留：

- `MODEL_FAITHFULNESS_NOT_ESTABLISHED`
- `PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED`
- `PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD`

方案 C 需要未来新增 **dev/validation 范围内的上下文反事实推理**；本轮没有执行。若该审计产生阴性或基线依赖结果，应如实作为独立阴性结果报告，不能回退到 HC-SeqLIME、MF-SeqLIME，也不能把上下文响应并入 90 维 attribution。

## 2. 对 SleepXViT 的可借鉴边界

参考论文：H. Lee, Y. R. Choi, H. K. Lee 等，*Explainable vision transformer for automatic visual sleep staging on multimodal PSG signals*，npj Digital Medicine 8, 55 (2025)，DOI: [10.1038/s41746-024-01378-0](https://doi.org/10.1038/s41746-024-01378-0)。

论文提供了两条与当前项目可分离借鉴的叙事：

- 用 reliability diagram 与置信度分布检查预测概率和实际正确率的对应关系，并把低置信度视为需要谨慎复核的信号；
- 将相邻 epoch 对目标 epoch 的影响作为独立的序列上下文解释，而不是当作单 epoch 热图的一部分。

本项目只借鉴这两种**问题定义与证据组织方式**，不借鉴以下做法：

- 不采用其 ViT 架构、10-epoch 设置或 stride-1 推理；当前模型固定为 75 epochs、stride 5；
- 不采用 LRP，也不建立 LRP/LIME Hybrid Attribution；
- 不采用 softmax 加权的 relevance 融合；
- 不把置信度、上下文影响或生理概念乘入 LIME；
- 不复现论文基于 test set 的“well-calibrated”强结论；本项目当前只能形成 dev/validation 范围内的结果；
- 不提出临床分流、人机协作效率或医生复核收益结论。

因此，“借鉴”是模块化研究问题的借鉴，不是方法移植。

## 3. 第一部分：现有资产盘点

### 3.1 Canonical 状态与 manifest

| 资产 | 路径 | 静态结论 |
|---|---|---|
| 当前论文材料 manifest | `docs/paper_xai/CURRENT_RESULTS_MANIFEST.json` | 记录现有论文材料的文件大小与 SHA-256；不应由本规划反向重写 |
| authoritative XAI 状态 | `outputs/xai_authoritative_status/isruc_seed42/CURRENT_XAI_STATUS.json` | canonical 方法为 `CONTEXT_PRESERVING_CLASSIC_SEQUENCE_LIME`；checkpoint 为只读；三项强制状态均已固定 |
| canonical 运行/选择来源 | `outputs/hierarchical_context_xai_feasibility/isruc_seed42/run_manifest.json` 与 `frozen_validation_selection.json` | 冻结 validation 30 样本；Classic baseline 为 `B0_CLASSIC_90`；固定 test 未执行 |
| 数据划分 | `dataset_protocols/ISRUC/split.json` | train 70 subjects、validation 15 subjects、test 15 subjects；可从 recording 轴确定性映射 subject ID |

关键来源哈希：

- split predictions：`53cd6b51c008e0dd47dd187654c8eddacd5d0a69037c48aeecafe5efa9594e39`
- authoritative status：`55c74b36113f438f0a94b8e9d2c679f6cdfd2b85825b8c63ecd80f43f60092bc`
- reliability sample table：`eeee6650c39332a4be8dc20f42e59900b58d364639e6fc477893fe6ef7b7b312`
- frozen validation selection：`a4d64fbab98f0f9f3e3421ede525ea8975426c38797ca7a7b942892304f0db99`

### 3.2 Pre-HMM 概率、标签、ID 与质量字段

主资产为：

`outputs/predictions/result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42_split_predictions.npz`

静态读取结果如下：

| split | 样本数 | Pre-HMM 概率 | 标签 | recording / epoch 轴 | 质量字段 |
|---|---:|---|---|---|---|
| train（可作为 development 资产池） | 61,115 | `train_probs_raw`，形状 `(61115, 5)` | `train_y_true`，五类完整 | `train_recording_id`, `train_epoch_index` | `train_quality` |
| validation | 13,000 | `val_probs_raw`，形状 `(13000, 5)` | `val_y_true`，五类完整 | `val_recording_id`, `val_epoch_index` | `val_quality` |

所有概率均为有限值，逐行概率和与 1 的最大绝对误差为 `2.3841858e-07`。质量分数范围为 `[0.145, 1.0]`。`run_main.py` 中的保存路径表明，`*_probs_raw` 来自冻结模型的 sequence prediction aggregation，温度缩放与 HMM/postprocessor 发生在其后；因此它们可作为 Pre-HMM 神经概率。`*_pred` / `*_y_pred_final` 是后处理结果，不能与 Pre-HMM confidence 混用。

subject ID 没有作为单独数组嵌入 NPZ，但并不缺失：`*_recording_id` 是 split 内的零基 recording 轴，可按 `dataset_protocols/ISRUC/split.json` 中对应 split 的有序 `split_files` / `split_subjects` 确定性连接为 `Sxxx`。论文分析表必须物化这一连接并保留 recording 与 subject 两列。

质量分数由 `src/data_processing.py::compute_epoch_quality_scores` 从原始 epoch 的 non-finite、flat、step 和 noise 指标计算。它是信号质量置信度，不是预测置信度，也不能作为 LIME 权重。

### 3.3 离线可靠性指标的可计算性

现有 dev/validation 资产**足以在不运行模型的情况下**计算：

- ECE：使用 Pre-HMM argmax confidence 与对应的 Pre-HMM argmax correctness；
- multiclass Brier score：使用完整五类概率与 one-hot 标签；
- NLL：使用真实类别的 Pre-HMM 概率；
- risk–coverage：按 `1 - max(val_probs_raw)` 从低风险到高风险排序，以 Pre-HMM argmax 的 0/1 error 为 risk；
- AURC：对上述 risk–coverage 曲线积分。

建议 primary 结果使用未经 validation 温度拟合的 `val_probs_raw`。现有 `val_probs_calibrated` 可以作为“已存在但可能在同一 validation 上调参”的敏感性分析，不能当作独立 held-out 校准证据。

重要限制：固定 test-36 本轮及推荐最小实验均不运行，因此不能声称独立 test calibration、跨数据集 calibration、临床可用阈值或低置信度分流收益。当前可支持的措辞仅是“在冻结 validation 上评估了预测可靠性”。

### 3.4 完整 90 维 attribution 与解释可靠性字段

Canonical validation 路径模式为：

`outputs/hierarchical_context_xai_feasibility/isruc_seed42/runs/validation_classic_sample*_R0_seed*_zero/classic_attribution.csv`

静态完整性检查得到：

- 90 个 R0 文件；
- 30 个冻结 validation 样本；
- seeds `3101/3102/3103`；
- 每个文件包含 T0 与 T1 两行；
- `feature_count=90`；
- 每个 `importance` 数组恰有 90 个有限值；
- 顺序为 3 个 channel × 每 channel 30 个 temporal segments，可从现有论文 figure data 的 EEG、EOG、EMG 顺序核对。

对应解释诊断位于：

`outputs/hierarchical_context_xai_feasibility/isruc_seed42/reliability_sample_level.csv`

其中 B0 的 30 个样本具有 confidence、correct、transition boundary、local surrogate R²、response variance、effective sample size、mask-seed stability、R7 similarity、reliability state、reliable 与 coverage weight。因 T0/T1 对当前 frozen canonical 向量重复，样本级统计必须预先指定一个 protocol（建议 T0）或先在样本内合并，不能把 60 行当作 60 个独立样本。

因此，现有资产足够对**冻结 30 样本**开展跨种子 90 维一致性（Spearman/cosine、top-k overlap）以及置信度与解释诊断的并列分析；不够支持全量 13,000 validation epochs 的解释一致性。

### 3.5 明确缺失项

只报告以下缺失，不回退至 HC/MF 历史候选：

1. 没有全量 train/dev 或 13,000 validation epochs 的 canonical 90 维向量；现有完整向量仅覆盖冻结 validation 30 样本；
2. 没有独立于 model-development validation 的 held-out calibration 评估；在不使用 test 的边界下只能做 validation-level 结论；
3. split prediction NPZ 未直接存 subject string，需要用 frozen split manifest 确定性连接；
4. 没有邻近 epoch 替换后的 Pre-HMM 反事实概率，也没有 per-window context response；Context Influence 未来需要新推理；
5. 没有证据允许把预测可靠性、信号质量或上下文影响解释为 canonical LIME 的权重可靠度。

`HC_SEQLIME_FAIL` 与 `SUPERSEDED_HISTORICAL_CANDIDATE` 资产不用于补齐这些缺失。

## 4. 第二部分：Sequence Context Influence 可行性

### 4.1 已定位的 canonical 代码路径

| 功能 | 路径 / 位置 | 现状 |
|---|---|---|
| 模型序列长度锁定 | `run_main.py` 顶部 protocol env 与 sequence 常量 | `LFTS_SEQ_LEN=75` |
| 记录内窗口构造 | `tools/sequence_lime_paper_evaluation.py::build_sequence_windows` | 75 epochs、stride 5、包含末窗、禁止跨 recording |
| 短记录边界 | `tools/sequence_lime_paper_evaluation.py::window_epoch_indices` | 仅短记录使用最后有效 epoch 填充；有效长度单独保存 |
| 全序列目标聚合 | `tools/sequence_lime_paper_evaluation.py::predict_sequence_probs` | 各重叠窗口 target-position 的 Pre-HMM 五类概率做等权平均；默认 `batch_windows=16` |
| 目标覆盖窗定位 | `target_windows` / `target_formal_window_batches` | 找出覆盖指定 global epoch 的正式批次与相对位置 |
| 目标 epoch 反事实查询 | `sequence_target_vector` / `infer_precomputed_epochs_for_target` | 已有复制窗口、替换输入、抽取 target-position、跨覆盖窗等权平均的安全结构 |
| `run_main.py` wrapper | `SequenceContextModelWrapperForLIME` | canonical LIME 只替换目标 epoch，周围 75-epoch context 保持不变 |

可行性结论：无需修改模型即可复用“覆盖窗口—目标位置—等权聚合”的基础设施，但必须新建**隔离的 context audit 工具**；不能改变现有 wrapper，使其成为 LIME 的一部分。

### 4.2 审计 estimand

对目标 epoch `t`，先用未干预的冻结模型得到所有覆盖窗等权聚合后的 Pre-HMM 概率向量 `p_t`，并令 `c* = argmax(p_t)`。对合法邻近 offset `d != 0`，仅替换物理 epoch `t+d`，保持目标 epoch 不变，得到 `p_t^(d,b)`。

Primary response：

`Influence(d,b) = p_t[c*] - p_t^(d,b)[c*]`

同时保存：

- 完整五类概率差；
- Jensen–Shannon divergence；
- Pre-HMM argmax 是否翻转；
- 原始与反事实概率；
- target 与 context epoch 的 recording/epoch ID；
- 覆盖 target 的窗口数，以及同时覆盖 target/context 的窗口数。

`c*` 必须来自原始 Pre-HMM argmax，不能使用真实标签、HMM 最终标签、置信度分层或生理概念来选 target class。

### 4.3 Leave-one-epoch-out / 受控替换方案

建议采用两层、预注册的替换：

1. **Primary：canonical normalized zero replacement**。只在确认当前标准化空间中 zero 是既有冻结中性基线后使用。替换 raw epoch，并通过 canonical feature extractor 重新生成匹配的 30-segment features；不能只改 raw 或只改 segment tensor。
2. **Sensitivity：recording-neutral medoid replacement**。在同一 recording 内，以不使用标签、预测、置信度或生理概念的冻结规则选择最接近 recording feature median 的真实 epoch；排除 target 与被审计邻域，按最小 epoch index 决定并列。复制该 donor 的 raw 与 segment pair。

不把 local temporal mean 作为 primary，因为它是合成波形并可能过度平滑；如使用，只能作为附录敏感性分析。

建议最小 offset 集为 `{-3,-2,-1,+1,+2,+3}`，即目标前后 90 秒；开发阶段可另用 `{-10,-5,+5,+10}` 检查影响是否局限于近邻。不得根据 validation 结果重新选择 offset。

### 4.4 边界与多窗口聚合

- 不跨 recording，不环绕；缺失 offset 记为 structural missing，不用另一侧补齐；
- 短记录 padding 位置不是物理 epoch，不允许被干预；
- 主报告使用“完整 canonical aggregation effect”：在所有覆盖 target 的窗口中，只改变同时包含 `t+d` 的窗口，其余窗口保持原样，然后按 canonical 等权规则聚合；
- 附加报告使用“shared-window paired effect”：仅在同时覆盖 target 与 `t+d` 的窗口上做原始/替换配对，以避免未受影响窗口稀释效应；
- 两种 estimand 必须分列，不能选择效果更大的一个；
- 原始无干预复算应与保存的 `*_probs_raw` 在预注册容差内一致，建议 `max_abs_diff < 1e-6` 后才允许解释反事实结果。

### 4.5 随机种子、批处理与输出隔离

- zero 与 deterministic medoid 是无随机 primary；
- 若附加随机 neutral donor 敏感性分析，使用与 LIME 分离的 seeds `4201/4202/4203`，不得复用其结果来挑选 offset 或样本；
- 使用现有 `batch_windows=16` 和正式覆盖窗批处理；不得为凑 batch 引入伪窗口；
- 先通过现有 memory guard，再运行 dev；本规划不执行该步骤；
- 新输出根建议固定为 `outputs/sequence_context_influence_audit/isruc_seed42/v1/`，并写入 checkpoint、source prediction、split、代码与配置 SHA-256；
- 任何 audit 结果只能形成独立表/图，不写入 `classic_attribution.csv`、不改变 90 维排序、不覆盖历史输出。

## 5. 第三部分：三种论文写法比较

| 方案 | 创新性 | 实验成本 | 审稿风险 | 新推理 | 可支持结论 | 不能支持结论 |
|---|---|---|---|---|---|---|
| A. 仅当前 LIME | 低至中；主要是序列窗口保持与透明失败审计 | 最低；零新增实验 | 中；审稿人可能认为只是工程适配，且现有 R7 阴性结果限制解释强度 | 否 | 复现的 context-preserving query path；90 维局部扰动敏感性；现有工程与阴性审计 | model-faithful、因果、临床有效、优于其他归因 |
| B. LIME + reliability evaluation | 中；把预测可靠性与解释诊断并列但不融合 | 低；概率指标全部离线，30 个 explanation 可静态分析 | 中；ECE binning、同一 validation 上 calibration、30 样本选择偏倚需充分披露 | 否 | validation 上的 Pre-HMM ECE/Brier/NLL/AURC；置信度与现有解释诊断的描述性关系 | 独立 test calibration、临床阈值、置信度能证明 explanation faithful、置信度加权 LIME |
| C. LIME + reliability + independent context audit | 中至高；形成 within-epoch / reliability / inter-epoch 三轴证据 | 中至高；reliability 离线，context 需 dev/validation 新推理 | 中至高；替换 OOD、窗口稀释、将 sensitivity 误写成 causality 是主要风险 | **是，仅 context audit** | validation 上的预测可靠性；相邻 epoch 替换对 target Pre-HMM 响应的独立敏感性、方向与距离模式 | context 是 LIME 权重、真实因果贡献、模型性能提升、HMM 决策解释、临床工作流收益 |

## 6. 推荐方案 C 的最小实验清单

以下是未来执行清单，不是本轮已运行结果。

### A. Reliability-aware Explanation Evaluation（零新推理）

1. 冻结并记录 split prediction、split manifest、authoritative status 和 canonical 90D 文件哈希。
2. 在 13,000 个 validation epochs 的 `val_probs_raw` 上计算：
   - primary ECE：15 个 equal-mass bins；固定宽度 15 bins 作为敏感性分析；
   - multiclass Brier、NLL；
   - Pre-HMM confidence 排序的 risk–coverage 与 AURC；
   - subject-cluster bootstrap 95% CI，建议 2,000 次；
   - 按 true stage、transition boundary 与 signal-quality 分层的描述性结果，但不据此调参。
3. 在冻结 30 个 canonical validation 样本上，以 T0 为唯一统计行：
   - 三 seed 的 90D Spearman/cosine 与 top-10 overlap；
   - confidence 与 local R²、response variance、ESS、mask-seed stability、R7 similarity 的相关/分层分析；
   - 显式标注 selected-30、exploratory、非全量 validation。
4. 不用 reliability 改变 LIME，不生成 reliability-weighted attribution。

### B. Sequence Context Influence Audit（需要新 dev/validation 推理）

1. 先在冻结 development 16 样本上做原始无干预 parity 与 2 样本 smoke；不触碰 test。
2. 冻结 offsets `±1/±2/±3`、primary zero、sensitivity recording-neutral medoid、Pre-HMM response 与 aggregation 规则。
3. 对 development 16 样本完成 protocol check 后，一次性运行冻结 validation 30 样本；不得按 validation 结果改 offset、baseline 或 target class。
4. 输出 sample × offset × baseline 的 signed probability drop、JSD、argmax flip、shared-window count 和完整聚合 effect，并给出 subject-cluster bootstrap CI。
5. 报告 baseline agreement 与边界 coverage；若 zero/medoid 结论不一致，结论必须降格为 replacement-dependent。
6. 运行后只比较 canonical 90D 文件哈希，确认 attribution identity；不重新计算或重排 90D LIME。

### C. 硬停止条件

- fixed test-36 始终不运行；
- 不训练、不微调、不改 checkpoint/default classifier；
- 不修改 canonical attribution、训练代码或正常 inference；
- 不用 test、置信度、标签、生理概念或 context effect 做方法/样本/offset 选择；
- 不覆盖任何历史输出；
- parity 或 source hash 不一致时停止 context audit；
- 无论结果正负，三项 authoritative 状态均保持不变。

## 7. 可用于论文的最终定位

若最小实验完成，最稳妥的论文表述是：

> We retained Context-Preserving Classic Sequence-LIME as the sole within-epoch attribution and evaluated it alongside two non-fusing audits: validation-level predictive reliability from frozen pre-HMM probabilities and an independent neighbor-epoch intervention analysis of target-epoch pre-HMM responses.

不能写成：

> Our reliability- and context-weighted LIME faithfully explains the classifier.

前者准确反映三个独立模块；后者同时违反 attribution identity、faithfulness 状态和禁止重新加权的边界。
