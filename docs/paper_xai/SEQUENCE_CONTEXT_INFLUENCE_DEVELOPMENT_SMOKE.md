# Sequence Context Influence Audit — Development 2-Sample Engineering Smoke

## 1. 最终状态

**`DEVELOPMENT_SMOKE_BLOCKED`**

本轮没有满足进入 frozen development-16 的全部工程门禁。唯一一次正式 smoke 在第一个冻结 development 样本 `8809 / S011`、offset `-3`、primary `normalized_zero` replacement 的 batch parity 控制处停止。batch=1 与计划 batch=16 未满足预先冻结的 `atol=1e-6, rtol=0` 判据。脚本按 fail-fast 规则立即终止，没有自动重跑，也没有启动 development-16。

该失败是工程门禁失败，不是科学阴性结果。本轮没有可报告的 Context Influence profile，不能据此判断上下文机制、offset 重要性、替换基线或生理意义。

## 2. 起点与推理前协议冻结

开始时分支为 `codex/a1-zero-mainline`，HEAD 同时包含 Reliability 审计代码提交 `4283935be4f0449c04ce138594c275fdb2667084` 和报告提交 `276ca1041be07213c455fa7250ac7bb725bceccc`，工作树干净。Context smoke 的独立代码与测试提交为 `4b0e3b1a6c695e7d5b49dea153102d22f6685211`。

模型推理前生成并设为只读的 `protocol_snapshot.json` 记录了 source HEAD `4b0e3b1`、checkpoint、development manifest、两个样本、offset、replacement、response、aggregation、batch 与容差。正式进程只在该快照存在、source HEAD 未变化且工作树干净时才允许加载模型。

冻结 development-16 manifest 为：

`outputs/hierarchical_context_xai_feasibility/isruc_seed42/development_samples.json`

其 SHA-256 为 `c7717212c92bfd655ea728643e5e8161fe3006b1e3be8e94e57fb817948eca5e`。该 manifest 没有另行指定 2-sample smoke，因此在任何推理前按“manifest 原顺序中的前两个、且所有冻结 offset 均位于同一 recording 内的有效样本”确定：

| 顺序 | Sample ID | Subject | Recording ID | Recording | Target epoch |
|---:|---:|---|---:|---|---:|
| 1 | 8809 | S011 | 9 | S011.npz | 813 |
| 2 | 58657 | S097 | 67 | S097.npz | 60 |

该选择规则不使用本轮预测结果、correctness、confidence、LIME 质量、Context Influence 或生理现象。两个样本不构成类别平衡或代表性样本。

实际执行中，样本 8809 进入推理并在首个正式反事实的 batch parity 处阻断；样本 58657 从未进入推理。因此不能声称“两个样本均已完成”。

## 3. 冻结的 Context Influence 协议

- Offset：`{-3,-2,-1,+1,+2,+3}`；`Δ=0` 不属于 influence，只用于 no-op control。
- Primary replacement：canonical normalized zero waveform；其 30-segment features 必须由替换后的 zero waveform 经 canonical extractor 重算。
- Sensitivity replacement：在同一 recording 内，排除 target 与 `t-3…t+3` 后，选择最接近 recording segment-feature median 的真实 epoch；精确并列时取最小 epoch index。不得使用标签、预测、置信度或生理概念。
- 原始响应类别：`c* = argmax(p_original)`，只来自未扰动的五类 Pre-HMM 聚合概率，不使用真实标签或 HMM 标签。
- Primary response：`p[c*]`。
- Signed influence：`p_original[c*] - p_counterfactual[c*]`；absolute influence 单独保存，不与 signed 值融合。
- Primary aggregation：所有覆盖 target 的 canonical 75-epoch、stride-5 窗口等权平均；只有同时覆盖 context epoch 的窗口被修改，其余 target-covering 窗口保持原样。
- Additional aggregation：只在同时覆盖 target/context 的窗口上计算 shared-window paired mean，与 primary 分列。
- 边界：不跨 recording、不环绕，不干预非物理 padding；缺失 offset 记 structural missing。
- 模型：冻结 A1-zero，`training=False`，Pre-HMM only，不运行 HMM、梯度或反向传播。
- Batch：计划 batch 16；对照 batch 1。
- Parity：`atol=1e-6, rtol=0`，不得看到结果后放宽。
- Replacement 随机性：zero 与 feature-medoid 都是确定性的；没有随机 donor。

## 4. 唯一正式运行的停止位置

唯一正式进程按以下顺序执行到首个硬失败：

1. 所有输入、checkpoint、manifest 与权威状态 lineage 通过；
2. 加载冻结模型，未启用训练或 HMM；
3. 对样本 8809 重建未扰动 target-window 路径；
4. baseline parity 门通过后继续；
5. 对 offset `-3` 走完整 no-op 路径；no-op 门通过后继续；
6. 生成 normalized-zero waveform 并从该 waveform 重算 segment features；
7. 单物理 epoch 隔离、raw/segment 双视图与 overlapping-window 门通过后继续；
8. 对同一反事实比较 batch=1 与 batch=16；batch parity 判据失败，立即停止。

由于 fail-fast 发生时失败行仍在内存中，精确 `max_abs_full_vector_diff` 和 `max_abs_window_vector_diff` 没有在抛错前持久化。这是本次失败诊断的明确限制：能够确定失败 scope 和失败 predicate，但不能诚实给出具体差值。没有为补齐该差值而重跑模型。

## 5. 工程控制结果

### 5.1 原始输出 parity

样本 8809 的控制流通过了 baseline parity 门后才进入 no-op 和正式 replacement，说明已达到预注册条件；但精确差值未在后续 fail-fast 前写盘。因此状态记录为 `PARTIAL_PASS_NOT_PERSISTED`，不能替代完整的两样本 parity 报告。样本 58657 未运行。

### 5.2 No-op control

样本 8809、offset `-3` 的 no-op 控制通过后才进入正式 normalized-zero intervention；精确 influence 未持久化。状态为 `PARTIAL_PASS_NOT_PERSISTED`。其余 offsets 和第二个样本未运行。

### 5.3 单 epoch、双视图与窗口重建

在失败 scope 内，代码已用实际 recording 副本核对只修改 context epoch，且 replacement segment features 从替换 waveform 重算；未受影响 epoch 由 exact array equality 检查。控制流在这些硬门通过后才进入 batch parity，但 exact scalar diff counts 未持久化，因此保持 `PARTIAL_PASS_NOT_PERSISTED`，不提升为完整 PASS。

静态窗口审计可完整复现该 scope：target epoch 813、context epoch 810 共有 15 个 target-covering windows，15 个均同时覆盖 context；window starts 为 `740,745,…,810`，无跨 recording、遗漏或重复。该静态结果不能替代已经丢失的运行时数值审计行。

### 5.4 Batch parity 与 deterministic rerun

Batch parity 为 **FAIL**：样本 8809、offset `-3`、normalized-zero 的 batch=1 与 batch=16 未同时满足 full-vector 和 per-window vector 的 `1e-6` 上限。不能根据现有产物区分是 full-vector、per-window 或二者均超限。

完整样本 deterministic rerun 位于 batch parity 之后，因硬失败没有执行，状态为 `NOT_RUN_AFTER_HARD_GATE`。

## 6. 时间、显存与资源门

正式进程运行期间，从外部 `nvidia-smi` 观察到单进程使用 22,582 MiB；该数值只是运行中的外部快照，不是 TensorFlow allocator peak。由于 fail-fast 发生在 runtime summary 持久化之前，无法提供可信的单样本、单 offset、总时间或 allocator 峰值，也不能推算完整 development-16 成本。

因此资源可行性门为 `NOT_EVALUATED_AFTER_BATCH_PARITY_FAIL`，不能以“没有 OOM”替代完整资源 PASS。

## 7. 数据访问与冻结资产

实际读取的预测 NPZ 键仅包括：`class_labels`、`train_y_true`、`train_probs_raw`、`train_recording_id`、`train_epoch_index`、`train_file_sample_counts`。实际进入推理的数据 scope 只有冻结 development 样本 8809/S011。

- development-16 其余样本：未运行；
- 第二个预选样本 58657/S097：未运行；
- validation-30：未访问；
- test-36：未访问；
- validation/test prediction arrays：未读取；
- HMM、训练、梯度、参数更新：未执行。

停止后的源哈希仍为：

| 资产 | SHA-256 |
|---|---|
| Checkpoint | `e86a84790fd7351c1864158eef79fbf28437a7a8a5f26662b93bcf1aef14b6ed` |
| Prediction NPZ | `53cd6b51c008e0dd47dd187654c8eddacd5d0a69037c48aeecafe5efa9594e39` |
| split.json | `8363546a86b81676fe6171f2e50405d2d7447cbef1f48dbf397cd53bf3f037d9` |
| Authoritative XAI status | `55c74b36113f438f0a94b8e9d2c679f6cdfd2b85825b8c63ecd80f43f60092bc` |
| Development manifest | `c7717212c92bfd655ea728643e5e8161fe3006b1e3be8e94e57fb817948eca5e` |

Canonical B0_CLASSIC_90、Reliability Audit、checkpoint、prediction asset、模型训练代码和三项权威状态均未修改。

## 8. 硬门总表

| 硬门 | 状态 |
|---|---|
| 输入/checkpoint/development manifest lineage | PASS |
| 推理前冻结样本与协议 | PASS |
| 完成恰好两个 development 样本 | **FAIL — 只到达第一个样本** |
| 原始输出 parity | PARTIAL_PASS_NOT_PERSISTED |
| No-op influence 接近 0 | PARTIAL_PASS_NOT_PERSISTED |
| 单物理 epoch 无泄漏 | PARTIAL_PASS_NOT_PERSISTED |
| Raw/segment 双视图同步 | PARTIAL_PASS_NOT_PERSISTED |
| Overlapping windows 无遗漏重复 | PARTIAL_PASS_NOT_PERSISTED |
| Target-position 映射 | PARTIAL_PASS_NOT_PERSISTED |
| Batch parity | **FAIL** |
| Deterministic rerun | NOT_RUN_AFTER_HARD_GATE |
| 数值/概率语义 | PARTIAL_PASS_FOR_REACHED_SCOPE |
| 无 HMM、梯度、训练、权重更新 | PASS |
| 无 validation/test 访问 | PASS |
| Canonical LIME 与权威状态不变 | PASS |
| Checkpoint/prediction hashes 不变 | PASS |
| 单 RTX 3090 时间/显存可行性 | NOT_EVALUATED_AFTER_HARD_GATE |

并非所有硬门均为 PASS，因此唯一允许的最终状态是：

**`DEVELOPMENT_SMOKE_BLOCKED`**

## 9. 为什么不能形成科学结论

本轮原本就是 n=2 的工程 smoke，而实际只到达一个样本的第一个 offset/baseline，且 batch-size 数值稳定性失败。没有完整 profile、没有第二个样本、没有两种 baseline/六个 offsets 的封闭执行，也没有 deterministic 和资源门。因此不能声称上下文机制有效或无效、某个 offset 重要、邻近 epoch 具有稳定生理意义、Context Influence 优于 LIME、解释 faithfulness 成立，或结果可进入论文正式实验。

若未来需要继续，只能在新的明确授权下先修正“失败行在 fail-fast 前落盘”的诊断持久化问题，并把任何重新执行标记为新的工程诊断；不得直接进入 development-16，也不得放宽 `1e-6` 容差来获得通过。
