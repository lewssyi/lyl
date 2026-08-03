# Sequence Context Influence — Batch Parity Failure Diagnostic Run 1

**`BATCH_PARITY_DIAGNOSTIC_ROOT_CAUSE_IDENTIFIED`**

该诊断不解除 **`DEVELOPMENT_SMOKE_BLOCKED`**，也不授权 development-16。

## 诊断结论

- 精确最大概率绝对差：`1.8417835235595703e-05`。
- 首个有序 probe 差异阶段：`raw_encoder` （max `1.6689300537109375e-06`）；并行 `segment_encoder` 也已差异（max `5.0067901611328125e-06`），因此层级结论是两条 epoch encoder 分支已经出现 batch-shape 差异。
- 根因分类：`BATCH_SHAPE_NUMERICAL_DRIFT`。
- 输入、窗口顺序、target position：`EXACT / EXACT / EXACT`。
- batch-1 A1/A2 exact stable：`True`。
- batch-16 B1/B2 exact stable：`True`。
- 最大概率差位于 window start `810`、target position `3`、class `W`：batch-1 `0.89874064922332764`，batch-16 `0.89875906705856323`。
- logits 最大绝对差：`0.00028181076049804688`。
- 聚合概率最大绝对差：`5.0187110900434817e-06`；response `0.9511457244555156` → `0.95115074316660564`。
- signed influence `0.0089731653531391897` → `0.0089681466420491462`；argmax 改变：`False`；influence 符号改变：`False`。
- batch-1 TensorFlow allocator peak：`139143936` bytes (`132.698` MiB)。
- batch-16 TensorFlow allocator peak：`1659473408` bytes (`1582.597` MiB)。
- TensorFlow allocator 不暴露与 PyTorch `max_memory_reserved` 等价的 reserved 指标，因此该项明确记为 N/A。
- 原 smoke 设置保持不变：float32、TF32 enabled、无 mixed precision 切换、training=False、seed=42。

## 边界与资产

- 第二个 development 样本访问：`False`。
- validation/test 访问：`False / False`。
- 模型权重不变：`True`；冻结资产哈希不变：`True`。
- 未运行 HMM、训练、梯度或权重更新；未修改 canonical attribution。
- 原 `development_smoke_2` 与原阻断报告未覆盖。

## 下一步

同 batch 重复若 exact stable 且输入身份完全一致，则应单独审议 batch-shape 数值协议；本轮禁止修复、改 batch、切换 TF32/dtype 或放宽 1e-6 容差。

所有详细数值、逐层差值、输入哈希和访问审计位于独立 diagnostic 输出目录。
