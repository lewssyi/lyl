# Sequence Context Influence — Batch-Shape Numerical Equivalence Protocol Amendment V1

**`BATCH_SHAPE_NUMERICAL_PROTOCOL_FROZEN`**

原 smoke 继续保持 **`DEVELOPMENT_SMOKE_BLOCKED`**。原 `atol=1e-6, rtol=0` 失败不被追溯撤销；本协议只适用于未来获得单独授权后启动的全新 smoke。

## 协议与理由

新协议命名为 `BATCH_SHAPE_NUMERICAL_EQUIVALENCE_V1`，依次执行 Structural Identity、Same-Shape Repeatability、Cross-Batch Probability Equivalence 和 Derived-Response Equivalence 四层门禁。结构身份和同 batch 重复仍要求 exact；中间 activation 与 logits 只保留为诊断信息。

诊断显示 A1/A2 与 B1/B2 分别逐元素完全稳定，输入 hash、窗口顺序和 target-position 完全一致；差异已经在并行 raw/segment encoder 输出处出现，聚合不是首个差异点，argmax 与 influence 符号均未改变。因此 `1e-6/0` 不适合作为不同 batch shape 的逐元素等价硬门。

概率门固定 batch-1 为 reference，对每个五类 Pre-HMM 元素要求：

`abs(p16 - p1) <= 1e-5 + 1e-5 * abs(p1)`

`1e-5/1e-5` 来自 2026-08-03 进入 smoke 前的授权文本 fallback（该文本早于 09:28:49 UTC 的 smoke 协议冻结），不是根据本次 `1.84178e-5` 反推。usage 超过 0.8 只产生 warning，绝不自动扩容差。

响应容差为 `epsilon_R(x)=1e-5+1e-5*abs(x)`；influence 容差严格由误差传播得到：`epsilon_I=epsilon_R(R_original_batch1)+epsilon_R(R_counterfactual_batch1)`。若任一 influence 的绝对值不超过 epsilon_I，则记为 `NUMERICALLY_NEAR_ZERO`，其符号变化不得解释为科学方向变化。

## 纯离线复核

- 历史协议：FAIL；per-window max `1.8417835235595703e-05`，超过元素数 `30`；聚合 max `5.0187110900434817e-06`。
- 新协议概率门：`False`；max diff `1.8417835235595703e-05`，该位置 allowed `1.8987406492233279e-05`。
- 绝对差最大元素自身 usage 为 `0.97000268273233858`，该元素通过并触发 near-tolerance warning。
- 全局最大 normalized tolerance usage：`1.609448262296669`，位置 `[14, 1]`；reference `0.09945310652256012`，candidate `0.099435411393642426`，actual diff `1.7695128917694092e-05`，allowed `1.0994531065225602e-05`。该元素超过硬门，所以当前诊断在 V1 下整体仍为 FAIL。
- Near-tolerance warning：`BATCH_SHAPE_NEAR_TOLERANCE_WARNING`。
- 聚合响应门：`True`；actual `5.0187110900434817e-06`，allowed `1.9511457244555156e-05`。
- Influence 门：`True`；actual `5.0187110900434817e-06`，epsilon_I `3.9112646142641705e-05`。
- 语义不变门：`True`；整体离线结果：`False`。
- logits 最大差继续记录，但 logits 量纲不同，不套用概率容差，也不作为硬门。

## 资源与科学边界

历史诊断记录的 TensorFlow allocator peak 为 batch-1 `139143936` bytes、batch-16 `1659473408` bytes；reserved-memory 为 N/A。这些 allocator 指标不能与外部观察的整个 GPU 进程 `22582 MiB`直接等同，也不能据此宣称 development-16 资源门通过。

离线复核工具本身没有导入 TensorFlow，也没有加载 checkpoint/原始波形或执行模型 forward。相关协议测试共 79 项通过：其中 75 项为纯数组/协议测试，另 4 项既有 batch-cap 单元测试仅导入 TensorFlow 并运行测试桩，显式禁用 GPU，未加载 checkpoint、波形或研究样本。

本轮没有访问第二个 development 样本、sample 58657、validation 或 test。该协议不证明 Context Influence 有效，不建立 model faithfulness，也不改变 canonical LIME 或三项权威科学状态。
