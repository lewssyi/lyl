# Context Influence Paper Convergence Report

**`PAPER_IMPLEMENTATION_READY`**

## 正式实现配置

正式执行策略为 `CONTEXT_INFLUENCE_CANONICAL_BATCH1_EXECUTION_V1`：inference batch size 固定为 1，float32、`training=False`、冻结 checkpoint、75-epoch 上下文、Pre-HMM 响应、raw/segment 同步更新均保持不变；普通浮点对照沿用项目已有 `atol=1e-5, rtol=1e-5`，不创建新协议。不调用 batch-16，也不声称 batch-size invariance。Canonical XAI 名称仍为 **Context-Preserving Classic Sequence-LIME**；Context Influence 仅是可选诊断/补充分析。

## 两样本 smoke

| Sample | Subject | Target | Class | Original response | Baseline max diff | No-op max | Repeat max diff | Windows (cover/shared) | Runtime (s) |
|---:|---|---:|---|---:|---:|---:|---:|---|---:|
| 8809 | S011 | 813 | W | 0.9601204475 | 1.55766805e-06 | 0 | 0 | 15/14,15 | 14.07 |
| 58657 | S097 | 60 | N1 | 0.650488764 | 3.010034561e-06 | 0 | 0 | 13/12,13 | 9.72 |

## 十项功能检查

| 检查 | 结果 |
|---|---|
| 程序完整运行 | PASS |
| Baseline 响应恢复 | PASS |
| No-op 无实质性 Influence | PASS |
| 仅修改指定 physical epoch | PASS |
| Raw/segment 同步更新 | PASS |
| 重叠窗口完整重建 | PASS |
| Target position 正确 | PASS |
| 无 NaN/Inf | PASS |
| 同配置重复稳定 | PASS |
| 结果文件可读取 | PASS |

全部正式 forward 的 leading batch dimension 均为 1；未发现 batch-16 调用。两个冻结样本以既定顺序访问，未访问其他 development、validation 或 test。checkpoint、prediction、split、development manifest、canonical attribution manifest 与模型内存权重均保持不变。

历史事实 **`BATCH_SHAPE_NUMERICAL_EQUIVALENCE_V1 = FAIL`** 原样保留（max usage `1.609448262296669`，位置 `[14, 1]`）。它表示历史 batch-16 性能优化路径未证明跨 batch 数值等价，不影响固定 batch=1 的论文实现；本轮没有新建 V2 或修改容差。

## 论文声明边界

可以声明：target-epoch channel–segment perturbation；3×30=90 维解释空间；cross-view consistent feature update；context-preserving overlapping-window inference；weighted local Ridge；reliability audit；independent physiological annotation；no training feedback。

不能声明：优于传统 LIME；已建立 model faithfulness；生理标注提高 attribution 真实性；batch-1 与 batch-16 数值等价；已完成临床有效性验证。

## 尚缺论文材料与结论

Context Influence 若进入补充材料，仍需基于既有协议的定量汇总图与清晰图注；若无充分定量证据，应仅保留为可选诊断，不提升为核心创新。当前不需要继续扩展 Context Influence 工程或协议层，可转入论文写作。
