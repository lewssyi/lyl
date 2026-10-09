# 投稿开放项 V1

本清单只记录影响论文正确性、完整性或可审稿性的事项。历史 batch parity、形式化审计扩展和未晋级 XAI 方法不作为论文工程阻塞项。

## 必须补齐

1. **确定论文是 ISRUC 单数据集还是双数据集主验证。** 若摘要、Table II 或结论要声称 ISRUC + SleepEDF，必须找到与 A1-zero 主架构、正式 split、selector 和 decoder 同构的现有 SleepEDF 证据，或在未来获得明确实验授权后补做。当前 SCR-Lite 结果不能替代 A1-zero，名称为 `edf_c0` 的旧日志实际使用 ISRUC Sxxx 划分。若改为 ISRUC 单数据集论文，则需删除所有 SleepEDF 性能声明，而不是保留空泛跨数据集结论。

2. **补齐能支撑主架构贡献的严格配对基础消融，或收缩贡献表述。** 当前 `recommended_ablation_structure.md` 将 base multimodal epoch encoder 和 isolated sequence backbone 标为 `MISSING_REQUIRED_ABLATION`。若论文把双视图与 TCN–Transformer 作为性能创新点，缺少这两项会直接削弱因果归因；若不再运行实验，则必须把论文定位改为冻结系统与 post-hoc evaluation study，并避免声称各组件带来性能提升。

3. **建立正式参考文献与数据集引用表。** 当前写作包只整理项目内证据，没有验证 ISRUC、SleepEDF、LIME、TCN、Transformer、ECE/Brier/AURC 等外部引用。相关工作、数据许可和方法出处在投稿前均为 `TO_VERIFY`。

4. **核实并填写训练优化器。** 冻结 `canonical_mainline.json` 明确记录 `optimizer: UNKNOWN_FROM_LOG`。方法章节在确认前不得猜测；若投稿模板要求完整训练细节，必须从既有日志或代码配置中确定并留下文件路径。

## 建议补齐

1. **完成真实盲评。** 当前 evaluator count 为 0，因此 enhanced presentation 的可读性、可靠性判断、偏好、时间与误读率均不可声明。若不做盲评，论文应把 Physiological Annotation 定位为自动审计与可复现展示工具。

2. **补充 28 维 segment feature 的字段字典。** 当前代码确认形状与 encoder，但写作包没有发现足够稳定的字段说明。缺少字典不阻止模型结构描述，却会降低方法可复现性。

3. **增加严格配对的第三训练 seed。** A1-zero 当前只有 seed42 与 seed2024 的汇总，n=2 只能做描述性稳定性说明，不能形成稳健的方差结论。

4. **统一投稿表中的 decoder 标签。** 分类主结果写 validation-selected temperature argmax；Reliability 与 XAI 写 uncalibrated Pre-HMM raw probabilities。不要只写“model probability”而省略阶段。

## 可放补充材料

1. ISRUC 两 seed 的 mean/std 与 Pair SupCon、local contrast 等负比较。
2. Equal-width ECE/MCE sensitivity、subject-level reliability 极值和全部 risk-coverage operating points。
3. Selected-30 的三 seed Spearman、cosine、top-10 overlap 与 confidence–diagnostic 描述性相关；必须保留 exploratory、selection bias、n=30 与重复测量说明。
4. Context Influence 两个 development smoke 的十项功能检查与固定 batch=1 配置。没有更大规模结果时不进入核心贡献或优越性结论。
5. HC-SeqLIME、MF-SeqLIME、structured neighborhood 与 boundary-adaptive sampling 的失败历史，可作为透明的 negative methods appendix，但不得进入当前主方法。
6. 生理标注的自动覆盖、一致性、冲突案例与空白 blinded package；不得替代真实专家评价。

## 明确不是阻塞项

- 历史 `BATCH_SHAPE_NUMERICAL_EQUIVALENCE_V1=FAIL` 不阻塞固定 batch=1 的正式论文实现。
- 不需要证明 batch=1 与 batch=16 逐元素等价，也不需要创建新容差协议。
- 不需要继续扩展 Context Influence 形式化审计、HC/MF 变体、状态机或大规模哈希清单。
- 不需要把 `MODEL_FAITHFULNESS_NOT_ESTABLISHED` 改写为正结论；透明报告该边界即可。
