# PR-SeqXAI Reviewer Q&A

## 1. 没有专家评价，为什么仍称为贡献？

贡献对象不是“人类已经更容易理解”，而是一个可复现、可审计的多层事后解释框架：
它保持重叠窗口推理语义，隔离模型敏感性与派生生理注释，并把失败的可靠性证据显式呈现。
工程隔离、自动生理指标和负结果均有冻结证据；人类获益被明确留作未来验证。

## 2. 为什么不声称比 LIME 更好？

PR-SeqXAI 的底层 attribution 就是冻结的 Context-Preserving Classic
Sequence-LIME。本研究没有产生可支持总体 superiority 的比较证据，且统一随机化门控未通过。
因此贡献是序列适配、信息分层和审计边界，而不是新的归因算法。

## 3. 生理一致性如何量化？

30 个冻结 validation 案例得到：concept coverage 100.0%、unsupported
0.0%、conflict 10.0%、channel-concept consistency 96.7%、top-region
evidence 100.0% 和 cross-seed stability 0.973。true-stage agreement 仅在
注释生成后描述性连接，未参与注释。

## 4. Annotation 是否改变 attribution？

没有。30/30 个 V0/V1 对的 attribution 数组 bitwise identical，ranking
和 color scale 也完全一致。CCA 与 reliability 均为下游层，不读取 true stage
来生成注释，也不能改变预测或训练。

## 5. 100% coverage 是否可能代表过度注释？

可能，因此本文只把 coverage 定义为固定公式找到了正的有限证据，而不把它解释为正确率。
临床正确性未知；10% conflict、36.7% reporting-only stage agreement 和尚未进行的专家评价
共同限制该结果。`NONE`/`INSUFFICIENT_EVIDENCE` 在协议中仍是合法输出。

## 6. 10% conflict 如何处理？

三个冲突案例被保留、逐案列入 `annotation_conflicts.csv`，并在增强展示中触发警告。
它们不被删除、不被平均掩盖，也不用于重新选择样本或调整 attribution。冲突是透明度结果，
不是失败后清理的数据。

## 7. Randomization 失败，为什么仍展示 LIME？

随机化失败阻止的是 model-faithfulness 声明，不会抹去局部扰动响应作为诊断量的存在。
因此该图只能称为 exploratory Local Perturbation Sensitivity，并必须与 reliability state
和 failed-randomization warning 一起阅读。

## 8. 为什么这是框架创新，而不是多画几张图？

框架定义了三个有类型边界的信息层：模型查询得到的 sensitivity、原始信号得到的 annotation、
以及审计得到的 reliability。身份守卫证明后两层不能改变第一层。创新点是可复现的组合规则、
信息来源隔离和声明边界，不是颜色或版式数量。

## 9. 人类可理解性为何留作未来工作？

目前 evaluator count 为零。填写空白模板、偏好、完成时间或一致性数值都会构成伪造。
现有 30-case 盲法包和 10-case pilot 只证明未来评价已经可执行，不能替代真实观察。

## 10. PR-SeqXAI 是否参与训练或正常推理？

不参与。正常推理默认不读取 XAI artifact；真实 explanation target、正
`lambda_xai` 或启用 feedback 均被守卫阻断。当前状态为
`PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD`。

## Concise English response

PR-SeqXAI contributes a guarded representation and evaluation framework, not
a superior attribution estimator. It preserves the frozen sequence-query
semantics, separates model-, signal-, and audit-derived information, and
reports negative reliability evidence. Human interpretability, clinical
utility, model faithfulness, and causality remain unestablished.
