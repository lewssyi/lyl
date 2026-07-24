# XAI Final Post-hoc Freeze Report

## Final decision

XAI development is frozen. The only canonical display method is
**Context-Preserving Classic Sequence-LIME**, restricted to exploratory
**Local Perturbation Sensitivity**. Engineering and audit infrastructure is
valid, while the scientific status remains
`MODEL_FAITHFULNESS_NOT_ESTABLISHED`. No new LIME algorithm, fixed-36
superiority run, classifier training, or explanation-guided training was
performed.

## Required questions

1. **当前唯一 canonical 解释方法是什么？**
   `CONTEXT_PRESERVING_CLASSIC_SEQUENCE_LIME`，展示名为
   Context-Preserving Classic Sequence-LIME。
2. **它是否参与训练？**
   否。真实 LIME target、正 `lambda_xai` 或 feedback enabled 均立即报错。
3. **它是否参与正常推理？**
   否。正常推理不读取 sensitivity、reliability 或 CCA artifact。
4. **当前能否称为 model-faithful？**
   不能；权威状态是 `MODEL_FAITHFULNESS_NOT_ESTABLISHED`。
5. **当前能否称为优于 Classic LIME？**
   不能。canonical 本身就是冻结的 context-preserving Classic 参考方法。
6. **MF-SeqLIME 状态是什么？**
   `SUPERSEDED_HISTORICAL_CANDIDATE`。
7. **HC-SeqLIME 状态是什么？**
   `VALIDATION_FAILED_HISTORICAL_CANDIDATE`；
   `VALIDATION_NO_QUALIFIED_HIERARCHICAL_CANDIDATE` / `HC_SEQLIME_FAIL`。
8. **CCA 状态是什么？**
   `IMPLEMENTATION_VALID_EXPERT_VALIDATION_PENDING`；仅为独立派生注释。
9. **Feedback training 是否永久 blocked？**
   是，`PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD`。
10. **哪些图进入正文？**
    Figure A 架构、Figure B reliability audit、Figure C coverage、Figure D
    固定规则案例。
11. **哪些负结果进入 supplement？**
    Figure E、negative-ablation table、PC prior、weak-zero、structured /
    boundary-adaptive、MF 与 HC-G9/G25 的失败记录。
12. **论文可声明的 XAI 贡献是什么？**
    可复现的 context-preserving 查询流程、严格 parity/memory guard、
    reliability taxonomy、协议一致的负结果、固定案例选择和独立 CCA 层。
13. **论文不可声明什么？**
    不可声明 model-faithful、causal、validated/superior attribution、
    临床已验证、性能提升或 explanation-guided training 收益。
14. **所有案例是否按固定规则选择？**
    是。validation-only、先选后画，禁止按美观、生理预期或论文故事选择。
15. **Canonical parity 是否保持？**
    权威同后端 GPU batch-16 guard 为
    `EXACT_ARTIFACT_PARITY_PASS`，wrapper 最大差 `5.96e-08`。最终服务器
    GPU 驱动失配，CPU 诊断仍保持 100% argmax、零 Accuracy/Macro-F1/Kappa
    差和 wrapper PASS，但概率最大差 `8.52e-4`；该结果已隔离，不替代权威
    GPU guard。
16. **完整测试结果是什么？**
    152/152：131/131 unittest，21/21 pytest；新增/确认的最终冻结守卫
    超过要求的 16 项。
17. **Branch、HEAD、worktree 和 commits 是什么？**
    Branch `codex/a1-zero-mainline`。内容冻结提交依次为：
    `2996ddb`、`9c0de99`、`809dd71`、`d48a19e`、`bd7d5f7`。
    最终 provenance 提交及 clean worktree 由交付时的 `git log/status`
    记录。

## Provenance and non-overwrite statement

Historical MF, HC, PC and protocol-consistency results were not deleted or
rewritten. The failed final CPU parity attempt is preserved under
`parity_cpu_driver_mismatch_20260724`; the authoritative parity directory was
restored from the unchanged protocol-consistency guard. The previously
disclosed legacy guard-artifact incident remains disclosed. The checkpoint
SHA-256 is unchanged.

## 不超过 500 字总结

最终XAI定位：具有严格审计边界的事后分析框架。
Canonical方法：Context-Preserving Classic Sequence-LIME，仅报告局部扰动敏感性。
模型忠实性：未建立。论文级状态：工程管线有效，归因优越性不成立。
MF状态：已取代的历史候选。HC状态：validation失败的历史候选。
CCA状态：实现有效、专家验证待完成。Feedback状态：对当前方法永久阻断。
正文内容：架构、可靠性审计、覆盖率和固定规则案例。
补充材料：MF/structured/HC等负结果与失败原因。
可声明贡献：可复现post-hoc流程、parity/memory guard、可靠性分类、失败分析及独立临床语义注释。
不可声明贡献：model-faithful、causal、优于Classic、临床已验证、训练或性能提升。
