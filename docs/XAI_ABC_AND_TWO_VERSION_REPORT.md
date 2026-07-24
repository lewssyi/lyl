# PC-SeqLIME A/B/C 根因筛查与条件化双版本报告

> **STATUS: SUPERSEDED HISTORICAL CANDIDATE**
>
> This report is preserved as diagnostic history. The candidate did not
> establish model faithfulness under the authoritative protocol and is not the
> current paper method or a permitted training target.

## 结论摘要

本次审计的唯一主判定为 `MIXED_A_PLUS_C`：原 model-randomization sanity failure 同时包含真实实现/缓存缺陷（CASE C）和历史 PC 融合中 predicted-stage prior 对模型无关证据的污染（CASE A）。修复路由、图捕获和缓存身份后，Classic Sequence-LIME（M0）通过诊断集 full-model randomization sanity，历史 PC-SeqLIME（M4）失败；因此 CASE B 不成立。

在独立 validation split 上冻结的候选为 `V_ZERO + Y2 logit margin`，不含 predicted/true-stage prior 或固定 stage-channel prior。固定 36 个测试样本只评估一次。该候选在 faithfulness 和 mask-seed stability 上优于 M0，但外部 validation surrogate R² 与 M0 完全相同，并且 T1 随机模型类别协议仅 1/3 seeds 通过。因此最终判定是 `MF_SEQLIME_PARTIAL_PASS`，不是 paper-level pass。

Clinical Concept Annotation（CCA）已经与 attribution 类型和执行路径完全分离，状态为 `IMPLEMENTATION_VALID / EXPERT_VALIDATION_PENDING`。它不是 concept bottleneck，也不是因果贡献估计。

## 冻结边界与 provenance

- 分支：`codex/a1-zero-mainline`。
- 冻结锚点：`9978a3b5c3f75eb8d56a24b583894a0658b4f6ce`；tag：`a1-zero-xai-audit-v1.1.3`。
- checkpoint SHA256：`e86a84790fd7351c1864158eef79fbf28437a7a8a5f26662b93bcf1aef14b6ed`。
- split SHA256：`8363546a86b81676fe6171f2e50405d2d7447cbef1f48dbf397cd53bf3f037d9`。
- probability artifact SHA256：`53cd6b51c008e0dd47dd187654c8eddacd5d0a69037c48aeecafe5efa9594e39`。
- evaluator：`sequence_lime_v1.1.3`；TensorFlow 2.15.0；CUDA 12.2；cuDNN 8；8×RTX 3090。
- 固定 36 样本来源 manifest SHA256：`e9226560bc8ba23212e439bff75406d67075998082a6a3d056ae5e1fa956bdf2`；24 correct、12 incorrect、17 transition、19 stable，覆盖五个 true/predicted stages。
- 正式 v1.1.3 目录未被改写；其 `CURRENT_RESULTS_MANIFEST.json` 中 13 个有效文件的 SHA256 全部复核不变。

## CASE C：实现和缓存根因

旧 `final_layer` 路由匹配不存在或无权重的 `main_output_logits`，因此改变 0 个变量且输出最大变化为 0。旧 cache key 没有完整模型权重 hash，sample resume 也未校验 evaluated-model identity。这些缺陷足以让“随机化模型”实际复用原模型或旧 attribution。

修复后 R1 指向真实 `main_output`，R2–R6 按预注册层组从同一 canonical checkpoint 独立构建，R7 为全模型 fresh initialization。R1 改变 2 个变量、输出最大变化 0.967574；R7 改变 159 个变量、输出最大变化 0.960230。所有随机化 forward、zero/physiological marginal、surrogate response、deletion/insertion 均使用传入模型。每个缓存身份包含 checkpoint、完整权重 fingerprint、对象身份、随机化层级/seed、method、T0/T1、replacement、sample、mask seed、query budget 和 evaluator version；不匹配即拒绝恢复。

同一随机模型 eager 与 fixed-graph forward 一致，而与 original forward 明显不同；compiled runner 为 model-local。新增测试覆盖相同 seed 可复现、不同 seed 权重不同、非随机变量不变、图不捕获旧模型、缓存身份隔离。

## A/B 组件诊断

诊断在固定 12 样本上运行 M0–M8、Y0/Y1/Y2、T0/T1、R0–R7、seeds 3101/3102/3103，共 15,552 行；每条 R7 对照使用 1,000 次 permutation null，共 1,944 条比较。M0–M4 查询预算相等。

| 方法 | R7 signed Spearman | null 97.5% 上界 | R0 faithfulness advantage | R7 advantage | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| M0 classic | 0.0067 | 0.2076 | 0.1578 | 0.0145 | 3/3 seeds 通过 |
| M1 + zero | 0.1304 | 0.2062 | 0.1965 | 0.0106 | faithfulness 增强、敏感性弱于 M0 |
| M2 + physiological | 0.0596 | 0.2060 | 0.1229 | 0.0103 | 两方面均弱于 M0 |
| M3 zero + physiological | 0.1208 | 0.2066 | 0.1765 | 0.0101 | 3/3 seeds 通过但保留更多 R7 相似度 |
| M4 historical PC | 0.3111 | 0.2053 | 0.1718 | 0.0075 | 0/3 seeds，失败 |
| M5 predicted-stage prior only | 0.5263 | 0.2071 | 0.0879 | -0.0032 | 0/3；T0 R7 similarity=1 |

M6 stage-independent annotation 和 M7 signal energy 是预期的模型无关控制，不声称为 model attribution。证据支持 predicted-stage prior 是历史融合失败的主污染源。M0 已通过，故不触发 CASE B 的 B1–B4 层级重构。

## Validation-only 冻结选择

Validation 样本由 seed 42017 按 stage×correctness 分层后补齐 transition/stable，共 12 个；`test_data_used=false`。候选为 pure M0、pure zero、pure physiological、equal blend、25% blends，均比较 Y0/Y1/Y2、faithfulness、stability、randomization sensitivity 和等查询预算。

冻结结果为 `V_ZERO + Y2`，validation weighted R²=0.667663，查询数 2,106，randomization 2/3 seeds 通过。它不使用任何 stage prior。测试前将选择写入 `frozen_validation_selection.json`，之后未因测试结果更改配置。

## 固定 36 测试结果

正式比较包含 M0、M3、M4、M5、M7、M8 和冻结 V_ZERO；R0/R7 × 3 seeds × T0/T1，合计 9,072 行。主要方法使用 Y2，查询预算均为 1,530。

| 指标（T0；T1 的 R0 faithfulness 相同） | M0 | 冻结 MF 候选 | 差值 / 检验 |
| --- | ---: | ---: | --- |
| 外部 validation weighted R² | 0.667663 | 0.667663 | 无改善 |
| test local validation MAE | 0.123183 | 0.123183 | 无改善 |
| test local validation RMSE | 0.189646 | 0.189646 | 无改善 |
| deletion AUC（低为好） | 0.462331 | 0.408616 | -0.053715，p=5.53e-14 |
| insertion AUC（高为好） | 0.742768 | 0.769504 | +0.026735，p=7.97e-11 |
| advantage vs random | 0.139740 | 0.179966 | +0.040225，p=2.81e-15 |
| mask-seed Spearman | 0.418463 | 1.000000 | 改善 |
| Top-10 Jaccard | 0.335309 | 1.000000 | 改善 |
| T0 R0→R7 similarity | -0.024962 | 0.107103 | null upper=0.207151；2/3 seeds |
| T1 R0→R7 similarity | 0.052463 | 0.146881 | null upper=0.205673；仅 1/3 seeds |

分层结果没有隐藏失败亚组。Incorrect 12 样本中 deletion 0.438802→0.377651、insertion 0.703733→0.729743；correct 24 样本中 0.474095→0.424098、0.762286→0.789384。Transition 17 样本中 0.456122→0.419417、0.708091→0.727882；stable 19 样本中 0.467886→0.398951、0.773796→0.806744。样本量小，因此这些分层只作描述性报告。

最终 `MF_SEQLIME_PARTIAL_PASS` 的原因是：faithfulness 和稳定性改善，T0 full-model randomization 达到 2/3 seeds；但 surrogate fidelity 未优于 classic，且 T1 仅 1/3 seeds。不能升级为 `MF_SEQLIME_PAPER_LEVEL_PASS`。

## Clinical Concept Annotation

CCA 仅在冻结 MF attribution 完成后读取 Top-10 channel×segment 区域，再从 `raw_physiology_v1` 计算 EEG delta/theta/alpha/sigma/beta、EOG slow/rapid eye、EMG low/active tone。输出严格分为 `model_attribution_score`、`clinical_concept_activation`、`clinical_annotation_score`；108 个 sample-seed 行均满足 attribution bitwise unchanged、`ranking_modified=false`、true stage 未参与、predicted stage 仅作显示上下文。平均 coverage=0.9969，三 attribution seeds 的 concept stability=1.0，model-region overlap=1.0。

这些是 derived clinical annotations，不是模型内部 concept bottleneck，不证明临床有效性或因果机制。必须完成盲法专家审核后才能做 clinical-validity 声明。

## 最终工程验证

- canonical batch-16：最大概率差 0，argmax agreement 1.0，Accuracy/Macro-F1/Kappa 差均为 0。
- sequence wrapper：最大差 `5.960464477539063e-08`。
- fixed graph：warmup 20 后 1,200 forwards；current GPU memory 增长 0 bytes，后半程跨度 0 bytes；checkpoint 和权重哈希不变。
- 完整维护测试：58/58 通过。
- 必需输出：13 个 A/B/C 文件、11 个双版本文件、3 PDF + 3 PNG 全部非空。

## 18 项直接回答

1. **原 sanity failure 是否含实现/缓存错误？** 是。旧 final-layer 路由改变 0 变量，cache/resume 不验证完整模型身份。
2. **路线 C 是否成立？** 成立，已修复并用回归测试与重跑证据确认。
3. **Classic Sequence-LIME 是否通过 randomization sanity？** 是；诊断 M0 为 3/3 seeds，固定 36 的 T0/T1 也均为 3/3。
4. **zero marginal 的作用？** 提高 R0 faithfulness，但相对 M0 保留更多 R7 相似度，削弱模型敏感性。
5. **physiological marginal 的作用？** 相对 M0 同时降低 R0 faithfulness 并削弱随机化敏感性。
6. **predicted-stage prior 是否为失败主因？** 是；M5 T0 在 R7 仍为 similarity 1，M4 超过 permutation null。
7. **属于 A/B/C 还是 MIXED？** `MIXED_A_PLUS_C`；主判定 A，次判定 C，B 不成立。
8. **是否建立 MF-SeqLIME？** 建立了 validation-frozen、无 prior 的候选，但科学等级仅 `MF_SEQLIME_PARTIAL_PASS`。
9. **相对 classic 改进了什么？** 删除/插入 AUC、相对随机 faithfulness、mask-seed Spearman 和 Top-10 Jaccard。
10. **哪些未改善或变差？** 外部 validation R²、local MAE/RMSE 无改善；T1 randomization 仅 1/3 seeds，弱于要求。
11. **是否通过 full-model randomization？** 预注册主协议 T0 以 2/3 seeds 通过；T1 仅 1/3，必须保留该限制。
12. **CCA 是否完全独立于排名？** 是；类型、文件、调用顺序和 exact-unchanged guard 均独立。
13. **能否声称优于传统 LIME？** 不能作总体或论文级 superiority 声明；只能声称固定样本/替换规则下特定 faithfulness 与稳定性指标改善。
14. **能否进入 explanation-guided training？** 不能。
15. **最关键阻塞是什么？** surrogate fidelity 没有 validation 改善，T1 randomization 不足，且临床专家验证未完成。
16. **当前 branch/HEAD/工作树？** branch 为 `codex/a1-zero-mainline`；代码与 guard 的 pre-doc HEAD 为 `b119186`。本报告的 docs commit 是交付 HEAD，最终 clean 状态由交付命令复核。
17. **新增 commits？** `770d2cc`、`d7c7090`、`efa9daf`、`bf4d288`、`b119186`，以及最终 docs/results commit。
18. **输出与测试路径？** A/B/C：`outputs/sequence_lime_abc_diagnosis/isruc_seed42/`；双版本：`outputs/sequence_lime_two_versions/isruc_seed42/`；最终 guard：其 `final_validation/`；测试：`tests/test_xai_randomization_route.py`、`tests/test_xai_two_versions.py` 和完整 `tests/`。旧正式 `outputs/sequence_lime_paper_evaluation/isruc_seed42/` 的 v1.1.3 数据与 13 个索引哈希均未变。

## 不超过 500 字总结

A/B/C 判定：MIXED_A_PLUS_C（主 A、次 C，B 不成立）。
核心原因：旧随机化层路由及 cache identity 有缺陷；修复后历史 predicted-stage prior 仍造成模型无关排名。
是否修复实现问题：是，路由、模型 hash、resume 和 fixed-graph 均已修复并测试。
模型忠实性版本：冻结 V_ZERO+Y2；MF_SEQLIME_PARTIAL_PASS。
临床注释版本：CCA 实现有效，专家验证待完成，绝不回写 attribution。
相比传统 LIME：固定 36 上删除/插入 faithfulness 与 seed 稳定性改善，surrogate fidelity 未改善。
论文可声明内容：指定替换规则下的局部敏感性和部分指标改善。
论文不可声明内容：总体优越、临床有效、因果贡献、paper-level PASS。
是否允许反馈训练：否。
下一步唯一优先事项：在独立 validation 设计能提高 surrogate fidelity 且同时通过 T0/T1 2/3 seeds 的候选，再做新的预注册测试。

## Protocol-consistency supersession (2026-07-23)

The historical Classic Sequence-LIME M0 3/3 sanity statement above is superseded by
`docs/SEQUENCE_LIME_PROTOCOL_CONSISTENCY_REPORT.md`. The old A/B/C and newer validation
runs differed in mask configuration, seed, query budget, null/similarity handling,
faithfulness, pass aggregation and cache protocol identity. Under one current evaluator,
both the old 12-sample group and new 12-sample group pass 0/3 T0 and 0/3 T1, with no
material A-C difference. The old conclusion is `superseded_by_protocol_consistency_audit`.
Feedback training remains prohibited.
