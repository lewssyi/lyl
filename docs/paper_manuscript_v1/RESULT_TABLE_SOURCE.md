# 结果表数值来源 V1

## ISRUC 主结果（正文正式结果）

| Model / setting | Seed | Test N | Accuracy | Macro-F1 | Kappa | W F1 | N1 F1 | N2 F1 | N3 F1 | REM F1 | 来源 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1-zero canonical | 42 | 13,072 | 0.830860 | 0.819873 | 0.782580 | 0.918388 | 0.594013 | 0.805556 | 0.882239 | 0.899167 | `outputs/final_evidence_synthesis/isruc/canonical_mainline.json`；`outputs/final_evidence_synthesis/isruc/paired_isruc_result_table.csv` |

该行是当前唯一可直接进入 Table II 的主模型结果。正式 decoder 为 validation-selected temperature argmax；不得与 HMM 候选或 Pre-HMM reliability 数值混淆。

## SleepEDF 主结果

| Model / setting | 状态 | 可用数值 | 来源与处理 |
|---|---|---|---|
| 与 ISRUC A1-zero 同构的 SleepEDF 主结果 | **TO_VERIFY** | 不填 | `outputs/final_evidence_synthesis/isruc/cross_dataset_evidence_gap.csv` 指出 seed 数与严格配对证据缺口；未找到可无歧义归入相同 checkpoint/protocol 的结果。 |
| SleepEDF SCR-Lite seed42 | 非主线、不同模型 | Accuracy 0.842762；Macro-F1 0.795101；kappa 0.785059；N=37,758 | `outputs/logs/result_20260710_070633_edf_A1_scr_lite_seed42.txt`。该 run 启用 stage concept router，不得作为 A1-zero 主结果。 |
| 名称含 `edf_c0_no_concept` 的旧 run | 排除 | Accuracy 0.826117；Macro-F1 0.811548；kappa 0.775777 | `outputs/logs/result_20260627_180622_edf_c0_no_concept.txt` 实际列出 ISRUC Sxxx train/val/test 文件，不能标为 SleepEDF。 |

因此，若论文以 ISRUC 单数据集为主，SleepEDF 只能作为待验证扩展；若论文声明双数据集主验证，则缺失同构 SleepEDF 结果是投稿阻塞项。

## 已有模型消融

| Variant | Seed | Accuracy | Macro-F1 | Kappa | Δ Macro-F1 vs A1-zero seed42 | 论文定位 | 来源 |
|---|---:|---:|---:|---:|---:|---|---|
| A1-zero canonical | 42 | 0.830860 | 0.819873 | 0.782580 | 0 | 参照 | `outputs/final_evidence_synthesis/isruc/paired_isruc_result_table.csv` |
| SCR-Lite | 42 | 0.830707 | 0.819024 | 0.782055 | -0.000849 | 同 seed 负比较 | `outputs/final_evidence_synthesis/isruc/paired_isruc_result_table.csv` |
| Epoch-UTF-Lite | 2024 | 0.826882 | 0.814291 | 0.777152 | -0.005582 | seed 不同，只作描述性比较 | `outputs/final_evidence_synthesis/isruc/paired_isruc_result_table.csv` |
| N1/N2 Pair SupCon | 42 | 0.830401 | 0.819073 | 0.781481 | -0.000800 | 补充消融 | `outputs/final_evidence_synthesis/isruc/paired_isruc_result_table.csv` |
| N1 context local contrast | 42 | 0.830248 | 0.819614 | 0.781461 | -0.000259 | 补充消融 | `outputs/final_evidence_synthesis/isruc/paired_isruc_result_table.csv` |

`outputs/final_evidence_synthesis/isruc/recommended_ablation_structure.md` 明确指出 base multimodal epoch encoder 与 isolated sequence backbone 的严格配对消融缺失；不得用不同 split、预处理、selector 或 decoder 的历史 run 填补。HC-SeqLIME、MF-SeqLIME、structured neighborhood 与 boundary-adaptive sampling 属于 XAI 负结果，不是分类架构消融。

## Prediction reliability（正文正式结果）

| Endpoint | Result | 95% subject-bootstrap CI | 来源 |
|---|---:|---:|---|
| Validation Accuracy | 0.830154 | — | `outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/overall_metrics.json` |
| Validation Macro-F1 | 0.816125 | — | `outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/overall_metrics.json` |
| NLL | 0.424695 | [0.382625, 0.463887] | `outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/overall_metrics.json`；`outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/bootstrap_confidence_intervals.json` |
| Multiclass Brier | 0.241856 | [0.215918, 0.266965] | `outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/overall_metrics.json`；`outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/bootstrap_confidence_intervals.json` |
| ECE, 15 equal-mass, primary | 0.010686 | [0.007074, 0.021069] | `outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/overall_metrics.json`；`outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/bootstrap_confidence_intervals.json` |
| ECE, 15 equal-width, sensitivity | 0.009938 | — | `outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/overall_metrics.json` |
| AURC, discrete mean risk | 0.050646 | [0.039866, 0.063314] | `outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/overall_metrics.json`；`outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/bootstrap_confidence_intervals.json` |

以上均来自 ISRUC validation 的 13,000 个唯一 epoch、15 subjects、A1-zero seed42、uncalibrated Pre-HMM raw probabilities。Risk-coverage 全曲线来源为 `outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/risk_coverage.csv`，预先指定工作点来源为 `outputs/sleepxvit_inspired_audit/isruc_seed42/reliability_v1/coverage_operating_points.csv`。Coverage cutoff 不得称为部署或临床阈值。

## Canonical XAI 与解释可靠性（正文正式结果）

| Endpoint | Result | 范围 / 协议 | 来源 |
|---|---:|---|---|
| Local surrogate weighted R² | 0.811 | validation T0, frozen selected-30 | `docs/paper_xai/TABLE_XAI_RELIABILITY_RESULTS.md` |
| Mask-seed stability median | 0.906 | validation T0, 30 samples | `docs/paper_xai/TABLE_XAI_RELIABILITY_RESULTS.md` |
| Complete reliable coverage | 1/30 = 3.3% | validation T0 | `docs/paper_xai/TABLE_XAI_RELIABILITY_RESULTS.md` |
| Randomization-unreliable | 22/30 = 73.3% | validation T0 | `docs/paper_xai/TABLE_XAI_RELIABILITY_RESULTS.md` |
| Mean R7 similarity | 0.240 | 3 seeds | `docs/paper_xai/TABLE_XAI_RELIABILITY_RESULTS.md` |
| Mean permutation-null q95 | 0.104 | 1,000 permutations | `docs/paper_xai/TABLE_XAI_RELIABILITY_RESULTS.md` |
| Randomization seed passes | 0/3 | T0；统一审计 T1 亦为 0/3 | `docs/paper_xai/TABLE_XAI_RELIABILITY_RESULTS.md`；`docs/SEQUENCE_LIME_PROTOCOL_CONSISTENCY_REPORT.md` |

这些结果支持透明披露局部 surrogate fit、seed consistency 和 randomization failure，不支持 model faithfulness。Selected-30 seeds 3101/3102/3103 是重复测量，不得当作 90 个独立样本。

## 生理标注与盲评（辅助分析）

| Endpoint | Result | 论文解释 | 来源 |
|---|---:|---|---|
| Physiological concept coverage | 30/30 = 100.0% | 固定公式产生正有限派生证据；非临床正确率 | `docs/paper_xai/PHYSIOLOGY_AWARE_DISPLAY_REPORT.md` |
| Unsupported annotation | 0/30 = 0.0% | 自动规则审计 | `docs/paper_xai/PHYSIOLOGY_AWARE_DISPLAY_REPORT.md`；`outputs/physiology_aware_display_validation/isruc_seed42/physiology_metrics.csv` |
| Opposing-concept conflict | 3/30 = 10.0% | 冲突必须保留显示 | `docs/paper_xai/PHYSIOLOGY_AWARE_DISPLAY_REPORT.md`；`outputs/physiology_aware_display_validation/isruc_seed42/physiology_metrics.csv` |
| Dominant-channel consistency | 29/30 = 96.7% | attribution channel 与 top concept channel 描述性一致 | `docs/paper_xai/PHYSIOLOGY_AWARE_DISPLAY_REPORT.md`；`outputs/physiology_aware_display_validation/isruc_seed42/physiology_metrics.csv` |
| Cross-seed concept stability | 0.973 | frozen 30 cases | `docs/paper_xai/PHYSIOLOGY_AWARE_DISPLAY_REPORT.md`；`outputs/physiology_aware_display_validation/isruc_seed42/physiology_metrics.csv` |
| Attribution identity | 30/30 bitwise unchanged | annotation 为下游层 | `outputs/physiology_aware_display_validation/isruc_seed42/final_decision.json` |
| Genuine blinded ratings | 0 evaluators | 未完成人因验证，expert agreement 不可估 | `outputs/physiology_aware_display_validation/isruc_seed42/final_decision.json` |

“生理标注改善解释”“更易判断可靠性”“降低误读”均不得进入结果结论。

## Context Influence（辅助分析）

| Evidence | Result | 论文定位 | 来源 |
|---|---|---|---|
| Development smoke sample 8809 / S011 | 完成；baseline max diff 1.56e-6；no-op 0；repeat 0 | 工程可用性 | `docs/paper_xai/CONTEXT_INFLUENCE_PAPER_CONVERGENCE_REPORT.md` |
| Development smoke sample 58657 / S097 | 完成；baseline max diff 3.01e-6；no-op 0；repeat 0 | 工程可用性 | `docs/paper_xai/CONTEXT_INFLUENCE_PAPER_CONVERGENCE_REPORT.md` |
| 十项功能检查 | 10/10 PASS | 程序与关键科学操作正确 | `docs/paper_xai/CONTEXT_INFLUENCE_PAPER_CONVERGENCE_REPORT.md` |
| 正式执行策略 | batch=1, float32, `training=False`, Pre-HMM | 固定论文实现 | `docs/paper_xai/CONTEXT_INFLUENCE_PAPER_CONVERGENCE_REPORT.md` |
| 历史 batch shape | `BATCH_SHAPE_NUMERICAL_EQUIVALENCE_V1=FAIL` | 非论文阻塞项；不声称 batch invariance | `docs/paper_xai/CONTEXT_INFLUENCE_PAPER_CONVERGENCE_REPORT.md` |

当前没有更大规模 Context Influence profile，不能报告群体效应、offset 优越性或相对 LIME 的性能提升。
