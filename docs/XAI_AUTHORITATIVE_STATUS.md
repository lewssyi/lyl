# XAI Authoritative Status

This file is the sole current scientific-status entry point for the frozen
ISRUC seed-42 A1-zero classifier. Historical reports and output directories are
preserved as provenance, but later protocol-consistency evidence supersedes
their incompatible model-randomization claims.

## Current status

- Framework display name: **Physiology-Aware and Reliability-Controlled
  Post-hoc Sequence Explanation Framework (PR-SeqXAI)**.
- Framework definition: Context-Preserving Classic Sequence-LIME +
  independent Clinical Concept Annotation + Reliability-Aware Visualization.
- Engineering status: `IMPLEMENTATION_AND_AUDIT_PIPELINE_VALID`.
- Sequence adaptation status:
  `CONTEXT_PRESERVING_SEQUENCE_INFERENCE_ESTABLISHED`.
- Physiology annotation status:
  `AUTOMATIC_PHYSIOLOGY_CONSISTENCY_SUPPORTED`.
- Visualization status:
  `PHYSIOLOGY_AND_RELIABILITY_ENHANCED_PRESENTATION_IMPLEMENTED`.
- Human interpretability status: `NOT_EVALUATED`.
- Clinical validation status: `NOT_VALIDATED`.
- Model faithfulness status: `NOT_ESTABLISHED`.
- Classifier: frozen A1-zero checkpoint; read-only.
- Scientific status: `MODEL_FAITHFULNESS_NOT_ESTABLISHED`.
- Canonical post-hoc method:
  `CONTEXT_PRESERVING_CLASSIC_SEQUENCE_LIME`.
- Display name: **Context-Preserving Classic Sequence-LIME**.
- Method role: `EXPLORATORY_LOCAL_PERTURBATION_SENSITIVITY`.
- MF-SeqLIME status: `SUPERSEDED_HISTORICAL_CANDIDATE`.
- HC-SeqLIME status: `VALIDATION_FAILED_HISTORICAL_CANDIDATE`.
- Clinical annotation status:
  `IMPLEMENTATION_VALID_EXPERT_VALIDATION_PENDING`.
- Protocol audit: `CASE_PROTOCOL_DRIFT`.
- Unified Classic Sequence-LIME randomization: T0 `0/3`, T1 `0/3`.
- Classic Sequence-LIME generalization: not supported.
- HC-SeqLIME feasibility:
  `VALIDATION_NO_QUALIFIED_HIERARCHICAL_CANDIDATE`.
- HC-SeqLIME fixed-test execution count: `0`.
- HC-SeqLIME paper-level decision: `HC_SEQLIME_FAIL`.
- Feedback training: `PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD`.
- Explanation-guided training: prohibited.
- Clinical Concept Annotation: a separate derived annotation only; it cannot
  alter model attribution, prediction, ranking, or training.
- Physiology-aware visualization decision:
  `PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED`.

The visualization decision records that genuine human evaluation has not been
performed. It does not mark the engineering pipeline, automatic physiology
audit, or display implementation as failed. PR-SeqXAI is a framework name,
not a new attribution algorithm; the underlying attribution name and role
remain unchanged.

`PAPER_LEVEL_INTERPRETABILITY_FAIL`, historical Classic 3/3 results, and the
former `MF_SEQLIME_PARTIAL_PASS` remain historical records. They are not the
current scientific conclusion and cannot justify a model-faithfulness or
paper-level method claim.

## Evidence priority

1. `outputs/xai_authoritative_status/isruc_seed42/CURRENT_XAI_STATUS.json`
2. `docs/paper_xai/PR_SEQXAI_FINAL_POSITIONING_REPORT.md`
3. `docs/paper_xai/PHYSIOLOGY_AWARE_DISPLAY_REPORT.md`
4. `docs/XAI_CURRENT_PAPER_CONCLUSION.md`
5. `docs/HIERARCHICAL_CONTEXT_XAI_FEASIBILITY_REPORT.md`
6. `outputs/hierarchical_context_xai_feasibility/isruc_seed42/`
7. `docs/SEQUENCE_LIME_PROTOCOL_CONSISTENCY_REPORT.md`
8. `outputs/sequence_lime_protocol_consistency_audit/isruc_seed42/`

The protocol-consistency audit completed 288/288 unified A/C shards. Both
independent 12-sample groups passed 0/3 for T0 and 0/3 for T1, and the A-C
bootstrap intervals included zero. The conditional old fixed-36 rerun was not
triggered. Canonical batch-16 parity, sequence-wrapper parity, the 1,200-forward
memory guard, and 95/95 maintained checks passed.

The later HC-SeqLIME study completed 1,080/1,080 validation jobs under equal
1,530-query budgets. H3-G25-F45 and H3-G9-F45 both passed 0/3 T0 and 0/3 T1
seeds, retained R7 similarity above their 1,000-permutation null q95, and were
inferior to Classic-90 on joint faithfulness and seed stability. No candidate
passed all ten validation gates, so the fixed 36-sample test and CCA rerun were
not executed. Final isolated parity and memory guards passed, as did 122/122
maintained checks. The final freeze added 30 focused guard checks and completed
152/152 maintained checks in total (131 unittest checks plus 21 pytest checks).

At final freeze, the server NVIDIA user-space library (580.173) did not match
the loaded kernel driver (580.159), so a same-backend GPU rerun was not
possible. A CPU batch-16 diagnostic preserved 100% argmax agreement, zero
Accuracy/Macro-F1/Kappa differences, and wrapper parity, but its probability
rounding exceeded the GPU artifact threshold; it is isolated as a
non-authoritative backend diagnostic. The authoritative GPU batch-16 parity
and 1,200-forward memory guards remain the frozen evidence.

## Allowed and prohibited use

The frozen paper method is a post-hoc local perturbation sensitivity analysis:
it applies channel-by-segment perturbations to the target epoch inside a
75-epoch context-preserving wrapper, reruns every covering sequence window
through the frozen A1-zero model, aggregates overlapping pre-HMM neural
probabilities, and fits a weighted local Ridge surrogate. It uses no stage or
concept prior, fusion, hierarchical context attribution, gradient, or
explanation feedback.

Future post-hoc work may query the frozen model and use isolated
development/validation/test splits. It must not overwrite historical outputs,
select on test data, weaken gates, use stage priors, modify normal inference,
feed explanations into loss, or start feedback training. The canonical method
must not be described as Model-Faithful, Validated Model Attribution, or Causal
Attribution.

Authoritative correction base: branch `codex/a1-zero-mainline`, commit
`72b4258fafa32f6579e7eb4de9c2bd6621cc70f1`.
