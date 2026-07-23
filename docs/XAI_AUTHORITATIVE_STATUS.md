# XAI Authoritative Status

This file is the sole current scientific-status entry point for the frozen
ISRUC seed-42 A1-zero classifier. Historical reports and output directories are
preserved as provenance, but later protocol-consistency evidence supersedes
their incompatible model-randomization claims.

## Current status

- Classifier: frozen A1-zero checkpoint; read-only.
- Scientific status: `MODEL_FAITHFULNESS_NOT_ESTABLISHED`.
- Protocol audit: `CASE_PROTOCOL_DRIFT`.
- Unified Classic Sequence-LIME randomization: T0 `0/3`, T1 `0/3`.
- Classic Sequence-LIME generalization: not supported.
- Current method boundary:
  `MF_SEQLIME_PARTIAL_PASS_SUPERSEDED_BY_PROTOCOL_CONSISTENCY_AUDIT`.
- Feedback training: `BLOCKED`.
- Explanation-guided training: prohibited.
- Clinical Concept Annotation: a separate derived annotation only; it cannot
  alter model attribution, prediction, ranking, or training.

`PAPER_LEVEL_INTERPRETABILITY_FAIL`, historical Classic 3/3 results, and the
former `MF_SEQLIME_PARTIAL_PASS` remain historical records. They are not the
current scientific conclusion and cannot justify a model-faithfulness or
paper-level method claim.

## Evidence priority

1. `outputs/xai_authoritative_status/isruc_seed42/CURRENT_XAI_STATUS.json`
2. `docs/XAI_CURRENT_PAPER_CONCLUSION.md`
3. `docs/SEQUENCE_LIME_PROTOCOL_CONSISTENCY_REPORT.md`
4. `outputs/sequence_lime_protocol_consistency_audit/isruc_seed42/`

The protocol-consistency audit completed 288/288 unified A/C shards. Both
independent 12-sample groups passed 0/3 for T0 and 0/3 for T1, and the A-C
bootstrap intervals included zero. The conditional old fixed-36 rerun was not
triggered. Canonical batch-16 parity, sequence-wrapper parity, the 1,200-forward
memory guard, and 95/95 maintained checks passed.

## Allowed and prohibited use

Future post-hoc feasibility work may query the frozen model and use isolated
development/validation/test splits. It must not overwrite historical formal
outputs, select on test data, weaken randomization gates, use stage priors,
modify normal inference, feed explanations into loss, or start feedback
training. A new explanation prototype remains unvalidated until it passes its
preregistered validation and fixed-test gates.

Authoritative correction base: branch `codex/a1-zero-mainline`, commit
`72b4258fafa32f6579e7eb4de9c2bd6621cc70f1`.
