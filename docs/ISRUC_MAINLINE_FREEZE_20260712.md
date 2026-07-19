# ISRUC Mainline Freeze (2026-07-12)

## Decision

**ISRUC classification-module search is frozen. A1-zero seed42 is the canonical ISRUC classification mainline.**

Canonical test metrics are Accuracy 0.830860, Macro-F1 0.819873, Kappa 0.782580, N1-F1 0.594013, and REM-F1 0.899167 on 13,072 fixed test epochs.

## Stopped Classification Directions

Direct clinical concept feedback, SCR classification routing, Epoch-UTF-Lite, P1 transfer as a performance route, N1/N2 Pair SupCon, N1 context local contrast, N1 multi-prototype decoding, simple probability ensembles, multi-teacher distillation, and complementarity gating are stopped as performance routes.

## Retained Mechanism Evidence

Stage-specific concept responsibility, physiological plausibility, representation misclustering, N1 contextual multimodality, oracle model complementarity, and persistent shared-error analysis remain valid analysis results within their audited scope.

## Claims

Allowed claims must follow `claim_evidence_matrix.csv`. In particular, ISRUC seed42 representation misclustering and N1 contextual multimodality are supported. Direct concept classification gains, confidence-independent trust, deployable complementarity, and distillation feasibility are unsupported. The statement that real LIME attribution is fed into training is false/forbidden for the current mainline; LIME remains post-hoc.

## Minimal Remaining Work

Only fixed-protocol seed completion, strict C0 baselines, necessary structural ablations after implementation review, SleepEDF counterpart runs, and final statistical aggregation are allowed. No new classification module or performance sweep is authorized.

## Boundary

Future work must not introduce SCR-v2, new contrastive losses, concept feedback, distillation, gating, prototype heads, preprocessing sweeps, or new temporal/frequency modules under the frozen classification mainline.
