# Physiology-aware and Reliability-aware Display Validation Protocol

## Registration status

This protocol is frozen before generation of the display-validation package.
It evaluates presentation and reader judgment, not a new attribution method.
No LIME masks, surrogate, groups, model inference, classifier parameter,
checkpoint, or feedback path may be changed or executed.

The frozen scientific boundary remains:

- canonical method: `CONTEXT_PRESERVING_CLASSIC_SEQUENCE_LIME`;
- method role: `EXPLORATORY_LOCAL_PERTURBATION_SENSITIVITY`;
- scientific status: `MODEL_FAITHFULNESS_NOT_ESTABLISHED`;
- feedback status: `PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD`;
- CCA status: `IMPLEMENTATION_VALID_EXPERT_VALIDATION_PENDING`.

## Study question

The paired study asks whether adding a downstream physiological annotation
and reliability context to an unchanged heatmap improves physiological
readability, reliability judgment, objective task accuracy, and resistance to
misinterpretation. It does not test model faithfulness, attribution
superiority, causal physiology, model performance, or clinical validity.

## Frozen inputs

All cases come from the existing 30-case validation pool at
`outputs/hierarchical_context_xai_feasibility/isruc_seed42/`. The displayed
prediction, target class, channel-by-segment sensitivity, ranking, and model
output are read from the already completed authoritative artifacts.
`R0/seed3101/T0` is the display attribution. Existing R0 seeds 3101, 3102 and
3103 are used only for the prespecified concept-stability audit.

No probability or attribution is recomputed. The cached validation waveform
is read only to apply the already implemented, stage-independent
`raw_physiology_v1` annotation mapping. True stage is excluded from annotation
generation and may appear only in the locked analysis table after the display
package is generated.

## Paired displays

Each anonymous case has two presentations with an identical heatmap:

- Base presentation: prediction, EEG/EOG/EMG channel-by-segment Local
  Perturbation Sensitivity heatmap, and a signed sensitivity legend. If the
  frozen artifact contains no negative entries, the negative half of the
  shared scale remains visible and is explicitly empty.
- Context presentation: the same prediction and bitwise-identical heatmap,
  plus derived clinical concept annotation, reliability state, surrogate
  fidelity, faithfulness, seed stability, randomization status,
  target/context summary, and an unsupported-evidence warning.

The context presentation cannot reorder, rescale independently, recolor
independently, suppress, or modify a heatmap value. Clinical annotation,
reliability text, and warnings are downstream display layers.

## Blinding and allocation

The evaluator-facing materials use only anonymous case IDs and `Panel A` /
`Panel B`. They contain none of: Classic, Enhanced, New method, Our method,
V0, V1, or the research hypothesis. A fixed seed of `20260726` generates a
balanced random allocation: 15 cases have the context presentation in A and
15 in B. The restricted allocation manifest is not included in the
evaluator-facing package.

Within each case, both panels use the same color limits, axes, prediction, and
heatmap bytes. Case order is randomized once with the same fixed seed.

## Evaluator tasks

For each panel, evaluators record:

1. the channel receiving the greatest attention;
2. the most important time region;
3. whether a physiological pattern can be identified;
4. the physiological pattern category;
5. whether the explanation is reliable;
6. whether the explanation could mislead;
7. whether the display helps analyze an incorrect prediction.

The response form also records completion time, confidence, and 1–5 ratings
for physiological plausibility, interpretability, information clarity,
reliability awareness, error-analysis usefulness, clinical/research
usefulness, risk of misleading, and overall preference. Risk of misleading
is analyzed separately and is never added in the same direction to a total
score.

## Evaluators and evidence status

Priority evaluators are sleep-medicine physicians, researchers experienced
in manual PSG staging, and trained graduate students. The target pilot is at
least two to three evaluators with sleep-staging experience. Evaluator
background, experience, involvement in model development, and knowledge of
method identity must be reported.

Until genuine blinded ratings exist, all rating fields remain blank and the
only permitted decision is
`PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED`. Scores, completion times,
preferences, and agreement statistics must never be simulated or imputed.

## Automatic guards

The package must pass all of the following:

- base/context attribution arrays are bitwise identical;
- rank order is identical;
- the same shared color limits are used;
- annotation and reliability metadata cannot alter attribution;
- true stage is absent from annotation inputs;
- the A/B allocation is reproducible and balanced;
- evaluator-facing materials hide method identity;
- unreliable and NONE/INSUFFICIENT_EVIDENCE cases are retained;
- no GPU, model, inference, or feedback code path is invoked;
- the canonical attribution source hash is preserved;
- CPU diagnostic artifacts cannot enter a formal panel;
- every required artifact is nonempty.

## GPU boundary

The NVIDIA user-space/kernel mismatch is an environment limitation, not an
authorization to recompute results. If a required panel input is absent from
the frozen artifacts, that field is recorded as
`BLOCKED_BY_GPU_BACKEND_MISMATCH`; CPU probabilities cannot replace
authoritative GPU probabilities.
