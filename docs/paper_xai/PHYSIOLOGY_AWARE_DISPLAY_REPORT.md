# Physiology-aware and Reliability-aware Display Report

## Decision

`PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED`

The frozen 30-case validation package, 10-case pilot subset, automatic
physiology audit, blank response templates, and randomized blinded pairs are
complete. No genuine evaluator ratings exist, so no claim about improved
interpretability, reliability judgment, completion accuracy, completion
time, preference, or reduced misinterpretation is permitted.

## Automatic audit

The audit used only the existing canonical B0 R0/T0 attribution artifacts,
existing R0 seeds 3101/3102/3103, and the read-only validation waveform
cache. It did not load a model, call inference, start a LIME experiment, use
the frozen 36-case test set, or use a CPU probability as an authoritative
substitute.

Across 30 cases:

- physiological concept coverage: 30/30, 100.0%;
- unsupported annotation rate: 0/30, 0.0%;
- annotation conflict rate: 3/30, 10.0%;
- attribution-dominant channel / top-concept channel consistency:
  29/30, 96.7%;
- mean top-region evidence rate: 100.0%;
- mean cross-mask-seed concept stability: 0.973;
- reporting-only true-stage concept agreement: 11/30, 36.7%;
- mean concept sparsity: 40.0%.

Coverage means that the fixed formula produced at least one positive finite
derived evidence value; it is not clinical correctness. The modest
true-stage agreement and three opposing-concept conflicts reinforce the
expert-validation requirement.

## Required questions

1. **Are the V0 and V1 attributions identical?**
   Yes. All 30 pairs passed bitwise identity, stable ranking identity, and
   shared-color-limit guards. CCA and reliability fields are downstream only.

2. **What is physiological annotation coverage?**
   30/30 cases, or 100.0%, under the preregistered positive finite evidence
   definition.

3. **What are the unsupported and conflict rates?**
   Unsupported: 0/30 (0.0%). Opposing-concept conflict: 3/30 (10.0%).

4. **Is the context presentation easier for identifying physiology?**
   Not yet known. No real blinded evaluator responses are available.

5. **Is reliability easier to judge?**
   Not yet known. Reliability-judgment accuracy awaits genuine paired ratings.

6. **Does the context presentation reduce misinterpretation?**
   Not yet known. No misinterpretation outcome has been observed.

7. **Was a genuine blinded evaluation completed?**
   No. A complete anonymous 30-case package and 10-case pilot subset exist,
   but all response templates remain blank.

8. **What is expert agreement?**
   Not estimable; evaluator count is zero and no agreement value was imputed.

9. **Can the display be called clinically validated?**
   No. CCA remains `IMPLEMENTATION_VALID_EXPERT_VALIDATION_PENDING`.

10. **Can the attribution be called model-faithful?**
    No. The authoritative status remains
    `MODEL_FAITHFULNESS_NOT_ESTABLISHED`.

11. **Can this be a paper contribution?**
    It can be presented as a reproducible evaluation instrument and
    explanation-display design with explicit failure and reliability
    boundaries. Human benefit cannot be claimed before the blinded study.

12. **Is this algorithm innovation or display innovation?**
    Display and evaluation-protocol innovation. The underlying attribution
    algorithm is unchanged and no superiority experiment was run.

13. **Does the GPU backend issue affect this conclusion?**
    No new GPU result was required. The package reuses hashed attribution,
    prediction, reliability, and waveform-cache artifacts. The
    non-authoritative CPU diagnostic is excluded. Any missing model-derived
    field would have been marked `BLOCKED_BY_GPU_BACKEND_MISMATCH`.

14. **What is the final scientific decision?**
    `PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED`, because automatic
    metrics and a blinded package cannot replace genuine blinded evaluation.

## Paper positioning

The current defensible statement is:

> We constructed a blinded, paired physiology-annotated and
> reliability-aware display evaluation while holding the underlying Local
> Perturbation Sensitivity bitwise fixed.

The stronger statement that the presentation improves physiological
interpretability or reliability awareness remains prospective. It may be
used only after every preregistered human-validation gate passes.

The display cannot be described as a more model-faithful attribution, a
superior LIME algorithm, a causal physiological explanation, a performance
module, or a clinically validated system.

## Verification and provenance

- All 30 anonymous cases have readable PDF and PNG pairs; the evaluator
  package contains 92 nonempty files and 30 readable PDFs.
- The isolated result directory contains 198 nonempty files, including the
  source-hash manifest, attribution-identity guard, automatic metrics,
  randomized allocation, blank response templates, and final decision.
- The dedicated display guards pass 22/22 checks. The complete repository
  regression passes 153/153 `unittest` checks and 21/21 `pytest` checks
  (174 total).
- Figure A remains a native-vector fallback and visibly states `Frozen
  Model`, `Pre-HMM Neural Response`, `No Gradient`, `No Training Feedback`,
  and `Post-hoc Only`.
- The NVIDIA userspace/kernel mismatch appears only as an existing
  CPU-fallback diagnostic warning during regression testing. No new
  attribution, model inference, LIME run, classifier change, or feedback
  training was performed.
