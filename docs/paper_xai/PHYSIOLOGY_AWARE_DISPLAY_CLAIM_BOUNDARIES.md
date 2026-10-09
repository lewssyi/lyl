# Physiology-aware Display Claim Boundaries

## Permitted current statement

The project has produced a reproducible, blinded evaluation package that
compares a conventional heatmap presentation with a physiology-annotated and
reliability-aware presentation while holding the underlying attribution
bitwise fixed.

Until genuine blinded ratings are complete, the package is a validation
instrument, not evidence of improved human interpretation.

## Conditionally permitted statement

Only if every prespecified human-validation gate passes:

> The physiology-annotated and reliability-aware presentation improved
> physiological interpretability and reliability awareness relative to a
> conventional LIME heatmap.

This statement concerns presentation, not attribution quality.

## Prohibited statements

The materials cannot support any of the following:

- the attribution is more model-faithful;
- the attribution algorithm outperforms LIME;
- the display reveals causal physiological mechanisms;
- the model, explanation, annotation, or system is clinically validated;
- the display improves classifier performance;
- the annotation reflects the model's internal reasoning;
- the reliability layer corrects a prediction;
- the annotation justifies explanation-guided training.

## Layer separation

The immutable order is:

`frozen model output → frozen Local Perturbation Sensitivity → reliability audit`

and independently:

`frozen Local Perturbation Sensitivity → raw_physiology_v1 activation → derived annotation`.

No arrow returns to attribution, inference, loss, attention, FiLM, or the
classifier. All CCA captions must state:
`Derived annotation; expert validation pending.`

## Current decision vocabulary

The only allowed final decisions are:

- `PHYSIOLOGY_AWARE_VISUALIZATION_VALIDATED`;
- `PHYSIOLOGY_AWARE_VISUALIZATION_PARTIAL`;
- `PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED`.

Without genuine blinded evaluator data, the decision is
`PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED`.
