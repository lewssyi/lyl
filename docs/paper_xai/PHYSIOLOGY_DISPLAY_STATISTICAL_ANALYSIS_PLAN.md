# Statistical Analysis Plan: Physiology-aware Display

## Evidence stage

This is a preregistered pilot expert-validation plan. Automatic annotation
audits may be reported immediately, but no human-comparison effect is
estimated until genuine blinded ratings are entered. Empty templates are not
data.

## Analysis populations

- Main display set: all 30 frozen validation cases.
- Pilot set: deterministic 10-case coverage subset.
- Primary human population: evaluators with documented sleep-staging
  experience.
- Sensitivity population: all eligible evaluators, stratified by background.

Cases and evaluators are paired across the two anonymous panels. The
restricted allocation key is joined only after data lock.

## Primary endpoints

1. Objective physiological-pattern task correctness.
2. Reliability-judgment accuracy relative to the frozen reliability label.
3. Misinterpretation rate.
4. Completion time.

Secondary endpoints are the eight 1–5 ratings. `risk_of_misleading` is a
lower-is-better endpoint and is analyzed independently; it is never summed in
the same direction as the seven higher-is-better ratings.

## Paired comparisons

For binary endpoints, report the paired difference in proportions and an
evaluator-clustered bootstrap 95% confidence interval. Use McNemar's exact
test when the number of discordant pairs permits. For ordinal ratings, report
the paired median difference, a paired bootstrap interval, and the
rank-biserial effect size from a two-sided Wilcoxon signed-rank comparison.
For completion time, analyze paired log time, report the geometric mean ratio
and a paired bootstrap 95% interval, and retain the raw median and IQR.

Bootstrap resampling uses 10,000 replicates, sampling evaluators first and
cases within evaluator to preserve clustering. The seed is `20260726`.

## Agreement

Report Fleiss' kappa for categorical reliability and pattern judgments when
three or more evaluators are available; with two evaluators, report Cohen's
kappa. Report a two-way random-effects, absolute-agreement ICC for ordinal
ratings when estimable. Always include raw percent agreement and the number
of raters/cases. Do not report an agreement number when no genuine ratings
exist.

## Multiple comparisons

The four primary endpoints form one family controlled with Holm correction
at two-sided alpha 0.05. The eight subjective ratings form an exploratory
family with Benjamini–Hochberg FDR at 0.05. Automatic annotation audits are
descriptive and are not treated as evidence that the display improves human
interpretation.

## Missing data

No rating, time, or answer is imputed. The primary paired analysis includes
only case-evaluator pairs with both panels complete for that endpoint.
Missingness is reported by panel, case, evaluator, and reason. A sensitivity
analysis treats missing binary answers as incorrect only when the evaluator
opened the panel and explicitly abandoned the item.

## Automatic physiology metrics

All thresholds and definitions are fixed before computation:

- `physiological_concept_coverage`: fraction of cases with at least one
  finite positive derived concept-evidence score.
- `unsupported_annotation_rate`: fraction of cases for which no finite
  positive evidence exists; these display `NONE / INSUFFICIENT_EVIDENCE`.
- `annotation_conflict_rate`: fraction with both members of an opposing pair
  (EOG Slow/Rapid or EMG Low/Active) at least 80% of the larger score.
- `channel_concept_consistency`: fraction whose highest-evidence concept
  belongs to the attribution-dominant channel.
- `top_region_evidence_rate`: fraction of top-10 attribution regions with a
  positive raw-physiology activation on the same channel and segment.
- `cross_mask_seed_concept_stability`: mean pairwise Spearman correlation of
  the nine concept-evidence scores from frozen R0 seeds 3101/3102/3103.
- `stage_concept_agreement`: reporting-only fraction whose top concept is in
  the prespecified true-stage set: W={Alpha, Beta, EMG Active},
  N1={Theta, EOG Slow}, N2={Theta, Sigma}, N3={Delta},
  REM={EOG Rapid, EMG Low}. True stage is joined only after annotation.
- `concept_sparsity`: fraction of nine concept-evidence entries that are
  non-finite or non-positive.

Per-case metrics and aggregate numerators, denominators, means, and bootstrap
95% intervals are retained. Stage-concept agreement is a descriptive
stratification and cannot alter or validate an annotation.

## Decision rule

`PHYSIOLOGY_AWARE_VISUALIZATION_VALIDATED` requires genuine blinded ratings,
randomized order, unchanged attribution, improved physiological
interpretability, improved reliability accuracy, non-inferior
misinterpretation, and complete annotation audit reporting.

With no genuine blinded ratings, the forced decision is
`PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED`. Automatic metrics alone
cannot produce `VALIDATED`.
