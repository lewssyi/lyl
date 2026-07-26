# Researcher Scoring and User-task Instructions

## Before distribution

Distribute only `docs/paper_xai/expert_evaluation_package/`. Do not distribute
the restricted allocation manifest or scoring key under
`outputs/physiology_aware_display_validation/isruc_seed42/`.

Assign each evaluator an anonymous ID. Record background, staging experience,
model-development involvement, and whether the evaluator knows panel
identity. For the pilot, use the 10 case IDs frozen in
`PACKAGE_MANIFEST.json`.

## Evaluator sequence

For every case:

1. start a timer before opening Panel A;
2. complete all objective tasks and ratings for Panel A;
3. lock Panel A responses;
4. start a new timer for Panel B;
5. complete Panel B;
6. record confidence and optional rationale;
7. do not disclose the allocation until all ratings are locked.

## Locked scoring

After data lock, join responses to
`scoring_key_restricted.json`. Score main channel and top time region against
the immutable attribution-derived key. Score physiological category against
the stage-independent derived annotation key; `NONE` is a valid answer when
the key is `NONE / INSUFFICIENT_EVIDENCE`.

Reliability judgment is correct only when it matches the frozen reliability
criterion. Misinterpretation is coded from the preregistered answer rule, not
from reviewer preference. Retain raw responses alongside every derived score.

Do not use true stage to rescore, revise, or adjudicate the annotation. True
stage is joined only for the reporting-only stage-concept agreement
stratification.

## Data integrity

- Never fill blank ratings on behalf of an evaluator.
- Never impute missing answers or times.
- Never remove unreliable or visually sparse cases.
- Never relabel Panel A or Panel B after allocation.
- Never inspect panel preference before the response database is locked.
- Analyze `risk_of_misleading` separately as lower-is-better.

## Reporting threshold

Describe the study as `pilot expert validation` when evaluator or case counts
are small. Do not use `PHYSIOLOGY_AWARE_VISUALIZATION_VALIDATED` unless every
hard gate in the preregistered protocol is satisfied.
