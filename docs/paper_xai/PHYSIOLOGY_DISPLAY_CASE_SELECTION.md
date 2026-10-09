# Physiology-aware Display Case Selection

## Locked case pool

The main evaluation package contains all 30 unique cases in the frozen
validation reliability pool. Using the complete pool avoids an aesthetic or
narrative-based sample filter. The pool was fixed before display generation
and does not use the frozen 36-case test set.

Each record preserves:

- anonymous case ID and source sample ID;
- source recording and epoch within recording;
- true stage for locked final stratification only;
- displayed prediction;
- correctness and transition status;
- canonical reliability state;
- any auxiliary context-dominated audit flag;
- attribution-dominant channel;
- selection rule and source artifact hash.

## Main package

All 30 validation cases are retained. This guarantees the available coverage
of W/N1/N2/N3/REM, correct/incorrect, transition/stable, locally explainable,
randomization-unreliable, surrogate-unreliable, low-response, and
EEG/EOG/EMG-dominant cases. A historical hierarchical audit may supply an
auxiliary context-dominated selection flag, but it never changes the
canonical B0 reliability label or the displayed attribution.

## Pilot package

The 10-case pilot is a deterministic subset of the main package. Starting
from an empty set, the selector repeatedly chooses the lowest-sample-ID case
that maximizes the number of uncovered tokens:

`stage`, `correctness`, `transition status`, `canonical reliability state`,
`dominant channel`, and `auxiliary context-dominated flag`.

After all available tokens are covered or no further gain is possible,
lowest sample IDs fill the remaining slots. The algorithm never reads a
rendered image, annotation label, physiological plausibility, or anticipated
paper narrative.

## Attribution and channel dominance

The display vector is the frozen B0 `R0/seed3101/T0` 90-feature vector. It is
reshaped into EEG/EOG/EMG by 30 segments without transformation. Dominant
channel is the largest channel sum of absolute frozen sensitivity. Ranking
uses stable descending absolute sensitivity. Neither calculation changes the
stored vector.

## Exclusions

Cases are excluded only if a required frozen attribution file is missing,
malformed, non-finite, not 90 elements, or fails source-hash validation.
Unreliable, incorrect, transition, low-response, unsupported-annotation, and
visually sparse cases are never excluded for those properties.

The following are prohibited selection variables:

- whether a heatmap is attractive or easy to narrate;
- whether a physiological pattern matches expectation;
- whether the case supports the research hypothesis;
- true-stage agreement of the annotation;
- evaluator outcome or preference.

If a requested stratum has no eligible case, the manifest must say:
`No reliable case available under the authoritative criterion.`
