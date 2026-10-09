# XAI Methods: PR-SeqXAI

PR-SeqXAI denotes the **Physiology-Aware and Reliability-Controlled Post-hoc
Sequence Explanation Framework**. It is a composition of three independently
typed layers and is not a new attribution estimator.

## Model-derived sensitivity layer

**Context-Preserving Classic Sequence-LIME** is the sole canonical
attribution. Its role is restricted to exploratory **Local Perturbation
Sensitivity** for the frozen A1-zero sequence classifier. For a target epoch,
the method constructs channel-by-segment masks over 30 temporal segments and
three channels while retaining the surrounding 75-epoch context. The frozen
artifact was produced by rebuilding every sequence window covering the target
epoch and aggregating target-position pre-HMM neural probabilities across
overlapping windows.

A distance-weighted Ridge surrogate relates binary perturbation masks to the
aggregated probability of the fixed target class. The canonical configuration
contains no predicted-stage, true-stage, fixed stage-channel, or clinical-
concept prior; no MF or hierarchical fusion; and no gradient or feedback path.

## Signal-derived annotation layer

Clinical Concept Annotation is downstream and optional. It maps
`raw_physiology_v1` measurements for nine EEG/EOG/EMG concepts to the frozen
high-sensitivity regions. Sensitivity, concept activation, and annotation are
stored separately. True stage is not used to generate annotation, and
predicted stage is display context only. Unsupported evidence and opposing-
concept conflicts are reported. The layer cannot change sensitivity, ranking,
prediction, or model parameters.

## Audit-derived reliability layer

Reliability is evaluated separately from sensitivity magnitude. The frozen
audit includes held-out surrogate fit, insertion/deletion response, mask-seed
stability, full-model R7 randomization, and a 1,000-permutation null. Mutually
exclusive states include locally explainable, low response, surrogate
unreliable, and randomization unreliable. The visualization places these
states and conflict warnings beside, rather than inside, the attribution.

Normal inference does not read PR-SeqXAI artifacts. Feedback training is
disabled and `PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD`.
