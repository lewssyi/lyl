# XAI Limitations

Authoritative model faithfulness has not been established. In particular,
high local surrogate fit and mask-seed stability did not imply successful
full-model randomization: R7 similarity remained above the permutation-null
q95 for all three canonical T0 seeds. Reliable coverage was therefore low, and
individual maps outside the reliable subset should be read as diagnostic
perturbation responses rather than dependable model explanations.

The estimates are replacement-dependent. Changing the perturbation baseline,
mask granularity, channel grouping, or temporal segmentation changes the local
estimand. This study freezes one implementation for reproducibility but does
not show that it is uniquely correct. Results also depend on a local linear
surrogate and cannot be extrapolated to global behavior.

Clinical Concept Annotation remains an implementation-valid derived layer
without clinical expert validation. A concept label is neither a causal
explanation nor evidence that the classifier used that physiological concept.
No clinical annotation modifies sensitivity, ranking, prediction, or model
parameters.

The analysis is post-hoc only. It does not provide gradients, training
supervision, prediction correction, or performance improvement. Explanation
feedback is permanently blocked for the current method. Negative MF and
hierarchical studies remain informative failure analyses but cannot be used to
claim superiority over Classic Sequence-LIME.
