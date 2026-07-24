# XAI Methods

We used **Context-Preserving Classic Sequence-LIME** as the sole canonical
display method. Its role was restricted to exploratory **Local Perturbation
Sensitivity** analysis of the frozen A1-zero sequence classifier. For a target
epoch, the method constructed channel-by-segment masks over 30 temporal
segments and three input channels while retaining the surrounding 75-epoch
context. Every sequence window covering the target epoch was reinferred by the
unchanged classifier. The resulting pre-HMM neural probabilities were
aggregated over overlapping windows so that the queried response followed the
same context and overlap convention as canonical inference.

A distance-weighted local Ridge surrogate related the binary perturbation masks
to the aggregated probability of the fixed target class. Its coefficients were
normalized and reported as Local Perturbation Sensitivity. The canonical
configuration contains no predicted-stage, true-stage, fixed stage-channel, or
clinical-concept prior; no zero, physiological, MF, or hierarchical fusion; and
no gradient or feedback path. The method never changes the classifier,
attention, prediction, post-processing, or normal inference.

Reliability was evaluated separately from sensitivity magnitude. The audit
included held-out local surrogate fit, insertion–deletion response,
mask-seed stability, full-model randomization at R7, and a 1,000-permutation
null. Samples were assigned mutually exclusive reliability states, including
locally explainable, low response, surrogate unreliable, and randomization
unreliable. Context-dominated status was available only as an auxiliary
HC-SeqLIME diagnostic and was not part of the canonical method.

Clinical Concept Annotation was downstream and optional. It mapped a
high-sensitivity region to `raw_physiology_v1` concept activation and then to a
derived annotation. The sensitivity array, concept activation, and annotation
were stored separately. Concepts could not modify sensitivity ranking, model
inference, or training. Feedback training was disabled and permanently
blocked; only forward-only head checks, zero-weight parity, and random or
shuffled dummy-target unit controls were permitted.
