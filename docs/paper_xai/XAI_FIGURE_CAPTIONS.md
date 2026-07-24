# XAI Figure Captions

**Figure A | Guarded post-hoc XAI architecture.** The frozen A1-zero classifier
is queried through target-epoch perturbations in all covering 75-epoch windows.
Overlapping pre-HMM neural probabilities feed a weighted local surrogate and a
separate reliability audit. There is no gradient or training-feedback edge.
Clinical Concept Annotation is optional and downstream. **Derived annotation;
expert validation pending.**

**Figure B | Reliability audit for Context-Preserving Classic Sequence-LIME.**
All four panels use the retained canonical B0 validation T0 protocol: local
surrogate fit, insertion–deletion response across mask seeds, mask-seed
stability, and R7 similarity against a 1,000-permutation null q95. The R7
result does not support model faithfulness.

**Figure C | Reliability coverage.** Mutually exclusive reliability states for
the canonical validation T0 pool are shown overall and stratified by true
stage, prediction correctness, and transition status. A zero-width category is
retained in the legend rather than omitted.

**Figure D | Deterministically selected Local Perturbation Sensitivity cases.**
One relatively reliable, one randomization-unreliable, one
incorrect/transition-priority, and one auxiliary context-dominated case were
selected before plotting under the frozen validation-only protocol. The last
case uses HC status only for selection; the displayed map is canonical B0.
These maps are not causal importance or ground-truth explanations.

**Figure E | Negative method-development record.** Protocol-separated outcomes
show why Classic remained a sensitivity reference and why MF, structured
sampling, HC-G9, and HC-G25 were not promoted. Numeric endpoints from
incompatible protocols are not pooled.

**Clinical Concept Annotation caption rule.** Any present or future CCA panel
must include: **Derived annotation; expert validation pending.**
