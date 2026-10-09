# XAI Figure Captions

**Figure A | Guarded PR-SeqXAI architecture.** The frozen A1-zero classifier
is queried through target-epoch perturbations in all covering 75-epoch windows.
Overlapping pre-HMM neural probabilities feed a weighted local surrogate and a
separate reliability audit. There is no gradient or training-feedback edge.
Clinical Concept Annotation is optional and downstream. **Derived annotation;
human and expert validation not performed.**

**Figure B | Reliability audit for Context-Preserving Classic Sequence-LIME.**
All four panels use the retained canonical B0 validation T0 protocol: local
surrogate fit, insertion–deletion response across mask seeds, mask-seed
stability, and R7 similarity against a 1,000-permutation null q95. The R7
result does not support model faithfulness.

**Figure C | Automatic physiology annotation and reliability coverage.** The
manuscript composite places the frozen automatic physiology-metric summary
beside the existing canonical validation T0 reliability-coverage panel.
Coverage, unsupported output, conflict, channel-concept consistency, and
cross-seed stability describe the automatic annotation rule; mutually
exclusive reliability states are stratified separately. These panels do not
report human interpretability, clinical validity, or model faithfulness.

**Figure D | Deterministically selected Local Perturbation Sensitivity cases.**
One relatively reliable, one randomization-unreliable, one
incorrect/transition-priority, and one auxiliary context-dominated case were
selected before plotting under the frozen validation-only protocol. The last
case uses HC status only for selection; the displayed map is canonical B0.
These maps are not causal importance or ground-truth explanations.
The sensitivity arrays and rankings are identical across base and enhanced
display variants.

**Figure E | Negative method-development record.** Protocol-separated outcomes
show why Classic remained a sensitivity reference and why MF, structured
sampling, HC-G9, and HC-G25 were not promoted. Numeric endpoints from
incompatible protocols are not pooled.

**Clinical Concept Annotation caption rule.** Any present or future CCA panel
must include: **Derived physiology-oriented annotation; unchanged Local
Perturbation Sensitivity; human and clinical validation not performed.**
