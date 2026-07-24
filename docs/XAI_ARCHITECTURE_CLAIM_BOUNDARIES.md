# XAI Architecture and Claim Boundaries

## Allowed post-hoc flow

```text
Frozen A1-zero Model
        ↓
Target Epoch
        ↓
Context-Preserving Perturbation
        ↓
Covering Sequence Windows
        ↓
Repeated Frozen Inference
        ↓
Weighted Local Surrogate
        ↓
Local Perturbation Sensitivity
        ↓
Reliability Audit
        ↓
Optional Clinical Annotation
```

The inference response is the **pre-HMM neural probability**. The path is
query-only and contains **No Training Feedback** and **No Gradient**. Normal
inference never reads an explanation artifact. There is no edge from LIME,
sensitivity, reliability, or annotation to a loss, classifier, attention
module, prediction correction, FiLM path, concept modulation path, or model
parameter.

The sole canonical display method is **Context-Preserving Classic
Sequence-LIME**, used only as exploratory local perturbation sensitivity
analysis. It perturbs target-epoch channel×segment features within the frozen
75-epoch wrapper, reinfers all covering windows, aggregates overlapping
probabilities, and fits a weighted local Ridge surrogate. It excludes clinical
priors, stage priors, fusion, MF-SeqLIME components, HC-SeqLIME components, and
feedback.

Clinical Concept Annotation is downstream and optional: sensitivity region →
`raw_physiology_v1` concept activation → derived annotation. It cannot alter
ranking, prediction, training, or causal interpretation. Every CCA figure
caption must state: **Derived annotation; expert validation pending.**

MF-SeqLIME is a superseded historical candidate. HC-SeqLIME is a validation
failed historical candidate and may appear only as a supplementary negative
study. Neither is part of the canonical architecture.
