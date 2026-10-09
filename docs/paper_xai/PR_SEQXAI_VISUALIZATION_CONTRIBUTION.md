# PR-SeqXAI Visualization Contribution

## Contribution definition

The visualization contribution is a **multi-layer explanation
representation**, not a larger number of plots, richer colors, or a
better-looking dashboard. The representation keeps three evidence types
separate so that a reader can distinguish a model query from a downstream
interpretation and from a warning about explanation quality.

## Layer 1: model-derived channel-temporal perturbation sensitivity

**Scientific question:** Which channel-by-time perturbations are locally
associated with the frozen target-class response under the specified
replacement rule?

This layer displays the unchanged Context-Preserving Classic Sequence-LIME
array, its sign, channel, temporal segment, target class, and common color
scale. It is the only layer produced by queries to the frozen model. It is
local and replacement-dependent; it is not causal importance or global model
reasoning.

## Layer 2: physiological concept annotation

**Scientific question:** Which prespecified raw-signal measurements provide a
physiological vocabulary for the displayed high-sensitivity regions?

This layer reports nine fixed EEG/EOG/EMG concepts computed by
`raw_physiology_v1`. It is an independent derived annotation. True stage is
not an annotation input, and predicted stage is display context only. The
layer can report `NONE` or `INSUFFICIENT_EVIDENCE`; conflicts are retained.
It cannot modify the sensitivity array or ranking.

## Layer 3: reliability and conflict disclosure

**Scientific question:** Which audit outcomes limit how strongly the displayed
map may be interpreted?

This layer places surrogate fit, perturbation response, mask-seed stability,
randomization status, mutually exclusive reliability state, and
unsupported/conflict warnings beside the map. It makes negative evidence
visible but does not claim that reliability awareness has improved for human
readers.

## Why this is a framework contribution

PR-SeqXAI defines typed interfaces and claim boundaries among the three
layers. Layer 1 is model-derived, Layer 2 is signal-derived, and Layer 3 is
audit-derived. The identity guard proves that Layers 2 and 3 do not alter
Layer 1. The representation therefore prevents semantic enrichment from
masquerading as a better attribution estimator and prevents attractive
examples from hiding failed reliability tests.

This separation is the innovative object that can be evaluated and reused.
Human usability remains an open empirical question because no evaluator data
exist.

## Caption rule

Every physiology-aware panel must state:

> Derived physiology-oriented annotation and reliability disclosure are
> downstream of an unchanged Local Perturbation Sensitivity map. Human and
> clinical validation were not performed.
