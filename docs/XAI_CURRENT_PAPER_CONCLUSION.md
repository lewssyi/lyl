# Current XAI Paper Conclusion

The frozen A1-zero classifier retains its established predictive evaluation,
but current evidence does not establish that Classic Sequence-LIME or the
superseded MF-SeqLIME candidate is reliably model-faithful. The authoritative
scientific status is `MODEL_FAITHFULNESS_NOT_ESTABLISHED`.

The prior apparent conflict between historical Classic Sequence-LIME 3/3
randomization and newer validation 0/3 was caused by material protocol drift.
Under one protocol-versioned evaluator, the historical diagnostic group and an
independent validation group both pass 0/3 T0 and 0/3 T1 seeds. R7 attribution
similarity remains above its 1,000-permutation null bound, despite marked decay
of faithfulness advantage. Accordingly, neither general model sensitivity nor
a paper-level explanation-method claim is supported.

The paper may report specified-replacement local perturbation behavior,
descriptive context dependence, negative randomization results, protocol
drift, and the need for reliability-aware coverage. It must not claim global
causal contribution, general explanation superiority, clinical validity,
model-faithful reasoning, or eligibility for explanation-guided training.

Any Hierarchical Context Attribution work is a new strictly post-hoc
feasibility study. It cannot revise this conclusion unless it passes the frozen
validation gates and, only then, a one-time fixed test. Until such evidence
exists, feedback training remains `BLOCKED`.
