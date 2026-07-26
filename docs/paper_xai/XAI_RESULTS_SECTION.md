# XAI Results

PR-SeqXAI was reproducible at the engineering and sequence-query levels.
Canonical probabilities and argmax labels were unchanged, the 75-epoch
wrapper differed by at most \(5.96\times10^{-8}\), the checkpoint hash was
unchanged, and the fixed-graph 1,200-forward guard showed zero current-memory
growth. These checks establish implementation validity and
overlapping-window consistency, not scientific model faithfulness.

For the canonical validation T0 analysis (30 frozen samples), the local
surrogate weighted \(R^2\) was 0.811 and median mask-seed Spearman stability was
0.906. The mean R7 attribution similarity across three randomization seeds was
0.240, whereas the mean 1,000-permutation-null q95 was 0.104. Consequently,
none of the three seeds passed the authoritative randomization criterion.
Only 1 of 30 samples (3.3%) met the complete reliability rule. The remaining
samples were predominantly randomization-unreliable, with additional
low-response and surrogate-unreliable cases. Stratified counts by sleep stage,
correctness, and transition status are reported without pooling protocols.

The unified protocol-consistency audit yielded 0/3 passing seeds for both T0
and T1. Thus, `MODEL_FAITHFULNESS_NOT_ESTABLISHED` remains the authoritative
scientific status. MF-SeqLIME is a superseded historical candidate. Structured
neighborhood and boundary-adaptive variants admitted no qualified candidate.
HC-G9 and HC-G25 each passed 0/3 T0 and 0/3 T1 validation seeds; therefore the
frozen 36-sample test was not run and HC-SeqLIME was classified as a validation
failed historical candidate.

The independent physiology audit used 30 frozen validation cases without new
model inference. The base and enhanced displays retained bitwise-identical
attributions, rankings, and color scales. Concept coverage was 100.0%,
unsupported annotation was 0.0%, opposing-concept conflict was 10.0%,
channel-concept consistency was 96.7%, and cross-seed concept stability was
0.973. These automatic endpoints measure coverage, consistency, and
stability; they do not establish clinical correctness, human interpretability,
or model faithfulness.

The case-study panel follows a deterministic validation-only protocol.
Selected sample IDs were 7861 (relatively reliable), 3080
(randomization-unreliable), 178 (incorrect/transition priority), and 355
(auxiliary HC context-dominated flag, with canonical B0 sensitivity
displayed). Conflict cases and unreliable states are retained rather than
filtered. The cases illustrate framework outputs and audit boundaries rather
than physiological correctness or causal mechanisms.

No genuine human or expert evaluation was performed. The 30-case blinded
package is an evaluation instrument, not evidence that the enhanced
presentation is easier to interpret, preferred, clinically useful, or less
likely to be misread.
