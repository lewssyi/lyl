# PR-SeqXAI Results Narrative

## Evidence class A: sequence consistency

The first result is an engineering and query-semantics result. The canonical
batch-16 artifact parity guard passed, the 75-epoch sequence wrapper differed
by at most \(5.96\times10^{-8}\), the checkpoint hash remained unchanged, and
the 1,200-forward fixed-graph guard showed zero current-memory growth. These
checks establish that the post-hoc query path preserves the frozen
overlapping-window inference convention and is reproducible under the
authoritative backend.

The later CPU diagnostic retained 100% argmax agreement and wrapper parity but
was not probability-equivalent to the authoritative GPU artifact because the
server NVIDIA userspace and kernel drivers were mismatched. It remains a
non-authoritative diagnostic and does not replace the frozen GPU parity
evidence.

**Interpretation boundary:** sequence-inference consistency is established;
model faithfulness is not.

## Evidence class B: automatic physiological annotation quality

The automatic audit used 30 frozen validation cases. It did not run LIME,
load the classifier, perform inference, select on the test set, or modify the
expert package. The base and enhanced presentations used bitwise-identical
attribution arrays, identical rankings, and identical color scales.

| Automatic endpoint | Frozen result | Supported interpretation |
| --- | ---: | --- |
| Physiological concept coverage | 100.0% (30/30) | The fixed annotation rule returned supported finite evidence for every case |
| Unsupported annotation rate | 0.0% (0/30) | No label was emitted without the preregistered evidence condition |
| Opposing-concept conflict rate | 10.0% (3/30) | Conflicting evidence was detected and retained |
| Channel-concept consistency | 96.7% (29/30) | The attribution-dominant channel usually matched the top-concept channel |
| Top-region evidence rate | 100.0% | Selected high-sensitivity regions contained finite concept evidence |
| Cross-mask-seed concept stability | 0.973 | Derived concept scores were stable across frozen mask seeds |
| Reporting-only stage agreement | 36.7% (11/30) | Descriptive join after annotation; not an annotation input or validation target |
| Mean concept sparsity | 40.0% | Descriptive density of the derived concept representation |

These metrics establish automatic **coverage, consistency, and stability**
under a fixed formula. They do not establish physiological correctness,
clinical usefulness, human interpretability, causal model reasoning, or model
faithfulness. In particular, 100% coverage is not proof that every annotation
is correct. The 10% conflict rate is reported as an explicit limitation and
warning rather than filtered away.

## Evidence class C: reliability and negative-result disclosure

The reliability evidence is deliberately reported next to the automatic
annotation results. Under the retained canonical B0 validation T0 protocol,
weighted local-surrogate \(R^2\) was 0.811 and median mask-seed Spearman
stability was 0.906. However, the mean R7 attribution similarity was 0.240
against a mean 1,000-permutation-null q95 of 0.104; no randomization seed
passed. Only 1/30 cases met the complete locally-explainable rule.

The unified protocol-consistency audit passed 0/3 seeds for both T0 and T1.
MF-SeqLIME was superseded, structured and boundary-adaptive neighborhoods
produced no qualified candidate, and HC-G9/HC-G25 each passed 0/3 T0 and 0/3
T1 seeds. The frozen 36-case HC test was therefore not run.

These negative findings are part of the framework contribution because they
prevent high surrogate fit or visually coherent annotations from being
misrepresented as model-faithful explanations. The authoritative status
remains `MODEL_FAITHFULNESS_NOT_ESTABLISHED`, and feedback training remains
`PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD`.

## What the results do and do not support

The evidence supports:

- a reproducible context-preserving sequence query path;
- exact isolation of attribution from annotation and visualization;
- automatic physiological coverage, consistency, stability, and conflict
  reporting;
- explicit reliability states and transparent retention of failed
  randomization evidence;
- implementation of a multilayer explanation representation.

The evidence does not support:

- superiority over LIME or a new attribution algorithm;
- model-faithful or causal attribution;
- easier interpretation, improved task accuracy, preference, or reduced
  misinterpretation;
- clinical validity or clinical utility;
- any training, inference, or performance benefit from XAI.
