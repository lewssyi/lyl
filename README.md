# A1-zero Sleep Staging and Sequence-XAI

This repository's only active mainline is `A1_concept_film_zero` (A1-zero): a
five-class (`W`, `N1`, `N2`, `N3`, `REM`) ISRUC sleep-staging model using EEG,
EOG, and EMG. The working branch is `codex/a1-zero-mainline`.

The frozen ISRUC seed-42 checkpoint is:

```text
outputs/checkpoints/result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42.weights.h5
SHA256 e86a84790fd7351c1864158eef79fbf28437a7a8a5f26662b93bcf1aef14b6ed
```

The complete reconstruction record, split hash, environment, enabled/default-off
parameters, and canonical metrics are in
[`docs/A1_ZERO_MAINLINE_MANIFEST.md`](docs/A1_ZERO_MAINLINE_MANIFEST.md).

## Scientific boundary

- Canonical decoding is the validation-selected temperature-scaled argmax.
- HMM/Viterbi decoding is implemented only as an optional validation-selected
  candidate; it is not the canonical A1-zero decoder.
- LIME explains pre-HMM neural probabilities.
- XAI results are local attribution, perturbation sensitivity, derived concept
  evidence, or interventional faithfulness under a named replacement rule.
  They are not global causal contributions or model-reasoning ground truth.
- Raw-waveform concepts (`raw_physiology_v1`) and segment-feature concepts
  (`segment_proxy_v1`) are related proxies, not mathematically identical
  clinical variables.

## Main entry points

```text
run_main.py
  Canonical training, validation-selected decoding, evaluation, and optional
  default-off auxiliary XAI head.

tools/run_main_with_protocol_split.py
  Fixed subject/recording split launcher. The canonical preset is
  isruc_A1_concept_film_zero.

tools/sequence_lime_paper_evaluation.py
  Read-only Sequence-LIME parity, smoke, confirmation, full evaluation, and
  paper-table pipeline.

tools/generate_sequence_xai_training_targets.py
  Train-split-only offline explanation teacher. Self-distillation and explicitly
  supplied out-of-fold checkpoints are distinguished in every manifest.

tools/xai_feedback_head_smoke.py
  Verifies checkpoint compatibility, default-off behavior, the [B,T,30,3]
  explanation output, and head-only gradient isolation.
```

The smaller, audited tooling map is in [`tools/README.md`](tools/README.md).

## Fixed-split A1-zero run

The project-specific Python environment is:

```bash
/home/b502/app/anaconda3/envs/l_venv/bin/python3.11
```

Example fixed-split launcher:

```bash
/home/b502/app/anaconda3/envs/l_venv/bin/python3.11 \
  tools/run_main_with_protocol_split.py \
  --split-json dataset_protocols/ISRUC/split.json \
  --gpus 0 \
  --dataset-preset isruc_A1_concept_film_zero
```

The canonical preprocessing remains the checkpoint-compatible full-recording
robust z-score. Deployment-safe alternatives are available as explicit modes;
they do not silently replace the frozen preprocessing.

## Sequence-LIME evaluation

Versioned JSON-compatible YAML profiles live under `configs/xai/`:

```text
sequence_lime_smoke.yaml
sequence_lime_functional.yaml
sequence_lime_confirmation.yaml
sequence_lime_full.yaml
```

Each profile has an isolated run ID, resolved configuration, fingerprint,
resume namespace, completion state, provenance, and result hashes. A current
result index is written to:

```text
outputs/sequence_lime_paper_evaluation/isruc_seed42/CURRENT_RESULTS_MANIFEST.json
```

The completed formal result is documented in
[`docs/XAI_FULL_EVALUATION_RESULTS.md`](docs/XAI_FULL_EVALUATION_RESULTS.md).
Its implementation/measurement chain passed, but the paper-level reliability
decision failed the project fidelity and randomization criteria; no universal
reliability or superiority claim is made.

Run parity before XAI:

```bash
/home/b502/app/anaconda3/envs/l_venv/bin/python3.11 \
  tools/sequence_lime_paper_evaluation.py \
  --dataset isruc --seed 42 --mode parity-diagnostic
```

Run a resumable functional check:

```bash
/home/b502/app/anaconda3/envs/l_venv/bin/python3.11 \
  tools/sequence_lime_paper_evaluation.py \
  --dataset isruc --seed 42 \
  --mode functional-smoke \
  --config configs/xai/sequence_lime_functional.yaml \
  --run-id functional_smoke --resume
```

The evaluator reports zero, physiological, Gaussian-noise, and local-mean
replacement endpoints separately. Any PC-SeqLIME superiority claim is limited
to the physiological-perturbation estimand when supported by the data.

## Default-off explanation-guided training

LIME is not run inside a training batch. The optional prototype is:

```text
train-only offline reliable explanation teacher
  -> reliability score and hard gate
  -> differentiable [B,T,30,3] explanation student head
  -> weak scheduled auxiliary alignment loss
```

Defaults preserve A1-zero behavior:

```text
LFTS_XAI_FEEDBACK_ENABLED=0
LFTS_XAI_FEEDBACK_LOSS_WEIGHT=0.0
```

The classification path never reads the explanation-head output. Supported
alignment losses are normalized MSE, KL, JS, a rank-correlation surrogate, and
top-k BCE. Supported gradient scopes are `head_only`, `fusion_and_head`, and
`full_backbone`; experiments must begin with `head_only`.

The E0-E7 controls and the lambda sweep are declared in:

```text
configs/xai/xai_feedback_experiments.yaml
```

Predicted-stage-prior supervision is rejected by default and exists only as the
explicit circularity control E7. True-stage priors and test-derived training
targets are rejected.

## Clinical review status

Code can generate a blinded, stratified expert-review pack and analyze completed
ratings, but it cannot supply clinical judgment. See
[`docs/CLINICAL_EXPERT_REVIEW_PROTOCOL.md`](docs/CLINICAL_EXPERT_REVIEW_PROTOCOL.md).

**Clinical expert validation is pending.**

## Tests

Run the maintained suite from the repository root:

```bash
/home/b502/app/anaconda3/envs/l_venv/bin/python3.11 -m unittest discover -s tests
```

Generated checkpoints, large caches, raw data, review images, and full result
directories are not committed. Their versioned manifests record paths and
SHA256 hashes instead.
