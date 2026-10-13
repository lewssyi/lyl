# A1-zero Sleep Staging and Sequence-XAI

The maintained classifier is A1-zero (`isruc_A1_concept_film_zero`), with the
checkpoint, protocol and inference settings documented in
[the mainline manifest](docs/A1_ZERO_MAINLINE_MANIFEST.md). Current paper evidence
also includes the paired architecture experiments stored under `outputs/`.

The canonical post-hoc method is **Context-Preserving Classic Sequence-LIME**.
Reliability evaluation, fixed batch-1 Context Influence and physiological
annotation are independent modules. The current scientific boundaries are in
[the authoritative status](docs/XAI_AUTHORITATIVE_STATUS.md): model faithfulness
is not established, real clinical expert validation is pending, and explanation
feedback training remains permanently blocked.

## Working directory

```text
run_main.py           Training, evaluation and checkpoint-compatible inference
src/                  Shared model, preprocessing and post-hoc implementation
tools/                Maintained experiment and paper entry points
configs/xai/          Canonical evaluation/annotation profiles and freeze guard
dataset_protocols/    Existing fixed subject/recording splits and metadata
tests/                Maintained regression and frozen-evidence checks
docs/                 Paper evidence, methods, reports, figures and review package
paper_outputs/        Existing manuscript documents
EEG_data/             Existing data and extraction helpers
tmp/                  Existing cross-dataset NPZ view; kept for protocol paths
outputs/              ALL existing experiments, checkpoints, predictions and logs
logs/                 ALL existing external/training logs
experiment_records/   Existing experiment provenance
models/               Existing saved model artifacts
.deps/, venv/         Preserved runtime dependencies
```

The detailed entry-point map and commands are in [tools/README.md](tools/README.md).
Frozen paper writing materials begin at
[docs/paper_manuscript_v1/](docs/paper_manuscript_v1/PAPER_PROJECT_FREEZE.md).
Historical reports retain their original scientific context; they do not activate
retired experiment routes.

## Mainline commands

Use `/home/b502/app/anaconda3/envs/l_venv/bin/python3.11`.

```bash
/home/b502/app/anaconda3/envs/l_venv/bin/python3.11 tools/run_main_with_protocol_split.py \
  --split-json dataset_protocols/ISRUC/split.json --gpus 0 \
  --dataset-preset isruc_A1_concept_film_zero

/home/b502/app/anaconda3/envs/l_venv/bin/python3.11 tools/sequence_lime_paper_evaluation.py \
  --dataset isruc --seed 42 --mode parity-diagnostic
```

Sequence-LIME JSON-compatible YAML profiles remain in `configs/xai/`:
`sequence_lime_smoke.yaml`, `sequence_lime_functional.yaml`,
`sequence_lime_confirmation.yaml`, and `sequence_lime_full.yaml`.
The `clinical_annotation.yaml` profile and permanently disabled
`xai_feedback_experiments.yaml` freeze guard are also retained.

## Workspace cleanup

[Cleanup record](docs/WORKSPACE_CLEANUP_20261010.md) lists removed tools, their
verified external backup and preservation checks. All experimental results and
logs remain at their original locations, including code snapshots contained in
result packages. Model/preprocessing source and runtime environments are unchanged.
