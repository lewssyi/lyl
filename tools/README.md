# Maintained tools

`tools/` contains only A1-zero, unified Sequence-XAI, protocol, and clinical
review entry points. One-off C10/B2/B3/SCR/UTF-Lite/SupCon/multi-teacher/
channel-residual and superseded diagnostic scripts were removed after an
external, hash-verified backup and a reference scan.

## A1-zero and protocol

- `run_main_with_protocol_split.py` — fixed-split launcher; canonical ISRUC
  preset: `isruc_A1_concept_film_zero`.
- `build_sleep_dataset_protocol.py` — subject/recording-disjoint dataset split
  protocol builder.
- `build_cross_dataset_protocol.py` — explicit cross-dataset protocol builder.

## Sequence-XAI

- `sequence_lime_paper_evaluation.py` — parity, isolated/resumable smoke,
  confirmation/full evaluation, stability, randomization, faithfulness,
  A-G prior ablation, runtime/query accounting, and paper tables.
- `generate_sequence_xai_training_targets.py` — train-only offline teacher with
  self-distillation/OOF provenance, multiple target controls, and a reliability
  score plus hard gate.
- `xai_feedback_head_smoke.py` — default-off head shape, checkpoint parity, and
  gradient-isolation smoke test.
- `audit_concept_definition_parity.py` — real-sample comparison of
  `segment_proxy_v1` and `raw_physiology_v1`.

## Clinical review

- `build_clinical_expert_review_pack.py` — blinded, stratified review pack.
- `analyze_clinical_expert_ratings.py` — agreement and rating analysis after
  human experts complete the template.

Clinical expert validation is pending.

The current guarded formal result, validation counts, scientific decision, and
failure/recovery provenance are recorded in
`docs/XAI_FULL_EVALUATION_RESULTS.md`.

## Canonical commands

Run prediction/wrapper parity:

```bash
/home/b502/app/anaconda3/envs/l_venv/bin/python3.11 \
  tools/sequence_lime_paper_evaluation.py \
  --dataset isruc --seed 42 --mode parity-diagnostic
```

Run the isolated functional profile:

```bash
/home/b502/app/anaconda3/envs/l_venv/bin/python3.11 \
  tools/sequence_lime_paper_evaluation.py \
  --dataset isruc --seed 42 \
  --mode functional-smoke \
  --config configs/xai/sequence_lime_functional.yaml \
  --run-id functional_smoke --resume
```

Generate a train-only teacher smoke artifact:

```bash
/home/b502/app/anaconda3/envs/l_venv/bin/python3.11 \
  tools/generate_sequence_xai_training_targets.py \
  --split train --target-type pc_seqlime_without_stage_prior \
  --teacher-mode self_distillation_in_sample \
  --max-recordings 1 --max-samples 1 --run-id feedback_target_smoke
```

The canonical decoder is validation-selected temperature argmax. HMM is an
optional validation-selected candidate, and LIME explains pre-HMM neural
probabilities.

The pre-cleanup usage decision, reason, replacement, and backup location for
every audited Python entry are recorded in
`docs/TOOLS_INVENTORY_BEFORE_CLEANUP.csv`.
