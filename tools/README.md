# Maintained mainline tools

The active classifier is A1-zero. The canonical attribution method is
Context-Preserving Classic Sequence-LIME; reliability evaluation and physiological
annotation remain independent post-hoc modules. Historical results and logs are
preserved in full. See `docs/WORKSPACE_CLEANUP_20261010.md` for cleanup provenance.

## Classifier, data protocol and paper ablations

- `run_main_with_protocol_split.py`: fixed-split A1-zero training/evaluation launcher.
- `build_sleep_dataset_protocol.py`: subject-disjoint dataset protocol builder.
- `build_cross_dataset_protocol.py`: cross-dataset protocol and linked NPZ view builder.
- `prepare_paper_combinations.py`: prepares the paired paper architecture ablations;
  relocated without content changes from the old root-level temporary helper.

## Canonical Sequence-LIME and validation

- `sequence_lime_paper_evaluation.py`: canonical checkpoint/wrapper parity, smoke,
  evaluation, faithfulness, stability and paper tables.
- `sequence_lime_protocol_consistency_audit.py`: unified Classic randomization protocol
  underlying the current paper conclusion.
- `aggregate_sequence_lime_protocol_consistency.py`: aggregate the unified audit.
- `finalize_sequence_lime_protocol_consistency.py`: finalize the unified audit decision.
- `sequence_lime_abc_diagnosis.py`: shared response/faithfulness helpers required by
  the retained unified audit. Its historical A/B/C campaign is not an active route.
- `final_xai_guards.py`: isolated checkpoint/wrapper parity and fixed-graph memory guard.

## Paper materials and independent post-hoc modules

- `build_paper_xai_materials.py`: paper tables, figures and deterministic case materials.
- `build_physiology_aware_display_validation.py`: frozen blinded display-validation pack.
- `sleepxvit_offline_reliability_audit.py`: inference-free reliability statistics.
- `audit_concept_definition_parity.py`: raw-physiology versus segment-proxy audit.
- `context_influence_paper_convergence.py`: fixed batch-1 paper implementation check.
- `sequence_context_influence_development_smoke.py`: shared context-window/replacement
  helpers used by the batch-1 entry point. Historical batch-16 runs are not the
  canonical paper implementation.
- `build_clinical_expert_review_pack.py`: expert review package generation.
- `analyze_clinical_expert_ratings.py`: analyze real expert ratings when available.

Retired HC/MF campaign launchers, superseded batch numerical diagnostics, dummy
training-target generators and one-off source cleaners were removed after a
hash-verified external backup. Their result directories, manifests and reports
remain at their original paths. `src/` is unchanged to preserve checkpoint and
inference compatibility. The disabled feedback YAML is retained as an executable
freeze guard exercised by existing tests, not as an experiment entry point.

## Commands

```bash
/home/b502/app/anaconda3/envs/l_venv/bin/python3.11 tools/run_main_with_protocol_split.py \
  --split-json dataset_protocols/ISRUC/split.json --gpus 0 \
  --dataset-preset isruc_A1_concept_film_zero

/home/b502/app/anaconda3/envs/l_venv/bin/python3.11 tools/sequence_lime_paper_evaluation.py \
  --dataset isruc --seed 42 --mode parity-diagnostic

PYTHONPATH=/home/b502/wssyi/project_backups/LIME-for-Time-Series_cleanup_20261010_021513/validation_dependencies \
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  /home/b502/app/anaconda3/envs/l_venv/bin/python3.11 -B -m pytest -q -p no:cacheprovider tests
```
