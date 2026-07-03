# Tooling Map

Current direction:

- Keep C0 as the no-concept baseline.
- Keep C10 (`*_c10_late_concept_residual`) as the main model.
- Stop C11/C11b post-hoc calibration and C12/C13/C14 training variants.
- Do not add more seed sweeps as the main route.

## Kept Tools

- `build_sleep_dataset_protocol.py`
  - Builds fixed train/val/test protocol split JSON files.
- `run_main_with_protocol_split.py`
  - Main fixed-split launcher.
  - Active presets:
    - `edf_c0_no_concept`
    - `edf_c0_accuracy_selector_only`
    - `edf_c10_late_concept_residual`
    - `isruc_c0_no_concept`
    - `isruc_c0_accuracy_selector_only`
    - `isruc_c10_late_concept_residual`
- `c10_error_diagnostics.py`
  - Reads C10 prediction NPZ files and reports error patterns.
- `c10_targeted_error_mining.py`
  - Second-pass C10 error analysis for focused confusion pairs and recording-level failures.
- `b2_physiology_rule_teacher_disagreement.py`
  - AISleep-inspired post-hoc rule teacher for W/N1/N2/N3/REM from clinical concepts.
  - Reports model-rule disagreement, uncertainty-weighted top-risk error rates, high-confidence conflicts,
    N1-focused rates, and ranked conflict examples.
- `b3_risk_aware_corrector.py`
  - Leakage-aware post-hoc corrector trained on train split, selected on val split, and evaluated on test.
  - Uses model probabilities, sequence context, optional B2 rule/risk features, gated probability blending,
    and a lightweight Viterbi pass.
- `export_c10a_split_predictions.py`
  - Optional exporter when a saved C10 checkpoint must be replayed over train/val/test files.
- `generate_c10_framework_visio.py`
- `generate_c10_nature_framework.py`
  - C10 figure-generation helpers.

## Retired Routes

The following routes should not be launched from this workspace unless they are
explicitly restored for an ablation appendix:

- C1-C8 concept-prior exploration presets.
- C10b confidence-gated residual.
- C10c post-hoc confusion-aware protection.
- C11 full 5-class recalibration.
- C11b conservative pairwise correction.
- C12 transition-boundary enhancement.
- C13 stability-first late residual.
- C14 N1/N2 pair-focus training.
- Probability ensembling.
- ISRUC concept-prior batch suite.

## Example Commands

Run C0 on ISRUC:

```bash
python tools/run_main_with_protocol_split.py \
  --split-json dataset_protocols/ISRUC/split.json \
  --gpus 0,1 \
  --per-replica-batch 16 \
  --predict-per-replica-batch 16 \
  --no-preset-batch-clamp \
  --dataset-preset isruc_c0_no_concept
```

Run C10 on ISRUC:

```bash
python tools/run_main_with_protocol_split.py \
  --split-json dataset_protocols/ISRUC/split.json \
  --gpus 0,1 \
  --per-replica-batch 16 \
  --predict-per-replica-batch 16 \
  --no-preset-batch-clamp \
  --dataset-preset isruc_c10_late_concept_residual
```

Run C10 on SleepEDF:

```bash
python tools/run_main_with_protocol_split.py \
  --split-json dataset_protocols/SleepEDF/split.json \
  --gpus 0,1 \
  --per-replica-batch 16 \
  --predict-per-replica-batch 16 \
  --no-preset-batch-clamp \
  --dataset-preset edf_c10_late_concept_residual
```

Analyze a C10 prediction NPZ:

```bash
python tools/c10_error_diagnostics.py \
  --prediction-npz outputs/predictions/<prediction_file>.npz \
  --dataset isruc \
  --output-dir outputs/diagnostics/<run_name> \
  --splits test \
  --baseline-pred-source saved
```

Run B2 physiology rule-teacher disagreement on an existing prediction file:

```bash
python tools/b2_physiology_rule_teacher_disagreement.py \
  --prediction-npz outputs/predictions/<prediction_file>.npz \
  --dataset isruc \
  --split test \
  --split-json dataset_protocols/ISRUC/split.json \
  --concept-extractor direct_epoch \
  --concept-cache outputs/diagnostics/b2_rule_teacher/isruc_test_direct_epoch_concepts_v2.npz \
  --calibrate-prototypes \
  --prototype-split val \
  --prototype-cache outputs/diagnostics/b2_rule_teacher/isruc_val_direct_epoch_stage_prototypes_v2.npz \
  --prototype-weight 0.90 \
  --output-dir outputs/diagnostics/b2_rule_teacher/<run_name>
```

Key B2 outputs:

- `summary.json`: rule metrics, disagreement rates, `model_error_rate_when_top_risk_10pct`,
  and `retained_accuracy_after_review_top_risk_10pct`.
- `top_conflicts.csv`: disagreement examples ranked by `risk_score` with model/rule margins and probability distance.
- `rule_teacher_outputs.npz`: `p_rule`, `rule_pred`, `risk_score`, `model_margin`, `prob_js`, and related arrays.

Run B3 risk-aware post-hoc corrector on a split prediction file:

```bash
python tools/b3_risk_aware_corrector.py \
  --prediction-npz outputs/predictions/<split_prediction_file>.npz \
  --dataset isruc \
  --split-json dataset_protocols/ISRUC/split.json \
  --concept-cache-dir outputs/diagnostics/b3_corrector/concept_cache \
  --concept-extractor direct_epoch \
  --prototype-source train \
  --prototype-weight 0.90 \
  --epochs 20 \
  --output-dir outputs/diagnostics/b3_corrector/<run_name>
```

B3 is a validation-selected correction layer. Treat it as a full-coverage
accuracy experiment, not as a replacement for B2 selective review metrics.
