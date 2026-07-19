#!/usr/bin/env python
"""Smoke-test the default-off differentiable XAI feedback head."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import sequence_lime_paper_evaluation as evaluation  # noqa: E402
from src.model_training import make_xai_alignment_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "xai_feedback_head_smoke" / "isruc_seed42",
    )
    parser.add_argument("--batch-windows", type=int, default=1)
    parser.add_argument(
        "--offline-target-path",
        type=Path,
        default=None,
        help="Optional train-only targets.npz used for a real gated XAI-only step.",
    )
    return parser.parse_args()


def tensor_map(model, outputs):
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return {name: np.asarray(value, dtype=np.float32) for name, value in zip(model.output_names, outputs)}


def load_offline_training_example(target_path: Path):
    from src.data_processing import load_npz_data  # noqa: WPS433

    target_path = target_path.resolve()
    manifest_path = target_path.parent / "target_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "split": "train",
        "test_data_accessed": False,
        "ground_truth_used_to_construct_targets": False,
        "predicted_stage_concept_prior_used": False,
        "ground_truth_stage_concept_prior_used": False,
        "label_conditioned_targets_allowed": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Unsafe smoke target manifest: {key}={manifest.get(key)!r}")
    with np.load(target_path, allow_pickle=False) as payload:
        global_idx = int(payload["global_epoch_index"][0])
        importance = np.asarray(payload["importance"][0], dtype=np.float32)
        reliability_score = float(payload["reliability_score"][0])
        gate_pass = bool(payload["reliability_gate_pass"][0])
        source_files = [str(value) for value in payload["source_files"].tolist()]
        counts = np.asarray(payload["file_sample_counts"], dtype=np.int32)
    if importance.shape != (30, 3):
        raise ValueError(f"Expected a [30,3] offline target, got {importance.shape}")
    if not gate_pass or reliability_score <= 0.0:
        raise ValueError("The selected smoke target did not pass its declared reliability gate")
    split_spec = evaluation.load_split_spec()
    x_raw, _labels = load_npz_data(
        str(Path(str(split_spec["source_npz_dir"])).expanduser()),
        file_subset=source_files,
        preprocess=True,
        filter_profile="clinical_canonical",
        filter_backend="scipy",
        normalization_mode="recording_robust_zscore",
        clip_sigma=None,
        enable_amplitude_clipping=False,
    )
    start, valid_len, relative_position = evaluation.target_windows(counts, global_idx)[0]
    indices = evaluation.window_epoch_indices(start, valid_len, 75)
    raw_window = np.asarray(x_raw[indices], dtype=np.float32)
    segment_window = np.asarray(
        [
            evaluation.extract_segment_features(
                epoch,
                evaluation.semantic_segmentation(epoch, target_segments=30),
                max_segments=30,
                include_spectral=True,
            )
            for epoch in raw_window
        ],
        dtype=np.float32,
    )
    target = np.zeros((1, 75, 30, 3), dtype=np.float32)
    epoch_weight = np.zeros((1, 75), dtype=np.float32)
    target[0, int(relative_position)] = importance
    epoch_weight[0, int(relative_position)] = reliability_score
    inputs = {
        "raw_input": raw_window[None, ...],
        "segment_input": segment_window[None, ...],
    }
    provenance = {
        "kind": "train_only_offline_teacher",
        "target_path": str(target_path),
        "target_sha256": evaluation.file_sha256(target_path),
        "target_type": manifest.get("target_type"),
        "teacher_mode": manifest.get("teacher_mode"),
        "global_epoch_index": global_idx,
        "relative_sequence_position": int(relative_position),
        "reliability_score": reliability_score,
        "reliability_gate_pass": gate_pass,
    }
    return inputs, target, epoch_weight, provenance


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    split_spec = evaluation.load_split_spec()
    split_files = evaluation.normalize_split_files(split_spec)
    with np.load(evaluation.PRED_PATH) as payload:
        artifact = evaluation.load_artifact_split(payload, "test")
    x_raw, x_seg, _quality = evaluation.load_or_build_test_inputs(out_dir, split_files, artifact)
    baseline = evaluation.build_model(tuple(x_raw.shape[1:]), tuple(x_seg.shape[1:]))
    feedback = evaluation.build_model(
        tuple(x_raw.shape[1:]),
        tuple(x_seg.shape[1:]),
        enable_xai_feedback_head=True,
        xai_feedback_loss_weight=1e-3,
        xai_alignment_loss_name="normalized_mse",
        xai_gradient_scope="head_only",
    )

    indices = evaluation.window_epoch_indices(0, min(75, len(x_raw)), 75)
    inputs = {
        "raw_input": np.asarray(x_raw[indices][None, ...], dtype=np.float32),
        "segment_input": np.asarray(x_seg[indices][None, ...], dtype=np.float32),
    }
    baseline_outputs = tensor_map(baseline, baseline(inputs, training=False))
    feedback_outputs = tensor_map(feedback, feedback(inputs, training=False))
    main_diff = np.abs(baseline_outputs["main_output"] - feedback_outputs["main_output"])
    head_shape = list(feedback_outputs["xai_importance_logits"].shape)

    backbone_before = {
        variable.name: np.asarray(variable.numpy()).copy()
        for variable in feedback.trainable_variables
        if not variable.name.startswith("xai_") and "/xai_" not in variable.name
    }
    head_before = {
        variable.name: np.asarray(variable.numpy()).copy()
        for variable in feedback.trainable_variables
        if variable.name.startswith("xai_") or "/xai_" in variable.name
    }
    if args.offline_target_path is not None:
        training_inputs, xai_target, xai_epoch_weight, target_provenance = (
            load_offline_training_example(args.offline_target_path)
        )
    else:
        training_inputs = inputs
        rng = np.random.default_rng(42)
        xai_target = rng.random(
            feedback_outputs["xai_importance_logits"].shape, dtype=np.float32
        )
        xai_target /= np.maximum(np.sum(xai_target, axis=(-2, -1), keepdims=True), 1e-8)
        xai_epoch_weight = np.ones((1, 75), dtype=np.float32)
        target_provenance = {"kind": "synthetic_shape_and_gradient_smoke"}
    loss_fn = make_xai_alignment_loss("normalized_mse")
    with tf.GradientTape() as tape:
        # This is an XAI-only isolation step. Use inference-state backbone
        # behavior so BatchNorm moving statistics cannot change without a
        # classifier loss; gradients through the differentiable head remain
        # available.
        live_outputs = feedback(training_inputs, training=False)
        live_map = {
            name: value for name, value in zip(feedback.output_names, live_outputs)
        }
        per_epoch_loss = loss_fn(
            tf.constant(xai_target), live_map["xai_importance_logits"]
        )
        xai_loss = 1e-3 * tf.reduce_sum(
            per_epoch_loss * tf.constant(xai_epoch_weight)
        ) / tf.maximum(
            tf.reduce_sum(tf.constant(xai_epoch_weight)), 1e-8
        )
    gradients = tape.gradient(xai_loss, feedback.trainable_variables)
    xai_gradient_pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, feedback.trainable_variables)
        if gradient is not None
    ]
    feedback.optimizer.apply_gradients(xai_gradient_pairs)
    post_step_outputs = tensor_map(feedback, feedback(inputs, training=False))
    post_step_main_diff = np.abs(
        feedback_outputs["main_output"] - post_step_outputs["main_output"]
    )
    backbone_max_change = max(
        (
            float(np.max(np.abs(variable.numpy() - backbone_before[variable.name])))
            for variable in feedback.trainable_variables
            if variable.name in backbone_before
        ),
        default=0.0,
    )
    head_max_change = max(
        (
            float(np.max(np.abs(variable.numpy() - head_before[variable.name])))
            for variable in feedback.trainable_variables
            if variable.name in head_before
        ),
        default=0.0,
    )
    checks = {
        "canonical_default_has_no_xai_output": "xai_importance_logits" not in baseline.output_names,
        "feedback_head_shape_is_B_T_30_3": head_shape == [1, 75, 30, 3],
        "old_checkpoint_main_output_parity": float(np.max(main_diff)) <= 1e-6,
        "head_only_backbone_unchanged_after_xai_only_step": backbone_max_change <= 1e-12,
        "head_parameters_updated_after_xai_only_step": head_max_change > 0.0,
        "classification_output_unchanged_after_xai_only_step": float(np.max(post_step_main_diff)) <= 1e-12,
        "offline_target_is_reliability_gated_when_requested": (
            args.offline_target_path is None
            or bool(target_provenance.get("reliability_gate_pass", False))
        ),
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "baseline_output_names": baseline.output_names,
        "feedback_output_names": feedback.output_names,
        "xai_head_shape": head_shape,
        "main_output_max_abs_diff": float(np.max(main_diff)),
        "main_output_mean_abs_diff": float(np.mean(main_diff)),
        "head_only_backbone_max_change": backbone_max_change,
        "head_parameter_max_change": head_max_change,
        "post_xai_step_main_output_max_abs_diff": float(np.max(post_step_main_diff)),
        "training_target": target_provenance,
        "default_enabled": False,
        "checkpoint": str(evaluation.CKPT_PATH.resolve()),
    }
    (out_dir / "xai_feedback_head_smoke.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
