#!/usr/bin/env python
"""Generate random/shuffled dummy targets for isolated XAI unit controls.

Real post-hoc sensitivity targets are permanently prohibited from training.
The resulting dummy artifact is aligned to ``[epoch, 30, 3]`` and must not be
used to optimize the frozen classifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import sequence_lime_paper_evaluation as evaluation  # noqa: E402
from src.data_processing import load_npz_data  # noqa: E402
from src.xai.posthoc_freeze import assert_offline_target_is_dummy_control  # noqa: E402

TARGET_TYPES = (
    "classic_sequence_lime",
    "physiological_single_feature",
    "lime_physiology_consensus",
    "pc_seqlime_without_stage_prior",
    "shuffled_target_control",
    "random_target_control",
    "pc_seqlime_with_predicted_stage_prior_ablation",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_counts(npz_dir: Path, files: Sequence[str]) -> np.ndarray:
    counts = []
    for filename in files:
        with np.load(npz_dir / filename, mmap_mode="r") as payload:
            counts.append(int(payload["x"].shape[0]))
    return np.asarray(counts, dtype=np.int32)


def selected_epoch_indices(counts: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    total = int(np.sum(counts))
    if max_samples <= 0 or max_samples >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    # Preserve recording coverage before filling the remaining budget randomly.
    offsets = np.concatenate([[0], np.cumsum(counts)])
    mandatory = []
    for record_index, count in enumerate(counts):
        if len(mandatory) >= max_samples:
            break
        mandatory.append(int(offsets[record_index] + count // 2))
    remaining = np.setdiff1d(np.arange(total, dtype=np.int64), np.asarray(mandatory))
    take = max_samples - len(mandatory)
    extra = rng.choice(remaining, size=take, replace=False) if take > 0 else np.asarray([], dtype=np.int64)
    return np.sort(np.concatenate([np.asarray(mandatory, dtype=np.int64), extra]))


def normalize_target(score: np.ndarray) -> np.ndarray:
    score = np.clip(np.asarray(score, dtype=np.float32), 0.0, None)
    total = float(np.sum(score))
    if total <= 1e-8:
        score = np.ones_like(score, dtype=np.float32)
        total = float(np.sum(score))
    return (score / total).astype(np.float32)


def channel_major_to_head(score: np.ndarray, segments: int = 30, channels: int = 3) -> np.ndarray:
    return normalize_target(score).reshape(channels, segments).T.astype(np.float32)


def calibration_scores(calibration: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    zero = calibration["modes"]["zero"]
    physiological = calibration["modes"]["physiological"]
    zero_score = 0.5 * (
        np.asarray(zero["deletion_rank_score"], dtype=np.float32)
        + np.asarray(zero["insertion_rank_score"], dtype=np.float32)
    )
    physiological_score = 0.5 * (
        np.asarray(physiological["deletion_rank_score"], dtype=np.float32)
        + np.asarray(physiological["insertion_rank_score"], dtype=np.float32)
    )
    return zero_score, physiological_score


def build_target_candidates(
    classic_score: np.ndarray,
    sequence_score: np.ndarray,
    calibration: Mapping[str, object],
    rng: np.random.Generator,
    predicted_stage_prior_score: np.ndarray,
) -> Dict[str, np.ndarray]:
    zero_score, physiological_score = calibration_scores(calibration)
    pc_no_stage_prior = (
        0.20 * np.asarray(sequence_score, dtype=np.float32)
        + 0.40 * zero_score
        + 0.40 * physiological_score
    )
    consensus = np.minimum(np.asarray(sequence_score, dtype=np.float32), physiological_score)
    shuffled = rng.permutation(pc_no_stage_prior)
    random_control = rng.random(pc_no_stage_prior.shape, dtype=np.float32)
    return {
        "classic_sequence_lime": classic_score,
        "physiological_single_feature": physiological_score,
        "lime_physiology_consensus": consensus,
        "pc_seqlime_without_stage_prior": pc_no_stage_prior,
        "shuffled_target_control": shuffled,
        "random_target_control": random_control,
        "pc_seqlime_with_predicted_stage_prior_ablation": (
            0.80 * pc_no_stage_prior
            + 0.20 * np.asarray(predicted_stage_prior_score, dtype=np.float32)
        ),
    }


def load_oof_checkpoint_map(path: Path) -> Dict[str, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("recording_to_checkpoint", {})
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("OOF checkpoint JSON must contain a non-empty recording_to_checkpoint object")
    result = {str(key): Path(str(value)).expanduser().resolve() for key, value in mapping.items()}
    missing = [str(value) for value in result.values() if not value.exists()]
    if missing:
        raise FileNotFoundError(f"Missing OOF checkpoints: {missing}")
    return result


def recording_for_epoch(counts: np.ndarray, global_index: int) -> Tuple[int, int]:
    offset = 0
    for record_index, count in enumerate(counts):
        if global_index < offset + int(count):
            return int(record_index), int(global_index - offset)
        offset += int(count)
    raise IndexError(global_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=("train",))
    parser.add_argument("--target-type", default="pc_seqlime_without_stage_prior", choices=TARGET_TYPES)
    parser.add_argument("--teacher-mode", default="self_distillation_in_sample", choices=("self_distillation_in_sample", "out_of_fold"))
    parser.add_argument("--oof-checkpoints-json", type=Path)
    parser.add_argument("--max-recordings", type=int, default=0, help="0 means all train recordings")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all loaded train epochs")
    parser.add_argument("--train-masks", type=int, default=300)
    parser.add_argument("--validation-masks", type=int, default=100)
    parser.add_argument("--batch-windows", type=int, default=16)
    parser.add_argument("--reliability-min-r2", type=float, default=None)
    parser.add_argument("--reliability-min-stability", type=float, default=None)
    parser.add_argument("--reliability-original-mask-error-max", type=float, default=None)
    parser.add_argument(
        "--reliability-min-physiological-deletion-advantage",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--reliability-min-physiological-insertion-advantage",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--allow-predicted-stage-prior-ablation",
        action="store_true",
        help="Required only for the explicitly circular E7 control target.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="train_targets")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "xai" / "xai_feedback_experiments.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "xai_feedback_targets" / "isruc_seed42")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_offline_target_is_dummy_control(args.target_type)
    feedback_config = json.loads(args.config.read_text(encoding="utf-8"))
    if feedback_config.get("schema_version") != "xai_feedback_experiments_v1":
        raise ValueError("Feedback config must use xai_feedback_experiments_v1")
    gate_config = feedback_config.get("reliability_gate", {})
    gate_bindings = {
        "reliability_min_r2": "validation_weighted_r2_min",
        "reliability_min_stability": "mask_seed_stability_spearman_min",
        "reliability_original_mask_error_max": "original_mask_error_max",
        "reliability_min_physiological_deletion_advantage": (
            "physiological_deletion_advantage_vs_random_min"
        ),
        "reliability_min_physiological_insertion_advantage": (
            "physiological_insertion_advantage_vs_random_min"
        ),
    }
    for argument_name, config_name in gate_bindings.items():
        if getattr(args, argument_name) is None:
            if config_name not in gate_config:
                raise ValueError(f"Feedback config lacks reliability_gate.{config_name}")
            setattr(args, argument_name, float(gate_config[config_name]))
    if args.train_masks < 20 or args.validation_masks < 10:
        raise ValueError("Target generation requires at least 20 train and 10 validation masks")
    if args.teacher_mode == "out_of_fold" and args.oof_checkpoints_json is None:
        raise ValueError("out_of_fold mode requires --oof-checkpoints-json; it is never inferred or mislabeled")
    stage_prior_ablation = (
        args.target_type == "pc_seqlime_with_predicted_stage_prior_ablation"
    )
    if stage_prior_ablation and not args.allow_predicted_stage_prior_ablation:
        raise ValueError(
            "The predicted-stage-prior E7 control requires "
            "--allow-predicted-stage-prior-ablation"
        )

    split_spec = evaluation.load_split_spec()
    split_files = evaluation.normalize_split_files(split_spec)
    files = list(split_files["train"])
    if args.max_recordings > 0:
        files = files[: args.max_recordings]
    if not files:
        raise RuntimeError("The frozen protocol has no train recordings")
    npz_dir = Path(str(split_spec["source_npz_dir"])).expanduser()
    counts = file_counts(npz_dir, files)
    x_raw, y_true = load_npz_data(
        str(npz_dir),
        file_subset=files,
        preprocess=True,
        filter_profile="clinical_canonical",
        filter_backend="scipy",
        normalization_mode="recording_robust_zscore",
        clip_sigma=None,
        enable_amplitude_clipping=False,
    )
    if int(np.sum(counts)) != len(x_raw):
        raise RuntimeError("Train recording counts do not match loaded epochs")

    evaluation.ensure_lime_imports()
    evaluation_segments = [(index * 100, (index + 1) * 100) for index in range(30)]
    x_seg = np.asarray(
        [
            evaluation.extract_segment_features(
                epoch,
                evaluation.semantic_segmentation(epoch, target_segments=30),
                max_segments=30,
                include_spectral=True,
            )
            for epoch in x_raw
        ],
        dtype=np.float32,
    )
    canonical_model = evaluation.build_model(tuple(x_raw.shape[1:]), tuple(x_seg.shape[1:]))
    oof_map = load_oof_checkpoint_map(args.oof_checkpoints_json) if args.teacher_mode == "out_of_fold" else {}
    model_cache: Dict[str, object] = {str(evaluation.CKPT_PATH.resolve()): canonical_model}

    config = replace(
        evaluation.mode_config("functional-smoke"),
        mode="xai-target-generation",
        train_masks=int(args.train_masks),
        val_masks=int(args.validation_masks),
        batch_windows=int(args.batch_windows),
        target_segments=30,
        perturbation_modes=("zero", "physiological"),
        max_samples=int(args.max_samples),
    )
    selected = selected_epoch_indices(counts, int(args.max_samples), int(args.seed))
    out_dir = args.output_root.resolve() / str(args.run_id)
    cache_dir = out_dir / "teacher_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    importance = []
    reliability = []
    reliability_scores = []
    reliability_gate_passes = []
    teacher_confidences = []
    prediction_correctness = []
    predicted_class = []
    recording_index = []
    epoch_in_recording = []
    target_components = []
    rng = np.random.default_rng(int(args.seed))

    for position, global_index_value in enumerate(selected):
        global_index = int(global_index_value)
        record_index, local_index = recording_for_epoch(counts, global_index)
        source_file = files[record_index]
        if args.teacher_mode == "out_of_fold":
            checkpoint = oof_map.get(source_file)
            if checkpoint is None:
                raise KeyError(f"No OOF checkpoint declared for held-out recording {source_file!r}")
            checkpoint_key = str(checkpoint)
            if checkpoint_key not in model_cache:
                model_cache[checkpoint_key] = evaluation.build_model(
                    tuple(x_raw.shape[1:]), tuple(x_seg.shape[1:]), weights_path=checkpoint
                )
            model = model_cache[checkpoint_key]
        else:
            model = canonical_model

        epoch = np.asarray(x_raw[global_index], dtype=np.float32)
        predicted_vector = evaluation.sequence_target_vector(
            model,
            x_raw,
            x_seg,
            counts,
            global_index,
            epoch,
            x_seg[global_index],
            batch_windows=config.batch_windows,
        )
        target_class = int(np.argmax(predicted_vector))
        semantic_segments = evaluation.semantic_segmentation(epoch, target_segments=30)
        classic = evaluation.fit_reliability_lime_ranking(
            "teacher_classic",
            epoch,
            evaluation_segments,
            evaluation_segments,
            "local_stratified",
            "zero",
            config.train_masks,
            config.val_masks,
            int(args.seed + global_index * 17),
            config,
            model,
            x_raw,
            x_seg,
            counts,
            global_index,
            target_class,
            cache_dir,
        )
        sequence = evaluation.fit_reliability_lime_ranking(
            "teacher_semantic",
            epoch,
            semantic_segments,
            evaluation_segments,
            "local_stratified",
            "zero",
            config.train_masks,
            config.val_masks,
            int(args.seed + global_index * 17 + 1),
            config,
            model,
            x_raw,
            x_seg,
            counts,
            global_index,
            target_class,
            cache_dir,
        )
        sequence_repeat = evaluation.fit_reliability_lime_ranking(
            "teacher_semantic_seed_repeat",
            epoch,
            semantic_segments,
            evaluation_segments,
            "local_stratified",
            "zero",
            config.train_masks,
            config.val_masks,
            int(args.seed + global_index * 17 + 2),
            config,
            model,
            x_raw,
            x_seg,
            counts,
            global_index,
            target_class,
            cache_dir,
        )
        calibration = evaluation.run_single_feature_calibration(
            epoch,
            evaluation_segments,
            config,
            model,
            x_raw,
            x_seg,
            counts,
            global_index,
            target_class,
            cache_dir,
            class_conditioned_perturbation=False,
        )
        predicted_stage_prior_score = evaluation.physiology_prior_rank_score(
            epoch,
            evaluation_segments,
            target_class,
        )
        candidates = build_target_candidates(
            classic["score"],
            sequence["score"],
            calibration,
            rng,
            predicted_stage_prior_score,
        )
        chosen = channel_major_to_head(candidates[args.target_type])
        chosen_evaluation = evaluation.evaluate_reliability_ranking(
            "teacher_selected_target",
            candidates[args.target_type],
            epoch,
            evaluation_segments,
            config,
            model,
            x_raw,
            x_seg,
            counts,
            global_index,
            target_class,
            cache_dir,
            class_conditioned_perturbation=False,
        )
        random_score = rng.random(candidates[args.target_type].shape, dtype=np.float32)
        random_evaluation = evaluation.evaluate_reliability_ranking(
            "teacher_random_reliability_control",
            random_score,
            epoch,
            evaluation_segments,
            config,
            model,
            x_raw,
            x_seg,
            counts,
            global_index,
            target_class,
            cache_dir,
            class_conditioned_perturbation=False,
        )
        validation_r2 = float(
            sequence["fidelity"].get("validation_weighted_r2", np.nan)
        )
        mask_seed_stability = evaluation.safe_spearman(
            np.asarray(sequence["score"], dtype=np.float32),
            np.asarray(sequence_repeat["score"], dtype=np.float32),
        )
        original_mask_error = abs(
            float(sequence["original_mask_probability"])
            - float(predicted_vector[target_class])
        )
        chosen_physiological = chosen_evaluation["metrics_by_mode"]["physiological"]
        random_physiological = random_evaluation["metrics_by_mode"]["physiological"]
        deletion_advantage = float(
            chosen_physiological["deletion_AOPC"]
            - random_physiological["deletion_AOPC"]
        )
        insertion_advantage = float(
            chosen_physiological["insertion_AOPC"]
            - random_physiological["insertion_AOPC"]
        )
        fidelity_component = float(
            np.clip(
                (validation_r2 - args.reliability_min_r2)
                / max(1.0 - args.reliability_min_r2, 1e-8),
                0.0,
                1.0,
            )
        )
        stability_component = float(
            np.clip(
                (mask_seed_stability - args.reliability_min_stability)
                / max(1.0 - args.reliability_min_stability, 1e-8),
                0.0,
                1.0,
            )
        )
        original_component = float(
            np.clip(
                1.0
                - original_mask_error
                / max(args.reliability_original_mask_error_max, 1e-12),
                0.0,
                1.0,
            )
        )
        deletion_component = float(
            deletion_advantage
            >= args.reliability_min_physiological_deletion_advantage
        )
        insertion_component = float(
            insertion_advantage
            >= args.reliability_min_physiological_insertion_advantage
        )
        reliability_score = float(
            np.mean(
                [
                    fidelity_component,
                    stability_component,
                    original_component,
                    deletion_component,
                    insertion_component,
                ]
            )
        )
        gate_pass = bool(
            validation_r2 >= args.reliability_min_r2
            and mask_seed_stability >= args.reliability_min_stability
            and original_mask_error <= args.reliability_original_mask_error_max
            and deletion_advantage
            >= args.reliability_min_physiological_deletion_advantage
            and insertion_advantage
            >= args.reliability_min_physiological_insertion_advantage
        )
        importance.append(chosen)
        reliability.append(
            [
                float(classic["fidelity"].get("validation_weighted_r2", np.nan)),
                validation_r2,
                mask_seed_stability,
                original_mask_error,
                deletion_advantage,
                insertion_advantage,
                float(np.max(predicted_vector)),
                float(int(target_class == int(y_true[global_index]))),
                float(calibration["modes"]["zero"]["calibration_strength_p90"]),
                float(calibration["modes"]["physiological"]["calibration_strength_p90"]),
                reliability_score,
            ]
        )
        reliability_scores.append(reliability_score)
        reliability_gate_passes.append(gate_pass)
        teacher_confidences.append(float(np.max(predicted_vector)))
        prediction_correctness.append(int(target_class == int(y_true[global_index])))
        predicted_class.append(target_class)
        recording_index.append(record_index)
        epoch_in_recording.append(local_index)
        target_components.append(
            {
                "global_epoch_index": global_index,
                "source_file": source_file,
                "predicted_class": target_class,
                "ground_truth_recorded_for_audit_only": int(y_true[global_index]),
                "classic_validation_weighted_r2": reliability[-1][0],
                "sequence_validation_weighted_r2": reliability[-1][1],
                "mask_seed_stability_spearman": mask_seed_stability,
                "original_mask_error": original_mask_error,
                "physiological_deletion_advantage_vs_random": deletion_advantage,
                "physiological_insertion_advantage_vs_random": insertion_advantage,
                "teacher_confidence": teacher_confidences[-1],
                "prediction_correctness_audit_only": bool(prediction_correctness[-1]),
                "reliability_score": reliability_score,
                "reliability_gate_pass": gate_pass,
            }
        )
        print(f"target {position + 1}/{len(selected)}: train epoch {global_index}, class={evaluation.CLASS_LABELS[target_class]}")

    target_path = out_dir / "targets.npz"
    np.savez_compressed(
        target_path,
        global_epoch_index=np.asarray(selected, dtype=np.int64),
        recording_index=np.asarray(recording_index, dtype=np.int32),
        epoch_index_in_recording=np.asarray(epoch_in_recording, dtype=np.int32),
        predicted_class=np.asarray(predicted_class, dtype=np.int16),
        importance=np.asarray(importance, dtype=np.float32),
        reliability=np.asarray(reliability, dtype=np.float32),
        reliability_score=np.asarray(reliability_scores, dtype=np.float32),
        reliability_gate_pass=np.asarray(reliability_gate_passes, dtype=bool),
        teacher_confidence=np.asarray(teacher_confidences, dtype=np.float32),
        prediction_correctness=np.asarray(prediction_correctness, dtype=np.int8),
        reliability_columns=np.asarray(
            [
                "classic_validation_weighted_r2",
                "sequence_validation_weighted_r2",
                "mask_seed_stability_spearman",
                "original_mask_error",
                "physiological_deletion_advantage_vs_random",
                "physiological_insertion_advantage_vs_random",
                "teacher_confidence",
                "prediction_correctness_audit_only",
                "zero_strength_p90",
                "physiological_strength_p90",
                "reliability_score",
            ]
        ),
        source_files=np.asarray(files),
        file_sample_counts=np.asarray(counts, dtype=np.int32),
    )
    manifest = {
        "schema_version": "sequence_xai_targets_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TRAIN_ONLY_TARGET_GENERATION_COMPLETE",
        "split": "train",
        "test_data_accessed": False,
        "target_type": args.target_type,
        "teacher_mode": args.teacher_mode,
        "self_distillation_in_sample": args.teacher_mode == "self_distillation_in_sample",
        "out_of_fold": args.teacher_mode == "out_of_fold",
        "ground_truth_used_to_construct_targets": False,
        "ground_truth_saved_for_audit_only": True,
        "predicted_class_used_as_explanation_output": True,
        "predicted_stage_concept_prior_used": bool(stage_prior_ablation),
        "ground_truth_stage_concept_prior_used": False,
        "class_conditioned_physiological_perturbation_used": False,
        "label_conditioned_targets_allowed": False,
        "reliability_gate": {
            "validation_weighted_r2_min": float(args.reliability_min_r2),
            "mask_seed_stability_spearman_min": float(args.reliability_min_stability),
            "original_mask_error_max": float(args.reliability_original_mask_error_max),
            "physiological_deletion_advantage_vs_random_min": float(
                args.reliability_min_physiological_deletion_advantage
            ),
            "physiological_insertion_advantage_vs_random_min": float(
                args.reliability_min_physiological_insertion_advantage
            ),
            "prediction_correctness_used_by_gate": False,
            "teacher_confidence_used_by_gate": False,
            "participating_target_count": int(np.sum(reliability_gate_passes)),
        },
        "target_shape": [int(len(selected)), 30, 3],
        "target_normalization": "nonnegative_unit_sum_per_epoch",
        "model_segment_feature_grid": "semantic_segmentation_v1_matching_main_training_and_inference",
        "xai_target_output_grid": "30_equal_width_bins_by_3_channels",
        "selected_train_epoch_count": int(len(selected)),
        "loaded_train_recording_count": int(len(files)),
        "loaded_train_epoch_count": int(len(x_raw)),
        "source_files": files,
        "config": asdict(config),
        "feedback_experiment_config_path": str(args.config.resolve()),
        "feedback_experiment_config_sha256": sha256_file(args.config),
        "seed": int(args.seed),
        "checkpoint": str(evaluation.CKPT_PATH.resolve()),
        "checkpoint_sha256": sha256_file(evaluation.CKPT_PATH),
        "targets_path": str(target_path),
        "targets_sha256": sha256_file(target_path),
        "target_components": target_components,
        "claim_boundary": (
            "In-sample self-distillation target; not OOF evidence."
            if args.teacher_mode == "self_distillation_in_sample"
            else "OOF target generated with the explicitly declared held-out-recording checkpoint map."
        ),
    }
    write_json(out_dir / "target_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "target_type", "teacher_mode", "target_shape", "targets_sha256")}, indent=2))


if __name__ == "__main__":
    main()
