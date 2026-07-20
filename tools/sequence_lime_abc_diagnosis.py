#!/usr/bin/env python
"""Sharded M0--M8 / Y0--Y2 / T0--T1 / R0--R7 A/B/C diagnosis."""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xai.model_faithful_sequence_lime import (  # noqa: E402
    compose_diagnostic_methods,
    positive_rank_score,
    resolve_target_protocols,
    transform_surrogate_targets,
)
from tools import sequence_lime_paper_evaluation as evaluator  # noqa: E402


ABC_ROOT = PROJECT_ROOT / "outputs/sequence_lime_abc_diagnosis/isruc_seed42"
FORMAL_MANIFEST = PROJECT_ROOT / (
    "outputs/sequence_lime_paper_evaluation/isruc_seed42/runs/"
    "full_evaluation_v113_merged_guarded/sample_manifest_full_evaluation.csv"
)
FIXED_MANIFEST = ABC_ROOT / "fixed_36_samples.json"
SMOKE_IDS = (284, 1336, 2633, 12972)
DIAGNOSTIC_IDS = (284, 1165, 1336, 2633, 3637, 4073, 4986, 5014, 6420, 7518, 10829, 12972)
METHODS = tuple(f"M{i}" for i in range(9))
TARGETS = ("Y0", "Y1", "Y2")
PROTOCOLS = ("T0", "T1")
FRACTIONS = np.asarray([0.0, 0.10, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_fixed_manifest() -> None:
    frame = pd.read_csv(FORMAL_MANIFEST)
    if len(frame) != 36 or int(frame["correct"].sum()) != 24 or int(frame["transition_boundary"].sum()) != 17:
        raise RuntimeError("formal v1.1.3 fixed-36 composition is invalid")
    records = json.loads(frame.to_json(orient="records"))
    payload = {
        "schema_version": "sequence_lime_fixed_36_v1",
        "source_manifest": str(FORMAL_MANIFEST.relative_to(PROJECT_ROOT)),
        "source_manifest_sha256": evaluator.file_sha256(FORMAL_MANIFEST),
        "sample_count": 36,
        "correct_count": 24,
        "incorrect_count": 12,
        "transition_count": 17,
        "stable_count": 19,
        "smoke_ids": list(SMOKE_IDS),
        "diagnostic_ids": list(DIAGNOSTIC_IDS),
        "samples": records,
    }
    write_json(FIXED_MANIFEST, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "samples"}, indent=2))


def load_context():
    artifact = evaluator.load_artifact_split(np.load(evaluator.PRED_PATH), "test")
    current = evaluator.OUT_ROOT / "cache/current_sequence_probs_float32.npy"
    if current.exists():
        artifact = dict(artifact)
        artifact["probs"] = np.load(current).astype(np.float32)
        artifact["pred"] = np.argmax(artifact["probs"], axis=1).astype(np.int16)
    split_files = evaluator.normalize_split_files(evaluator.load_split_spec())
    x_raw, x_seg, _ = evaluator.load_or_build_test_inputs(evaluator.OUT_ROOT, split_files, artifact)
    return artifact, x_raw, x_seg


def subset_frame(name: str) -> pd.DataFrame:
    fixed = json.loads(FIXED_MANIFEST.read_text(encoding="utf-8"))
    frame = pd.DataFrame(fixed["samples"])
    ids = SMOKE_IDS if name == "smoke" else DIAGNOSTIC_IDS if name == "diagnostic" else tuple(frame["global_epoch_index"])
    selected = frame.set_index("global_epoch_index").loc[list(ids)].reset_index()
    if name == "smoke":
        if int(selected["correct"].sum()) != 2 or set(selected["transition_boundary"].astype(bool)) != {False, True}:
            raise RuntimeError("smoke subset composition is invalid")
    return selected


def calibration_masks(feature_count: int) -> np.ndarray:
    identity = np.eye(feature_count, dtype=np.int32)
    return np.vstack([
        np.ones((1, feature_count), dtype=np.int32),
        np.zeros((1, feature_count), dtype=np.int32),
        np.ones((feature_count, feature_count), dtype=np.int32) - identity,
        identity,
    ])


def marginal_score(target_values: np.ndarray, offset: int, feature_count: int) -> np.ndarray:
    values = np.asarray(target_values, dtype=np.float32)
    original = float(values[offset])
    full = float(values[offset + 1])
    deletion = original - values[offset + 2: offset + 2 + feature_count]
    insertion = values[offset + 2 + feature_count: offset + 2 + 2 * feature_count] - full
    return positive_rank_score(
        0.5 * positive_rank_score(deletion) + 0.5 * positive_rank_score(insertion)
    )


def signal_energy(epoch: np.ndarray, segments: Sequence[Tuple[int, int]]) -> np.ndarray:
    result = np.zeros((epoch.shape[1], len(segments)), dtype=np.float32)
    for channel in range(epoch.shape[1]):
        for segment_index, (start, end) in enumerate(segments):
            values = np.asarray(epoch[int(start):int(end), channel], dtype=np.float32)
            result[channel, segment_index] = float(np.mean(values * values)) if values.size else 0.0
    return positive_rank_score(result.ravel())


def concept_scores(epoch, segments, target_class: int) -> Tuple[np.ndarray, np.ndarray]:
    evaluator.ensure_lime_imports()
    predicted = evaluator.physiology_prior_rank_score(epoch, segments, target_class)
    concept_prior = np.mean(
        np.vstack([evaluator.get_stage_concept_prior(stage) for stage in range(len(evaluator.CLASS_LABELS))]),
        axis=0,
    )
    channel_prior = np.mean(
        np.vstack([evaluator.get_clinical_prior(stage) for stage in range(len(evaluator.CLASS_LABELS))]),
        axis=0,
    )
    independent = evaluator.physiology_prior_rank_score(
        epoch,
        segments,
        target_class,
        concept_prior_override=concept_prior,
        channel_prior_override=channel_prior,
    )
    return positive_rank_score(predicted), positive_rank_score(independent)


def fit_score(masks, values, train_masks: int):
    ridge, fidelity, _ = evaluator.fit_surrogate(
        masks,
        np.asarray(values, dtype=np.float32),
        train_masks,
        alpha_grid=(1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0),
        kernel_width_grid=(0.15, 0.25, 0.5, 0.75),
        response_transform="probability",
    )
    return np.asarray(ridge.coef_, dtype=np.float32), fidelity


def faithfulness_masks(score_map: Mapping[Tuple[str, str, str], np.ndarray]) -> Tuple[np.ndarray, Dict[Tuple[str, str, str], Tuple[list, list]]]:
    rows = []
    locations = {}
    for key, score in score_map.items():
        order = np.argsort(-np.asarray(score), kind="stable")
        deletion_indices, insertion_indices = [], []
        for fraction in FRACTIONS:
            count = int(math.ceil(float(fraction) * len(order)))
            deletion = np.ones(len(order), dtype=np.int32)
            insertion = np.zeros(len(order), dtype=np.int32)
            if count:
                deletion[order[:count]] = 0
                insertion[order[:count]] = 1
            deletion_indices.append(len(rows))
            rows.append(deletion)
            insertion_indices.append(len(rows))
            rows.append(insertion)
        locations[key] = (deletion_indices, insertion_indices)
    return np.asarray(rows, dtype=np.int32), locations


def curve_summary(vectors, target_class: int, indices: Tuple[list, list]) -> Dict[str, float]:
    deletion_indices, insertion_indices = indices
    probabilities = np.asarray(vectors, dtype=np.float32)[:, int(target_class)]
    deletion = probabilities[deletion_indices]
    insertion = probabilities[insertion_indices]
    deletion_auc = float(np.trapz(deletion, FRACTIONS))
    insertion_auc = float(np.trapz(insertion, FRACTIONS))
    return {
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "deletion_aopc": float(deletion[0] - np.mean(deletion[1:])),
        "insertion_aopc": float(np.mean(insertion[1:]) - insertion[0]),
        "deletion_curve": json.dumps(deletion.astype(float).tolist(), separators=(",", ":")),
        "insertion_curve": json.dumps(insertion.astype(float).tolist(), separators=(",", ":")),
    }


def run_sample(
    sample: Mapping[str, object],
    model,
    model_level: str,
    seed: int,
    train_masks: int,
    val_masks: int,
    artifact,
    x_raw,
    x_seg,
    out_dir: Path,
) -> pd.DataFrame:
    started = time.perf_counter()
    evaluator.ensure_lime_imports()
    global_index = int(sample["global_epoch_index"])
    epoch = np.asarray(x_raw[global_index], dtype=np.float32)
    segments = evaluator.semantic_segmentation(epoch, target_segments=30)
    feature_count = int(epoch.shape[1] * len(segments))
    if feature_count != 90:
        raise RuntimeError(f"expected 90 channel-segment features, observed {feature_count}")
    base_masks = evaluator.make_masks(
        feature_count,
        train_masks,
        val_masks,
        seed + global_index,
        sampling="local_stratified",
    )
    calibration = calibration_masks(feature_count)
    combined_masks = np.vstack([base_masks, calibration])
    calibration_offset = len(base_masks)
    original_vector = np.asarray(artifact["probs"][global_index], dtype=np.float32)
    randomized_vector = evaluator.sequence_target_vector(
        model,
        x_raw,
        x_seg,
        artifact["file_sample_counts"],
        global_index,
        epoch,
        x_seg[global_index],
        batch_windows=16,
    )
    protocols = resolve_target_protocols(original_vector, randomized_vector)
    identities = {
        protocol: evaluator.evaluated_model_identity(
            model,
            f"{model_level}_seed{seed}",
            randomization_level=model_level,
            randomization_seed=seed,
            method_id="M0-M4_shared_component_pool",
            protocol_id=protocol,
        )
        for protocol in PROTOCOLS
    }
    zero_eval = evaluator.evaluate_masks_for_target_epoch(
        combined_masks,
        model,
        x_raw,
        x_seg,
        artifact["file_sample_counts"],
        global_index,
        protocols["T0"],
        epoch,
        segments,
        "zero",
        16,
        out_dir,
        "abc_shared_zero",
        model_state_id=f"{model_level}_seed{seed}",
        perturbation_target_class=protocols["T0"],
        model_identity=identities["T0"],
    )
    zero_vectors = np.asarray(zero_eval["full_probability_vectors"], dtype=np.float32)
    physiological_vectors = {}
    for protocol, target_class in protocols.items():
        physiological = evaluator.evaluate_masks_for_target_epoch(
            combined_masks,
            model,
            x_raw,
            x_seg,
            artifact["file_sample_counts"],
            global_index,
            target_class,
            epoch,
            segments,
            "physiological",
            16,
            out_dir,
            f"abc_shared_physiological_{protocol}",
            model_state_id=f"{model_level}_seed{seed}",
            perturbation_target_class=target_class,
            model_identity=identities[protocol],
        )
        physiological_vectors[protocol] = np.asarray(
            physiological["full_probability_vectors"], dtype=np.float32
        )
    energy = signal_energy(epoch, segments)
    score_map: Dict[Tuple[str, str, str], np.ndarray] = {}
    fidelity_map = {}
    for protocol, target_class in protocols.items():
        zero_targets = transform_surrogate_targets(zero_vectors, target_class)
        physiological_targets = transform_surrogate_targets(
            physiological_vectors[protocol], target_class
        )
        predicted_prior, independent_prior = concept_scores(epoch, segments, target_class)
        random_score = np.random.default_rng(seed * 100000 + global_index * 10 + target_class).random(feature_count)
        for target_name in TARGETS:
            coefficient, fidelity = fit_score(
                base_masks,
                zero_targets[target_name][: len(base_masks)],
                train_masks,
            )
            methods = compose_diagnostic_methods(
                coefficient,
                marginal_score(zero_targets[target_name], calibration_offset, feature_count),
                marginal_score(physiological_targets[target_name], calibration_offset, feature_count),
                predicted_prior,
                independent_prior,
                energy,
                random_score,
            )
            fidelity_map[(protocol, target_name)] = fidelity
            for method, score in methods.items():
                score_map[(protocol, target_name, method)] = np.asarray(score, dtype=np.float32)
    curve_masks, locations = faithfulness_masks(score_map)
    curve_eval = evaluator.evaluate_masks_for_target_epoch(
        curve_masks,
        model,
        x_raw,
        x_seg,
        artifact["file_sample_counts"],
        global_index,
        protocols["T0"],
        epoch,
        segments,
        "zero",
        16,
        out_dir,
        "abc_faithfulness_bundle_T0_T1",
        model_state_id=f"{model_level}_seed{seed}",
        perturbation_target_class=protocols["T0"],
        model_identity=identities["T0"],
    )
    curve_vectors = np.asarray(curve_eval["full_probability_vectors"], dtype=np.float32)
    runtime = float(time.perf_counter() - started)
    query_count = int(len(combined_masks) * 3 + len(curve_masks))
    rows = []
    for (protocol, target_name, method), score in score_map.items():
        fidelity = fidelity_map[(protocol, target_name)]
        curve = curve_summary(curve_vectors, protocols[protocol], locations[(protocol, target_name, method)])
        matrix = score.reshape(epoch.shape[1], len(segments))
        rows.append({
            "global_epoch_index": global_index,
            "true_stage": sample["true_stage"],
            "original_predicted_stage": sample["predicted_stage"],
            "correct": bool(sample["correct"]),
            "transition_boundary": bool(sample["transition_boundary"]),
            "randomization_level": model_level,
            "randomization_seed": int(seed),
            "target_class_protocol": protocol,
            "target_class": int(protocols[protocol]),
            "surrogate_target": target_name,
            "method": method,
            "model_weight_fingerprint": identities[protocol]["model_weight_fingerprint"],
            "cache_identity_digest": identities[protocol]["identity_digest"],
            "validation_weighted_r2": float(fidelity.get("validation_weighted_r2", np.nan)),
            "validation_mae": float(fidelity.get("validation_weighted_mae", np.nan)),
            "validation_rmse": float(fidelity.get("validation_weighted_rmse", np.nan)),
            "generalization_gap": float(fidelity.get("generalization_gap", np.nan)),
            "original_mask_error": float(fidelity.get("surrogate_original_mask_error", np.nan)),
            "effective_sample_size": float(fidelity.get("effective_sample_size", np.nan)),
            "attribution_l1_norm": float(np.linalg.norm(score, ord=1)),
            "attribution_l2_norm": float(np.linalg.norm(score, ord=2)),
            "attribution_entropy": float(-np.sum((score / max(float(score.sum()), 1e-8)) * np.log(np.clip(score / max(float(score.sum()), 1e-8), 1e-8, 1.0)))),
            "dominant_channel": int(np.argmax(np.sum(np.abs(matrix), axis=1))),
            "query_count": query_count,
            "runtime_seconds": runtime / max(len(score_map), 1),
            "attribution_score": json.dumps(score.astype(float).tolist(), separators=(",", ":")),
            "human_prior_used": bool(method in {"M4", "M5", "M6"}),
            "model_attribution_claim": bool(method not in {"M5", "M6", "M7", "M8"}),
            **curve,
        })
    return pd.DataFrame(rows)


def run_shard(args) -> None:
    if not FIXED_MANIFEST.exists():
        prepare_fixed_manifest()
    samples = subset_frame(args.subset)
    run_id = f"{args.subset}_{args.randomization_level}_seed{args.randomization_seed}"
    out_dir = ABC_ROOT / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sample_level_components.csv"
    completed = set()
    existing = pd.DataFrame()
    if args.resume and csv_path.exists():
        existing = pd.read_csv(csv_path)
        counts = existing.groupby("global_epoch_index").size()
        completed = set(int(index) for index, count in counts.items() if int(count) == 54)
    artifact, x_raw, x_seg = load_context()
    model = evaluator.randomized_model(
        x_raw.shape[1:],
        x_seg.shape[1:],
        args.randomization_level,
        seed=args.randomization_seed,
    )
    audit_rows = getattr(model, "_sequence_lime_randomization_weight_audit")
    pd.DataFrame(audit_rows).to_csv(out_dir / "randomization_weight_audit.csv", index=False)
    all_rows = [existing] if not existing.empty else []
    for _, sample in samples.iterrows():
        global_index = int(sample["global_epoch_index"])
        if global_index in completed:
            print(f"resume skip sample={global_index}", flush=True)
            continue
        print(f"start sample={global_index} run={run_id}", flush=True)
        result = run_sample(
            sample,
            model,
            args.randomization_level,
            args.randomization_seed,
            args.train_masks,
            args.validation_masks,
            artifact,
            x_raw,
            x_seg,
            out_dir,
        )
        all_rows.append(result)
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(csv_path, index=False)
        print(f"done sample={global_index} rows={len(result)}", flush=True)
    final = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if final.empty:
        raise RuntimeError("diagnostic shard produced no rows")
    manifest = {
        "schema_version": "abc_diagnostic_shard_v1",
        "status": "COMPLETE",
        "run_id": run_id,
        "subset": args.subset,
        "sample_count": int(final["global_epoch_index"].nunique()),
        "row_count": int(len(final)),
        "expected_rows_per_sample": 54,
        "randomization_level": args.randomization_level,
        "randomization_seed": int(args.randomization_seed),
        "train_masks": int(args.train_masks),
        "validation_masks": int(args.validation_masks),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "checkpoint_sha256": evaluator.file_sha256(evaluator.CKPT_PATH),
        "fixed_manifest_sha256": evaluator.file_sha256(FIXED_MANIFEST),
        "csv_sha256": evaluator.file_sha256(csv_path),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-manifest", action="store_true")
    parser.add_argument("--subset", choices=("smoke", "diagnostic", "full"), default="smoke")
    parser.add_argument("--randomization-level", choices=tuple(f"R{i}" for i in range(8)), default="R0")
    parser.add_argument("--randomization-seed", type=int, choices=(3101, 3102, 3103), default=3101)
    parser.add_argument("--train-masks", type=int, default=64)
    parser.add_argument("--validation-masks", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.prepare_manifest:
        prepare_fixed_manifest()
    else:
        run_shard(arguments)
