#!/usr/bin/env python
"""Validation-first MF-SeqLIME neighborhood/surrogate improvement evaluation."""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.xai.model_faithful_sequence_lime import (  # noqa: E402
    positive_rank_score,
    resolve_target_protocols,
    transform_surrogate_targets,
)
from src.xai.neighborhood_sampling import (  # noqa: E402
    calibration_masks,
    candidate_budget,
    compute_weak_zero_calibration,
    generate_structured_split_masks,
    refine_boundary_masks,
)
from tools import sequence_lime_abc_diagnosis as abc  # noqa: E402
from tools import sequence_lime_paper_evaluation as evaluator  # noqa: E402


ROOT = PROJECT / "outputs/mf_seqlime_surrogate_improvement/isruc_seed42"
VALIDATION_SAMPLES = ROOT / "validation_samples.json"
TEST_SAMPLES = ROOT / "test_samples.json"
FROZEN_SELECTION = ROOT / "frozen_validation_selection.json"
TRAIN_MASKS = 1200
VALIDATION_MASKS = 306
NEIGHBORHOOD_MASKS = 1506
FAITHFULNESS_MASKS = 24
TOTAL_QUERY_BUDGET = 1530
PROTOCOLS = ("T0", "T1")
ALPHAS = (0.0, 0.10, 0.20, 0.30)
PROFILE_CANDIDATES = {
    "smoke": (
        "C0_CLASSIC_Y2",
        "C1_STRUCTURED_M50",
        "C2_BOUNDARY_M50",
        "C3_WEAK_ZERO_A20",
    ),
    "full": (
        "C0_CLASSIC_Y2",
        "C1_STRUCTURED_M30",
        "C1_STRUCTURED_M50",
        "C1_STRUCTURED_M70",
        "C2_BOUNDARY_M50",
        "C3_WEAK_ZERO_A00",
        "C3_WEAK_ZERO_A10",
        "C3_WEAK_ZERO_A20",
        "C3_WEAK_ZERO_A30",
    ),
}


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_inputs(split: str):
    artifact_split = "val" if split == "validation" else "test"
    artifact = evaluator.load_artifact_split(np.load(evaluator.PRED_PATH), artifact_split)
    split_files = evaluator.normalize_split_files(evaluator.load_split_spec())
    if split == "test":
        if not FROZEN_SELECTION.exists():
            raise RuntimeError("test is blocked until validation selection is frozen")
        frozen = json.loads(FROZEN_SELECTION.read_text(encoding="utf-8"))
        if not bool(frozen.get("test_evaluation_allowed")):
            raise RuntimeError("validation decision forbids test evaluation")
        x_raw, x_seg, _ = evaluator.load_or_build_test_inputs(
            evaluator.OUT_ROOT, split_files, artifact
        )
        current = evaluator.OUT_ROOT / "cache/current_sequence_probs_float32.npy"
        if current.exists():
            artifact = dict(artifact)
            artifact["probs"] = np.load(current).astype(np.float32)
            artifact["pred"] = np.argmax(artifact["probs"], axis=1).astype(np.int16)
        return artifact, x_raw, x_seg

    cache_candidates = (
        ROOT / "cache/validation_inputs_segments.npz",
        PROJECT / "outputs/sequence_lime_two_versions/isruc_seed42/cache/validation_inputs_segments.npz",
    )
    for cache in cache_candidates:
        if cache.exists():
            cached = np.load(cache)
            return artifact, cached["x_raw"].astype(np.float32), cached["x_seg"].astype(np.float32)
    raise RuntimeError("frozen validation input cache is unavailable; do not rebuild inside an experiment shard")


def sample_frame(stage: str, split: str) -> pd.DataFrame:
    manifest = TEST_SAMPLES if split == "test" else VALIDATION_SAMPLES
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["samples"])
    if split == "test":
        return frame
    key = {
        "smoke": "smoke_sample_ids",
        "small": "small_validation_sample_ids",
        "final": "final_validation_sample_ids",
    }[stage]
    selected = [int(value) for value in payload[key]]
    return frame.set_index("global_epoch_index").loc[selected].reset_index()


def structured_split(
    seed: int,
    fraction: float,
    train_n: int,
    val_n: int,
    excluded_masks: np.ndarray | None = None,
) -> tuple[np.ndarray, Mapping[str, object]]:
    split = generate_structured_split_masks(
        3, 30, train_n, val_n, seed=seed,
        contiguous_fraction=fraction, excluded_masks=excluded_masks,
    )
    return split.masks.astype(np.int32), dict(split.provenance)


def logit_boundary_margin(vectors: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(vectors, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("full probability vectors are required for boundary refinement")
    top_two = np.sort(np.clip(probabilities, 1e-8, 1.0), axis=1)[:, -2:]
    return np.log(top_two[:, 1]) - np.log(top_two[:, 0])


def fit_y2(masks: np.ndarray, vectors: np.ndarray, target_class: int):
    targets = transform_surrogate_targets(vectors, int(target_class))["Y2"]
    ridge, fidelity, _ = evaluator.fit_surrogate(
        masks,
        np.asarray(targets, dtype=np.float32),
        TRAIN_MASKS,
        alpha_grid=(1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0),
        kernel_width_grid=(0.15, 0.25, 0.5, 0.75),
        response_transform="probability",
    )
    return np.asarray(ridge.coef_, dtype=np.float32), fidelity, np.asarray(targets, dtype=np.float32)


def _alpha_from_candidate(candidate: str) -> float:
    return int(candidate.rsplit("A", 1)[1]) / 100.0


def run_sample(sample, model, level: str, seed: int, candidates, artifact, x_raw, x_seg, out_dir: Path):
    started = time.perf_counter()
    evaluator.ensure_lime_imports()
    global_index = int(sample["global_epoch_index"])
    epoch = np.asarray(x_raw[global_index], dtype=np.float32)
    segments = evaluator.semantic_segmentation(epoch, target_segments=30)
    feature_count = int(epoch.shape[1] * len(segments))
    if feature_count != 90:
        raise RuntimeError(f"expected 90 channel-segment features, observed {feature_count}")
    original_vector = np.asarray(artifact["probs"][global_index], dtype=np.float32)
    evaluated_vector = evaluator.sequence_target_vector(
        model, x_raw, x_seg, artifact["file_sample_counts"], global_index,
        epoch, x_seg[global_index], batch_windows=16,
    )
    protocols = resolve_target_protocols(original_vector, evaluated_vector)
    identity = evaluator.evaluated_model_identity(
        model,
        f"{level}_seed{seed}",
        randomization_level=level,
        randomization_seed=seed,
        method_id="MF_SURROGATE_SHARED_FULL_VECTOR_CACHE",
        protocol_id="T0_T1_FROM_FULL_VECTORS",
    )

    accounting_rows = []
    mask_rows = []

    def evaluate(name: str, masks: np.ndarray):
        payload = evaluator.evaluate_masks_for_target_epoch(
            np.asarray(masks, dtype=np.int32), model, x_raw, x_seg,
            artifact["file_sample_counts"], global_index, protocols["T0"], epoch,
            segments, "zero", 16, out_dir, name,
            model_state_id=f"{level}_seed{seed}",
            perturbation_target_class=protocols["T0"], model_identity=identity,
        )
        row = {
            "global_epoch_index": global_index,
            "randomization_level": level,
            "randomization_seed": int(seed),
            "query_group": name,
            **payload["accounting"],
            **{f"timing_{key}": value for key, value in payload["timings_sec"].items()},
        }
        accounting_rows.append(row)
        if row["status"] != "WINDOW_INSTANCE_ACCOUNTING_PASS":
            raise RuntimeError(f"window accounting failed for {name}")
        return np.asarray(payload["full_probability_vectors"], dtype=np.float32)

    strategies = {}
    candidate_set = set(candidates)
    sample_seed = int(seed * 1000003 + global_index * 101)
    if "C0_CLASSIC_Y2" in candidate_set:
        masks = evaluator.make_masks(90, TRAIN_MASKS, VALIDATION_MASKS, sample_seed, sampling="bernoulli")
        vectors = evaluate("C0_BERNOULLI", masks)
        strategies["C0_CLASSIC_Y2"] = (masks, vectors, None, {"strategy": "S0_BERNOULLI"})

    for fraction, suffix in ((0.30, "M30"), (0.50, "M50"), (0.70, "M70")):
        candidate = f"C1_STRUCTURED_{suffix}"
        if candidate in candidate_set:
            masks, provenance = structured_split(sample_seed + int(fraction * 1000), fraction, TRAIN_MASKS, VALIDATION_MASKS)
            vectors = evaluate(candidate, masks)
            strategies[candidate] = (masks, vectors, None, provenance)

    if "C2_BOUNDARY_M50" in candidate_set:
        base, base_provenance = structured_split(sample_seed + 2000, 0.50, 840, VALIDATION_MASKS)
        base_train, heldout = base[:840], base[840:]
        base_vectors = evaluate("C2_BOUNDARY_BASE", base_train)
        adaptive = refine_boundary_masks(
            base_train, logit_boundary_margin(base_vectors), 360, sample_seed + 2001,
            maximum_total_queries=1200,
            excluded_masks=heldout,
        )
        masks = np.vstack([base_train, adaptive.masks, heldout]).astype(np.int32)
        vectors = evaluate("C2_BOUNDARY_M50", masks)
        provenance = {**base_provenance, **dict(adaptive.provenance), "base_train_masks": 840, "adaptive_train_masks": 360}
        strategies["C2_BOUNDARY_M50"] = (masks, vectors, None, provenance)

    c3_candidates = sorted(candidate_set.intersection({f"C3_WEAK_ZERO_A{int(alpha * 100):02d}" for alpha in ALPHAS}))
    if c3_candidates:
        calibration = calibration_masks(90)
        model_masks, base_provenance = structured_split(
            sample_seed + 3000, 0.50, 713, VALIDATION_MASKS,
            excluded_masks=calibration.masks,
        )
        base_train, heldout = model_masks[:713], model_masks[713:]
        base_vectors = evaluate("C3_BOUNDARY_BASE", base_train)
        adaptive = refine_boundary_masks(
            base_train, logit_boundary_margin(base_vectors), 305, sample_seed + 3001,
            maximum_total_queries=1018,
            excluded_masks=np.vstack([heldout, calibration.masks]),
        )
        masks = np.vstack([calibration.masks, base_train, adaptive.masks, heldout]).astype(np.int32)
        if len(masks) != NEIGHBORHOOD_MASKS:
            raise RuntimeError("C3 neighborhood allocation is not 1506")
        vectors = evaluate("C3_WEAK_ZERO_SHARED", masks)
        provenance = {
            **base_provenance, **dict(adaptive.provenance), "base_train_masks": 713,
            "adaptive_train_masks": 305, "zero_calibration_masks": 182,
            "calibration_offset": 0, "current_evaluated_model_only": True,
        }
        for candidate in c3_candidates:
            strategies[candidate] = (masks, vectors, _alpha_from_candidate(candidate), provenance)

    scores = {}
    fidelities = {}
    components = {}
    for candidate in candidates:
        masks, vectors, alpha, provenance = strategies[candidate]
        for protocol in PROTOCOLS:
            target_class = int(protocols[protocol])
            coefficient, fidelity, targets = fit_y2(masks, vectors, target_class)
            classic = positive_rank_score(coefficient)
            zero = np.zeros(feature_count, dtype=np.float32)
            if candidate.startswith("C3_"):
                zero = abc.marginal_score(targets, 0, feature_count)
                score = compute_weak_zero_calibration(classic, zero, float(alpha))
            else:
                score = classic
            scores[(candidate, protocol)] = np.asarray(score, dtype=np.float32)
            fidelities[(candidate, protocol)] = fidelity
            components[(candidate, protocol)] = (classic, zero)
        retained = np.mean(masks, axis=1)
        mask_rows.append({
            "global_epoch_index": global_index, "randomization_level": level,
            "randomization_seed": int(seed), "candidate": candidate,
            "mask_count": int(len(masks)), "unique_mask_count": int(len({row.tobytes() for row in masks})),
            "retained_fraction_mean": float(np.mean(retained)),
            "retained_fraction_std": float(np.std(retained)),
            "retained_fraction_min": float(np.min(retained)),
            "retained_fraction_max": float(np.max(retained)),
            "provenance": json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        })

    curve_masks, locations = abc.faithfulness_masks(scores)
    curve_vectors = evaluate("FAITHFULNESS_ALL_CANDIDATES", curve_masks)
    runtime = float(time.perf_counter() - started)
    rows = []
    for candidate in candidates:
        budget = candidate_budget(
            zero_calibration_queries=182 if candidate.startswith("C3_") else 0
        )
        if budget["total_query_budget"] != TOTAL_QUERY_BUDGET:
            raise RuntimeError("candidate query budget drifted")
        for protocol in PROTOCOLS:
            score = scores[(candidate, protocol)]
            classic, zero = components[(candidate, protocol)]
            fidelity = fidelities[(candidate, protocol)]
            curve = abc.curve_summary(curve_vectors, protocols[protocol], locations[(candidate, protocol)])
            rows.append({
                "global_epoch_index": global_index,
                "recording_id": int(sample["recording_id"]),
                "true_stage": sample["true_stage"],
                "original_predicted_stage": sample["predicted_stage"],
                "correct": bool(sample["correct"]),
                "transition_boundary": bool(sample["transition_boundary"]),
                "selection_split": sample.get("selection_split", "test"),
                "test_data_used_for_selection": False,
                "randomization_level": level,
                "randomization_seed": int(seed),
                "target_class_protocol": protocol,
                "target_class": int(protocols[protocol]),
                "surrogate_target": "Y2",
                "candidate": candidate,
                "candidate_category": candidate[:2],
                "model_weight_fingerprint": identity["model_weight_fingerprint"],
                "cache_identity_digest": identity["identity_digest"],
                "validation_weighted_r2": float(fidelity["validation_weighted_r2"]),
                "validation_mae": float(fidelity["validation_weighted_mae"]),
                "validation_rmse": float(fidelity["validation_weighted_rmse"]),
                "generalization_gap": float(fidelity["generalization_gap"]),
                "effective_sample_size": float(fidelity["effective_sample_size"]),
                "selected_ridge_alpha": float(fidelity["selected_ridge_alpha"]),
                "selected_kernel_width": float(fidelity["selected_kernel_width"]),
                "query_count": TOTAL_QUERY_BUDGET,
                "neighborhood_query_count": NEIGHBORHOOD_MASKS,
                "faithfulness_query_count": FAITHFULNESS_MASKS,
                "zero_calibration_query_count": int(budget["zero_calibration_queries"]),
                "model_driven_neighborhood_queries": int(budget["model_driven_neighborhood_queries"]),
                "runtime_seconds_amortized": runtime / max(len(candidates) * len(PROTOCOLS), 1),
                "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "attribution_score": json.dumps(score.astype(float).tolist(), separators=(",", ":")),
                "classic_component_score": json.dumps(classic.astype(float).tolist(), separators=(",", ":")),
                "zero_component_score": json.dumps(zero.astype(float).tolist(), separators=(",", ":")),
                "weak_zero_alpha": float(_alpha_from_candidate(candidate)) if candidate.startswith("C3_") else 0.0,
                "human_annotation_used": False,
                "true_label_used_by_method": False,
                **curve,
            })
    return pd.DataFrame(rows), pd.DataFrame(mask_rows), pd.DataFrame(accounting_rows)


def run_shard(args) -> None:
    split = args.data_split
    if split == "test" and args.stage != "final":
        raise ValueError("test evaluation only accepts the final stage")
    candidates = list(PROFILE_CANDIDATES[args.candidate_profile])
    if split == "test":
        frozen = json.loads(FROZEN_SELECTION.read_text(encoding="utf-8"))
        selected = str(frozen["selected_candidate"])
        candidates = ["C0_CLASSIC_Y2"] + ([] if selected == "C0_CLASSIC_Y2" else [selected])
    frame = sample_frame(args.stage, split)
    frame = frame.iloc[args.shard_index :: args.shard_count].reset_index(drop=True)
    run_id = (
        f"{split}_{args.stage}_{args.randomization_level}_seed{args.randomization_seed}_"
        f"shard{args.shard_index + 1}of{args.shard_count}"
    )
    if args.run_tag:
        if not args.run_tag.replace("_", "").replace("-", "").isalnum():
            raise ValueError("run_tag may contain only letters, digits, underscores, and hyphens")
        run_id = f"{run_id}_{args.run_tag}"
    out_dir = ROOT / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "sample_level_metrics.csv"
    mask_path = out_dir / "mask_distribution_summary.csv"
    accounting_path = out_dir / "query_accounting.csv"
    existing = pd.read_csv(sample_path) if args.resume and sample_path.exists() else pd.DataFrame()
    completed = set()
    expected_rows = len(candidates) * len(PROTOCOLS)
    if not existing.empty:
        counts = existing.groupby("global_epoch_index").size()
        completed = {int(index) for index, count in counts.items() if int(count) == expected_rows}
    artifact, x_raw, x_seg = load_inputs(split)
    model = evaluator.randomized_model(
        x_raw.shape[1:], x_seg.shape[1:], args.randomization_level,
        seed=args.randomization_seed,
    )
    pd.DataFrame(getattr(model, "_sequence_lime_randomization_weight_audit")).to_csv(
        out_dir / "randomization_weight_audit.csv", index=False
    )
    sample_frames = [existing] if not existing.empty else []
    mask_frames = [pd.read_csv(mask_path)] if args.resume and mask_path.exists() else []
    accounting_frames = [pd.read_csv(accounting_path)] if args.resume and accounting_path.exists() else []
    for _, sample in frame.iterrows():
        sample_id = int(sample.global_epoch_index)
        if sample_id in completed:
            print(f"resume skip sample={sample_id}", flush=True)
            continue
        print(f"start run={run_id} sample={sample_id} candidates={len(candidates)}", flush=True)
        rows, masks, accounting = run_sample(
            sample.to_dict(), model, args.randomization_level, args.randomization_seed,
            candidates, artifact, x_raw, x_seg, out_dir,
        )
        if len(rows) != expected_rows:
            raise RuntimeError(f"expected {expected_rows} result rows, observed {len(rows)}")
        sample_frames.append(rows)
        mask_frames.append(masks)
        accounting_frames.append(accounting)
        pd.concat(sample_frames, ignore_index=True).to_csv(sample_path, index=False)
        pd.concat(mask_frames, ignore_index=True).to_csv(mask_path, index=False)
        pd.concat(accounting_frames, ignore_index=True).to_csv(accounting_path, index=False)
        print(f"done run={run_id} sample={sample_id}", flush=True)
    final = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    if final.empty or final.global_epoch_index.nunique() != len(frame):
        raise RuntimeError("shard output is incomplete")
    manifest = {
        "schema_version": "mf_surrogate_improvement_shard_v1",
        "status": "COMPLETE",
        "run_id": run_id,
        "data_split": split,
        "stage": args.stage,
        "candidate_profile": args.candidate_profile,
        "candidates": candidates,
        "sample_count": int(final.global_epoch_index.nunique()),
        "row_count": int(len(final)),
        "expected_rows_per_sample": expected_rows,
        "randomization_level": args.randomization_level,
        "randomization_seed": int(args.randomization_seed),
        "query_budget_per_candidate": TOTAL_QUERY_BUDGET,
        "test_data_used_for_selection": False,
        "checkpoint_sha256": evaluator.file_sha256(evaluator.CKPT_PATH),
        "sample_csv_sha256": evaluator.file_sha256(sample_path),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-split", choices=("validation", "test"), default="validation")
    parser.add_argument("--stage", choices=("smoke", "small", "final"), default="smoke")
    parser.add_argument("--candidate-profile", choices=tuple(PROFILE_CANDIDATES), default="smoke")
    parser.add_argument("--randomization-level", choices=("R0", "R3", "R5", "R7"), default="R0")
    parser.add_argument("--randomization-seed", type=int, choices=(3101, 3102, 3103), default=3101)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index/count are invalid")
    if args.stage != "smoke" and args.candidate_profile == "smoke":
        args.candidate_profile = "full"
    return args


if __name__ == "__main__":
    run_shard(parse_args())
