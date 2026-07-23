#!/usr/bin/env python
"""Strict post-hoc Hierarchical Context Attribution feasibility evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.xai.context_replacement import (  # noqa: E402
    apply_context_mask_to_window,
    contiguous_context_groups,
    replacement_epoch,
    target_group_index,
)
from src.xai.hierarchical_context_attribution import (  # noqa: E402
    CONTEXT_QUERIES,
    FAITHFULNESS_QUERIES,
    PROTOCOL_VERSION,
    TOTAL_QUERY_BUDGET,
    WITHIN_EPOCH_QUERIES,
    cache_identity,
    deterministic_masks,
    query_budget,
    selected_epoch_offsets,
)
from src.xai.model_faithful_sequence_lime import (  # noqa: E402
    positive_rank_score,
    resolve_target_protocols,
    transform_surrogate_targets,
)
from tools import sequence_lime_paper_evaluation as evaluator  # noqa: E402


ROOT = PROJECT / "outputs/hierarchical_context_xai_feasibility/isruc_seed42"
AUTHORITY = PROJECT / "outputs/xai_authoritative_status/isruc_seed42/CURRENT_XAI_STATUS.json"
VALIDATION_SOURCE = PROJECT / "outputs/mf_seqlime_surrogate_improvement/isruc_seed42/validation_samples.json"
TEST_SOURCE = PROJECT / "outputs/sequence_lime_abc_diagnosis/isruc_seed42/fixed_36_samples.json"
FULL_VALIDATION_CACHE = PROJECT / "outputs/sequence_lime_two_versions/isruc_seed42/cache/validation_inputs_segments.npz"
FULL_TEST_CACHE = PROJECT / "outputs/sequence_lime_paper_evaluation/isruc_seed42/cache/test_inputs_segments.npz"
LEVELS = ("R0", "R3", "R5", "R7")
SEEDS = (3101, 3102, 3103)
PROTOCOLS = ("T0", "T1")
GROUP_COUNTS = {"G15": 15, "G9": 9, "G25": 25}
ALPHAS = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
KERNELS = (0.15, 0.25, 0.5, 0.75)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def authority_guard() -> Mapping[str, object]:
    if not AUTHORITY.exists() or AUTHORITY.stat().st_size <= 1:
        raise RuntimeError("authoritative status file is missing")
    payload = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if payload.get("scientific_status") != "MODEL_FAITHFULNESS_NOT_ESTABLISHED":
        raise RuntimeError("scientific status is not the frozen negative boundary")
    if payload.get("feedback_training_status") != "BLOCKED":
        raise RuntimeError("feedback training must remain BLOCKED")
    return payload


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_frame(split: str) -> pd.DataFrame:
    z = np.load(evaluator.PRED_PATH)
    artifact = evaluator.load_artifact_split(z, split)
    files = evaluator.normalize_split_files(evaluator.load_split_spec())[split]
    source_root = Path(evaluator.load_split_spec()["source_npz_dir"])
    source_hashes = {name: file_sha(source_root / name) for name in files}
    counts = np.asarray(artifact["file_sample_counts"], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    labels, predicted = artifact["y_true"], artifact["pred"]
    transition = np.zeros(len(labels), dtype=bool)
    for index in range(len(labels)):
        left = index > 0 and artifact["recording_id"][index - 1] == artifact["recording_id"][index] and labels[index - 1] != labels[index]
        right = index + 1 < len(labels) and artifact["recording_id"][index + 1] == artifact["recording_id"][index] and labels[index + 1] != labels[index]
        transition[index] = bool(left or right)
    rows = []
    for index in range(len(labels)):
        recording_id = int(artifact["recording_id"][index])
        local_index = int(index - offsets[recording_id])
        rows.append({
            "global_epoch_index": int(index),
            "recording_id": recording_id,
            "epoch_index_in_recording": local_index,
            "source_recording": str(files[recording_id]),
            "source_file_sha256": source_hashes[files[recording_id]],
            "true_stage": evaluator.CLASS_LABELS[int(labels[index])],
            "predicted_stage": evaluator.CLASS_LABELS[int(predicted[index])],
            "correct": bool(labels[index] == predicted[index]),
            "transition_boundary": bool(transition[index]),
            "confidence": float(np.max(artifact["probs"][index])),
            "selection_split": {"train": "development", "val": "validation", "test": "test"}[split],
        })
    return pd.DataFrame(rows)


def deterministic_development_samples(count: int = 16) -> pd.DataFrame:
    frame = artifact_frame("train")
    counts = frame.groupby("recording_id").size().to_dict()
    frame = frame.loc[
        frame.apply(lambda row: 40 <= int(row.epoch_index_in_recording) < int(counts[int(row.recording_id)]) - 40, axis=1)
    ].copy()
    rng = np.random.default_rng(73001)
    selected = []
    used = set()
    for stage in evaluator.CLASS_LABELS:
        for correct in (True, False):
            group = frame.loc[(frame.true_stage == stage) & (frame.correct == correct) & ~frame.recording_id.isin(used)]
            if group.empty:
                group = frame.loc[(frame.true_stage == stage) & (frame.correct == correct)]
            if not group.empty:
                row = group.iloc[int(rng.integers(0, len(group)))]
                selected.append(int(row.global_epoch_index)); used.add(int(row.recording_id))
    remaining = frame.loc[~frame.global_epoch_index.isin(selected)].copy()
    remaining["confidence_bin"] = pd.qcut(remaining.confidence, 2, labels=["low", "high"])
    for transition in (True, False):
        for confidence_bin in ("low", "high"):
            if len(selected) >= count:
                break
            group = remaining.loc[(remaining.transition_boundary == transition) & (remaining.confidence_bin == confidence_bin) & ~remaining.recording_id.isin(used)]
            if group.empty:
                group = remaining.loc[(remaining.transition_boundary == transition) & (remaining.confidence_bin == confidence_bin)]
            if not group.empty:
                row = group.iloc[int(rng.integers(0, len(group)))]
                selected.append(int(row.global_epoch_index)); used.add(int(row.recording_id))
    if len(selected) < count:
        pool = remaining.loc[~remaining.global_epoch_index.isin(selected)]
        selected.extend(int(x) for x in rng.choice(pool.global_epoch_index.to_numpy(), size=count - len(selected), replace=False))
    return frame.set_index("global_epoch_index").loc[selected[:count]].reset_index()


def selected_existing_samples(split: str) -> pd.DataFrame:
    frame = artifact_frame("val" if split == "validation" else "test")
    if split == "validation":
        payload = json.loads(VALIDATION_SOURCE.read_text())
        ids = [int(value) for value in payload["final_validation_sample_ids"]]
    else:
        payload = json.loads(TEST_SOURCE.read_text())
        ids = [int(row["global_epoch_index"]) for row in payload["samples"]]
    return frame.set_index("global_epoch_index").loc[ids].reset_index()


def prepare() -> None:
    authority = authority_guard()
    ROOT.mkdir(parents=True, exist_ok=True)
    development = deterministic_development_samples(16)
    validation = selected_existing_samples("validation")
    test = selected_existing_samples("test")
    source_sets = {
        "development": set(development.source_recording),
        "validation": set(validation.source_recording),
        "test": set(test.source_recording),
    }
    overlaps = {
        "development_validation": sorted(source_sets["development"] & source_sets["validation"]),
        "development_test": sorted(source_sets["development"] & source_sets["test"]),
        "validation_test": sorted(source_sets["validation"] & source_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"physical source overlap: {overlaps}")
    for name, frame in (("development", development), ("validation", validation), ("test", test)):
        write_json(ROOT / f"{name}_samples.json", {
            "schema_version": f"hc_seqlime_{name}_samples_v1",
            "selection_split": name,
            "sample_count": int(len(frame)),
            "selection_uses_attribution": False,
            "selection_uses_test_endpoint": False,
            "source_recordings": sorted(frame.source_recording.unique().tolist()),
            "samples": json.loads(frame.to_json(orient="records")),
        })
    manifest = {
        "schema_version": "hc_seqlime_feasibility_run_manifest_v1",
        "status": "PREPARED",
        "protocol_version": PROTOCOL_VERSION,
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT, text=True).strip(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "checkpoint_sha256": evaluator.file_sha256(evaluator.CKPT_PATH),
        "checkpoint_policy": "READ_ONLY",
        "scientific_status": authority["scientific_status"],
        "feedback_training_status": authority["feedback_training_status"],
        "sample_counts": {"development": len(development), "validation": len(validation), "test": len(test)},
        "physical_source_overlap": overlaps,
        "test_execution_count": 0,
        "test_eligible": False,
        "query_budget": dict(query_budget("H0")),
        "target_uses_true_label": False,
        "stage_prior": False,
        "historical_formal_outputs_written": False,
    }
    write_json(ROOT / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def load_sample(split: str, sample_id: int):
    manifest = json.loads((ROOT / f"{split}_samples.json").read_text())
    matches = [row for row in manifest["samples"] if int(row["global_epoch_index"]) == int(sample_id)]
    if len(matches) != 1:
        raise ValueError("sample is not in the frozen split manifest")
    row = matches[0]
    artifact_split = {"development": "train", "validation": "val", "test": "test"}[split]
    artifact = evaluator.load_artifact_split(np.load(evaluator.PRED_PATH), artifact_split)
    recording_id = int(row["recording_id"])
    counts = np.asarray(artifact["file_sample_counts"], dtype=np.int32)
    start = int(np.sum(counts[:recording_id])); stop = start + int(counts[recording_id])
    if split == "development":
        cache = ROOT / "cache/development" / (Path(row["source_recording"]).stem + ".npz")
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            evaluator.ensure_lime_imports()
            from src.data_processing import load_npz_data
            spec = evaluator.load_split_spec()
            x_raw, labels, _quality = load_npz_data(
                str(Path(spec["source_npz_dir"])), file_subset=[row["source_recording"]], preprocess=True,
                filter_profile="clinical_canonical", filter_backend="scipy",
                normalization_mode="recording_robust_zscore", clip_sigma=None,
                enable_amplitude_clipping=False, return_quality=True,
            )
            features = [evaluator.extract_segment_features(epoch, evaluator.semantic_segmentation(epoch, target_segments=30), max_segments=30, include_spectral=True) for epoch in x_raw]
            np.savez_compressed(cache, x_raw=np.asarray(x_raw, dtype=np.float32), x_seg=np.asarray(features, dtype=np.float32), labels=np.asarray(labels, dtype=np.int16))
        cached = np.load(cache); x_raw, x_seg = cached["x_raw"].astype(np.float32), cached["x_seg"].astype(np.float32)
    else:
        cached = np.load(FULL_VALIDATION_CACHE if split == "validation" else FULL_TEST_CACHE)
        x_raw, x_seg = cached["x_raw"][start:stop].astype(np.float32), cached["x_seg"][start:stop].astype(np.float32)
    if len(x_raw) != int(counts[recording_id]):
        raise RuntimeError("recording cache count mismatch")
    local_id = int(sample_id) - start
    return row, artifact, x_raw, x_seg, local_id, start


def predict_specs(model, x_raw, x_seg, target_id: int, modifiers: Sequence[Callable], batch_windows: int = 16):
    covers = evaluator.target_windows(np.asarray([len(x_raw)], dtype=np.int32), target_id)
    if not covers:
        raise RuntimeError("no covering sequence windows")
    sums = np.zeros((len(modifiers), len(evaluator.CLASS_LABELS)), dtype=np.float64)
    observations = np.zeros(len(modifiers), dtype=np.int32)
    pending_raw, pending_seg, pending_meta = [], [], []
    forward_calls = 0

    def flush():
        nonlocal forward_calls
        if not pending_raw:
            return
        vectors = evaluator.compiled_perturbation_output(model, np.asarray(pending_raw, dtype=np.float32), np.asarray(pending_seg, dtype=np.float32))
        forward_calls += 1
        for row_index, (spec_index, relative) in enumerate(pending_meta):
            sums[spec_index] += vectors[row_index, relative].astype(np.float64)
            observations[spec_index] += 1
        pending_raw.clear(); pending_seg.clear(); pending_meta.clear()

    for spec_index, modifier in enumerate(modifiers):
        for start, valid_len, relative in covers:
            indices = evaluator.window_epoch_indices(start, valid_len, 75)
            raw = np.asarray(x_raw[indices], dtype=np.float32).copy()
            segment = np.asarray(x_seg[indices], dtype=np.float32).copy()
            raw, segment = modifier(raw, segment, indices)
            pending_raw.append(raw); pending_seg.append(segment); pending_meta.append((spec_index, int(relative)))
            if len(pending_raw) >= evaluator.perturbation_window_batch_size(batch_windows):
                flush()
    flush()
    if np.any(observations != len(covers)):
        raise RuntimeError("window accounting failed")
    return (sums / observations[:, None]).astype(np.float32), {"covering_windows": len(covers), "window_instances": int(np.sum(observations)), "forward_calls": forward_calls}


def fit_y2(masks: np.ndarray, vectors: np.ndarray, target_class: int, train_count: int):
    response = transform_surrogate_targets(vectors, int(target_class))["Y2"]
    ridge, fidelity, _ = evaluator.fit_surrogate(masks, response, int(train_count), alpha_grid=ALPHAS, kernel_width_grid=KERNELS, response_transform="probability")
    return positive_rank_score(np.asarray(ridge.coef_, dtype=np.float32)), fidelity, np.asarray(response, dtype=np.float32)


def epoch_mask_modifiers(x_raw, x_seg, epoch_id: int, masks: np.ndarray, target_class: int, feature_count: int):
    segment_count = int(feature_count) // 3
    segments = evaluator.uniform_segmentation(x_raw.shape[1], segment_count)
    perturbed_raw, perturbed_seg = evaluator.perturb_epochs_for_masks(x_raw[epoch_id], segments, masks, target_class, "zero")
    result = []
    for raw_epoch, segment_epoch in zip(perturbed_raw, perturbed_seg):
        def modifier(raw, segment, indices, raw_epoch=raw_epoch, segment_epoch=segment_epoch):
            matches = np.flatnonzero(np.asarray(indices) == int(epoch_id))
            for match in matches:
                raw[int(match)] = raw_epoch; segment[int(match)] = segment_epoch
            return raw, segment
        result.append(modifier)
    return result


def run_one(args: argparse.Namespace) -> None:
    authority_guard()
    row, artifact, x_raw, x_seg, target_id, artifact_start = load_sample(args.data_split, args.sample_id)
    model = evaluator.randomized_model(x_raw.shape[1:], x_seg.shape[1:], args.level, seed=args.seed)
    current_vector = evaluator.sequence_target_vector(model, x_raw, x_seg, np.asarray([len(x_raw)]), target_id, x_raw[target_id], x_seg[target_id], batch_windows=16)
    original_vector = np.asarray(artifact["probs"][int(args.sample_id)], dtype=np.float32)
    protocols = resolve_target_protocols(original_vector, current_vector)
    model_identity = evaluator.evaluated_model_identity(model, f"{PROTOCOL_VERSION}_{args.level}_{args.seed}", randomization_level=args.level, randomization_seed=args.seed, method_id="HC_SEQLIME_PROTOTYPE", protocol_id="T0_T1_Y2")
    groups = contiguous_context_groups(75, GROUP_COUNTS[args.group_scheme])
    target_group = target_group_index(groups)
    replacement_raw, replacement_segment, neutral_index = replacement_epoch(x_raw, x_seg, args.replacement)
    context_seed = args.seed * 1000003 + args.sample_id * 101 + GROUP_COUNTS[args.group_scheme]
    context_masks = deterministic_masks(len(groups), CONTEXT_QUERIES, context_seed)
    context_modifiers = []
    for mask in context_masks:
        def modifier(raw, segment, indices, mask=mask):
            return apply_context_mask_to_window(raw, segment, indices, target_id, mask, groups, replacement_raw, replacement_segment, keep_target=True)
        context_modifiers.append(modifier)
    context_vectors, context_accounting = predict_specs(model, x_raw, x_seg, target_id, context_modifiers)
    zero_target = deterministic_masks(30, 3, context_seed + 17)[1:2]
    target_zero_vector, _ = predict_specs(model, x_raw, x_seg, target_id, epoch_mask_modifiers(x_raw, x_seg, target_id, zero_target, protocols["T0"], 30))

    candidates = tuple(value.strip() for value in args.candidates.split(",") if value.strip())
    output_rows, context_rows, within_rows, runtime_rows = [], [], [], []
    for protocol in PROTOCOLS:
        target_class = int(protocols[protocol])
        context_score, context_fidelity, context_response = fit_y2(context_masks, context_vectors, target_class, 240)
        ranked = [int(index) for index in np.argsort(-np.abs(context_score)) if int(index) != target_group]
        group_centers = [int((start + stop - 1) // 2) for start, stop in groups]
        target_response = float(transform_surrogate_targets(current_vector[None, :], target_class)["Y2"][0])
        target_zero_response = float(transform_surrogate_targets(target_zero_vector, target_class)["Y2"][0])
        target_drop = target_response - target_zero_response
        neighbor_indices = {value for value in (target_group - 1, target_group + 1) if 0 <= value < len(groups)}
        neighbor = float(sum(abs(float(context_score[index])) for index in neighbor_indices))
        distant = float(sum(abs(float(context_score[index])) for index in range(len(groups)) if index != target_group and index not in neighbor_indices))
        ratio = float((neighbor + distant) / (abs(target_drop) + neighbor + distant + 1e-8))
        context_rows.append({
            "data_split": args.data_split, "sample_id": args.sample_id, "group_scheme": args.group_scheme,
            "replacement": args.replacement, "randomization_level": args.level, "randomization_seed": args.seed,
            "target_protocol": protocol, "target_class": target_class, "target_group_index": target_group,
            "selected_group_indices": json.dumps(ranked[:2]), "context_group_importance": json.dumps(context_score.astype(float).tolist()),
            "context_surrogate_R2": float(context_fidelity["validation_weighted_r2"]), "target_epoch_drop": target_drop,
            "target_contribution": abs(target_drop), "neighbor_context_contribution": neighbor,
            "distant_context_contribution": distant, "context_dependence_ratio": ratio,
            "response_variance": float(np.var(context_response)), "effective_sample_size": float(context_fidelity["effective_sample_size"]),
        })
        for candidate in candidates:
            offsets = selected_epoch_offsets(candidate, ranked, group_centers)
            epoch_ids = tuple(int(np.clip(target_id + offset, 0, len(x_raw) - 1)) for offset in offsets)
            per_epoch_count = WITHIN_EPOCH_QUERIES // len(epoch_ids)
            all_scores, all_fidelity, all_vectors = [], [], []
            for position, epoch_id in enumerate(epoch_ids):
                masks = deterministic_masks(args.feature_count, per_epoch_count, context_seed + 1009 * (position + 1) + 31 * args.feature_count)
                modifiers = epoch_mask_modifiers(x_raw, x_seg, epoch_id, masks, target_class, args.feature_count)
                vectors, accounting = predict_specs(model, x_raw, x_seg, target_id, modifiers)
                score, fidelity, response = fit_y2(masks, vectors, target_class, int(round(per_epoch_count * 0.8)))
                all_scores.append(score); all_fidelity.append(fidelity); all_vectors.append(vectors)
                within_rows.append({
                    "data_split": args.data_split, "sample_id": args.sample_id, "candidate": candidate,
                    "group_scheme": args.group_scheme, "feature_count": args.feature_count,
                    "explained_epoch_index": artifact_start + epoch_id, "explained_epoch_offset": epoch_id - target_id,
                    "randomization_level": args.level, "randomization_seed": args.seed, "target_protocol": protocol,
                    "target_class": target_class, "within_epoch_importance": json.dumps(score.astype(float).tolist()),
                    "local_surrogate_R2": float(fidelity["validation_weighted_r2"]), "response_variance": float(np.var(response)),
                    "effective_sample_size": float(fidelity["effective_sample_size"]), "logical_query_count": per_epoch_count,
                })
            fractions = np.linspace(0.0, 1.0, 12)
            deletion_modifiers, insertion_modifiers = [], []
            for fraction in fractions:
                deletion_epochs, insertion_epochs = {}, {}
                for epoch_id, score in zip(epoch_ids, all_scores):
                    order = np.argsort(-np.abs(score)); count = int(round(float(fraction) * len(score)))
                    deletion_mask = np.ones(len(score), dtype=np.int32); deletion_mask[order[:count]] = 0
                    insertion_mask = np.zeros(len(score), dtype=np.int32); insertion_mask[order[:count]] = 1
                    deletion_epochs[epoch_id] = epoch_mask_modifiers(x_raw, x_seg, epoch_id, deletion_mask[None, :], target_class, args.feature_count)[0]
                    insertion_epochs[epoch_id] = epoch_mask_modifiers(x_raw, x_seg, epoch_id, insertion_mask[None, :], target_class, args.feature_count)[0]
                def combine(mapping):
                    def modifier(raw, segment, indices):
                        for epoch_id, single in mapping.items():
                            raw, segment = single(raw, segment, indices)
                        return raw, segment
                    return modifier
                deletion_modifiers.append(combine(deletion_epochs)); insertion_modifiers.append(combine(insertion_epochs))
            faith_vectors, faith_accounting = predict_specs(model, x_raw, x_seg, target_id, deletion_modifiers + insertion_modifiers)
            faith_response = transform_surrogate_targets(faith_vectors, target_class)["Y2"]
            deletion_auc = float(np.trapz(faith_response[:12], fractions)); insertion_auc = float(np.trapz(faith_response[12:], fractions))
            local_r2 = float(np.mean([value["validation_weighted_r2"] for value in all_fidelity]))
            output_rows.append({
                "data_split": args.data_split, "source_recording": row["source_recording"], "sample_id": args.sample_id,
                "true_stage": row["true_stage"], "predicted_stage": row["predicted_stage"], "correct": row["correct"],
                "transition_boundary": row["transition_boundary"], "confidence": row["confidence"],
                "candidate": candidate, "group_scheme": args.group_scheme, "feature_count": args.feature_count,
                "replacement": args.replacement, "randomization_level": args.level, "randomization_seed": args.seed,
                "target_protocol": protocol, "target_class": target_class, "context_surrogate_R2": float(context_fidelity["validation_weighted_r2"]),
                "local_surrogate_R2": local_r2, "hierarchical_surrogate_R2": float(np.mean([local_r2, context_fidelity["validation_weighted_r2"]])),
                "deletion_auc": deletion_auc, "insertion_auc": insertion_auc, "faithfulness": insertion_auc - deletion_auc,
                "target_epoch_drop": target_drop, "context_dependence_ratio": ratio, "response_variance": float(np.var(context_response)),
                "effective_sample_size": float(context_fidelity["effective_sample_size"]), "selected_group_indices": json.dumps(ranked[:2]),
                "explained_epoch_indices": json.dumps([artifact_start + value for value in epoch_ids]),
                "context_importance": json.dumps(context_score.astype(float).tolist()),
                "within_epoch_importance": json.dumps([value.astype(float).tolist() for value in all_scores]),
                "query_budget": TOTAL_QUERY_BUDGET, "context_queries": CONTEXT_QUERIES,
                "within_epoch_queries": WITHIN_EPOCH_QUERIES, "faithfulness_queries": FAITHFULNESS_QUERIES,
                "actual_shared_context_queries": CONTEXT_QUERIES, "protocol_version": PROTOCOL_VERSION,
                "model_weight_fingerprint": model_identity["model_weight_fingerprint"],
                "cache_key": cache_identity(model_identity["model_weight_fingerprint"], args.sample_id, candidate, args.level, args.seed, protocol),
                "true_label_used_for_attribution": False, "stage_prior_used": False,
            })
            runtime_rows.append({"sample_id": args.sample_id, "candidate": candidate, "target_protocol": protocol, "logical_queries": TOTAL_QUERY_BUDGET, "context_window_instances": context_accounting["window_instances"], "faithfulness_window_instances": faith_accounting["window_instances"]})

    run_id = f"{args.stage}_{args.data_split}_sample{args.sample_id}_{args.group_scheme}_f{args.feature_count}_{args.level}_seed{args.seed}_{args.replacement}"
    out_dir = ROOT / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(out_dir / "hierarchical_sample_level.csv", index=False)
    pd.DataFrame(context_rows).to_csv(out_dir / "context_group_importance.csv", index=False)
    pd.DataFrame(within_rows).to_csv(out_dir / "within_epoch_attribution.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(out_dir / "runtime_query_summary.csv", index=False)
    write_json(out_dir / "run_manifest.json", {
        "status": "COMPLETE", "run_id": run_id, "stage": args.stage, "data_split": args.data_split,
        "sample_id": args.sample_id, "candidates": list(candidates), "group_scheme": args.group_scheme,
        "feature_count": args.feature_count, "replacement": args.replacement, "randomization_level": args.level,
        "randomization_seed": args.seed, "target_protocols": list(PROTOCOLS), "logical_query_budget_per_candidate": TOTAL_QUERY_BUDGET,
        "neutral_epoch_index": neutral_index, "checkpoint_sha256": evaluator.file_sha256(evaluator.CKPT_PATH),
        "model_weight_fingerprint": model_identity["model_weight_fingerprint"], "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "all_outputs_nonempty": all(path.stat().st_size > 1 for path in out_dir.glob("*.csv")),
    })
    print(json.dumps({"status": "COMPLETE", "run_id": run_id, "rows": len(output_rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--stage", choices=("smoke", "development", "validation", "test"), default="smoke")
    parser.add_argument("--data-split", choices=("development", "validation", "test"), default="development")
    parser.add_argument("--sample-id", type=int)
    parser.add_argument("--group-scheme", choices=tuple(GROUP_COUNTS), default="G15")
    parser.add_argument("--feature-count", type=int, choices=(30, 45, 90), default=30)
    parser.add_argument("--candidates", default="H0,H3")
    parser.add_argument("--replacement", choices=("zero", "local_temporal_mean", "recording_neutral_epoch"), default="zero")
    parser.add_argument("--level", choices=LEVELS, default="R0")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=3101)
    args = parser.parse_args()
    if not args.prepare and args.sample_id is None:
        parser.error("--sample-id is required")
    if args.stage == "test":
        frozen = ROOT / "frozen_validation_selection.json"
        if not frozen.exists() or not json.loads(frozen.read_text()).get("fixed_test_allowed"):
            parser.error("fixed test is blocked until validation hard gates pass")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    prepare() if arguments.prepare else run_one(arguments)
