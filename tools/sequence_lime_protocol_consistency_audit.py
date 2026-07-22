#!/usr/bin/env python
"""Classic Sequence-LIME old/new protocol-consistency audit.

This module deliberately evaluates one frozen Classic Sequence-LIME protocol
for both historical test samples and the newer validation samples.  It does
not alter attribution construction, train a classifier, or select a method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

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
from tools import sequence_lime_paper_evaluation as evaluator  # noqa: E402
from tools import sequence_lime_abc_diagnosis as abc  # noqa: E402


ROOT = PROJECT / "outputs/sequence_lime_protocol_consistency_audit/isruc_seed42"
OLD_MANIFEST = PROJECT / "outputs/sequence_lime_abc_diagnosis/isruc_seed42/fixed_36_samples.json"
NEW_MANIFEST = PROJECT / "outputs/mf_seqlime_surrogate_improvement/isruc_seed42/validation_samples.json"
TEST_MANIFEST = PROJECT / "outputs/mf_seqlime_surrogate_improvement/isruc_seed42/test_samples.json"
TEST_CACHE = PROJECT / "outputs/sequence_lime_paper_evaluation/isruc_seed42/cache/test_inputs_segments.npz"
VALIDATION_CACHE = PROJECT / "outputs/mf_seqlime_surrogate_improvement/isruc_seed42/cache/validation_inputs_segments.npz"
VALIDATION_CACHE_FALLBACK = PROJECT / "outputs/sequence_lime_two_versions/isruc_seed42/cache/validation_inputs_segments.npz"
PROTOCOL_VERSION = "classic_sequence_lime_consistency_v1"
METHOD_ID = "M0_CLASSIC_Y2"
LEVELS = ("R0", "R3", "R5", "R7")
SEEDS = (3101, 3102, 3103)
PROTOCOLS = ("T0", "T1")
TRAIN_MASKS = 1200
VALIDATION_MASKS = 306
NEIGHBORHOOD_MASKS = 1506
FAITHFULNESS_MASKS = 24
QUERY_BUDGET = 1530
ALPHA_GRID = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
KERNEL_GRID = (0.15, 0.25, 0.5, 0.75)
DIAGNOSTIC_IDS = (284, 1165, 1336, 2633, 3637, 4073, 4986, 5014, 6420, 7518, 10829, 12972)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_bytes(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def y2_margin(vector: Sequence[float], target_class: int) -> float:
    probabilities = np.clip(np.asarray(vector, dtype=np.float64), 1e-8, 1.0)
    other = np.delete(probabilities, int(target_class))
    return float(np.log(probabilities[int(target_class)]) - np.log(np.max(other)))


def entropy(vector: Sequence[float]) -> float:
    probabilities = np.clip(np.asarray(vector, dtype=np.float64), 1e-8, 1.0)
    return float(-np.sum(probabilities * np.log(probabilities)))


def kl_divergence(first: Sequence[float], second: Sequence[float]) -> float:
    p = np.clip(np.asarray(first, dtype=np.float64), 1e-8, 1.0)
    q = np.clip(np.asarray(second, dtype=np.float64), 1e-8, 1.0)
    return float(np.sum(p * np.log(p / q)))


def unified_mask_seed(randomization_seed: int, global_epoch_index: int) -> int:
    return int(randomization_seed) * 1000003 + int(global_epoch_index) * 101


def unified_masks(randomization_seed: int, global_epoch_index: int) -> np.ndarray:
    return evaluator.make_masks(
        90,
        TRAIN_MASKS,
        VALIDATION_MASKS,
        unified_mask_seed(randomization_seed, global_epoch_index),
        sampling="bernoulli",
    )


def cache_protocol_identity(group: str, level: str, seed: int, sample_id: int) -> str:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "group": str(group),
        "randomization_level": str(level),
        "randomization_seed": int(seed),
        "sample_id": int(sample_id),
        "method": METHOD_ID,
        "query_budget": QUERY_BUDGET,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_payload(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_frame(group: str) -> pd.DataFrame:
    if group in {"A", "B"}:
        payload = _load_payload(OLD_MANIFEST)
        frame = pd.DataFrame(payload["samples"])
        if group == "A":
            frame = frame.set_index("global_epoch_index").loc[list(DIAGNOSTIC_IDS)].reset_index()
        split_name = "test"
        files = list(_load_payload(TEST_MANIFEST)["test_source_files"])
    elif group == "C":
        payload = _load_payload(NEW_MANIFEST)
        frame = pd.DataFrame(payload["samples"])
        ids = [int(value) for value in payload["small_validation_sample_ids"]]
        frame = frame.set_index("global_epoch_index").loc[ids].reset_index()
        split_name = "validation"
        files = list(payload["validation_source_files"])
    else:
        raise ValueError(group)
    frame = frame.copy()
    frame["audit_group"] = group
    frame["data_split"] = split_name
    frame["source_recording"] = [files[int(value)] for value in frame.recording_id]
    return frame


def recover_samples() -> tuple[Mapping[str, object], Mapping[str, object]]:
    old_payload = _load_payload(OLD_MANIFEST)
    new_payload = _load_payload(NEW_MANIFEST)
    old_frame = group_frame("B")
    diagnostic = group_frame("A")
    new_frame = pd.DataFrame(new_payload["samples"]).copy()
    new_files = list(new_payload["validation_source_files"])
    new_frame["source_recording"] = [new_files[int(value)] for value in new_frame.recording_id]
    test_files = set(_load_payload(TEST_MANIFEST)["test_source_files"])
    validation_files = set(new_files)
    old_ids = set(int(value) for value in old_frame.global_epoch_index)
    new_ids = set(int(value) for value in new_frame.global_epoch_index)
    if test_files.intersection(validation_files):
        raise RuntimeError("old/new physical source recordings overlap")
    if old_ids.intersection(new_ids):
        raise RuntimeError("old/new sample IDs overlap")

    def records(frame: pd.DataFrame) -> list[dict[str, object]]:
        result = []
        for row in frame.itertuples():
            result.append({
                "sample_id": int(row.global_epoch_index),
                "source_recording": str(row.source_recording),
                "recording_id": int(row.recording_id),
                "epoch_index": int(row.epoch_index_in_recording),
                "true_stage": str(row.true_stage),
                "predicted_stage": str(row.predicted_stage),
                "correct": bool(row.correct),
                "transition": bool(row.transition_boundary),
                "stability_group": "transition" if bool(row.transition_boundary) else "stable",
                "original_confidence": float(row.confidence),
                "original_logit_margin": None,
            })
        return result

    old_recovered = {
        "schema_version": "old_abc_samples_recovered_v1",
        "source_manifest": str(OLD_MANIFEST.relative_to(PROJECT)),
        "source_manifest_sha256": evaluator.file_sha256(OLD_MANIFEST),
        "fixed_36_sample_count": int(len(old_frame)),
        "diagnostic_12_sample_ids": [int(value) for value in diagnostic.global_epoch_index],
        "fixed_36_samples": records(old_frame),
        "diagnostic_12_samples": records(diagnostic),
        "resampled": False,
    }
    new_recovered = {
        "schema_version": "new_validation_samples_recovered_v1",
        "source_manifest": str(NEW_MANIFEST.relative_to(PROJECT)),
        "source_manifest_sha256": evaluator.file_sha256(NEW_MANIFEST),
        "small_validation_sample_ids": [int(value) for value in new_payload["small_validation_sample_ids"]],
        "final_validation_sample_ids": [int(value) for value in new_payload["final_validation_sample_ids"]],
        "samples": records(new_frame),
        "old_new_physical_source_overlap": 0,
        "old_new_sample_id_overlap": 0,
        "resampled": False,
    }
    write_json(ROOT / "old_abc_samples_recovered.json", old_recovered)
    write_json(ROOT / "new_validation_samples_recovered.json", new_recovered)
    return old_recovered, new_recovered


def write_config_audit() -> None:
    fields = [
        ("method ID", "M0", "C0_CLASSIC_Y2", METHOD_ID, "SEMANTICALLY_EQUIVALENT", False),
        ("surrogate target", "Y0/Y1/Y2; report retained Y2", "Y2", "Y2", "DIFFERENT", True),
        ("target-class protocol", "T0/T1", "T0/T1", "T0/T1", "IDENTICAL", False),
        ("train mask count", 256, 1200, TRAIN_MASKS, "DIFFERENT", True),
        ("local validation mask count", 64, 306, VALIDATION_MASKS, "DIFFERENT", True),
        ("mask seed", "seed + sample_id", "seed*1000003 + sample_id*101", "seed*1000003 + sample_id*101", "DIFFERENT", True),
        ("mask generator", "local_stratified", "bernoulli", "bernoulli", "DIFFERENT", True),
        ("retention probability", "local deletion mixture", 0.5, 0.5, "DIFFERENT", True),
        ("original mask inclusion", True, True, True, "IDENTICAL", False),
        ("distance metric", "cosine to all-ones", "cosine to all-ones", "cosine to all-ones", "IDENTICAL", False),
        ("kernel width", "CV grid 0.15/0.25/0.5/0.75", "same", "same", "IDENTICAL", False),
        ("Ridge alpha grid", "1e-4..100", "1e-4..100", "1e-4..100", "IDENTICAL", False),
        ("sample weighting", "LIME exponential kernel", "same", "same", "IDENTICAL", False),
        ("normalization", "positive rank score", "positive rank score", "positive rank score", "IDENTICAL", False),
        ("query accounting", 2154, 1530, QUERY_BUDGET, "DIFFERENT", True),
        ("replacement", "zero", "zero", "zero", "IDENTICAL", False),
        ("sequence wrapper", "covering-window mean, batch16", "same", "same", "IDENTICAL", False),
        ("covering-window aggregation", "float64 sum -> float32", "same", "same", "IDENTICAL", False),
        ("model randomization implementation", "v1.1.3 cumulative R0-R7", "v1.1.3 R0/R3/R5/R7", "v1.1.3 R0/R3/R5/R7", "SEMANTICALLY_EQUIVALENT", False),
        ("randomization seed", "3101/3102/3103", "3101/3102/3103", "3101/3102/3103", "IDENTICAL", False),
        ("permutation null", "1000, aggregate historical evaluator", "1000, current validation evaluator", "1000 sample-level then group aggregate", "DIFFERENT", True),
        ("similarity metric", "signed Spearman primary", "absolute Spearman primary", "signed and absolute Spearman", "DIFFERENT", True),
        ("pass threshold", "null direction + faith decay", "mean abs Spearman <= q95", "null-compatible + faith decay; >=2/3", "DIFFERENT", True),
        ("faithfulness endpoint", "advantage versus random", "deletion/insertion AUC", "insertion AUC - deletion AUC", "DIFFERENT", True),
        ("cache identity", "model hash; no explicit protocol version", "model hash; isolated output", "includes protocol version", "DIFFERENT", True),
        ("resume behavior", "sample row count", "candidate row count", "manifest+protocol cache key", "DIFFERENT", True),
    ]
    rows = []
    for field, old, new, unified, classification, can_change in fields:
        rows.append({
            "field": field,
            "old_abc_value": old,
            "new_validation_value": new,
            "unified_audit_value": unified,
            "classification": classification,
            "could_change_randomization_conclusion": bool(can_change),
        })
    pd.DataFrame(rows).to_csv(ROOT / "resolved_config_diff.csv", index=False)


def write_code_path_audit() -> None:
    old_commit = "e7a9e2ef6003c398303a40237d6451c1acd5f4cd"
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    report = f"""# Classic Sequence-LIME code-path consistency audit

Historical report commit: `{old_commit}`. Current audit start: `{current}`.

| Path | Old A/B/C | New validation | Unified audit |
|---|---|---|---|
| mask generation | `evaluator.make_masks(..., local_stratified)` | `evaluator.make_masks(..., bernoulli)` | one `unified_masks` Bernoulli entry |
| model response | `evaluate_masks_for_target_epoch` | same | same, protocol-versioned identity |
| sequence wrapper | `sequence_target_vector`, covering-window mean | same | same |
| surrogate fit | `fit_surrogate`; old fit Y0/Y1/Y2 | `fit_y2`; Y2 only | one Y2 fit route |
| randomization | `randomized_model` R0-R7 | same R0/R3/R5/R7 definitions | same current implementation |
| similarity | historical signed-Spearman aggregation | newer absolute-Spearman aggregation | both signed and absolute, never mixed |
| permutation null | historical aggregate null | newer aggregate-pair null | 1000 sample-level permutations plus frozen group aggregation |
| pass aggregation | historical method/seed directional rule | newer candidate/protocol/seed mean rule | sample -> protocol/seed -> >=2/3 seeds |

Material drift existed in mask counts, mask distribution, mask seed, query budget,
target multiplicity, null aggregation, similarity sign handling, faithfulness endpoint,
pass aggregation, and cache identity. R7 itself did not change after the v1.1.3
route fix. T0 and T1 definitions are now shared through `resolve_target_protocols`.
The historical `3/3` and newer `0/3` therefore were not protocol-identical results.
"""
    (ROOT / "code_path_consistency.md").write_text(report, encoding="utf-8")


def prepare() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    old_recovered, new_recovered = recover_samples()
    write_config_audit()
    write_code_path_audit()
    split_path = PROJECT / "outputs/protocol/isruc_seed42/split_manifest.json"
    if not split_path.exists():
        candidates = sorted((PROJECT / "outputs/protocol/isruc_seed42").glob("*.json"))
        split_path = candidates[0] if candidates else PROJECT / "outputs/protocol/isruc_seed42"
    probability_path = evaluator.PRED_PATH
    payload = {
        "schema_version": "sequence_lime_protocol_consistency_audit_manifest_v1",
        "status": "CONFIG_AND_CODE_AUDIT_COMPLETE",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT, text=True).strip(),
        "checkpoint_sha256": evaluator.file_sha256(evaluator.CKPT_PATH),
        "split_path": str(split_path),
        "split_sha256": evaluator.file_sha256(split_path) if split_path.is_file() else None,
        "probability_artifact": str(probability_path.relative_to(PROJECT)),
        "probability_artifact_sha256": evaluator.file_sha256(probability_path),
        "evaluator_version": evaluator.EVALUATOR_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "resolved_config": {
            "method": METHOD_ID,
            "surrogate_target": "Y2 log probability margin",
            "target_protocols": list(PROTOCOLS),
            "randomization_levels": list(LEVELS),
            "randomization_seeds": list(SEEDS),
            "train_masks": TRAIN_MASKS,
            "validation_masks": VALIDATION_MASKS,
            "mask_sampling": "bernoulli",
            "ridge_alpha_grid": list(ALPHA_GRID),
            "kernel_width_grid": list(KERNEL_GRID),
            "query_budget": QUERY_BUDGET,
            "permutation_count": 1000,
        },
        "old_sample_manifest_sha256": old_recovered["source_manifest_sha256"],
        "new_validation_manifest_sha256": new_recovered["source_manifest_sha256"],
        "cache_isolation": "group/level/seed/shard directories + protocol-versioned model identity",
        "protected_output_roots_written": False,
    }
    write_json(ROOT / "audit_manifest.json", payload)
    for required in ("resolved_config_diff.csv", "code_path_consistency.md"):
        if (ROOT / required).stat().st_size <= 1:
            raise RuntimeError(f"empty audit output: {required}")
    print(json.dumps({"status": payload["status"], "root": str(ROOT)}, indent=2))


def load_inputs(group: str):
    split = "validation" if group == "C" else "test"
    artifact = evaluator.load_artifact_split(np.load(evaluator.PRED_PATH), "val" if split == "validation" else "test")
    cache = VALIDATION_CACHE if split == "validation" else TEST_CACHE
    if split == "validation" and not cache.exists():
        cache = VALIDATION_CACHE_FALLBACK
    if not cache.exists():
        raise RuntimeError(f"frozen {split} input cache unavailable")
    cached = np.load(cache)
    return artifact, cached["x_raw"].astype(np.float32), cached["x_seg"].astype(np.float32)


def fit_y2(masks: np.ndarray, vectors: np.ndarray, target_class: int):
    values = np.asarray(transform_surrogate_targets(vectors, int(target_class))["Y2"], dtype=np.float32)
    ridge, fidelity, _ = evaluator.fit_surrogate(
        masks,
        values,
        TRAIN_MASKS,
        alpha_grid=ALPHA_GRID,
        kernel_width_grid=KERNEL_GRID,
        response_transform="probability",
    )
    return positive_rank_score(np.asarray(ridge.coef_, dtype=np.float32)), fidelity, values


def run_sample(sample: Mapping[str, object], model, group: str, level: str, seed: int, artifact, x_raw, x_seg, out_dir: Path) -> pd.DataFrame:
    evaluator.ensure_lime_imports()
    started = time.perf_counter()
    sample_id = int(sample["global_epoch_index"])
    epoch = np.asarray(x_raw[sample_id], dtype=np.float32)
    segments = evaluator.semantic_segmentation(epoch, target_segments=30)
    if int(epoch.shape[1] * len(segments)) != 90:
        raise RuntimeError("expected 90 features")
    masks = unified_masks(seed, sample_id)
    if len(masks) != NEIGHBORHOOD_MASKS or len({row.tobytes() for row in masks}) != len(masks):
        raise RuntimeError("unified masks are incomplete or duplicated")
    original_vector = np.asarray(artifact["probs"][sample_id], dtype=np.float32)
    randomized_vector = evaluator.sequence_target_vector(
        model, x_raw, x_seg, artifact["file_sample_counts"], sample_id,
        epoch, x_seg[sample_id], batch_windows=16,
    )
    protocols = resolve_target_protocols(original_vector, randomized_vector)
    identity = evaluator.evaluated_model_identity(
        model,
        f"{PROTOCOL_VERSION}_{group}_{level}_seed{seed}",
        randomization_level=level,
        randomization_seed=seed,
        method_id=f"{PROTOCOL_VERSION}:{METHOD_ID}",
        protocol_id="T0_T1_Y2_SHARED_FULL_VECTORS",
    )
    evaluated = evaluator.evaluate_masks_for_target_epoch(
        masks, model, x_raw, x_seg, artifact["file_sample_counts"], sample_id,
        protocols["T0"], epoch, segments, "zero", 16, out_dir,
        "unified_classic_y2_neighborhood",
        model_state_id=f"{level}_seed{seed}",
        perturbation_target_class=protocols["T0"], model_identity=identity,
    )
    if evaluated["accounting"]["status"] != "WINDOW_INSTANCE_ACCOUNTING_PASS":
        raise RuntimeError("neighborhood accounting failed")
    vectors = np.asarray(evaluated["full_probability_vectors"], dtype=np.float32)
    scores, fidelities, responses = {}, {}, {}
    for protocol in PROTOCOLS:
        score, fidelity, response = fit_y2(masks, vectors, int(protocols[protocol]))
        scores[(protocol,)] = score
        fidelities[protocol] = fidelity
        responses[protocol] = response
    curve_masks, locations = abc.faithfulness_masks(scores)
    if len(curve_masks) != FAITHFULNESS_MASKS:
        raise RuntimeError("faithfulness budget drifted")
    curves = evaluator.evaluate_masks_for_target_epoch(
        curve_masks, model, x_raw, x_seg, artifact["file_sample_counts"], sample_id,
        protocols["T0"], epoch, segments, "zero", 16, out_dir,
        "unified_classic_y2_faithfulness",
        model_state_id=f"{level}_seed{seed}",
        perturbation_target_class=protocols["T0"], model_identity=identity,
    )
    if curves["accounting"]["status"] != "WINDOW_INSTANCE_ACCOUNTING_PASS":
        raise RuntimeError("faithfulness accounting failed")
    curve_vectors = np.asarray(curves["full_probability_vectors"], dtype=np.float32)
    if len(masks) + len(curve_masks) != QUERY_BUDGET:
        raise RuntimeError("query budget drifted")
    rows = []
    for protocol in PROTOCOLS:
        target_class = int(protocols[protocol])
        score = scores[(protocol,)]
        fidelity = fidelities[protocol]
        response = responses[protocol]
        curve = abc.curve_summary(curve_vectors, target_class, locations[(protocol,)])
        faithfulness_advantage = float(curve["insertion_auc"] - curve["deletion_auc"])
        matrix = score.reshape(3, 30)
        rows.append({
            "audit_group": group,
            "data_split": sample["data_split"],
            "global_epoch_index": sample_id,
            "source_recording": sample["source_recording"],
            "recording_id": int(sample["recording_id"]),
            "epoch_index_in_recording": int(sample["epoch_index_in_recording"]),
            "true_stage": sample["true_stage"],
            "original_predicted_stage": sample["predicted_stage"],
            "correct": bool(sample["correct"]),
            "transition_boundary": bool(sample["transition_boundary"]),
            "randomization_level": level,
            "randomization_seed": int(seed),
            "target_class_protocol": protocol,
            "target_class": target_class,
            "original_confidence": float(np.max(original_vector)),
            "original_entropy": entropy(original_vector),
            "original_y2_margin": y2_margin(original_vector, target_class),
            "randomized_confidence": float(np.max(randomized_vector)),
            "randomized_entropy": entropy(randomized_vector),
            "randomized_y2_margin": y2_margin(randomized_vector, target_class),
            "argmax_changed": bool(np.argmax(original_vector) != np.argmax(randomized_vector)),
            "prediction_kl_divergence": kl_divergence(original_vector, randomized_vector),
            "surrogate_weighted_r2": float(fidelity["validation_weighted_r2"]),
            "mae": float(fidelity["validation_weighted_mae"]),
            "rmse": float(fidelity["validation_weighted_rmse"]),
            "effective_sample_size": float(fidelity["effective_sample_size"]),
            "local_response_standard_deviation": float(np.std(response)),
            "local_response_range": float(np.ptp(response)),
            "selected_ridge_alpha": float(fidelity["selected_ridge_alpha"]),
            "selected_kernel_width": float(fidelity["selected_kernel_width"]),
            "dominant_channel": int(np.argmax(np.sum(np.abs(matrix), axis=1))),
            "deletion_auc": float(curve["deletion_auc"]),
            "insertion_auc": float(curve["insertion_auc"]),
            "faithfulness_advantage": faithfulness_advantage,
            "deletion_curve": curve["deletion_curve"],
            "insertion_curve": curve["insertion_curve"],
            "attribution_score": json.dumps(score.astype(float).tolist(), separators=(",", ":")),
            "mask_seed": unified_mask_seed(seed, sample_id),
            "mask_sha256": sha256_bytes(masks),
            "query_budget": QUERY_BUDGET,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_cache_key": cache_protocol_identity(group, level, seed, sample_id),
            "model_weight_fingerprint": identity["model_weight_fingerprint"],
            "cache_identity_digest": identity["identity_digest"],
            "runtime_seconds": float(time.perf_counter() - started) / len(PROTOCOLS),
        })
    return pd.DataFrame(rows)


def run_shard(args: argparse.Namespace) -> None:
    if not (ROOT / "audit_manifest.json").exists():
        raise RuntimeError("run --prepare before inference")
    frame = group_frame(args.group)
    frame = frame.iloc[args.shard_index::args.shard_count].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("empty shard")
    run_id = f"group{args.group}_{args.level}_seed{args.seed}_shard{args.shard_index + 1}of{args.shard_count}"
    out_dir = ROOT / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sample_level_raw.csv"
    existing = pd.read_csv(csv_path) if args.resume and csv_path.exists() and csv_path.stat().st_size > 1 else pd.DataFrame()
    complete = set()
    if not existing.empty:
        counts = existing.groupby("global_epoch_index").size()
        complete = {int(index) for index, count in counts.items() if int(count) == len(PROTOCOLS)}
    artifact, x_raw, x_seg = load_inputs(args.group)
    model = evaluator.randomized_model(x_raw.shape[1:], x_seg.shape[1:], args.level, seed=args.seed)
    pd.DataFrame(getattr(model, "_sequence_lime_randomization_weight_audit")).to_csv(
        out_dir / "randomization_weight_audit.csv", index=False
    )
    all_frames = [existing] if not existing.empty else []
    for _, sample in frame.iterrows():
        sample_id = int(sample.global_epoch_index)
        if sample_id in complete:
            print(f"resume skip group={args.group} sample={sample_id}", flush=True)
            continue
        print(f"start group={args.group} level={args.level} seed={args.seed} sample={sample_id}", flush=True)
        result = run_sample(sample.to_dict(), model, args.group, args.level, args.seed, artifact, x_raw, x_seg, out_dir)
        all_frames.append(result)
        pd.concat(all_frames, ignore_index=True).to_csv(csv_path, index=False)
        print(f"done group={args.group} level={args.level} seed={args.seed} sample={sample_id}", flush=True)
    final = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    expected = 2 * len(frame)
    if len(final) != expected or final.global_epoch_index.nunique() != len(frame):
        raise RuntimeError(f"incomplete shard: rows={len(final)} expected={expected}")
    manifest = {
        "schema_version": "sequence_lime_protocol_consistency_shard_v1",
        "status": "COMPLETE",
        "run_id": run_id,
        "group": args.group,
        "randomization_level": args.level,
        "randomization_seed": int(args.seed),
        "sample_count": int(final.global_epoch_index.nunique()),
        "row_count": int(len(final)),
        "query_budget_per_sample": QUERY_BUDGET,
        "protocol_version": PROTOCOL_VERSION,
        "csv_sha256": evaluator.file_sha256(csv_path),
        "checkpoint_sha256": evaluator.file_sha256(evaluator.CKPT_PATH),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def context_occluded_vector(model, x_raw, x_seg, counts, sample_id: int) -> np.ndarray:
    """Predict the target with surrounding epochs zeroed, retaining the target."""
    batches = evaluator.target_formal_window_batches(counts, sample_id, 16)
    if not batches:
        raise RuntimeError(f"no sequence window covers sample {sample_id}")
    local_sum = None
    local_n = 0
    for batch, positions in batches:
        raw_batch, segment_batch = [], []
        for row_index, (start, valid_len) in enumerate(batch):
            indices = evaluator.window_epoch_indices(start, valid_len, 75)
            raw_window = np.zeros_like(np.asarray(x_raw[indices], dtype=np.float32))
            segment_window = np.zeros_like(np.asarray(x_seg[indices], dtype=np.float32))
            if row_index in positions:
                relative = int(positions[row_index])
                raw_window[relative] = x_raw[sample_id]
                segment_window[relative] = x_seg[sample_id]
            raw_batch.append(raw_window)
            segment_batch.append(segment_window)
        probabilities = evaluator.model_main_output(model({
            "raw_input": np.asarray(raw_batch, dtype=np.float32),
            "segment_input": np.asarray(segment_batch, dtype=np.float32),
        }, training=False))
        if local_sum is None:
            local_sum = np.zeros(probabilities.shape[-1], dtype=np.float64)
        for row_index, relative in positions.items():
            local_sum += probabilities[int(row_index), int(relative)].astype(np.float64)
            local_n += 1
    return (local_sum / max(local_n, 1)).astype(np.float32)


def run_occlusion_shard(args: argparse.Namespace) -> None:
    frame = group_frame(args.group)
    frame = frame.iloc[args.shard_index::args.shard_count].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("empty occlusion shard")
    artifact, x_raw, x_seg = load_inputs(args.group)
    model = evaluator.randomized_model(x_raw.shape[1:], x_seg.shape[1:], "R0", seed=3101)
    rows = []
    for sample in frame.to_dict(orient="records"):
        sample_id = int(sample["global_epoch_index"])
        original = evaluator.sequence_target_vector(
            model, x_raw, x_seg, artifact["file_sample_counts"], sample_id,
            x_raw[sample_id], x_seg[sample_id], batch_windows=16,
        )
        target_class = int(np.argmax(original))
        target_occluded = evaluator.sequence_target_vector(
            model, x_raw, x_seg, artifact["file_sample_counts"], sample_id,
            np.zeros_like(x_raw[sample_id]), np.zeros_like(x_seg[sample_id]), batch_windows=16,
        )
        context_occluded = context_occluded_vector(
            model, x_raw, x_seg, artifact["file_sample_counts"], sample_id
        )
        original_response = y2_margin(original, target_class)
        target_effect = float(original_response - y2_margin(target_occluded, target_class))
        context_effect = float(original_response - y2_margin(context_occluded, target_class))
        ratio = float(abs(context_effect) / (abs(target_effect) + abs(context_effect) + 1e-8))
        rows.append({
            "audit_group": args.group,
            "global_epoch_index": sample_id,
            "source_recording": sample["source_recording"],
            "target_class": target_class,
            "original_y2_response": original_response,
            "target_occluded_y2_response": y2_margin(target_occluded, target_class),
            "context_occluded_y2_response": y2_margin(context_occluded, target_class),
            "target_epoch_occlusion_drop": target_effect,
            "context_occlusion_effect": context_effect,
            "context_dependence_ratio": ratio,
            "diagnostic_only": True,
            "used_for_method_selection": False,
        })
        print(f"occlusion done group={args.group} sample={sample_id}", flush=True)
    out_dir = ROOT / "occlusion" / f"group{args.group}_shard{args.shard_index + 1}of{args.shard_count}"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "occlusion_diagnostics.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    write_json(out_dir / "run_manifest.json", {
        "status": "COMPLETE",
        "group": args.group,
        "sample_count": len(rows),
        "protocol_version": PROTOCOL_VERSION,
        "csv_sha256": evaluator.file_sha256(output),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--occlusion", action="store_true")
    parser.add_argument("--group", choices=("A", "B", "C"))
    parser.add_argument("--level", choices=LEVELS, default="R0")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=3101)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        return args
    if args.group is None:
        parser.error("--group is required unless --prepare is used")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.prepare:
        prepare()
    elif arguments.occlusion:
        run_occlusion_shard(arguments)
    else:
        run_shard(arguments)
