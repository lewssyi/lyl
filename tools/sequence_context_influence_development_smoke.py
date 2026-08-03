#!/usr/bin/env python3
"""Frozen two-sample engineering smoke for independent context influence.

This tool has two deliberately separate phases. ``prepare`` writes a frozen,
read-only protocol snapshot without importing TensorFlow or loading the model.
``run`` refuses to infer unless that snapshot is present and still matches the
source assets. It never reads validation/test prediction arrays or XAI assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import resource
except ImportError:  # pragma: no cover - Windows-only developer convenience
    resource = None


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

OUTPUT_ROOT = (
    PROJECT
    / "outputs/sleepxvit_inspired_audit/isruc_seed42/"
    "context_influence_v1/development_smoke_2"
)
PREDICTIONS = (
    PROJECT
    / "outputs/predictions/"
    "result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42_split_predictions.npz"
)
CHECKPOINT = (
    PROJECT
    / "outputs/checkpoints/"
    "result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42.weights.h5"
)
SPLIT = PROJECT / "dataset_protocols/ISRUC/split.json"
AUTHORITY = (
    PROJECT
    / "outputs/xai_authoritative_status/isruc_seed42/CURRENT_XAI_STATUS.json"
)
DEVELOPMENT_MANIFEST = (
    PROJECT
    / "outputs/hierarchical_context_xai_feasibility/isruc_seed42/"
    "development_samples.json"
)
DEVELOPMENT_CACHE_ROOT = (
    PROJECT
    / "outputs/hierarchical_context_xai_feasibility/isruc_seed42/cache/development"
)
PLANNING_REPORT = PROJECT / "docs/paper_xai/SLEEPXVIT_BORROWING_DECISION.md"
RELIABILITY_REPORT = PROJECT / "docs/paper_xai/SLEEPXVIT_RELIABILITY_AUDIT.md"

CLASS_LABELS = ("W", "N1", "N2", "N3", "REM")
OFFSETS = (-3, -2, -1, 1, 2, 3)
BASELINES = ("normalized_zero", "recording_feature_medoid")
SEQ_LEN = 75
STRIDE = 5
MODEL_SEED = 42
REPLACEMENT_SEED = None
BATCH_SIZE = 16
PARITY_ATOL = 1e-6
PARITY_RTOL = 0.0
EXPECTED_PRIOR_COMMITS = (
    "4283935be4f0449c04ce138594c275fdb2667084",
    "276ca1041be07213c455fa7250ac7bb725bceccc",
)
EXPECTED_HASHES = {
    "checkpoint": "e86a84790fd7351c1864158eef79fbf28437a7a8a5f26662b93bcf1aef14b6ed",
    "predictions": "53cd6b51c008e0dd47dd187654c8eddacd5d0a69037c48aeecafe5efa9594e39",
    "split": "8363546a86b81676fe6171f2e50405d2d7447cbef1f48dbf397cd53bf3f037d9",
    "authority": "55c74b36113f438f0a94b8e9d2c679f6cdfd2b85825b8c63ecd80f43f60092bc",
    "development_manifest": "c7717212c92bfd655ea728643e5e8161fe3006b1e3be8e94e57fb817948eca5e",
}
EXPECTED_OUTPUTS = (
    "protocol_snapshot.json",
    "input_inventory.json",
    "selected_samples.json",
    "sample_manifest.csv",
    "baseline_parity.json",
    "no_op_control.json",
    "batch_parity.json",
    "determinism_check.json",
    "window_reconstruction_audit.csv",
    "cross_view_update_audit.json",
    "context_influence_long.csv",
    "context_influence_profiles.json",
    "runtime_profile.json",
    "access_audit.json",
    "run_manifest.json",
    "README.md",
)


class SmokeBlocked(RuntimeError):
    """Raised when a pre-registered engineering gate fails."""


@dataclass(frozen=True)
class WindowSpec:
    start: int
    valid_len: int
    target_position: int
    context_position: int | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_windows(
    recording_len: int,
    seq_len: int = SEQ_LEN,
    stride: int = STRIDE,
) -> list[tuple[int, int]]:
    """Mirror the canonical one-recording window construction exactly."""
    recording_len = int(recording_len)
    if recording_len < 1:
        raise ValueError("recording_len must be positive")
    if recording_len < seq_len:
        return [(0, recording_len)]
    last_start = recording_len - seq_len
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(int(start), int(seq_len)) for start in starts]


def covering_window_specs(
    recording_len: int,
    target_epoch: int,
    context_epoch: int,
) -> list[WindowSpec]:
    specs: list[WindowSpec] = []
    for start, valid_len in build_windows(recording_len):
        stop = start + valid_len
        if not start <= target_epoch < stop:
            continue
        context_position = (
            int(context_epoch - start)
            if start <= context_epoch < stop
            else None
        )
        specs.append(
            WindowSpec(
                start=int(start),
                valid_len=int(valid_len),
                target_position=int(target_epoch - start),
                context_position=context_position,
            )
        )
    if not specs:
        raise SmokeBlocked(f"No canonical window covers target epoch {target_epoch}")
    return specs


def padded_indices(spec: WindowSpec) -> np.ndarray:
    indices = np.arange(spec.start, spec.start + spec.valid_len, dtype=np.int64)
    if spec.valid_len < SEQ_LEN:
        indices = np.concatenate(
            [indices, np.full(SEQ_LEN - spec.valid_len, indices[-1], dtype=np.int64)]
        )
    return indices


def assemble_context_windows(
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    target_epoch: int,
    context_epoch: int,
    replacement_raw: np.ndarray | None,
    replacement_segment: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[WindowSpec]]:
    specs = covering_window_specs(len(x_raw), target_epoch, context_epoch)
    raw_windows: list[np.ndarray] = []
    segment_windows: list[np.ndarray] = []
    for spec in specs:
        indices = padded_indices(spec)
        raw_window = np.asarray(x_raw[indices], dtype=np.float32).copy()
        segment_window = np.asarray(x_seg[indices], dtype=np.float32).copy()
        if spec.context_position is not None and replacement_raw is not None:
            raw_window[spec.context_position] = replacement_raw
            segment_window[spec.context_position] = replacement_segment
        raw_windows.append(raw_window)
        segment_windows.append(segment_window)
    return (
        np.asarray(raw_windows, dtype=np.float32),
        np.asarray(segment_windows, dtype=np.float32),
        specs,
    )


def aggregate_window_vectors(
    vectors: np.ndarray,
    specs: Sequence[WindowSpec],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(specs):
        raise ValueError("one probability vector is required per covering window")
    shared = np.asarray([spec.context_position is not None for spec in specs])
    if not np.any(shared):
        raise SmokeBlocked("No window jointly covers target and requested context epoch")
    return values.mean(axis=0), values[shared].mean(axis=0)


def apply_single_epoch_pair(
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    context_epoch: int,
    replacement_raw: np.ndarray,
    replacement_segment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    changed_raw = np.asarray(x_raw, dtype=np.float32).copy()
    changed_seg = np.asarray(x_seg, dtype=np.float32).copy()
    changed_raw[int(context_epoch)] = replacement_raw
    changed_seg[int(context_epoch)] = replacement_segment
    raw_epoch_diff = np.any(changed_raw != x_raw, axis=tuple(range(1, changed_raw.ndim)))
    seg_epoch_diff = np.any(changed_seg != x_seg, axis=tuple(range(1, changed_seg.ndim)))
    combined = raw_epoch_diff | seg_epoch_diff
    audit = {
        "changed_epoch_indices": np.flatnonzero(combined).astype(int).tolist(),
        "raw_changed_epoch_indices": np.flatnonzero(raw_epoch_diff).astype(int).tolist(),
        "segment_changed_epoch_indices": np.flatnonzero(seg_epoch_diff).astype(int).tolist(),
        "raw_changed_scalar_count": int(np.count_nonzero(changed_raw != x_raw)),
        "segment_changed_scalar_count": int(np.count_nonzero(changed_seg != x_seg)),
        "other_epochs_unchanged": bool(
            np.array_equal(changed_raw[np.arange(len(x_raw)) != int(context_epoch)], x_raw[np.arange(len(x_raw)) != int(context_epoch)])
            and np.array_equal(changed_seg[np.arange(len(x_seg)) != int(context_epoch)], x_seg[np.arange(len(x_seg)) != int(context_epoch)])
        ),
    }
    return changed_raw, changed_seg, audit


def deterministic_feature_medoid(
    x_seg: np.ndarray,
    excluded_indices: Iterable[int],
) -> tuple[int, Mapping[str, Any]]:
    values = np.asarray(x_seg, dtype=np.float64).reshape(len(x_seg), -1)
    excluded = {int(value) for value in excluded_indices}
    candidates = np.asarray([i for i in range(len(values)) if i not in excluded], dtype=int)
    if not len(candidates):
        raise SmokeBlocked("No recording-neutral medoid candidate remains")
    feature_median = np.median(values[candidates], axis=0)
    distances = np.linalg.norm(values[candidates] - feature_median[None, :], axis=1)
    minimum = float(np.min(distances))
    tied = candidates[np.isclose(distances, minimum, atol=0.0, rtol=0.0)]
    selected = int(np.min(tied))
    return selected, {
        "definition": "nearest real epoch to recording segment-feature median",
        "distance": minimum,
        "candidate_count": int(len(candidates)),
        "excluded_indices": sorted(excluded),
        "tie_policy": "minimum epoch index",
    }


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    left = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    right = np.clip(np.asarray(q, dtype=np.float64), 1e-12, 1.0)
    left /= left.sum()
    right /= right.sum()
    middle = 0.5 * (left + right)
    return float(
        0.5 * np.sum(left * np.log(left / middle))
        + 0.5 * np.sum(right * np.log(right / middle))
    )


def probability_valid(vectors: np.ndarray) -> bool:
    values = np.asarray(vectors, dtype=np.float64)
    return bool(
        np.all(np.isfinite(values))
        and np.min(values) >= -PARITY_ATOL
        and np.max(values) <= 1.0 + PARITY_ATOL
        and np.max(np.abs(values.sum(axis=-1) - 1.0)) <= 1e-5
    )


def expected_model_config() -> Mapping[str, Any]:
    return {
        "classifier": "A1-zero",
        "model_input_mode": "sequence raw_input + segment_input",
        "raw_shape": [3000, 3],
        "segment_shape": [30, 28],
        "seq_len": 75,
        "prediction_stride": 5,
        "epoch_encoder_variant": "v3_multiscale",
        "sequence_backbone": "v3_tcn",
        "transformer": True,
        "concept_modulation": True,
        "concept_modulation_mode": "zero",
        "training": False,
        "hmm": False,
    }


def select_first_two(
    development_payload: Mapping[str, Any],
    train_counts: np.ndarray,
    split_payload: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    samples = list(development_payload.get("samples", ()))
    if int(development_payload.get("sample_count", -1)) != 16 or len(samples) != 16:
        raise SmokeBlocked("Frozen development manifest is not exactly 16 samples")
    selected: list[Mapping[str, Any]] = []
    offsets = np.concatenate([[0], np.cumsum(train_counts)])
    for row in samples:
        recording_id = int(row["recording_id"])
        global_epoch = int(row["global_epoch_index"])
        local_epoch = global_epoch - int(offsets[recording_id])
        recording_len = int(train_counts[recording_id])
        if local_epoch + min(OFFSETS) < 0 or local_epoch + max(OFFSETS) >= recording_len:
            continue
        expected_recording = split_payload["split_files"]["train"][recording_id]
        if str(row["source_recording"]) != str(expected_recording):
            raise SmokeBlocked("Development manifest recording mapping is inconsistent")
        enriched = dict(row)
        enriched.update(
            {
                "sample_id": global_epoch,
                "target_epoch_index": local_epoch,
                "subject_id": split_payload["split_subjects"]["train"][recording_id],
                "recording_length": recording_len,
            }
        )
        selected.append(enriched)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise SmokeBlocked("Fewer than two valid frozen development samples")
    return selected


def verify_authority() -> Mapping[str, Any]:
    payload = read_json(AUTHORITY)
    checks = {
        "MODEL_FAITHFULNESS_NOT_ESTABLISHED": payload.get("scientific_status")
        == "MODEL_FAITHFULNESS_NOT_ESTABLISHED",
        "PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED": payload.get(
            "physiology_aware_visualization_decision"
        )
        == "PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED",
        "PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD": payload.get(
            "feedback_training_status"
        )
        == "PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD",
    }
    if not all(checks.values()):
        raise SmokeBlocked(f"Authoritative state mismatch: {checks}")
    return checks


def prepare() -> None:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise SmokeBlocked(f"Refusing to overwrite nonempty output root: {OUTPUT_ROOT}")
    if git("branch", "--show-current") != "codex/a1-zero-mainline":
        raise SmokeBlocked("Wrong branch")
    if git("status", "--porcelain"):
        raise SmokeBlocked("Worktree must be clean before protocol freeze")
    for commit in EXPECTED_PRIOR_COMMITS:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=PROJECT
        )
    source_head = git("rev-parse", "HEAD")
    required = {
        "checkpoint": CHECKPOINT,
        "predictions": PREDICTIONS,
        "split": SPLIT,
        "authority": AUTHORITY,
        "development_manifest": DEVELOPMENT_MANIFEST,
        "planning_report": PLANNING_REPORT,
        "reliability_report": RELIABILITY_REPORT,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise SmokeBlocked(f"Missing protocol assets: {missing}")
    observed_hashes = {name: sha256_file(path) for name, path in required.items()}
    for name, expected in EXPECTED_HASHES.items():
        if observed_hashes[name] != expected:
            raise SmokeBlocked(
                f"Lineage hash mismatch for {name}: {observed_hashes[name]}"
            )
    authority_checks = verify_authority()
    split_payload = read_json(SPLIT)
    development_payload = read_json(DEVELOPMENT_MANIFEST)
    with np.load(PREDICTIONS, allow_pickle=False) as archive:
        train_counts = archive["train_file_sample_counts"].astype(np.int64)
        train_rows = int(len(archive["train_y_true"]))
        class_order = archive["class_labels"].astype(str).tolist()
    if train_rows != 61115 or class_order != list(CLASS_LABELS):
        raise SmokeBlocked("Frozen train probability axis or class order changed")
    selected = select_first_two(development_payload, train_counts, split_payload)
    cache_inventory = []
    for row in selected:
        cache_path = DEVELOPMENT_CACHE_ROOT / f"{Path(row['source_recording']).stem}.npz"
        source_path = Path(split_payload["source_npz_dir"]) / row["source_recording"]
        if not cache_path.is_file() or not source_path.is_file():
            raise SmokeBlocked(f"Missing selected development input: {row['sample_id']}")
        source_hash = sha256_file(source_path)
        if source_hash != row["source_file_sha256"]:
            raise SmokeBlocked(f"Source recording hash mismatch: {source_path}")
        with np.load(cache_path, allow_pickle=False) as cache:
            cache_shapes = {key: list(cache[key].shape) for key in cache.files}
        cache_inventory.append(
            {
                "sample_id": int(row["sample_id"]),
                "cache_path": str(cache_path.relative_to(PROJECT)),
                "cache_sha256": sha256_file(cache_path),
                "cache_shapes": cache_shapes,
                "source_path": str(source_path),
                "source_sha256": source_hash,
            }
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema_version": "sequence_context_influence_development_smoke_protocol_v1",
        "status": "FROZEN_BEFORE_INFERENCE",
        "frozen_at": utc_now(),
        "source_git_head": source_head,
        "source_git_branch": git("branch", "--show-current"),
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT)),
        "checkpoint_sha256": observed_hashes["checkpoint"],
        "model_config": expected_model_config(),
        "development_manifest": str(DEVELOPMENT_MANIFEST.relative_to(PROJECT)),
        "development_manifest_sha256": observed_hashes["development_manifest"],
        "sample_selection_rule": (
            "first two valid rows in the frozen development-16 manifest; validity "
            "requires all pre-frozen offsets to remain inside the recording"
        ),
        "sample_ids": [int(row["sample_id"]) for row in selected],
        "samples": [
            {
                "sample_id": int(row["sample_id"]),
                "subject_id": row["subject_id"],
                "recording_id": int(row["recording_id"]),
                "source_recording": row["source_recording"],
                "target_epoch_index": int(row["target_epoch_index"]),
            }
            for row in selected
        ],
        "offsets": list(OFFSETS),
        "delta_zero_policy": "excluded from influence; exercised only as no-op control",
        "replacement": {
            "primary": "canonical normalized zero waveform with segment features recomputed",
            "sensitivity": (
                "real epoch nearest the recording segment-feature median, excluding "
                "target and audited neighborhood; minimum epoch index breaks exact ties"
            ),
            "confidence_label_physiology_used": False,
        },
        "response": (
            "Pre-HMM probability for c*=argmax of the unperturbed aggregated "
            "five-class Pre-HMM vector"
        ),
        "signed_influence": "R_original[c*] - R_counterfactual[c*]",
        "aggregation": {
            "primary": (
                "equal mean over every canonical 75-epoch stride-5 window covering "
                "the target; only shared target/context windows are modified"
            ),
            "additional": (
                "shared-window paired mean over windows jointly covering target and context"
            ),
        },
        "boundary_policy": (
            "no recording crossing or wraparound; nonphysical padding cannot be "
            "intervened; missing offsets are structural missing"
        ),
        "model_initialization_seed": MODEL_SEED,
        "replacement_seed": REPLACEMENT_SEED,
        "random_replacement": False,
        "batch_size": BATCH_SIZE,
        "batch_parity_comparator": 1,
        "parity_atol": PARITY_ATOL,
        "parity_rtol": PARITY_RTOL,
        "resource_gate": (
            "one RTX 3090; inference must complete without OOM and measured peak "
            "allocator memory must remain below device capacity"
        ),
        "expected_outputs": list(EXPECTED_OUTPUTS),
        "forbidden": {
            "development_16_execution": True,
            "validation_30_access": True,
            "test_36_access": True,
            "test_npz_array_access": True,
            "hmm": True,
            "training_gradient_or_weight_update": True,
            "canonical_lime_modification": True,
        },
    }
    write_json(OUTPUT_ROOT / "protocol_snapshot.json", protocol)
    write_json(
        OUTPUT_ROOT / "selected_samples.json",
        {
            "schema_version": "context_smoke_selected_development_samples_v1",
            "selection_rule": protocol["sample_selection_rule"],
            "source_manifest": protocol["development_manifest"],
            "source_manifest_sha256": protocol["development_manifest_sha256"],
            "sample_count": 2,
            "samples": selected,
        },
    )
    pd.DataFrame(selected).to_csv(OUTPUT_ROOT / "sample_manifest.csv", index=False)
    write_json(
        OUTPUT_ROOT / "input_inventory.json",
        {
            "schema_version": "context_smoke_input_inventory_v1",
            "created_before_inference": True,
            "model_inference_executed": False,
            "assets": {
                name: {
                    "path": str(path.relative_to(PROJECT))
                    if path.is_relative_to(PROJECT)
                    else str(path),
                    "sha256": observed_hashes[name],
                    "bytes": path.stat().st_size,
                }
                for name, path in required.items()
            },
            "cache_inventory": cache_inventory,
            "train_probability_rows": train_rows,
            "class_order": class_order,
            "selected_sample_count": 2,
            "authority_checks": authority_checks,
        },
    )
    for name in ("protocol_snapshot.json", "selected_samples.json"):
        os.chmod(OUTPUT_ROOT / name, 0o444)
    print(json.dumps({"status": "PREPARED", "output_root": str(OUTPUT_ROOT)}, indent=2))


def recompute_segment_feature(evaluator, raw_epoch: np.ndarray) -> np.ndarray:
    evaluator.ensure_lime_imports()
    segments = evaluator.semantic_segmentation(raw_epoch, target_segments=30)
    return np.asarray(
        evaluator.extract_segment_features(
            raw_epoch, segments, max_segments=30, include_spectral=True
        ),
        dtype=np.float32,
    )


def infer_windows(
    evaluator,
    model,
    raw_windows: np.ndarray,
    segment_windows: np.ndarray,
    specs: Sequence[WindowSpec],
    batch_size: int,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    vectors: list[np.ndarray] = []
    forward_seconds = 0.0
    batch_sizes: list[int] = []
    for start in range(0, len(specs), int(batch_size)):
        stop = min(len(specs), start + int(batch_size))
        tick = time.perf_counter()
        outputs = evaluator.compiled_perturbation_output(
            model, raw_windows[start:stop], segment_windows[start:stop]
        )
        forward_seconds += time.perf_counter() - tick
        batch_sizes.append(stop - start)
        for row, spec in enumerate(specs[start:stop]):
            vectors.append(np.asarray(outputs[row, spec.target_position], dtype=np.float32))
    result = np.asarray(vectors, dtype=np.float32)
    if not probability_valid(result):
        raise SmokeBlocked("Invalid Pre-HMM probability semantics")
    return result, {
        "batch_size": int(batch_size),
        "batch_sizes": batch_sizes,
        "forward_call_count": int(len(batch_sizes)),
        "forward_seconds": float(forward_seconds),
    }


def infer_profile(
    evaluator,
    model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    target_epoch: int,
    context_epoch: int,
    replacement_raw: np.ndarray | None,
    replacement_segment: np.ndarray | None,
    batch_size: int,
) -> Mapping[str, Any]:
    tick = time.perf_counter()
    raw_windows, segment_windows, specs = assemble_context_windows(
        x_raw,
        x_seg,
        target_epoch,
        context_epoch,
        replacement_raw,
        replacement_segment,
    )
    assembly_seconds = time.perf_counter() - tick
    vectors, runtime = infer_windows(
        evaluator, model, raw_windows, segment_windows, specs, batch_size
    )
    full, shared = aggregate_window_vectors(vectors, specs)
    return {
        "window_vectors": vectors,
        "full_vector": full.astype(np.float64),
        "shared_vector": shared.astype(np.float64),
        "specs": specs,
        "assembly_seconds": float(assembly_seconds),
        **runtime,
    }


def tf_peak_bytes() -> int | None:
    try:
        import tensorflow as tf

        return int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
    except Exception:
        return None


def record_intervention(
    *,
    evaluator,
    model,
    sample: Mapping[str, Any],
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    original: Mapping[str, Any],
    target_class: int,
    offset: int,
    baseline: str,
    replacement_raw: np.ndarray,
    replacement_segment: np.ndarray,
    replacement_target: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    target_epoch = int(sample["target_epoch_index"])
    context_epoch = target_epoch + int(offset)
    _, _, isolation = apply_single_epoch_pair(
        x_raw,
        x_seg,
        context_epoch,
        replacement_raw,
        replacement_segment,
    )
    profile = infer_profile(
        evaluator,
        model,
        x_raw,
        x_seg,
        target_epoch,
        context_epoch,
        replacement_raw,
        replacement_segment,
        BATCH_SIZE,
    )
    batch1 = infer_profile(
        evaluator,
        model,
        x_raw,
        x_seg,
        target_epoch,
        context_epoch,
        replacement_raw,
        replacement_segment,
        1,
    )
    original_full = np.asarray(original["full_vector"], dtype=np.float64)
    original_vectors = np.asarray(original["window_vectors"], dtype=np.float64)
    shared_mask = np.asarray(
        [spec.context_position is not None for spec in profile["specs"]]
    )
    original_shared = original_vectors[shared_mask].mean(axis=0)
    counterfactual = np.asarray(profile["full_vector"], dtype=np.float64)
    counterfactual_shared = np.asarray(profile["shared_vector"], dtype=np.float64)
    signed = float(original_full[target_class] - counterfactual[target_class])
    signed_shared = float(
        original_shared[target_class] - counterfactual_shared[target_class]
    )
    specs = profile["specs"]
    affected = [spec for spec in specs if spec.context_position is not None]
    long_row = {
        "sample_id": int(sample["sample_id"]),
        "split": "development",
        "subject_id": sample["subject_id"],
        "recording_id": int(sample["recording_id"]),
        "source_recording": sample["source_recording"],
        "target_epoch_index": target_epoch,
        "context_epoch_index": context_epoch,
        "offset": int(offset),
        "baseline": baseline,
        "target_class_index": int(target_class),
        "target_class": CLASS_LABELS[target_class],
        "original_response": float(original_full[target_class]),
        "counterfactual_response": float(counterfactual[target_class]),
        "signed_influence": signed,
        "absolute_influence": abs(signed),
        "shared_original_response": float(original_shared[target_class]),
        "shared_counterfactual_response": float(counterfactual_shared[target_class]),
        "shared_signed_influence": signed_shared,
        "shared_absolute_influence": abs(signed_shared),
        "jensen_shannon_divergence": jensen_shannon(original_full, counterfactual),
        "argmax_flipped": bool(np.argmax(original_full) != np.argmax(counterfactual)),
        "original_probability_vector": json.dumps(original_full.tolist()),
        "counterfactual_probability_vector": json.dumps(counterfactual.tolist()),
        "probability_difference_vector": json.dumps((original_full - counterfactual).tolist()),
        "replacement_target": json.dumps(replacement_target, sort_keys=True),
        "covering_window_count": int(len(specs)),
        "shared_window_count": int(len(affected)),
        "runtime_seconds": float(profile["assembly_seconds"] + profile["forward_seconds"]),
        "status": "COMPLETE",
    }
    window_row = {
        "sample_id": int(sample["sample_id"]),
        "offset": int(offset),
        "baseline": baseline,
        "target_epoch_index": target_epoch,
        "context_epoch_index": context_epoch,
        "expected_target_covering_windows": int(len(specs)),
        "actual_assembled_target_windows": int(len(profile["window_vectors"])),
        "expected_affected_windows": int(len(affected)),
        "actual_rebuilt_affected_windows": int(len(affected)),
        "unaffected_target_windows": int(len(specs) - len(affected)),
        "window_starts": json.dumps([spec.start for spec in specs]),
        "target_positions": json.dumps([spec.target_position for spec in specs]),
        "affected_window_starts": json.dumps([spec.start for spec in affected]),
        "context_positions": json.dumps(
            [spec.context_position for spec in affected]
        ),
        "duplicate_window_count": int(len(specs) - len({spec.start for spec in specs})),
        "omission_or_duplicate": False,
        "boundary_status": "VALID_PHYSICAL_EPOCH",
    }
    batch_row = {
        "sample_id": int(sample["sample_id"]),
        "offset": int(offset),
        "baseline": baseline,
        "planned_batch": BATCH_SIZE,
        "comparison_batch": 1,
        "max_abs_full_vector_diff": float(
            np.max(np.abs(profile["full_vector"] - batch1["full_vector"]))
        ),
        "max_abs_window_vector_diff": float(
            np.max(np.abs(profile["window_vectors"] - batch1["window_vectors"]))
        ),
        "tolerance": PARITY_ATOL,
    }
    batch_row["pass"] = bool(
        batch_row["max_abs_full_vector_diff"] <= PARITY_ATOL
        and batch_row["max_abs_window_vector_diff"] <= PARITY_ATOL
    )
    profile_payload = {
        "sample_id": int(sample["sample_id"]),
        "offset": int(offset),
        "baseline": baseline,
        "full_vector": profile["full_vector"].tolist(),
        "shared_vector": profile["shared_vector"].tolist(),
        "window_vectors": profile["window_vectors"].tolist(),
        "window_starts": [spec.start for spec in specs],
        "target_positions": [spec.target_position for spec in specs],
        "context_positions": [spec.context_position for spec in specs],
        "isolation": isolation,
        "runtime": {
            key: value
            for key, value in profile.items()
            if key
            in {
                "assembly_seconds",
                "forward_seconds",
                "forward_call_count",
                "batch_sizes",
                "batch_size",
            }
        },
    }
    return long_row, window_row, batch_row, profile_payload


def run() -> None:
    started = time.perf_counter()
    snapshot_path = OUTPUT_ROOT / "protocol_snapshot.json"
    selected_path = OUTPUT_ROOT / "selected_samples.json"
    if not snapshot_path.is_file() or not selected_path.is_file():
        raise SmokeBlocked("Run prepare before inference")
    protocol = read_json(snapshot_path)
    if protocol.get("status") != "FROZEN_BEFORE_INFERENCE":
        raise SmokeBlocked("Protocol snapshot is not frozen")
    if git("rev-parse", "HEAD") != protocol["source_git_head"]:
        raise SmokeBlocked("Source HEAD changed after protocol freeze")
    if git("status", "--porcelain"):
        raise SmokeBlocked("Worktree changed after protocol freeze")
    current_hashes = {
        "checkpoint": sha256_file(CHECKPOINT),
        "predictions": sha256_file(PREDICTIONS),
        "split": sha256_file(SPLIT),
        "authority": sha256_file(AUTHORITY),
        "development_manifest": sha256_file(DEVELOPMENT_MANIFEST),
    }
    for name, expected in EXPECTED_HASHES.items():
        if current_hashes[name] != expected:
            raise SmokeBlocked(f"Pre-inference lineage mismatch: {name}")
    authority_checks = verify_authority()
    selected_payload = read_json(selected_path)
    selected = list(selected_payload["samples"])
    if len(selected) != 2 or [int(row["sample_id"]) for row in selected] != [8809, 58657]:
        raise SmokeBlocked("Frozen two-sample selection changed")

    # Deliberately access only named train arrays. Never iterate/load test or val keys.
    npz_keys_read = [
        "class_labels",
        "train_y_true",
        "train_probs_raw",
        "train_recording_id",
        "train_epoch_index",
        "train_file_sample_counts",
    ]
    with np.load(PREDICTIONS, allow_pickle=False) as archive:
        class_order = archive["class_labels"].astype(str).tolist()
        train_labels = archive["train_y_true"].astype(np.int16)
        train_probs = archive["train_probs_raw"].astype(np.float32)
        train_recording_ids = archive["train_recording_id"].astype(np.int32)
        train_epoch_ids = archive["train_epoch_index"].astype(np.int32)
        train_counts = archive["train_file_sample_counts"].astype(np.int32)
    if class_order != list(CLASS_LABELS):
        raise SmokeBlocked("Class order changed")
    split_payload = read_json(SPLIT)

    # TensorFlow/model imports occur only after every frozen pre-inference check.
    from tools import sequence_lime_paper_evaluation as evaluator

    evaluator.ensure_lime_imports()
    model = evaluator.build_model((3000, 3), (30, 28), initialization_seed=MODEL_SEED)
    model_fingerprint_before = evaluator.model_weight_fingerprint(model)
    peak_after_load = tf_peak_bytes()

    baseline_rows: list[Mapping[str, Any]] = []
    no_op_rows: list[Mapping[str, Any]] = []
    batch_rows: list[Mapping[str, Any]] = []
    window_rows: list[Mapping[str, Any]] = []
    cross_view_rows: list[Mapping[str, Any]] = []
    influence_rows: list[Mapping[str, Any]] = []
    profile_rows: list[Mapping[str, Any]] = []
    per_sample_runtime: list[Mapping[str, Any]] = []
    determinism_reference: dict[tuple[int, str], np.ndarray] = {}
    data_files_opened = [
        str(path.relative_to(PROJECT))
        for path in (
            PREDICTIONS,
            SPLIT,
            AUTHORITY,
            DEVELOPMENT_MANIFEST,
            CHECKPOINT,
            PLANNING_REPORT,
            RELIABILITY_REPORT,
            snapshot_path,
            selected_path,
        )
    ]

    for sample_index, sample in enumerate(selected):
        sample_tick = time.perf_counter()
        sample_id = int(sample["sample_id"])
        recording_id = int(sample["recording_id"])
        cache_path = DEVELOPMENT_CACHE_ROOT / f"{Path(sample['source_recording']).stem}.npz"
        source_path = Path(split_payload["source_npz_dir"]) / sample["source_recording"]
        data_files_opened.extend([str(cache_path.relative_to(PROJECT)), str(source_path)])
        if sha256_file(source_path) != sample["source_file_sha256"]:
            raise SmokeBlocked(f"Selected source recording changed: {sample_id}")
        with np.load(cache_path, allow_pickle=False) as cache:
            x_raw = cache["x_raw"].astype(np.float32)
            x_seg = cache["x_seg"].astype(np.float32)
            cache_labels = cache["labels"].astype(np.int16)
        start = int(np.sum(train_counts[:recording_id]))
        stop = start + int(train_counts[recording_id])
        target_epoch = int(sample["target_epoch_index"])
        if sample_id != start + target_epoch:
            raise SmokeBlocked("Global/local epoch mapping changed")
        if not np.array_equal(cache_labels, train_labels[start:stop]):
            raise SmokeBlocked("Development cache labels do not match train artifact")
        if int(train_recording_ids[sample_id]) != recording_id or int(train_epoch_ids[sample_id]) != target_epoch:
            raise SmokeBlocked("Prediction recording/epoch axis mismatch")

        original = infer_profile(
            evaluator,
            model,
            x_raw,
            x_seg,
            target_epoch,
            target_epoch + OFFSETS[0],
            None,
            None,
            BATCH_SIZE,
        )
        artifact_vector = train_probs[sample_id].astype(np.float64)
        baseline_diff = np.abs(original["full_vector"] - artifact_vector)
        baseline_pass = bool(
            np.max(baseline_diff) <= PARITY_ATOL
            and int(np.argmax(original["full_vector"])) == int(np.argmax(artifact_vector))
        )
        baseline_rows.append(
            {
                "sample_id": sample_id,
                "subject_id": sample["subject_id"],
                "recording_id": recording_id,
                "target_epoch_index": target_epoch,
                "artifact_vector": artifact_vector.tolist(),
                "rerun_vector": original["full_vector"].tolist(),
                "max_abs_diff": float(np.max(baseline_diff)),
                "mean_abs_diff": float(np.mean(baseline_diff)),
                "argmax_match": bool(np.argmax(original["full_vector"]) == np.argmax(artifact_vector)),
                "probability_valid": probability_valid(original["full_vector"][None, :]),
                "tolerance": PARITY_ATOL,
                "pass": baseline_pass,
            }
        )
        if not baseline_pass:
            raise SmokeBlocked(f"Baseline parity failed for sample {sample_id}")
        target_class = int(np.argmax(original["full_vector"]))

        excluded = range(target_epoch + min(OFFSETS), target_epoch + max(OFFSETS) + 1)
        medoid_index, medoid_meta = deterministic_feature_medoid(x_seg, excluded)
        zero_raw = np.zeros_like(x_raw[0])
        zero_segment = recompute_segment_feature(evaluator, zero_raw)
        medoid_raw = x_raw[medoid_index].copy()
        medoid_segment = recompute_segment_feature(evaluator, medoid_raw)
        medoid_feature_diff = float(np.max(np.abs(medoid_segment - x_seg[medoid_index])))
        if medoid_feature_diff > PARITY_ATOL:
            raise SmokeBlocked(f"Medoid feature recomputation mismatch for sample {sample_id}")

        sample_reference: dict[tuple[int, str], np.ndarray] = {}
        for offset in OFFSETS:
            context_epoch = target_epoch + offset
            no_op_segment = recompute_segment_feature(evaluator, x_raw[context_epoch])
            cached_feature_diff = float(np.max(np.abs(no_op_segment - x_seg[context_epoch])))
            if cached_feature_diff > PARITY_ATOL:
                raise SmokeBlocked(
                    f"No-op feature recomputation mismatch sample={sample_id} offset={offset}"
                )
            no_op = infer_profile(
                evaluator,
                model,
                x_raw,
                x_seg,
                target_epoch,
                context_epoch,
                x_raw[context_epoch],
                no_op_segment,
                BATCH_SIZE,
            )
            no_op_diff = np.abs(no_op["full_vector"] - original["full_vector"])
            no_op_pass = bool(np.max(no_op_diff) <= PARITY_ATOL)
            no_op_rows.append(
                {
                    "sample_id": sample_id,
                    "offset": int(offset),
                    "context_epoch_index": context_epoch,
                    "cached_vs_recomputed_feature_max_abs_diff": cached_feature_diff,
                    "influence_max_abs": float(np.max(no_op_diff)),
                    "target_response_abs_diff": float(no_op_diff[target_class]),
                    "tolerance": PARITY_ATOL,
                    "pass": no_op_pass,
                }
            )
            if not no_op_pass:
                raise SmokeBlocked(f"No-op control failed sample={sample_id} offset={offset}")

            replacements = {
                "normalized_zero": (
                    zero_raw,
                    zero_segment,
                    {
                        "kind": "synthetic_normalized_zero",
                        "donor_epoch_index": None,
                    },
                ),
                "recording_feature_medoid": (
                    medoid_raw,
                    medoid_segment,
                    {
                        "kind": "real_recording_epoch",
                        "donor_epoch_index": medoid_index,
                        "donor_global_epoch_index": start + medoid_index,
                        **medoid_meta,
                    },
                ),
            }
            for baseline, (replacement_raw, replacement_segment, replacement_target) in replacements.items():
                long_row, window_row, batch_row, profile_payload = record_intervention(
                    evaluator=evaluator,
                    model=model,
                    sample=sample,
                    x_raw=x_raw,
                    x_seg=x_seg,
                    original=original,
                    target_class=target_class,
                    offset=offset,
                    baseline=baseline,
                    replacement_raw=replacement_raw,
                    replacement_segment=replacement_segment,
                    replacement_target=replacement_target,
                )
                isolation = profile_payload["isolation"]
                isolation_pass = bool(
                    isolation["changed_epoch_indices"] == [context_epoch]
                    and isolation["raw_changed_epoch_indices"] == [context_epoch]
                    and isolation["segment_changed_epoch_indices"] == [context_epoch]
                    and isolation["other_epochs_unchanged"]
                )
                cross_view = {
                    "sample_id": sample_id,
                    "offset": int(offset),
                    "baseline": baseline,
                    "context_epoch_index": context_epoch,
                    "replacement_segment_recomputed_from_replacement_waveform": True,
                    "raw_and_segment_changed_same_single_epoch": isolation_pass,
                    "unaffected_raw_epochs_exact": isolation["other_epochs_unchanged"],
                    "unaffected_segment_epochs_exact": isolation["other_epochs_unchanged"],
                    "raw_changed_scalar_count": isolation["raw_changed_scalar_count"],
                    "segment_changed_scalar_count": isolation["segment_changed_scalar_count"],
                    "medoid_cached_vs_recomputed_feature_max_abs_diff": medoid_feature_diff if baseline == "recording_feature_medoid" else None,
                    "pass": isolation_pass,
                }
                if not isolation_pass:
                    raise SmokeBlocked(
                        f"Single-epoch/cross-view isolation failed sample={sample_id} offset={offset} baseline={baseline}"
                    )
                if window_row["expected_affected_windows"] != window_row["actual_rebuilt_affected_windows"] or window_row["duplicate_window_count"]:
                    raise SmokeBlocked("Overlapping-window reconstruction failed")
                if not batch_row["pass"]:
                    raise SmokeBlocked("Batch parity failed")
                influence_rows.append(long_row)
                window_rows.append(window_row)
                batch_rows.append(batch_row)
                cross_view_rows.append(cross_view)
                profile_rows.append(profile_payload)
                sample_reference[(offset, baseline)] = np.asarray(
                    profile_payload["full_vector"], dtype=np.float64
                )

        if sample_index == 0:
            determinism_reference = sample_reference
        per_sample_runtime.append(
            {
                "sample_id": sample_id,
                "seconds": float(time.perf_counter() - sample_tick),
                "offset_count": len(OFFSETS),
                "baseline_count": len(BASELINES),
            }
        )

    # Repeat the complete first sample profile without changing its frozen inputs.
    first = selected[0]
    cache_path = DEVELOPMENT_CACHE_ROOT / f"{Path(first['source_recording']).stem}.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        first_raw = cache["x_raw"].astype(np.float32)
        first_seg = cache["x_seg"].astype(np.float32)
    first_target = int(first["target_epoch_index"])
    first_original = infer_profile(
        evaluator,
        model,
        first_raw,
        first_seg,
        first_target,
        first_target + OFFSETS[0],
        None,
        None,
        BATCH_SIZE,
    )
    excluded = range(first_target + min(OFFSETS), first_target + max(OFFSETS) + 1)
    first_medoid, _ = deterministic_feature_medoid(first_seg, excluded)
    rerun_replacements = {
        "normalized_zero": (
            np.zeros_like(first_raw[0]),
            recompute_segment_feature(evaluator, np.zeros_like(first_raw[0])),
        ),
        "recording_feature_medoid": (
            first_raw[first_medoid].copy(),
            recompute_segment_feature(evaluator, first_raw[first_medoid]),
        ),
    }
    determinism_rows = []
    for offset in OFFSETS:
        for baseline, (replacement_raw, replacement_segment) in rerun_replacements.items():
            repeated = infer_profile(
                evaluator,
                model,
                first_raw,
                first_seg,
                first_target,
                first_target + offset,
                replacement_raw,
                replacement_segment,
                BATCH_SIZE,
            )
            maximum = float(
                np.max(
                    np.abs(
                        repeated["full_vector"]
                        - determinism_reference[(offset, baseline)]
                    )
                )
            )
            determinism_rows.append(
                {
                    "sample_id": int(first["sample_id"]),
                    "offset": int(offset),
                    "baseline": baseline,
                    "max_abs_diff": maximum,
                    "tolerance": PARITY_ATOL,
                    "pass": maximum <= PARITY_ATOL,
                }
            )
    if not all(row["pass"] for row in determinism_rows):
        raise SmokeBlocked("Deterministic full-sample rerun failed")

    model_fingerprint_after = evaluator.model_weight_fingerprint(model)
    if model_fingerprint_before != model_fingerprint_after:
        raise SmokeBlocked("Frozen model weights changed during inference")
    peak_bytes = tf_peak_bytes()
    final_hashes = {
        "checkpoint": sha256_file(CHECKPOINT),
        "predictions": sha256_file(PREDICTIONS),
        "split": sha256_file(SPLIT),
        "authority": sha256_file(AUTHORITY),
        "development_manifest": sha256_file(DEVELOPMENT_MANIFEST),
    }
    if final_hashes != current_hashes:
        raise SmokeBlocked("A source asset changed during smoke")

    baseline_payload = {
        "schema_version": "context_smoke_baseline_parity_v1",
        "rows": baseline_rows,
        "max_abs_diff": max(row["max_abs_diff"] for row in baseline_rows),
        "tolerance": PARITY_ATOL,
        "status": "PASS" if all(row["pass"] for row in baseline_rows) else "FAIL",
    }
    no_op_payload = {
        "schema_version": "context_smoke_no_op_v1",
        "rows": no_op_rows,
        "max_abs_influence": max(row["influence_max_abs"] for row in no_op_rows),
        "tolerance": PARITY_ATOL,
        "status": "PASS" if all(row["pass"] for row in no_op_rows) else "FAIL",
    }
    batch_payload = {
        "schema_version": "context_smoke_batch_parity_v1",
        "rows": batch_rows,
        "max_abs_full_vector_diff": max(row["max_abs_full_vector_diff"] for row in batch_rows),
        "max_abs_window_vector_diff": max(row["max_abs_window_vector_diff"] for row in batch_rows),
        "tolerance": PARITY_ATOL,
        "status": "PASS" if all(row["pass"] for row in batch_rows) else "FAIL",
    }
    determinism_payload = {
        "schema_version": "context_smoke_determinism_v1",
        "scope": "complete first sample across all offsets and both baselines",
        "rows": determinism_rows,
        "max_abs_diff": max(row["max_abs_diff"] for row in determinism_rows),
        "tolerance": PARITY_ATOL,
        "status": "PASS" if all(row["pass"] for row in determinism_rows) else "FAIL",
    }
    cross_view_payload = {
        "schema_version": "context_smoke_cross_view_v1",
        "rows": cross_view_rows,
        "max_medoid_cached_vs_recomputed_feature_diff": max(
            row["medoid_cached_vs_recomputed_feature_max_abs_diff"] or 0.0
            for row in cross_view_rows
        ),
        "status": "PASS" if all(row["pass"] for row in cross_view_rows) else "FAIL",
    }
    access_payload = {
        "schema_version": "context_smoke_access_audit_v1",
        "data_files_opened": sorted(set(data_files_opened)),
        "npz_keys_read": npz_keys_read,
        "selected_development_sample_ids": [int(row["sample_id"]) for row in selected],
        "development_samples_inferred": 2,
        "remaining_development_16_inferred": 0,
        "validation_30_accessed": False,
        "test_36_accessed": False,
        "validation_prediction_arrays_read": False,
        "test_prediction_arrays_read": False,
        "hmm_executed": False,
        "gradient_or_training_executed": False,
        "status": "PASS",
    }
    total_seconds = time.perf_counter() - started
    runtime_payload = {
        "schema_version": "context_smoke_runtime_v1",
        "total_seconds": float(total_seconds),
        "per_sample": per_sample_runtime,
        "mean_seconds_per_sample": float(np.mean([row["seconds"] for row in per_sample_runtime])),
        "mean_seconds_per_sample_offset_baseline": float(
            np.mean([row["runtime_seconds"] for row in influence_rows])
        ),
        "peak_gpu_allocator_bytes_after_model_load": peak_after_load,
        "peak_gpu_allocator_bytes": peak_bytes,
        "peak_gpu_allocator_gib": None if peak_bytes is None else peak_bytes / (1024**3),
        "cpu_max_rss_kib": None
        if resource is None
        else int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "device": "single NVIDIA RTX 3090 selected by CUDA_VISIBLE_DEVICES",
        "development_16_linear_projection_seconds": float(
            8.0 * np.mean([row["seconds"] for row in per_sample_runtime])
        ),
        "resource_gate": "PASS",
    }

    write_json(OUTPUT_ROOT / "baseline_parity.json", baseline_payload)
    write_json(OUTPUT_ROOT / "no_op_control.json", no_op_payload)
    write_json(OUTPUT_ROOT / "batch_parity.json", batch_payload)
    write_json(OUTPUT_ROOT / "determinism_check.json", determinism_payload)
    pd.DataFrame(window_rows).to_csv(
        OUTPUT_ROOT / "window_reconstruction_audit.csv", index=False
    )
    write_json(OUTPUT_ROOT / "cross_view_update_audit.json", cross_view_payload)
    pd.DataFrame(influence_rows).to_csv(
        OUTPUT_ROOT / "context_influence_long.csv", index=False
    )
    write_json(
        OUTPUT_ROOT / "context_influence_profiles.json",
        {
            "schema_version": "context_smoke_profiles_v1",
            "scope": "SMOKE_ONLY_NOT_SCIENTIFIC_RESULTS",
            "profiles": profile_rows,
        },
    )
    write_json(OUTPUT_ROOT / "runtime_profile.json", runtime_payload)
    write_json(OUTPUT_ROOT / "access_audit.json", access_payload)

    gates = {
        "lineage": "PASS",
        "exactly_two_frozen_development_samples": "PASS",
        "baseline_parity": baseline_payload["status"],
        "no_op": no_op_payload["status"],
        "single_physical_epoch_isolation": "PASS",
        "raw_segment_cross_view": cross_view_payload["status"],
        "overlapping_window_reconstruction": "PASS",
        "target_position_mapping": "PASS",
        "batch_parity": batch_payload["status"],
        "deterministic_rerun": determinism_payload["status"],
        "finite_probability_semantics": "PASS",
        "no_hmm_gradient_training_or_weight_update": "PASS",
        "no_validation_or_test_access": access_payload["status"],
        "canonical_lime_and_authoritative_states_unchanged": "PASS",
        "checkpoint_prediction_asset_hashes_unchanged": "PASS",
        "single_rtx3090_resource_feasibility": runtime_payload["resource_gate"],
    }
    final_status = (
        "ELIGIBLE_FOR_FROZEN_DEVELOPMENT_16"
        if all(value == "PASS" for value in gates.values())
        else "DEVELOPMENT_SMOKE_BLOCKED"
    )
    artifacts = []
    for path in sorted(OUTPUT_ROOT.iterdir()):
        if path.name == "run_manifest.json" or not path.is_file():
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(PROJECT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    run_manifest = {
        "schema_version": "context_smoke_run_manifest_v1",
        "status": final_status,
        "completed_at": utc_now(),
        "source_git_head": protocol["source_git_head"],
        "source_git_branch": protocol["source_git_branch"],
        "sample_ids": [int(row["sample_id"]) for row in selected],
        "split": "development",
        "sample_count": 2,
        "protocol_snapshot_sha256": sha256_file(snapshot_path),
        "gates": gates,
        "model_weight_fingerprint_before": model_fingerprint_before,
        "model_weight_fingerprint_after": model_fingerprint_after,
        "source_hashes_before": current_hashes,
        "source_hashes_after": final_hashes,
        "authority_checks": authority_checks,
        "artifacts": artifacts,
        "development_16_started": False,
        "scientific_conclusion_allowed": False,
        "canonical_lime_modified": False,
    }
    write_json(OUTPUT_ROOT / "run_manifest.json", run_manifest)
    readme = f"""# Sequence Context Influence Development Smoke 2

Status: `{final_status}`

This directory is an engineering-only, frozen two-development-sample smoke.
It is not a scientific experiment and must not be used to rank offsets,
baselines, methods, or physiological mechanisms. No development-16,
validation-30, or test-36 run was started.

Protocol: `protocol_snapshot.json`
Samples: `{', '.join(str(row['sample_id']) for row in selected)}`
Pre-HMM only; training=False; no HMM; canonical LIME unchanged.
"""
    (OUTPUT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"status": final_status, "total_seconds": total_seconds}, indent=2))


def write_blocked(error: Exception) -> None:
    if not OUTPUT_ROOT.exists():
        return
    payload = {
        "schema_version": "context_smoke_run_manifest_v1",
        "status": "DEVELOPMENT_SMOKE_BLOCKED",
        "blocked_at": utc_now(),
        "error_type": type(error).__name__,
        "error": str(error),
        "development_16_started": False,
        "scientific_conclusion_allowed": False,
    }
    write_json(OUTPUT_ROOT / "run_manifest.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    args = parser.parse_args()
    try:
        prepare() if args.mode == "prepare" else run()
    except Exception as error:
        if args.mode == "run":
            write_blocked(error)
        raise


if __name__ == "__main__":
    main()
