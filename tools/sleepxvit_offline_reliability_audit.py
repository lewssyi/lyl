#!/usr/bin/env python3
"""Offline reliability audit for frozen A1-zero pre-HMM probabilities.

This tool is intentionally inference-free. It reads only frozen train/validation
arrays, validates physical-epoch lineage, computes validation reliability
statistics, performs subject-level bootstrap, and audits the selected-30
canonical B0_CLASSIC_90 explanations as repeated measures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs" / "predictions" / (
    "result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42_split_predictions.npz"
)
DEFAULT_SPLIT = PROJECT_ROOT / "dataset_protocols" / "ISRUC" / "split.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / (
    "result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42.weights.h5"
)
DEFAULT_STATUS = PROJECT_ROOT / "outputs" / "xai_authoritative_status" / "isruc_seed42" / "CURRENT_XAI_STATUS.json"
DEFAULT_RELIABILITY = PROJECT_ROOT / "outputs" / "hierarchical_context_xai_feasibility" / "isruc_seed42" / "reliability_sample_level.csv"
DEFAULT_ATTRIBUTION_RUNS = PROJECT_ROOT / "outputs" / "hierarchical_context_xai_feasibility" / "isruc_seed42" / "runs"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "sleepxvit_inspired_audit" / "isruc_seed42" / "reliability_v1"

EXPECTED_PREDICTION_SHA256 = "53cd6b51c008e0dd47dd187654c8eddacd5d0a69037c48aeecafe5efa9594e39"
EXPECTED_CHECKPOINT_SHA256 = "e86a84790fd7351c1864158eef79fbf28437a7a8a5f26662b93bcf1aef14b6ed"
EXPECTED_SPLIT_SHA256 = "8363546a86b81676fe6171f2e50405d2d7447cbef1f48dbf397cd53bf3f037d9"
PLANNING_SOURCE_HEAD = "a834bf9"
PLANNING_REPORT_COMMIT = "ad0dc77"
EXPECTED_CLASS_ORDER = ("W", "N1", "N2", "N3", "REM")
AUTHORITATIVE_STATES = {
    "scientific_status": "MODEL_FAITHFULNESS_NOT_ESTABLISHED",
    "physiology_aware_visualization_decision": "PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED",
    "feedback_training_status": "PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD",
}
DEFAULT_BOOTSTRAP_SEED = 20260802
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_BINS = 15
HIGH_CONFIDENCE_THRESHOLD = 0.90
PROBABILITY_TOLERANCE = 1e-5


class AuditBlocked(RuntimeError):
    """Raised when a critical input-lineage requirement is not satisfied."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_value(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def manifest_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ensure_new_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def canonical_subject_mapping(split_payload: dict[str, Any], split_name: str) -> tuple[list[str], list[str]]:
    files = [str(item) for item in split_payload["split_files"][split_name]]
    subjects = [str(item) for item in split_payload["split_subjects"][split_name]]
    if len(files) != len(subjects):
        raise AuditBlocked(f"{split_name}: split_files and split_subjects lengths differ")
    inferred = [Path(filename).stem for filename in files]
    if inferred != subjects:
        raise AuditBlocked(f"{split_name}: ordered split_files do not map exactly to split_subjects")
    if files != sorted(files):
        raise AuditBlocked(f"{split_name}: split_files are not in canonical sorted order")
    return files, subjects


def materialize_subject_ids(recording_id: np.ndarray, subjects: Sequence[str]) -> np.ndarray:
    recording_id = np.asarray(recording_id, dtype=np.int64).reshape(-1)
    if len(recording_id) == 0:
        return np.asarray([], dtype="U1")
    if int(recording_id.min()) < 0 or int(recording_id.max()) >= len(subjects):
        raise AuditBlocked("recording_id contains values outside ordered subject mapping")
    return np.asarray([subjects[int(index)] for index in recording_id])


def validate_probability_array(probabilities: np.ndarray, expected_rows: int) -> dict[str, Any]:
    probabilities = np.asarray(probabilities)
    finite = bool(np.isfinite(probabilities).all())
    row_sum_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))) if probabilities.ndim == 2 else math.inf
    minimum = float(np.min(probabilities)) if probabilities.size else math.nan
    maximum = float(np.max(probabilities)) if probabilities.size else math.nan
    normalized = bool(
        probabilities.shape == (expected_rows, len(EXPECTED_CLASS_ORDER))
        and finite
        and minimum >= -PROBABILITY_TOLERANCE
        and maximum <= 1.0 + PROBABILITY_TOLERANCE
        and row_sum_error <= PROBABILITY_TOLERANCE
    )
    return {
        "shape": list(probabilities.shape),
        "dtype": str(probabilities.dtype),
        "finite": finite,
        "minimum": minimum,
        "maximum": maximum,
        "maximum_row_sum_absolute_error": row_sum_error,
        "probability_semantics": "NORMALIZED_PROBABILITY" if normalized else "UNCONFIRMED",
        "softmax_required": False if normalized else None,
        "pass": normalized,
    }


def validate_split_axes(
    split_name: str,
    labels: np.ndarray,
    recording_id: np.ndarray,
    epoch_index: np.ndarray,
    file_counts: np.ndarray,
    subject_ids: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    recording_id = np.asarray(recording_id, dtype=np.int64).reshape(-1)
    epoch_index = np.asarray(epoch_index, dtype=np.int64).reshape(-1)
    file_counts = np.asarray(file_counts, dtype=np.int64).reshape(-1)
    n = len(labels)
    aligned = len(recording_id) == n == len(epoch_index) == len(subject_ids)
    contiguous_recordings = set(np.unique(recording_id).tolist()) == set(range(len(file_counts)))
    counts_match = int(file_counts.sum()) == n
    count_axis_match = True
    epoch_axis_match = True
    for rid, expected_count in enumerate(file_counts):
        mask = recording_id == rid
        if int(mask.sum()) != int(expected_count):
            count_axis_match = False
        if not np.array_equal(epoch_index[mask], np.arange(int(expected_count), dtype=np.int64)):
            epoch_axis_match = False
    keys = list(zip(subject_ids.tolist(), epoch_index.tolist()))
    unique_epochs = len(set(keys)) == n
    label_range = sorted(np.unique(labels).tolist())
    return {
        "split": split_name,
        "row_count": n,
        "array_alignment": aligned,
        "file_sample_counts_sum": int(file_counts.sum()),
        "file_sample_counts_match": counts_match,
        "recording_axis_contiguous": contiguous_recordings,
        "recording_counts_match": count_axis_match,
        "epoch_axis_zero_based_contiguous_within_recording": epoch_axis_match,
        "unique_physical_epoch_count": len(set(keys)),
        "physical_epoch_keys_unique": unique_epochs,
        "subject_count": int(len(np.unique(subject_ids))),
        "label_values": label_range,
        "label_range_valid": label_range == list(range(len(EXPECTED_CLASS_ORDER))),
        "pass": bool(
            aligned
            and counts_match
            and contiguous_recordings
            and count_axis_match
            and epoch_axis_match
            and unique_epochs
            and label_range == list(range(len(EXPECTED_CLASS_ORDER)))
        ),
    }


def per_sample_losses(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    nll = -np.log(clipped[np.arange(len(labels)), labels])
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    brier = np.sum((probabilities - one_hot) ** 2, axis=1)
    return nll, brier


def calibration_bins_equal_width(
    confidence: np.ndarray, correctness: np.ndarray, n_bins: int = DEFAULT_BINS
) -> tuple[pd.DataFrame, float, float]:
    confidence = np.asarray(confidence, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: list[dict[str, Any]] = []
    weighted_gap = 0.0
    gaps = []
    for index in range(int(n_bins)):
        lower, upper = float(edges[index]), float(edges[index + 1])
        mask = (confidence >= lower) & ((confidence < upper) if index < n_bins - 1 else (confidence <= upper))
        count = int(mask.sum())
        if count:
            mean_confidence = float(confidence[mask].mean())
            accuracy = float(correctness[mask].mean())
            gap = abs(mean_confidence - accuracy)
            weighted_gap += (count / len(confidence)) * gap
            gaps.append(gap)
        else:
            mean_confidence = accuracy = gap = math.nan
        rows.append({
            "binning": "equal_width",
            "bin_index": index + 1,
            "lower_bound": lower,
            "upper_bound": upper,
            "lower_inclusive": True,
            "upper_inclusive": index == n_bins - 1,
            "n": count,
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
            "absolute_gap": gap,
        })
    return pd.DataFrame(rows), float(weighted_gap), float(max(gaps) if gaps else math.nan)


def calibration_bins_equal_mass(
    confidence: np.ndarray, correctness: np.ndarray, n_bins: int = DEFAULT_BINS
) -> tuple[pd.DataFrame, float, float]:
    confidence = np.asarray(confidence, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    if len(confidence) == 0:
        return pd.DataFrame(), math.nan, math.nan
    actual_bins = min(int(n_bins), len(confidence))
    order = np.argsort(confidence, kind="stable")
    groups = np.array_split(order, actual_bins)
    rows: list[dict[str, Any]] = []
    weighted_gap = 0.0
    gaps = []
    rank_start = 0
    for index, group in enumerate(groups):
        values = confidence[group]
        count = len(group)
        mean_confidence = float(values.mean())
        accuracy = float(correctness[group].mean())
        gap = abs(mean_confidence - accuracy)
        weighted_gap += (count / len(confidence)) * gap
        gaps.append(gap)
        rows.append({
            "binning": "equal_mass",
            "bin_index": index + 1,
            "actual_bin_count": actual_bins,
            "rank_start_inclusive": rank_start,
            "rank_end_exclusive": rank_start + count,
            "lower_bound": float(values.min()),
            "upper_bound": float(values.max()),
            "n": int(count),
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
            "absolute_gap": gap,
        })
        rank_start += count
    return pd.DataFrame(rows), float(weighted_gap), float(max(gaps))


def risk_coverage_curve(confidence: np.ndarray, correctness: np.ndarray) -> tuple[pd.DataFrame, float]:
    confidence = np.asarray(confidence, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    order = np.argsort(-confidence, kind="stable")
    sorted_confidence = confidence[order]
    cumulative_correct = np.cumsum(correctness[order], dtype=np.float64)
    retained = np.arange(1, len(confidence) + 1, dtype=np.int64)
    selective_accuracy = cumulative_correct / retained
    risk = 1.0 - selective_accuracy
    coverage = retained / len(confidence)
    frame = pd.DataFrame({
        "retained_n": retained,
        "coverage": coverage,
        "confidence_threshold": sorted_confidence,
        "selective_accuracy": selective_accuracy,
        "error_risk": risk,
    })
    # Discrete AURC definition used throughout this audit: mean risk over all
    # top-k coverages k/n, k=1,...,n.
    aurc = float(np.mean(risk))
    return frame, aurc


def coverage_operating_points(
    confidence: np.ndarray,
    correctness: np.ndarray,
    subject_ids: np.ndarray,
    requested: Iterable[float] = (1.0, 0.9, 0.8, 0.7, 0.5, 0.3),
) -> pd.DataFrame:
    confidence = np.asarray(confidence, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    subject_ids = np.asarray(subject_ids)
    sorted_confidence = np.sort(confidence)[::-1]
    rows = []
    for target in requested:
        rank = min(max(int(math.ceil(float(target) * len(confidence))), 1), len(confidence))
        threshold = float(sorted_confidence[rank - 1])
        mask = confidence >= threshold
        actual = float(mask.mean())
        accuracy = float(correctness[mask].mean())
        rows.append({
            "requested_coverage": float(target),
            "actual_coverage": actual,
            "retained_n": int(mask.sum()),
            "subject_count": int(len(np.unique(subject_ids[mask]))),
            "selective_accuracy": accuracy,
            "error_risk": 1.0 - accuracy,
            "confidence_threshold": threshold,
            "threshold_role": "VALIDATION_DESCRIPTIVE_OPERATING_POINT",
        })
    return pd.DataFrame(rows)


def compute_metric_bundle(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predictions == labels
    nll, brier = per_sample_losses(probabilities, labels)
    _, ece_equal_mass, mce_equal_mass = calibration_bins_equal_mass(confidence, correct)
    _, ece_equal_width, mce_equal_width = calibration_bins_equal_width(confidence, correct)
    _, aurc = risk_coverage_curve(confidence, correct)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "nll": float(nll.mean()),
        "multiclass_brier_sum": float(brier.mean()),
        "ece_equal_mass_15": ece_equal_mass,
        "ece_equal_width_15": ece_equal_width,
        "mce_equal_mass_15": mce_equal_mass,
        "mce_equal_width_15": mce_equal_width,
        "aurc_discrete_mean_risk": aurc,
        "mean_confidence": float(confidence.mean()),
        "confidence_minus_accuracy": float(confidence.mean() - correct.mean()),
    }


def classwise_tables(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predictions == labels
    nll, brier = per_sample_losses(probabilities, labels)
    metric_rows: list[dict[str, Any]] = []
    ovr_mass, ovr_width = [], []
    for class_index, class_name in enumerate(class_names):
        binary_truth = labels == class_index
        class_probability = probabilities[:, class_index]
        mass, mass_ece, mass_mce = calibration_bins_equal_mass(class_probability, binary_truth)
        width, width_ece, width_mce = calibration_bins_equal_width(class_probability, binary_truth)
        for frame, sink in ((mass, ovr_mass), (width, ovr_width)):
            frame.insert(0, "class_index", class_index)
            frame.insert(1, "class_name", class_name)
            sink.append(frame)
        binary_brier = float(np.mean((class_probability - binary_truth.astype(float)) ** 2))
        binary_nll = float(np.mean(-(
            binary_truth * np.log(np.clip(class_probability, 1e-12, 1.0))
            + (~binary_truth) * np.log(np.clip(1.0 - class_probability, 1e-12, 1.0))
        )))
        metric_rows.append({
            "analysis_scope": "one_vs_rest",
            "stage": class_name,
            "n": int(len(labels)),
            "positive_n": int(binary_truth.sum()),
            "accuracy": math.nan,
            "mean_confidence": float(class_probability.mean()),
            "nll": binary_nll,
            "brier": binary_brier,
            "ece_equal_mass_15": mass_ece,
            "ece_equal_width_15": width_ece,
            "mce_equal_mass_15": mass_mce,
            "mce_equal_width_15": width_mce,
        })
        for scope, mask in (("true_stage", labels == class_index), ("predicted_stage", predictions == class_index)):
            error_mask = mask & (~correct)
            metric_rows.append({
                "analysis_scope": scope,
                "stage": class_name,
                "n": int(mask.sum()),
                "positive_n": math.nan,
                "accuracy": float(correct[mask].mean()) if mask.any() else math.nan,
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else math.nan,
                "nll": float(nll[mask].mean()) if mask.any() else math.nan,
                "brier": float(brier[mask].mean()) if mask.any() else math.nan,
                "ece_equal_mass_15": math.nan,
                "ece_equal_width_15": math.nan,
                "mce_equal_mass_15": math.nan,
                "mce_equal_width_15": math.nan,
                "error_n": int(error_mask.sum()),
                "high_confidence_error_n": int((error_mask & (confidence >= HIGH_CONFIDENCE_THRESHOLD)).sum()),
                "high_confidence_error_fraction_of_stage": float(
                    (error_mask & (confidence >= HIGH_CONFIDENCE_THRESHOLD)).sum() / max(mask.sum(), 1)
                ),
                "high_confidence_fraction_among_errors": float(
                    (error_mask & (confidence >= HIGH_CONFIDENCE_THRESHOLD)).sum() / max(error_mask.sum(), 1)
                ),
            })
    return (
        pd.DataFrame(metric_rows),
        pd.concat(ovr_mass, ignore_index=True),
        pd.concat(ovr_width, ignore_index=True),
    )


def confidence_distribution_summary(confidence: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    rows = []
    for label, mask in (("correct", correct), ("incorrect", ~correct)):
        values = np.asarray(confidence)[mask]
        rows.append({
            "group": label,
            "n": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
            "median": float(np.median(values)),
            "q10": float(np.quantile(values, 0.10)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "q90": float(np.quantile(values, 0.90)),
            "high_confidence_fraction_ge_0_90": float(np.mean(values >= HIGH_CONFIDENCE_THRESHOLD)),
        })
    return pd.DataFrame(rows)


def subject_level_metrics(
    probabilities: np.ndarray, labels: np.ndarray, subject_ids: np.ndarray
) -> pd.DataFrame:
    rows = []
    for subject in sorted(np.unique(subject_ids).tolist()):
        mask = subject_ids == subject
        metrics = compute_metric_bundle(probabilities[mask], labels[mask])
        rows.append({"subject_id": subject, "n": int(mask.sum()), **metrics})
    return pd.DataFrame(rows)


def subject_bootstrap(
    probabilities: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    subjects = np.asarray(sorted(np.unique(subject_ids).tolist()))
    if len(subjects) == 0:
        return {"status": "NOT_COMPUTABLE", "reason": "subject IDs unavailable"}
    indices = [np.flatnonzero(subject_ids == subject) for subject in subjects]
    rng = np.random.default_rng(int(seed))
    tracked = {name: [] for name in ("nll", "multiclass_brier_sum", "ece_equal_mass_15", "aurc_discrete_mean_risk")}
    failures: list[str] = []
    for iteration in range(int(iterations)):
        try:
            sampled_subjects = rng.integers(0, len(subjects), size=len(subjects))
            sample_index = np.concatenate([indices[int(index)] for index in sampled_subjects])
            metrics = compute_metric_bundle(probabilities[sample_index], labels[sample_index])
            for name in tracked:
                tracked[name].append(float(metrics[name]))
        except Exception as exc:  # pragma: no cover - guarded reporting path
            failures.append(f"iteration={iteration}: {type(exc).__name__}: {exc}")
    point = compute_metric_bundle(probabilities, labels)
    confidence_intervals = {}
    for name, values in tracked.items():
        array = np.asarray(values, dtype=np.float64)
        confidence_intervals[name] = {
            "point_estimate": float(point[name]),
            "lower_95_percentile": float(np.quantile(array, 0.025)) if len(array) else None,
            "upper_95_percentile": float(np.quantile(array, 0.975)) if len(array) else None,
            "bootstrap_mean": float(array.mean()) if len(array) else None,
        }
    return {
        "status": "COMPLETE" if not failures else "COMPLETE_WITH_FAILURES",
        "resampling_unit": "subject",
        "sampling": "sample validation subjects with replacement and retain all epochs for every draw",
        "seed": int(seed),
        "requested_iterations": int(iterations),
        "effective_iterations": int(iterations - len(failures)),
        "failed_iterations": len(failures),
        "subject_count_per_iteration": int(len(subjects)),
        "confidence_interval": "95% percentile",
        "metrics": confidence_intervals,
        "failure_examples": failures[:10],
    }


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 0 else math.nan


def top_k_overlap(first: np.ndarray, second: np.ndarray, k: int = 10) -> float:
    first_top = set(np.argsort(-np.abs(first), kind="stable")[:k].tolist())
    second_top = set(np.argsort(-np.abs(second), kind="stable")[:k].tolist())
    return float(len(first_top & second_top) / k)


def selected30_analysis(
    attribution_runs: Path,
    reliability_path: Path,
    output_dir: Path,
    val_probabilities: np.ndarray,
    val_labels: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    paths = sorted(attribution_runs.glob("validation_classic_sample*_R0_seed*_zero/classic_attribution.csv"))
    attribution_rows = []
    input_files = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame[
            frame["candidate"].eq("B0_CLASSIC_90")
            & frame["target_protocol"].eq("T0")
            & frame["randomization_level"].eq("R0")
        ]
        for row in frame.itertuples(index=False):
            vector = np.asarray(json.loads(row.importance), dtype=np.float64)
            attribution_rows.append({
                "sample_id": int(row.sample_id),
                "seed": int(row.randomization_seed),
                "vector": vector,
                "path": manifest_path(path),
            })
        input_files.append({
            "path": str(path.relative_to(PROJECT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if len(attribution_rows) != 90:
        raise AuditBlocked(f"selected-30 expected 90 T0 repeated-measure rows, found {len(attribution_rows)}")
    samples = sorted({row["sample_id"] for row in attribution_rows})
    if len(samples) != 30:
        raise AuditBlocked(f"selected-30 expected 30 samples, found {len(samples)}")
    if any(row["vector"].shape != (90,) or not np.isfinite(row["vector"]).all() for row in attribution_rows):
        raise AuditBlocked("selected-30 contains a non-finite or non-90D canonical vector")
    pair_rows = []
    sample_rows = []
    for sample_id in samples:
        sample_items = sorted((row for row in attribution_rows if row["sample_id"] == sample_id), key=lambda row: row["seed"])
        seeds = [row["seed"] for row in sample_items]
        if seeds != [3101, 3102, 3103]:
            raise AuditBlocked(f"sample {sample_id}: expected seeds 3101/3102/3103, found {seeds}")
        for first_index in range(3):
            for second_index in range(first_index + 1, 3):
                first, second = sample_items[first_index], sample_items[second_index]
                rho = spearmanr(first["vector"], second["vector"]).statistic
                pair_rows.append({
                    "sample_id": sample_id,
                    "seed_a": first["seed"],
                    "seed_b": second["seed"],
                    "spearman": float(rho) if np.isfinite(rho) else math.nan,
                    "cosine": cosine_similarity(first["vector"], second["vector"]),
                    "top10_overlap_fraction": top_k_overlap(first["vector"], second["vector"], 10),
                })
        sample_pairs = [row for row in pair_rows if row["sample_id"] == sample_id]
        sample_rows.append({
            "sample_id": sample_id,
            "repeated_measure_seed_count": 3,
            "pair_count": 3,
            "mean_pairwise_spearman": float(np.nanmean([row["spearman"] for row in sample_pairs])),
            "mean_pairwise_cosine": float(np.nanmean([row["cosine"] for row in sample_pairs])),
            "mean_pairwise_top10_overlap_fraction": float(np.nanmean([row["top10_overlap_fraction"] for row in sample_pairs])),
        })
    pair_frame = pd.DataFrame(pair_rows)
    sample_frame = pd.DataFrame(sample_rows)
    reliability = pd.read_csv(reliability_path)
    reliability = reliability[
        reliability["method"].eq("B0_CLASSIC_90") & reliability["target_protocol"].eq("T0")
    ].copy()
    if len(reliability) != 30 or reliability["sample_id"].nunique() != 30:
        raise AuditBlocked("selected-30 reliability table is not one T0 B0 row per sample")
    sample_index = reliability["sample_id"].to_numpy(dtype=np.int64)
    if np.any(sample_index < 0) or np.any(sample_index >= len(val_labels)):
        raise AuditBlocked("selected-30 sample_id is outside the validation probability axis")
    expected_confidence = np.max(val_probabilities[sample_index], axis=1)
    confidence_max_abs_diff = float(
        np.max(np.abs(expected_confidence - reliability["confidence"].to_numpy(dtype=float)))
    )
    expected_prediction = np.argmax(val_probabilities[sample_index], axis=1)
    expected_prediction_name = np.asarray(class_names)[expected_prediction]
    expected_true_name = np.asarray(class_names)[val_labels[sample_index]]
    predicted_stage_matches = bool(
        np.array_equal(expected_prediction_name, reliability["predicted_stage"].to_numpy(dtype=str))
    )
    true_stage_matches = bool(
        np.array_equal(expected_true_name, reliability["true_stage"].to_numpy(dtype=str))
    )
    if confidence_max_abs_diff > 1e-8 or not predicted_stage_matches or not true_stage_matches:
        raise AuditBlocked("selected-30 reliability rows do not align with frozen validation probabilities/labels")
    joined = reliability.merge(sample_frame, on="sample_id", how="inner", validate="one_to_one")
    variables = [
        "local_surrogate_R2",
        "response_variance",
        "effective_sample_size",
        "mask_seed_stability",
        "R7_similarity",
        "mean_pairwise_spearman",
        "mean_pairwise_cosine",
        "mean_pairwise_top10_overlap_fraction",
    ]
    correlation_rows = []
    for variable in variables:
        valid = joined[["confidence", variable]].dropna()
        result = spearmanr(valid["confidence"], valid[variable]) if len(valid) >= 3 else None
        correlation_rows.append({
            "analysis_population": "selected_30_exploratory",
            "n": int(len(valid)),
            "x": "confidence",
            "y": variable,
            "spearman_rho": float(result.statistic) if result is not None and np.isfinite(result.statistic) else math.nan,
            "p_value_descriptive_only": float(result.pvalue) if result is not None and np.isfinite(result.pvalue) else math.nan,
            "selection_bias_warning": True,
        })
    pair_frame.to_csv(output_dir / "selected30_seed_consistency_pairs.csv", index=False)
    joined.to_csv(output_dir / "selected30_exploratory_sample_level.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(output_dir / "selected30_descriptive_correlations.csv", index=False)
    inventory = {
        "status": "COMPLETE",
        "analysis_population": "selected_30_exploratory",
        "selection_bias_warning": True,
        "generalization_to_full_validation_allowed": False,
        "clustering_executed": False,
        "canonical_method": "B0_CLASSIC_90",
        "target_protocol": "T0",
        "randomization_level": "R0",
        "sample_count": 30,
        "seed_count_per_sample": 3,
        "seed_rows_are_repeated_measures": True,
        "independent_sample_count": 30,
        "vector_length": 90,
        "attribution_files": input_files,
        "reliability_path": str(reliability_path.relative_to(PROJECT_ROOT)),
        "reliability_sha256": sha256_file(reliability_path),
        "validation_probability_alignment": {
            "confidence_max_abs_diff": confidence_max_abs_diff,
            "predicted_stage_matches_raw_argmax": predicted_stage_matches,
            "true_stage_matches_validation_label": true_stage_matches,
            "pass": True,
        },
        "summary": {
            "mean_pairwise_spearman": float(pair_frame["spearman"].mean()),
            "mean_pairwise_cosine": float(pair_frame["cosine"].mean()),
            "mean_pairwise_top10_overlap_fraction": float(pair_frame["top10_overlap_fraction"].mean()),
        },
    }
    write_json(output_dir / "selected30_input_inventory.json", inventory)
    return inventory


def figure_context(n_epochs: int, n_subjects: int, checkpoint_hash: str) -> str:
    return (
        f"ISRUC validation | n={n_epochs:,} unique epochs | {n_subjects} subjects | "
        f"Pre-HMM, uncalibrated | A1-zero {checkpoint_hash[:8]}"
    )


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linestyle": "-",
    })


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("pdf", "png", "svg"):
        path = figure_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300)
        paths.append(str(path))
    plt.close(fig)
    return paths


def generate_figures(
    output_dir: Path,
    class_names: Sequence[str],
    probabilities: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    equal_mass_bins: pd.DataFrame,
    ovr_mass_bins: pd.DataFrame,
    risk_curve: pd.DataFrame,
    operating_points: pd.DataFrame,
    classwise: pd.DataFrame,
    checkpoint_hash: str,
) -> list[str]:
    configure_plot_style()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    context = figure_context(len(labels), len(np.unique(subject_ids)), checkpoint_hash)
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
    generated: list[str] = []

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.vlines(
        equal_mass_bins["mean_confidence"],
        equal_mass_bins["mean_confidence"],
        equal_mass_bins["accuracy"],
        color="#56B4E9",
        linewidth=1.0,
        alpha=0.55,
    )
    ax.plot(
        equal_mass_bins["mean_confidence"],
        equal_mass_bins["accuracy"],
        color="#0072B2",
        marker="o",
        markersize=4.2,
        linewidth=1.6,
        label="Observed accuracy",
    )
    ax.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1.1, label="Perfect calibration")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy", title="Overall reliability (15 equal-mass bins)")
    ax.legend(loc="upper left")
    fig.text(0.5, -0.015, context, ha="center", fontsize=7.5, color="#444444")
    generated += save_figure(fig, figure_dir, "fig_overall_reliability")

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), sharex=True, sharey=True)
    for index, (class_name, color) in enumerate(zip(class_names, colors)):
        ax = axes.flat[index]
        frame = ovr_mass_bins[ovr_mass_bins["class_name"].eq(class_name)]
        ax.plot(frame["mean_confidence"], frame["accuracy"], marker="o", markersize=3.4, color=color)
        ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=0.8)
        ax.set_title(f"{class_name} vs rest")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes.flat[5].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed frequency")
    for ax in axes[1, :2]:
        ax.set_xlabel("Mean class probability")
    fig.suptitle("One-vs-rest reliability (15 equal-mass bins)", y=1.01)
    fig.text(0.5, -0.01, context, ha="center", fontsize=7.5, color="#444444")
    fig.tight_layout()
    generated += save_figure(fig, figure_dir, "fig_classwise_ovr_reliability")

    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    ax.plot(risk_curve["coverage"], risk_curve["error_risk"], color="#0072B2", linewidth=1.8)
    ax.scatter(operating_points["actual_coverage"], operating_points["error_risk"], color="#D55E00", s=24, zorder=3)
    for row in operating_points.itertuples(index=False):
        ax.annotate(f"{int(round(row.requested_coverage * 100))}%", (row.actual_coverage, row.error_risk),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set(xlim=(0, 1.01), ylim=(0, max(0.02, float(risk_curve["error_risk"].max()) * 1.05)),
           xlabel="Coverage", ylabel="Error risk (1 − selective accuracy)", title="Risk–coverage curve")
    fig.text(0.5, -0.015, context, ha="center", fontsize=7.5, color="#444444")
    generated += save_figure(fig, figure_dir, "fig_risk_coverage")

    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predictions == labels
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    bins = np.linspace(0, 1, 31)
    ax.hist(confidence[correct], bins=bins, density=True, histtype="step", linewidth=1.8,
            color="#009E73", label=f"Correct (n={int(correct.sum()):,})")
    ax.hist(confidence[~correct], bins=bins, density=True, histtype="step", linewidth=1.8,
            color="#D55E00", label=f"Incorrect (n={int((~correct).sum()):,})")
    ax.axvline(HIGH_CONFIDENCE_THRESHOLD, color="#555555", linestyle=":", linewidth=1.0, label="0.90 reference")
    ax.set(xlim=(0, 1), xlabel="Maximum Pre-HMM probability", ylabel="Density", title="Confidence distribution by correctness")
    ax.legend(loc="upper left")
    fig.text(0.5, -0.015, context, ha="center", fontsize=7.5, color="#444444")
    generated += save_figure(fig, figure_dir, "fig_confidence_correct_incorrect")

    true_stage = classwise[classwise["analysis_scope"].eq("true_stage")].set_index("stage").loc[list(class_names)]
    fig, ax = plt.subplots(figsize=(5.1, 3.5))
    x = np.arange(len(class_names))
    width = 0.36
    ax.bar(x - width / 2, true_stage["mean_confidence"], width, color="#56B4E9", label="Mean confidence")
    ax.bar(x + width / 2, 1.0 - true_stage["accuracy"], width, color="#D55E00", label="Error rate")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_title("Confidence and error rate by true sleep stage")
    ax.legend(loc="upper right")
    fig.text(0.5, -0.015, context, ha="center", fontsize=7.5, color="#444444")
    generated += save_figure(fig, figure_dir, "fig_true_stage_confidence_error")

    fig, ax = plt.subplots(figsize=(6.5, 2.65))
    ax.axis("off")
    display = operating_points.copy()
    display["requested"] = display["requested_coverage"].map(lambda value: f"{100 * value:.0f}%")
    display["actual"] = display["actual_coverage"].map(lambda value: f"{100 * value:.2f}%")
    display["accuracy"] = display["selective_accuracy"].map(lambda value: f"{100 * value:.2f}%")
    display["risk"] = display["error_risk"].map(lambda value: f"{100 * value:.2f}%")
    display["threshold"] = display["confidence_threshold"].map(lambda value: f"{value:.4f}")
    table = ax.table(
        cellText=display[["requested", "actual", "retained_n", "subject_count", "accuracy", "risk", "threshold"]].values,
        colLabels=["Requested", "Actual", "Epochs", "Subjects", "Sel. accuracy", "Risk", "Confidence ≥"],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.25)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.6)
        cell.set_facecolor("#EAF2F8" if row == 0 else "white")
    ax.set_title("Validation-descriptive coverage operating points", pad=10)
    fig.text(0.5, 0.01, context, ha="center", fontsize=7.5, color="#444444")
    generated += save_figure(fig, figure_dir, "fig_coverage_operating_points")
    return generated


@dataclass
class AuditInputs:
    train_probabilities: np.ndarray
    train_labels: np.ndarray
    train_recording_id: np.ndarray
    train_epoch_index: np.ndarray
    train_subject_ids: np.ndarray
    val_probabilities: np.ndarray
    val_labels: np.ndarray
    val_recording_id: np.ndarray
    val_epoch_index: np.ndarray
    val_subject_ids: np.ndarray
    class_names: tuple[str, ...]


def load_and_validate_inputs(args: argparse.Namespace, output_dir: Path) -> tuple[AuditInputs, dict[str, Any], dict[str, Any]]:
    required_paths = [args.predictions, args.split, args.checkpoint, args.status, args.reliability, args.attribution_runs]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise AuditBlocked(f"Missing required input paths: {missing}")
    prediction_hash = sha256_file(args.predictions)
    checkpoint_hash = sha256_file(args.checkpoint)
    split_hash = sha256_file(args.split)
    split_payload = json.loads(args.split.read_text(encoding="utf-8"))
    status_payload = json.loads(args.status.read_text(encoding="utf-8"))
    train_files, train_subject_order = canonical_subject_mapping(split_payload, "train")
    val_files, val_subject_order = canonical_subject_mapping(split_payload, "val")
    test_files, test_subject_order = canonical_subject_mapping(split_payload, "test")
    split_overlaps = {
        "train_validation": sorted(set(train_subject_order) & set(val_subject_order)),
        "train_test": sorted(set(train_subject_order) & set(test_subject_order)),
        "validation_test": sorted(set(val_subject_order) & set(test_subject_order)),
    }
    with np.load(args.predictions, allow_pickle=False) as archive:
        required_keys = [
            "class_labels",
            "train_y_true", "train_probs_raw", "train_recording_id", "train_epoch_index", "train_file_sample_counts", "train_pred_argmax",
            "val_y_true", "val_probs_raw", "val_recording_id", "val_epoch_index", "val_file_sample_counts", "val_pred_argmax",
        ]
        absent = [key for key in required_keys if key not in archive.files]
        if absent:
            raise AuditBlocked(f"Prediction archive missing required non-test keys: {absent}")
        class_names = tuple(str(item) for item in archive["class_labels"].tolist())
        train_labels = archive["train_y_true"].astype(np.int64)
        train_probabilities = archive["train_probs_raw"].astype(np.float64)
        train_recording_id = archive["train_recording_id"].astype(np.int64)
        train_epoch_index = archive["train_epoch_index"].astype(np.int64)
        train_counts = archive["train_file_sample_counts"].astype(np.int64)
        train_pred_argmax = archive["train_pred_argmax"].astype(np.int64)
        val_labels = archive["val_y_true"].astype(np.int64)
        val_probabilities = archive["val_probs_raw"].astype(np.float64)
        val_recording_id = archive["val_recording_id"].astype(np.int64)
        val_epoch_index = archive["val_epoch_index"].astype(np.int64)
        val_counts = archive["val_file_sample_counts"].astype(np.int64)
        val_pred_argmax = archive["val_pred_argmax"].astype(np.int64)
        # Deliberately do not read test_* or *_probs_calibrated arrays.
    train_subject_ids = materialize_subject_ids(train_recording_id, train_subject_order)
    val_subject_ids = materialize_subject_ids(val_recording_id, val_subject_order)
    train_probability_check = validate_probability_array(train_probabilities, 61115)
    val_probability_check = validate_probability_array(val_probabilities, 13000)
    train_axis_check = validate_split_axes("train", train_labels, train_recording_id, train_epoch_index, train_counts, train_subject_ids)
    val_axis_check = validate_split_axes("validation", val_labels, val_recording_id, val_epoch_index, val_counts, val_subject_ids)
    train_keys = set(zip(train_subject_ids.tolist(), train_epoch_index.tolist()))
    val_keys = set(zip(val_subject_ids.tolist(), val_epoch_index.tolist()))
    argmax_checks = {
        "train_raw_argmax_matches_train_pred_argmax": bool(np.array_equal(np.argmax(train_probabilities, axis=1), train_pred_argmax)),
        "validation_raw_argmax_matches_val_pred_argmax": bool(np.array_equal(np.argmax(val_probabilities, axis=1), val_pred_argmax)),
        "postprocessed_prediction_arrays_used_for_metrics": False,
    }
    authoritative_checks = {
        key: {"expected": expected, "observed": status_payload.get(key), "pass": status_payload.get(key) == expected}
        for key, expected in AUTHORITATIVE_STATES.items()
    }
    lineage_checks = {
        "prediction_sha256": {"expected": EXPECTED_PREDICTION_SHA256, "observed": prediction_hash, "pass": prediction_hash == EXPECTED_PREDICTION_SHA256},
        "checkpoint_sha256": {"expected": EXPECTED_CHECKPOINT_SHA256, "observed": checkpoint_hash, "pass": checkpoint_hash == EXPECTED_CHECKPOINT_SHA256},
        "split_sha256": {"expected": EXPECTED_SPLIT_SHA256, "observed": split_hash, "pass": split_hash == EXPECTED_SPLIT_SHA256},
        "planning_source_head": PLANNING_SOURCE_HEAD,
        "planning_report_commit": PLANNING_REPORT_COMMIT,
        "different_planning_nodes_are_expected": True,
        "runtime_git_head": git_value(["rev-parse", "HEAD"]),
        "runtime_git_branch": git_value(["branch", "--show-current"]),
    }
    inventory = {
        "schema_version": "sleepxvit_offline_reliability_input_inventory_v1",
        "audit_type": "OFFLINE_RELIABILITY_ONLY",
        "model_inference_executed": False,
        "sequence_context_influence_executed": False,
        "explanation_clustering_executed": False,
        "test_36_accessed": False,
        "probability_calibration_fitted": False,
        "primary_probability_array": "val_probs_raw",
        "probability_semantics": "OVERLAPPING_75_EPOCH_WINDOW_AGGREGATED_PRE_HMM_NORMALIZED_PROBABILITY",
        "softmax_reapplied": False,
        "class_order": list(class_names),
        "assets": {
            "predictions": {"path": str(args.predictions.relative_to(PROJECT_ROOT)), "bytes": args.predictions.stat().st_size, "sha256": prediction_hash},
            "split": {"path": str(args.split.relative_to(PROJECT_ROOT)), "bytes": args.split.stat().st_size, "sha256": split_hash},
            "checkpoint_lineage_only_not_loaded": {"path": str(args.checkpoint.relative_to(PROJECT_ROOT)), "bytes": args.checkpoint.stat().st_size, "sha256": checkpoint_hash},
            "authoritative_status": {"path": str(args.status.relative_to(PROJECT_ROOT)), "sha256": sha256_file(args.status)},
        },
        "splits": {
            "train": {
                "role": "EXPLICIT_TRAIN_SPLIT_NOT_DEVELOPMENT",
                "row_count": len(train_labels),
                "probability_shape": list(train_probabilities.shape),
                "label_shape": list(train_labels.shape),
                "recording_count": len(train_files),
                "subject_count": len(train_subject_order),
                "subject_mapping": [{"recording_id": index, "file": filename, "subject_id": subject} for index, (filename, subject) in enumerate(zip(train_files, train_subject_order))],
            },
            "development": {
                "separate_full_probability_split_available": False,
                "train_relabelled_as_development": False,
                "used_for_threshold_selection": False,
            },
            "validation": {
                "role": "PRIMARY_DESCRIPTIVE_UNCALIBRATED_RELIABILITY_SET",
                "row_count": len(val_labels),
                "probability_shape": list(val_probabilities.shape),
                "label_shape": list(val_labels.shape),
                "recording_count": len(val_files),
                "subject_count": len(val_subject_order),
                "subject_mapping": [{"recording_id": index, "file": filename, "subject_id": subject} for index, (filename, subject) in enumerate(zip(val_files, val_subject_order))],
            },
        },
        "test_split_metadata_only": {"file_count": len(test_files), "subject_count": len(test_subject_order), "prediction_arrays_accessed": False},
    }
    critical_checks = {
        "class_order": {"expected": list(EXPECTED_CLASS_ORDER), "observed": list(class_names), "pass": class_names == EXPECTED_CLASS_ORDER},
        "train_probabilities": train_probability_check,
        "validation_probabilities": val_probability_check,
        "train_axes": train_axis_check,
        "validation_axes": val_axis_check,
        "validation_subject_count_is_15": {"observed": len(np.unique(val_subject_ids)), "pass": len(np.unique(val_subject_ids)) == 15},
        "validation_subject_mapping_matches_split": {"observed": sorted(np.unique(val_subject_ids).tolist()), "expected": sorted(val_subject_order), "pass": sorted(np.unique(val_subject_ids).tolist()) == sorted(val_subject_order)},
        "train_validation_physical_epoch_overlap": {"count": len(train_keys & val_keys), "pass": not (train_keys & val_keys)},
        "split_subject_overlaps": {"overlaps": split_overlaps, "pass": not any(split_overlaps.values())},
        "argmax_alignment": {"details": argmax_checks, "pass": all(value for key, value in argmax_checks.items() if key != "postprocessed_prediction_arrays_used_for_metrics") and not argmax_checks["postprocessed_prediction_arrays_used_for_metrics"]},
        "lineage": {"details": lineage_checks, "pass": all(item["pass"] for item in lineage_checks.values() if isinstance(item, dict) and "pass" in item)},
        "authoritative_states": {"details": authoritative_checks, "pass": all(item["pass"] for item in authoritative_checks.values())},
    }
    all_pass = all(bool(check["pass"]) for check in critical_checks.values())
    validation = {
        "schema_version": "sleepxvit_offline_reliability_input_validation_v1",
        "status": "PASS" if all_pass else "BLOCKED",
        "formal_statistics_allowed": all_pass,
        "critical_checks": critical_checks,
        "noncritical_limitations": [
            "No separate full development probability split is present; the 61,115 rows remain explicitly train.",
            "Validation is used for descriptive uncalibrated reliability, not deployment or clinical threshold selection.",
            "val_probs_calibrated is not used as primary or independent held-out evidence.",
        ],
    }
    write_json(output_dir / "input_inventory.json", inventory)
    write_json(output_dir / "input_validation.json", validation)
    if not all_pass:
        raise AuditBlocked("Critical input validation failed; formal statistics were not computed")
    return AuditInputs(
        train_probabilities=train_probabilities,
        train_labels=train_labels,
        train_recording_id=train_recording_id,
        train_epoch_index=train_epoch_index,
        train_subject_ids=train_subject_ids,
        val_probabilities=val_probabilities,
        val_labels=val_labels,
        val_recording_id=val_recording_id,
        val_epoch_index=val_epoch_index,
        val_subject_ids=val_subject_ids,
        class_names=class_names,
    ), inventory, validation


def summarize_risk_trend(risk_curve: pd.DataFrame, operating_points: pd.DataFrame) -> dict[str, Any]:
    ordered = operating_points.sort_values("actual_coverage")
    risk_values = ordered["error_risk"].to_numpy(dtype=float)
    adjacent_non_decreasing = np.diff(risk_values) >= -1e-12
    rho = spearmanr(ordered["actual_coverage"], ordered["error_risk"])
    return {
        "expected_direction": "risk should generally increase as coverage increases",
        "operating_point_adjacent_pairs": int(len(adjacent_non_decreasing)),
        "non_decreasing_pairs": int(adjacent_non_decreasing.sum()),
        "all_operating_points_monotonic": bool(adjacent_non_decreasing.all()),
        "spearman_coverage_vs_risk": float(rho.statistic),
        "spearman_p_value_descriptive": float(rho.pvalue),
        "full_curve_final_risk": float(risk_curve.iloc[-1]["error_risk"]),
        "full_curve_minimum_risk": float(risk_curve["error_risk"].min()),
    }


def artifact_manifest(output_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            artifacts.append({
                "path": manifest_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return artifacts


def run_audit(args: argparse.Namespace) -> int:
    output_dir = args.output.resolve()
    ensure_new_output_dir(output_dir)
    try:
        inputs, inventory, validation = load_and_validate_inputs(args, output_dir)
        probabilities = inputs.val_probabilities
        labels = inputs.val_labels
        predictions = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
        correct = predictions == labels
        overall = compute_metric_bundle(probabilities, labels)
        equal_mass_bins, _, _ = calibration_bins_equal_mass(confidence, correct, args.bins)
        equal_width_bins, _, _ = calibration_bins_equal_width(confidence, correct, args.bins)
        equal_mass_bins.to_csv(output_dir / "reliability_bins_equal_frequency.csv", index=False)
        equal_width_bins.to_csv(output_dir / "reliability_bins_equal_width.csv", index=False)
        risk_curve, aurc = risk_coverage_curve(confidence, correct)
        risk_curve.to_csv(output_dir / "risk_coverage.csv", index=False)
        operating_points = coverage_operating_points(confidence, correct, inputs.val_subject_ids)
        operating_points.to_csv(output_dir / "coverage_operating_points.csv", index=False)
        classwise, ovr_mass, ovr_width = classwise_tables(probabilities, labels, inputs.class_names)
        classwise.to_csv(output_dir / "classwise_metrics.csv", index=False)
        ovr_mass.to_csv(output_dir / "classwise_reliability_bins_equal_frequency.csv", index=False)
        ovr_width.to_csv(output_dir / "classwise_reliability_bins_equal_width.csv", index=False)
        distribution = confidence_distribution_summary(confidence, correct)
        distribution.to_csv(output_dir / "confidence_correct_incorrect.csv", index=False)
        subjects = subject_level_metrics(probabilities, labels, inputs.val_subject_ids)
        subjects.to_csv(output_dir / "subject_level_metrics.csv", index=False)
        selected30 = selected30_analysis(
            args.attribution_runs,
            args.reliability,
            output_dir,
            probabilities,
            labels,
            inputs.class_names,
        )
        bootstrap = subject_bootstrap(
            probabilities, labels, inputs.val_subject_ids, args.bootstrap_iterations, args.bootstrap_seed
        )
        write_json(output_dir / "bootstrap_confidence_intervals.json", bootstrap)
        risk_trend = summarize_risk_trend(risk_curve, operating_points)
        n1_row = classwise[
            classwise["analysis_scope"].eq("true_stage") & classwise["stage"].eq("N1")
        ].iloc[0].to_dict()
        overall_payload = {
            "schema_version": "sleepxvit_offline_reliability_overall_metrics_v1",
            "dataset": "ISRUC",
            "split": "validation",
            "unique_epoch_count": len(labels),
            "subject_count": int(len(np.unique(inputs.val_subject_ids))),
            "probability_stage": "PRE_HMM",
            "calibration_status": "UNCALIBRATED_RAW_PROBABILITY",
            "metrics": overall,
            "definitions": {
                "primary_ece": "15 equal-mass bins; weighted absolute confidence-accuracy gap",
                "sensitivity_ece": "15 equal-width bins on [0,1]; weighted absolute confidence-accuracy gap",
                "multiclass_brier_sum": "mean over epochs of sum over five classes of squared probability error",
                "aurc_discrete_mean_risk": "(1/n) * sum_{k=1}^n [1 - accuracy(top-k confidence epochs)]",
                "confidence": "maximum raw Pre-HMM class probability",
                "risk": "1 - selective accuracy",
            },
            "risk_trend": risk_trend,
            "n1_true_stage": n1_row,
            "selected30_summary": selected30["summary"],
        }
        write_json(output_dir / "overall_metrics.json", overall_payload)
        figure_paths = generate_figures(
            output_dir,
            inputs.class_names,
            probabilities,
            labels,
            inputs.val_subject_ids,
            equal_mass_bins,
            ovr_mass,
            risk_curve,
            operating_points,
            classwise,
            inventory["assets"]["checkpoint_lineage_only_not_loaded"]["sha256"],
        )
        readme = "# SleepXViT-inspired Offline Reliability Audit\n\n"
        readme += "This directory contains a validation-descriptive, inference-free audit of frozen raw Pre-HMM probabilities.\n\n"
        readme += f"- Validation: {len(labels):,} unique physical epochs, {len(np.unique(inputs.val_subject_ids))} subjects.\n"
        readme += "- Primary calibration: 15 equal-mass bins; sensitivity: 15 equal-width bins.\n"
        readme += f"- Subject bootstrap: {bootstrap['effective_iterations']}/{bootstrap['requested_iterations']} effective iterations, seed {bootstrap['seed']}.\n"
        readme += "- selected-30: exploratory B0_CLASSIC_90/T0 repeated-measure audit only; n=30, no clustering, no extrapolation.\n"
        readme += "- No model inference, training, calibration fitting, context audit, fixed test-36 access, or attribution reweighting occurred.\n\n"
        readme += "Authoritative states remain `MODEL_FAITHFULNESS_NOT_ESTABLISHED`, `PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED`, and `PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD`.\n"
        (output_dir / "README.md").write_text(readme, encoding="utf-8")
        manifest = {
            "schema_version": "sleepxvit_offline_reliability_run_manifest_v1",
            "status": "COMPLETE",
            "audit_type": "OFFLINE_RELIABILITY_ONLY",
            "dataset": "ISRUC",
            "primary_split": "validation",
            "unique_epoch_count": len(labels),
            "subject_count": int(len(np.unique(inputs.val_subject_ids))),
            "class_order": list(inputs.class_names),
            "probability_array": "val_probs_raw",
            "probability_stage": "PRE_HMM",
            "probability_calibration": "UNCALIBRATED",
            "softmax_reapplied": False,
            "model_inference_executed": False,
            "training_or_finetuning_executed": False,
            "probability_calibration_fitted": False,
            "sequence_context_influence_executed": False,
            "explanation_clustering_executed": False,
            "fixed_test_36_accessed": False,
            "canonical_attribution_modified": False,
            "confidence_reweighted_lime": False,
            "historical_candidate_used": False,
            "planning_source_head": PLANNING_SOURCE_HEAD,
            "planning_report_commit": PLANNING_REPORT_COMMIT,
            "runtime_git_head": git_value(["rev-parse", "HEAD"]),
            "runtime_git_branch": git_value(["branch", "--show-current"]),
            "audit_tool": {
                "path": "tools/sleepxvit_offline_reliability_audit.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "runtime_versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "bootstrap": {"seed": args.bootstrap_seed, "requested_iterations": args.bootstrap_iterations,
                          "effective_iterations": bootstrap["effective_iterations"], "failed_iterations": bootstrap["failed_iterations"]},
            "figure_files": [manifest_path(Path(path)) for path in figure_paths],
            "source_hashes": {
                "checkpoint": inventory["assets"]["checkpoint_lineage_only_not_loaded"]["sha256"],
                "predictions": inventory["assets"]["predictions"]["sha256"],
                "split": inventory["assets"]["split"]["sha256"],
            },
            "authoritative_states_preserved": AUTHORITATIVE_STATES,
            "artifacts": artifact_manifest(output_dir),
        }
        write_json(output_dir / "run_manifest.json", manifest)
        print(json.dumps({
            "status": "COMPLETE",
            "output_dir": str(output_dir),
            "metrics": overall,
            "bootstrap": bootstrap["metrics"],
            "risk_trend": risk_trend,
            "selected30": selected30["summary"],
        }, indent=2, sort_keys=True))
        return 0
    except AuditBlocked as exc:
        blocking = {
            "status": "BLOCKED",
            "reason": str(exc),
            "formal_statistics_computed": False,
            "model_inference_executed": False,
            "fixed_test_36_accessed": False,
        }
        write_json(output_dir / "BLOCKED.json", blocking)
        (output_dir / "README.md").write_text(
            "# BLOCKED\n\nFormal reliability statistics were not computed.\n\n" + str(exc) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(blocking, indent=2), file=sys.stderr)
        return 3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--reliability", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--attribution-runs", type=Path, default=DEFAULT_ATTRIBUTION_RUNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    args = parser.parse_args(argv)
    for attribute in ("predictions", "split", "checkpoint", "status", "reliability", "attribution_runs", "output"):
        value = getattr(args, attribute)
        if not value.is_absolute():
            setattr(args, attribute, (PROJECT_ROOT / value).resolve())
    if args.bins < 2:
        parser.error("--bins must be at least 2")
    if args.bootstrap_iterations < 1:
        parser.error("--bootstrap-iterations must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run_audit(parse_args()))
