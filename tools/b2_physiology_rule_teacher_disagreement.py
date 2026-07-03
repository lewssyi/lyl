#!/usr/bin/env python
"""B2 physiology rule-teacher disagreement analysis.

This tool is intentionally post-hoc: it does not change model predictions or
training. It turns clinical concept proxies into an AISleep-inspired soft rule
teacher, then measures where the model and rule teacher disagree.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CLASS_LABELS = ("W", "N1", "N2", "N3", "REM")
LABEL_KEYS = ("y_true", "labels", "targets", "target", "y")
PROB_KEYS = (
    "probs_calibrated",
    "probs_raw",
    "probs",
    "probabilities",
    "prediction_probs",
    "pred_probs",
    "test_probs_calibrated",
    "test_probs_raw",
)
PRED_KEYS = (
    "y_pred_final",
    "pred_final",
    "final_pred",
    "prediction",
    "pred",
    "predicted_labels",
    "y_pred",
    "test_pred",
)
COUNT_KEYS = ("file_sample_counts", "recording_sample_counts", "sample_counts")
RECORDING_ID_KEYS = ("recording_id", "recording_ids", "record_ids", "record_id")
EPOCH_INDEX_KEYS = ("epoch_index", "epoch_indices", "epoch_idx")
CONCEPT_KEYS = (
    "clinical_concept_proxy",
    "clinical_concept_proxies",
    "clinical_concept_evidence",
    "clinical_concepts",
    "concept_proxy",
    "concept_proxies",
    "concept_evidence",
    "concept_targets",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an AISleep-inspired physiology rule teacher from clinical "
            "concepts and report model-rule disagreement."
        )
    )
    parser.add_argument("--prediction-npz", required=True, help="Prediction NPZ produced by run_main or export tools.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--probs-key", default=None)
    parser.add_argument("--pred-key", default=None)
    parser.add_argument("--concept-key", default=None)
    parser.add_argument("--concept-cache", default=None, help="Optional NPZ cache containing a concepts array.")
    parser.add_argument("--split-json", default=None, help="Protocol split.json used to recompute concepts if absent.")
    parser.add_argument("--npz-dir", default=None, help="Override NPZ directory from split.json when recomputing concepts.")
    parser.add_argument("--concept-extractor", choices=("direct_epoch", "segment"), default="direct_epoch")
    parser.add_argument("--concept-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--filter-profile", default="clinical_canonical")
    parser.add_argument("--normalization-mode", default="recording_robust_zscore")
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--no-adaptive-concepts", action="store_true")
    parser.add_argument("--allow-probability-fallback", action="store_true")
    parser.add_argument("--calibrate-prototypes", action="store_true")
    parser.add_argument("--prototype-split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--prototype-cache", default=None, help="Optional NPZ cache for calibrated stage prototypes.")
    parser.add_argument("--prototype-weight", type=float, default=0.70)
    parser.add_argument("--rule-temperature", type=float, default=0.75)
    parser.add_argument("--n1-window", type=int, default=20)
    parser.add_argument("--transition-radius", type=int, default=2)
    parser.add_argument("--model-conf-threshold", type=float, default=0.70)
    parser.add_argument("--rule-conf-threshold", type=float, default=0.55)
    parser.add_argument("--top-examples", type=int, default=300)
    parser.add_argument("--save-npz", action="store_true", default=True)
    parser.add_argument("--no-save-npz", action="store_false", dest="save_npz")
    return parser.parse_args()


def split_candidates(split: str, base_keys: Sequence[str], include_unscoped: bool) -> list[str]:
    prefixes = [split]
    if split == "val":
        prefixes.append("validation")
    candidates: list[str] = []
    for prefix in prefixes:
        for key in base_keys:
            candidates.append(key if key.startswith(f"{prefix}_") else f"{prefix}_{key}")
    if split == "test":
        candidates.extend(key if key.startswith("test_") else f"test_{key}" for key in base_keys)
    if include_unscoped:
        candidates.extend(base_keys)
    return list(dict.fromkeys(candidates))


def find_key(
    data: np.lib.npyio.NpzFile,
    split: str,
    base_keys: Sequence[str],
    explicit: str | None = None,
    include_unscoped: bool = True,
) -> str | None:
    if explicit:
        explicit_candidates = [explicit]
        if not explicit.startswith(f"{split}_"):
            explicit_candidates.insert(0, f"{split}_{explicit}")
        for key in explicit_candidates:
            if key in data.files:
                return key
        raise KeyError(f"Explicit key {explicit!r} was not found in {data.files}.")
    for key in split_candidates(split, base_keys, include_unscoped):
        if key in data.files:
            return key
    return None


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    row_sums = np.sum(probs, axis=-1, keepdims=True)
    return probs / np.maximum(row_sums, 1e-12)


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    temperature = max(float(temperature), 1e-6)
    scaled = scores / temperature
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_scores = np.exp(scaled)
    return exp_scores / np.maximum(np.sum(exp_scores, axis=-1, keepdims=True), 1e-12)


def row_standardize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    center = np.mean(scores, axis=1, keepdims=True)
    scale = np.std(scores, axis=1, keepdims=True)
    return (scores - center) / np.maximum(scale, 1e-6)


def labels_to_indices(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return np.argmax(labels, axis=-1).astype(np.int32)
    return labels.reshape(-1).astype(np.int32)


def recording_axes_from_counts(split: str, counts: np.ndarray | None, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    if counts is None:
        return np.asarray([f"{split}_recording_000"] * n_rows, dtype=object), np.arange(n_rows, dtype=np.int32)
    values = [int(item) for item in np.asarray(counts).reshape(-1) if int(item) > 0]
    if sum(values) != int(n_rows):
        raise ValueError(f"file_sample_counts sum={sum(values)} does not match rows={n_rows}.")
    recording_ids: list[str] = []
    epoch_indices: list[int] = []
    for rec_idx, count in enumerate(values):
        recording_ids.extend([f"{split}_recording_{rec_idx:03d}"] * count)
        epoch_indices.extend(range(count))
    return np.asarray(recording_ids, dtype=object), np.asarray(epoch_indices, dtype=np.int32)


def read_prediction_file(
    path: Path,
    split: str,
    probs_key: str | None = None,
    pred_key: str | None = None,
) -> dict[str, np.ndarray | str | tuple[str, ...]]:
    with np.load(path, allow_pickle=True) as data:
        label_key = find_key(data, split, LABEL_KEYS, include_unscoped=True)
        if label_key is None:
            raise KeyError(f"No label key found in {path}.")
        score_key = find_key(data, split, PROB_KEYS, explicit=probs_key, include_unscoped=True)
        if score_key is None:
            raise KeyError(f"No probability key found in {path}.")
        pred_key_found = find_key(data, split, PRED_KEYS, explicit=pred_key, include_unscoped=True)
        count_key = find_key(data, split, COUNT_KEYS, include_unscoped=True)
        rec_key = find_key(data, split, RECORDING_ID_KEYS, include_unscoped=True)
        epoch_key = find_key(data, split, EPOCH_INDEX_KEYS, include_unscoped=True)

        y_true = labels_to_indices(np.asarray(data[label_key]))
        probs = normalize_probabilities(np.asarray(data[score_key]))
        if probs.ndim != 2 or probs.shape[0] != y_true.shape[0]:
            raise ValueError(f"{score_key} must have shape (n, classes), got {probs.shape}.")
        if pred_key_found is not None:
            model_pred = labels_to_indices(np.asarray(data[pred_key_found]))
        else:
            model_pred = np.argmax(probs, axis=1).astype(np.int32)
        if model_pred.shape[0] != y_true.shape[0]:
            raise ValueError(f"{pred_key_found} length does not match labels.")

        counts = np.asarray(data[count_key], dtype=np.int32) if count_key is not None else None
        if rec_key is not None and epoch_key is not None:
            recording_id = np.asarray(data[rec_key]).reshape(-1).astype(object)
            epoch_index = np.asarray(data[epoch_key]).reshape(-1).astype(np.int32)
            if recording_id.shape[0] != y_true.shape[0] or epoch_index.shape[0] != y_true.shape[0]:
                raise ValueError("recording_id and epoch_index must match prediction rows.")
        else:
            recording_id, epoch_index = recording_axes_from_counts(split, counts, len(y_true))

        if "class_labels" in data.files:
            class_labels = tuple(str(item) for item in np.asarray(data["class_labels"]).reshape(-1))
        else:
            class_labels = DEFAULT_CLASS_LABELS

        concept_key = find_key(data, split, CONCEPT_KEYS, include_unscoped=True)
        concepts = np.asarray(data[concept_key], dtype=np.float32) if concept_key is not None else None

    return {
        "y_true": y_true,
        "probs": probs,
        "model_pred": model_pred,
        "recording_id": recording_id,
        "epoch_index": epoch_index,
        "counts": counts,
        "class_labels": class_labels,
        "label_key": label_key,
        "score_key": score_key,
        "pred_key": pred_key_found or "",
        "concepts": concepts,
        "concept_key": concept_key or "",
    }


def moving_average_same(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    window = max(1, int(window))
    if window == 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def transition_context_from_stage(stage: np.ndarray, recording_id: np.ndarray, radius: int) -> np.ndarray:
    stage = np.asarray(stage).reshape(-1)
    recording_id = np.asarray(recording_id).reshape(-1)
    radius = max(0, int(radius))
    context = np.zeros(stage.shape[0], dtype=np.float32)
    for rec in dict.fromkeys(recording_id.tolist()):
        rows = np.flatnonzero(recording_id == rec)
        if rows.size <= 1:
            continue
        local = stage[rows]
        boundaries = np.flatnonzero(local[1:] != local[:-1]) + 1
        for boundary in boundaries:
            lo = max(0, int(boundary) - radius)
            hi = min(rows.size, int(boundary) + radius + 1)
            if hi > lo:
                context[rows[lo:hi]] = 1.0
    return context


def adaptive_percentile_scale(
    concepts: np.ndarray,
    recording_id: np.ndarray,
    lower: float = 10.0,
    upper: float = 90.0,
) -> np.ndarray:
    concepts = np.asarray(concepts, dtype=np.float32)
    scaled = np.zeros_like(concepts, dtype=np.float32)
    recording_id = np.asarray(recording_id).reshape(-1)
    for rec in dict.fromkeys(recording_id.tolist()):
        rows = np.flatnonzero(recording_id == rec)
        block = concepts[rows]
        lo = np.percentile(block, lower, axis=0, keepdims=True)
        hi = np.percentile(block, upper, axis=0, keepdims=True)
        denom = np.maximum(hi - lo, 1e-6)
        scaled[rows] = np.clip((block - lo) / denom, 0.0, 1.0)
    return scaled


def compute_rule_teacher(
    concepts: np.ndarray,
    recording_id: np.ndarray,
    adaptive_concepts: bool = True,
    rule_temperature: float = 0.75,
    n1_window: int = 20,
    transition_radius: int = 2,
    stage_prototypes: np.ndarray | None = None,
    prototype_weight: float = 0.0,
) -> dict[str, np.ndarray]:
    concepts = np.asarray(concepts, dtype=np.float32)
    if concepts.ndim != 2 or concepts.shape[1] < 9:
        raise ValueError(f"concepts must have shape (n, >=9), got {concepts.shape}.")
    concepts = concepts[:, :9]
    c = adaptive_percentile_scale(concepts, recording_id) if adaptive_concepts else np.clip(concepts, 0.0, 1.0)

    delta = c[:, 0]
    theta = c[:, 1]
    alpha = c[:, 2]
    sigma = c[:, 3]
    beta = c[:, 4]
    eog_slow = c[:, 5]
    eog_rapid = c[:, 6]
    emg_atonia = c[:, 7]
    emg_active = c[:, 8]
    high_freq_proxy = np.maximum(beta, emg_active)

    score_w = 1.10 * alpha + 0.75 * beta + 0.90 * emg_active + 0.20 * high_freq_proxy
    score_w -= 0.70 * delta + 0.25 * sigma + 0.35 * emg_atonia

    score_n2 = 1.25 * sigma + 0.35 * theta + 0.20 * eog_slow
    score_n2 -= 0.45 * alpha + 0.35 * emg_active + 0.25 * eog_rapid

    score_n3 = 1.35 * delta + 0.20 * sigma
    score_n3 -= 0.60 * alpha + 0.55 * emg_active + 0.25 * eog_rapid

    score_rem = 1.10 * eog_rapid + 1.05 * emg_atonia + 0.25 * theta
    score_rem -= 0.45 * delta + 0.25 * sigma + 0.35 * alpha + 0.65 * emg_active

    base_scores = np.stack([score_w, score_n2, score_n3, score_rem], axis=1)
    base_probs = softmax(base_scores, temperature=rule_temperature)
    base_idx = np.argmax(base_probs, axis=1)
    base_stage_map = np.asarray([0, 2, 3, 4], dtype=np.int32)
    base_stage = base_stage_map[base_idx]
    base_conf = np.max(base_probs, axis=1)
    residual = 1.0 - base_conf
    transition_context = transition_context_from_stage(base_stage, recording_id, radius=transition_radius)

    n1_hint = np.zeros(concepts.shape[0], dtype=np.float32)
    for rec in dict.fromkeys(np.asarray(recording_id).reshape(-1).tolist()):
        rows = np.flatnonzero(recording_id == rec)
        if rows.size == 0:
            continue
        smoothed_stage = moving_average_same(base_stage[rows].astype(np.float32), n1_window)
        rem_residual = (base_stage[rows] == 4) & (smoothed_stage < 2.5)
        ambiguous_transition = (base_conf[rows] < 0.55) & (transition_context[rows] > 0.0)
        n1_hint[rows] = np.maximum(rem_residual.astype(np.float32), 0.50 * ambiguous_transition.astype(np.float32))

    score_n1 = 0.45 * theta + 0.35 * eog_slow + 0.55 * residual + 0.40 * transition_context + 0.85 * n1_hint
    score_n1 -= 0.30 * delta + 0.25 * sigma + 0.25 * emg_active + 0.15 * eog_rapid

    clinical_rule_scores = np.stack([score_w, score_n1, score_n2, score_n3, score_rem], axis=1)
    rule_scores = clinical_rule_scores
    prototype_scores = np.zeros_like(rule_scores, dtype=np.float32)
    if stage_prototypes is not None and prototype_weight > 0.0:
        prototypes = np.asarray(stage_prototypes, dtype=np.float32)
        if prototypes.shape != (5, c.shape[1]):
            raise ValueError(f"stage_prototypes must have shape {(5, c.shape[1])}, got {prototypes.shape}.")
        distances = np.mean((c[:, np.newaxis, :] - prototypes[np.newaxis, :, :]) ** 2, axis=2)
        prototype_scores = -distances.astype(np.float32)
        weight = float(np.clip(prototype_weight, 0.0, 1.0))
        rule_scores = (1.0 - weight) * row_standardize(clinical_rule_scores) + weight * row_standardize(prototype_scores)

    p_rule = softmax(rule_scores, temperature=rule_temperature).astype(np.float32)
    rule_pred = np.argmax(p_rule, axis=1).astype(np.int32)
    rule_conf = np.max(p_rule, axis=1).astype(np.float32)
    return {
        "p_rule": p_rule,
        "rule_scores": rule_scores.astype(np.float32),
        "clinical_rule_scores": clinical_rule_scores.astype(np.float32),
        "prototype_scores": prototype_scores.astype(np.float32),
        "rule_pred": rule_pred,
        "rule_conf": rule_conf,
        "concepts_scaled": c.astype(np.float32),
        "base_stage": base_stage.astype(np.int32),
        "base_conf": base_conf.astype(np.float32),
        "residual": residual.astype(np.float32),
        "transition_context": transition_context.astype(np.float32),
        "n1_residual_hint": n1_hint.astype(np.float32),
    }


def compute_probability_fallback_rule(
    probs: np.ndarray,
    recording_id: np.ndarray,
    rule_temperature: float,
    n1_window: int,
    transition_radius: int,
) -> dict[str, np.ndarray]:
    probs = normalize_probabilities(probs)
    base = probs[:, [0, 2, 3, 4]]
    base_idx = np.argmax(base, axis=1)
    base_stage = np.asarray([0, 2, 3, 4], dtype=np.int32)[base_idx]
    base_conf = np.max(base, axis=1)
    residual = 1.0 - base_conf
    transition_context = transition_context_from_stage(base_stage, recording_id, radius=transition_radius)
    n1_hint = np.zeros(probs.shape[0], dtype=np.float32)
    for rec in dict.fromkeys(np.asarray(recording_id).reshape(-1).tolist()):
        rows = np.flatnonzero(recording_id == rec)
        smoothed_stage = moving_average_same(base_stage[rows].astype(np.float32), n1_window)
        n1_hint[rows] = ((base_stage[rows] == 4) & (smoothed_stage < 2.5)).astype(np.float32)
    scores = np.log(np.clip(probs, 1e-8, 1.0))
    scores[:, 1] = 0.45 * scores[:, 1] + 0.35 * residual + 0.30 * transition_context + 0.50 * n1_hint
    p_rule = softmax(scores, temperature=rule_temperature).astype(np.float32)
    return {
        "p_rule": p_rule,
        "rule_scores": scores.astype(np.float32),
        "rule_pred": np.argmax(p_rule, axis=1).astype(np.int32),
        "rule_conf": np.max(p_rule, axis=1).astype(np.float32),
        "concepts_scaled": np.zeros((probs.shape[0], 9), dtype=np.float32),
        "base_stage": base_stage.astype(np.int32),
        "base_conf": base_conf.astype(np.float32),
        "residual": residual.astype(np.float32),
        "transition_context": transition_context.astype(np.float32),
        "n1_residual_hint": n1_hint.astype(np.float32),
    }


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, pred in zip(y_true.astype(np.int32), y_pred.astype(np.int32)):
        if 0 <= truth < n_classes and 0 <= pred < n_classes:
            cm[truth, pred] += 1
    return cm


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, class_labels: Sequence[str]) -> dict[str, object]:
    n_classes = len(class_labels)
    cm = confusion_matrix_np(y_true, y_pred, n_classes)
    support = np.sum(cm, axis=1)
    pred_support = np.sum(cm, axis=0)
    tp = np.diag(cm).astype(np.float64)
    precision = tp / np.maximum(pred_support, 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    accuracy = float(np.sum(tp) / max(len(y_true), 1))
    expected = float(np.sum(support * pred_support) / max(len(y_true) ** 2, 1))
    kappa = float((accuracy - expected) / max(1.0 - expected, 1e-12))
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1)) if f1.size else 0.0,
        "kappa": kappa,
        "per_class": {
            str(class_labels[idx]): {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx in range(n_classes)
        },
    }


def safe_rate(mask: np.ndarray, denominator_mask: np.ndarray | None = None) -> float:
    mask = np.asarray(mask, dtype=bool)
    if denominator_mask is None:
        denom = mask.size
        return float(np.sum(mask) / max(denom, 1))
    denominator_mask = np.asarray(denominator_mask, dtype=bool)
    denom = int(np.sum(denominator_mask))
    if denom == 0:
        return 0.0
    return float(np.sum(mask & denominator_mask) / denom)


def probability_margin(probs: np.ndarray) -> np.ndarray:
    probs = normalize_probabilities(np.asarray(probs, dtype=np.float64))
    if probs.shape[1] <= 1:
        return np.ones(probs.shape[0], dtype=np.float32)
    sorted_probs = np.sort(probs, axis=1)
    return (sorted_probs[:, -1] - sorted_probs[:, -2]).astype(np.float32)


def percentile_unit(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    lower, upper = np.percentile(values, [float(low), float(high)])
    if upper <= lower + 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    scaled = (values - lower) / (upper - lower)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def probability_js_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_probabilities(left)
    right = normalize_probabilities(right)
    midpoint = 0.5 * (left + right)
    eps = 1e-12
    left_kl = np.sum(left * np.log((left + eps) / (midpoint + eps)), axis=1)
    right_kl = np.sum(right * np.log((right + eps) / (midpoint + eps)), axis=1)
    return ((0.5 * (left_kl + right_kl)) / np.log(2.0)).astype(np.float32)


def compute_disagreement_risk(
    probs: np.ndarray,
    p_rule: np.ndarray,
    model_pred: np.ndarray,
    rule_pred: np.ndarray,
    rule_conf: np.ndarray,
) -> dict[str, np.ndarray]:
    probs = normalize_probabilities(probs)
    p_rule = normalize_probabilities(p_rule)
    model_conf = np.max(probs, axis=1).astype(np.float32)
    rule_conf = np.asarray(rule_conf, dtype=np.float32).reshape(-1)
    model_margin = probability_margin(probs)
    rule_margin = probability_margin(p_rule)
    prob_l1 = (0.5 * np.sum(np.abs(probs - p_rule), axis=1)).astype(np.float32)
    prob_js = probability_js_distance(probs, p_rule)
    disagreement = (np.asarray(model_pred).reshape(-1) != np.asarray(rule_pred).reshape(-1)).astype(np.float32)

    model_uncertainty = 1.0 - percentile_unit(model_margin)
    rule_margin_unit = percentile_unit(rule_margin)
    prob_js_unit = percentile_unit(prob_js)
    risk_score = disagreement * (
        0.90 * model_uncertainty
        + 0.05 * rule_conf
        + 0.025 * rule_margin_unit
        + 0.025 * prob_js_unit
    )
    return {
        "risk_score": np.asarray(risk_score, dtype=np.float32),
        "model_conf": model_conf,
        "model_margin": model_margin.astype(np.float32),
        "model_uncertainty": model_uncertainty.astype(np.float32),
        "rule_margin": rule_margin.astype(np.float32),
        "prob_l1": prob_l1,
        "prob_js": prob_js,
    }


def risk_quantile_rows(
    y_true: np.ndarray,
    model_pred: np.ndarray,
    rule_pred: np.ndarray,
    disagreement: np.ndarray,
    risk: Mapping[str, np.ndarray],
    fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
) -> list[dict[str, object]]:
    risk_score = np.asarray(risk["risk_score"], dtype=np.float32).reshape(-1)
    model_margin = np.asarray(risk["model_margin"], dtype=np.float32).reshape(-1)
    rule_margin = np.asarray(risk["rule_margin"], dtype=np.float32).reshape(-1)
    model_error = np.asarray(model_pred).reshape(-1) != np.asarray(y_true).reshape(-1)
    rule_error = np.asarray(rule_pred).reshape(-1) != np.asarray(y_true).reshape(-1)
    disagreement = np.asarray(disagreement, dtype=bool).reshape(-1)
    order = np.argsort(-risk_score)
    rows: list[dict[str, object]] = []
    for fraction in fractions:
        n_top = max(1, int(round(len(risk_score) * float(fraction))))
        top_idx = order[:n_top]
        top_mask = np.zeros_like(disagreement, dtype=bool)
        top_mask[top_idx] = True
        retained_mask = ~top_mask
        rows.append(
            {
                "fraction": float(fraction),
                "n": int(n_top),
                "coverage_after_review": float(np.sum(retained_mask) / max(len(retained_mask), 1)),
                "min_risk_score": float(risk_score[top_idx[-1]]) if n_top else 0.0,
                "mean_risk_score": float(np.mean(risk_score[top_idx])) if n_top else 0.0,
                "model_error_rate": safe_rate(model_error, top_mask),
                "retained_model_error_rate": safe_rate(model_error, retained_mask),
                "retained_model_accuracy": 1.0 - safe_rate(model_error, retained_mask),
                "rule_error_rate": safe_rate(rule_error, top_mask),
                "disagreement_rate": safe_rate(disagreement, top_mask),
                "mean_model_margin": float(np.mean(model_margin[top_idx])) if n_top else 0.0,
                "mean_rule_margin": float(np.mean(rule_margin[top_idx])) if n_top else 0.0,
            }
        )
    return rows


def per_class_rows(
    y_true: np.ndarray,
    model_pred: np.ndarray,
    rule_pred: np.ndarray,
    disagreement: np.ndarray,
    high_conflict: np.ndarray,
    class_labels: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, label in enumerate(class_labels):
        mask = y_true == idx
        rows.append(
            {
                "class": str(label),
                "n": int(np.sum(mask)),
                "model_error_rate": safe_rate(model_pred != y_true, mask),
                "rule_error_rate": safe_rate(rule_pred != y_true, mask),
                "disagreement_rate": safe_rate(disagreement, mask),
                "high_conflict_rate": safe_rate(high_conflict, mask),
                "high_conflict_error_rate": safe_rate((model_pred != y_true) & high_conflict, high_conflict & mask),
            }
        )
    return rows


def focus_pair_rows(
    y_true: np.ndarray,
    model_pred: np.ndarray,
    rule_pred: np.ndarray,
    disagreement: np.ndarray,
    high_conflict: np.ndarray,
    class_labels: Sequence[str],
) -> list[dict[str, object]]:
    pairs = ((0, 1), (1, 2), (4, 1), (2, 3), (4, 2))
    rows: list[dict[str, object]] = []
    for left, right in pairs:
        if left >= len(class_labels) or right >= len(class_labels):
            continue
        true_pair = (y_true == left) | (y_true == right)
        model_pair_flip = ((y_true == left) & (model_pred == right)) | ((y_true == right) & (model_pred == left))
        rule_pair_flip = ((y_true == left) & (rule_pred == right)) | ((y_true == right) & (rule_pred == left))
        rows.append(
            {
                "pair": f"{class_labels[left]}<->{class_labels[right]}",
                "n_true_pair": int(np.sum(true_pair)),
                "model_pair_flip_rate": safe_rate(model_pair_flip, true_pair),
                "rule_pair_flip_rate": safe_rate(rule_pair_flip, true_pair),
                "disagreement_rate": safe_rate(disagreement, true_pair),
                "high_conflict_rate": safe_rate(high_conflict, true_pair),
                "model_error_rate": safe_rate(model_pred != y_true, true_pair),
                "high_conflict_error_rate": safe_rate((model_pred != y_true) & high_conflict, high_conflict & true_pair),
            }
        )
    return rows


def summarize_disagreement(
    y_true: np.ndarray,
    probs: np.ndarray,
    model_pred: np.ndarray,
    rule_pred: np.ndarray,
    rule_conf: np.ndarray,
    class_labels: Sequence[str],
    model_conf_threshold: float,
    rule_conf_threshold: float,
    risk: Mapping[str, np.ndarray] | None = None,
) -> dict[str, object]:
    model_conf = np.max(probs, axis=1)
    disagreement = model_pred != rule_pred
    high_conflict = disagreement & (model_conf >= model_conf_threshold) & (rule_conf >= rule_conf_threshold)
    model_error = model_pred != y_true
    rule_error = rule_pred != y_true
    summary = {
        "n": int(len(y_true)),
        "model_metrics": metric_bundle(y_true, model_pred, class_labels),
        "rule_metrics": metric_bundle(y_true, rule_pred, class_labels),
        "disagreement_rate": safe_rate(disagreement),
        "high_conflict_rate": safe_rate(high_conflict),
        "model_error_rate": safe_rate(model_error),
        "rule_error_rate": safe_rate(rule_error),
        "model_error_rate_when_agree": safe_rate(model_error, ~disagreement),
        "model_error_rate_when_disagree": safe_rate(model_error, disagreement),
        "model_error_rate_when_high_conflict": safe_rate(model_error, high_conflict),
        "rule_error_rate_when_high_conflict": safe_rate(rule_error, high_conflict),
        "n_disagreement": int(np.sum(disagreement)),
        "n_high_conflict": int(np.sum(high_conflict)),
        "model_conf_threshold": float(model_conf_threshold),
        "rule_conf_threshold": float(rule_conf_threshold),
        "per_class": per_class_rows(y_true, model_pred, rule_pred, disagreement, high_conflict, class_labels),
        "focus_pairs": focus_pair_rows(y_true, model_pred, rule_pred, disagreement, high_conflict, class_labels),
    }
    if risk is not None:
        quantiles = risk_quantile_rows(y_true, model_pred, rule_pred, disagreement, risk)
        summary["risk_quantiles"] = quantiles
        for row in quantiles:
            pct = int(round(float(row["fraction"]) * 100))
            summary[f"model_error_rate_when_top_risk_{pct}pct"] = float(row["model_error_rate"])
            summary[f"retained_accuracy_after_review_top_risk_{pct}pct"] = float(row["retained_model_accuracy"])
            summary[f"disagreement_rate_when_top_risk_{pct}pct"] = float(row["disagreement_rate"])
    return summary


def load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def split_files_from_json(split_data: Mapping[str, object], split: str) -> list[str]:
    raw = split_data.get("split_files")
    if not isinstance(raw, dict):
        raise ValueError("split.json must contain split_files.")
    values = raw.get(split) or raw.get(split.capitalize())
    if not isinstance(values, list):
        raise ValueError(f"split_files.{split} must be a list.")
    return [str(item) for item in values]


def resolve_npz_dir(split_data: Mapping[str, object], override: str | None) -> Path:
    raw = override or split_data.get("source_npz_dir") or split_data.get("npz_dir")
    if not raw:
        raise ValueError("NPZ directory not provided. Pass --npz-dir or include it in split.json.")
    path = Path(str(raw)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"NPZ directory does not exist: {path}")
    return path


def sanitize_signal_epochs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        values = values[..., np.newaxis]
    if values.ndim != 3:
        raise ValueError(f"Expected signal epochs with shape (epochs, samples, channels), got {values.shape}.")
    return np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def recording_robust_zscore(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = sanitize_signal_epochs(values)
    center = np.median(values, axis=(0, 1), keepdims=True)
    mad = np.median(np.abs(values - center), axis=(0, 1), keepdims=True)
    robust_scale = 1.4826 * mad
    fallback_scale = np.std(values, axis=(0, 1), keepdims=True)
    scale = np.where(robust_scale > eps, robust_scale, fallback_scale)
    scale = np.where(scale > eps, scale, 1.0)
    return sanitize_signal_epochs((values - center) / (scale + eps))


def profile_key(value: str | None) -> str:
    return str(value or "clinical_canonical").strip().lower().replace("-", "_")


def clinical_filter_bands(filter_profile: str = "clinical_canonical", fs: float = 100.0) -> dict[int, tuple[float, float]]:
    key = profile_key(filter_profile)
    if key in {"none", "off", "false", "no_filter", "nofilter"}:
        return {}
    if key in {"legacy", "baseline", "sleepedf_legacy"}:
        eeg_high = min(30.0, float(fs) * 0.5 * 0.95)
        return {0: (0.5, eeg_high), 1: (0.5, eeg_high)}
    if key in {"clinical_canonical", "clinical", "sleepedf_like", "harmonized"}:
        return {0: (0.5, 30.0), 1: (0.1, 15.0), 2: (10.0, 45.0)}
    raise ValueError(f"Unknown filter profile: {filter_profile!r}")


def clamped_band(low: float, high: float, fs: float) -> tuple[float, float] | None:
    fs = float(fs) if fs is not None else 100.0
    nyquist = fs / 2.0
    low = max(float(low), 1e-4)
    high = min(float(high), nyquist * 0.95)
    if high <= low:
        return None
    return low, high


def apply_bandpass_to_channel(values: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    band = clamped_band(low, high, fs)
    if band is None:
        return values.astype(np.float32)
    from scipy import signal  # noqa: WPS433

    low, high = band
    sos = signal.butter(4, [low, high], btype="bandpass", fs=float(fs), output="sos")
    try:
        return signal.sosfiltfilt(sos, values, axis=1).astype(np.float32)
    except ValueError:
        return signal.sosfilt(sos, values, axis=1).astype(np.float32)


def preprocess_recording(
    values: np.ndarray,
    fs: float,
    filter_profile: str,
    normalization_mode: str,
    preprocess: bool,
) -> np.ndarray:
    values = sanitize_signal_epochs(values)
    if not preprocess:
        return values
    if values.shape[1] > 15:
        processed = values.copy()
        for channel_idx, band in clinical_filter_bands(filter_profile, fs=fs).items():
            if channel_idx < processed.shape[-1]:
                processed[:, :, channel_idx] = apply_bandpass_to_channel(
                    processed[:, :, channel_idx],
                    band[0],
                    band[1],
                    fs,
                )
        values = processed
    mode = str(normalization_mode or "recording_robust_zscore").strip().lower()
    if mode in {"recording_robust_zscore", "robust", "robust_zscore"}:
        return recording_robust_zscore(values)
    if mode in {"recording_zscore", "zscore", "standard"}:
        mean = np.mean(values, axis=(0, 1), keepdims=True)
        std = np.std(values, axis=(0, 1), keepdims=True)
        return sanitize_signal_epochs((values - mean) / (std + 1e-6))
    raise ValueError(f"Unknown normalization mode: {normalization_mode!r}")


def load_npz_recording(
    path: Path,
    preprocess: bool,
    filter_profile: str,
    normalization_mode: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=True) as data:
        if "x" not in data.files:
            raise KeyError(f"{path} does not contain key 'x'.")
        labels_key = "y" if "y" in data.files else "labels" if "labels" in data.files else None
        labels = np.asarray(data[labels_key]).reshape(-1) if labels_key else np.zeros(np.asarray(data["x"]).shape[0], dtype=np.int32)
        fs = float(np.asarray(data["fs"]).reshape(-1)[0]) if "fs" in data.files else 100.0
        x = np.asarray(data["x"], dtype=np.float32)
    return preprocess_recording(x, fs, filter_profile, normalization_mode, preprocess), labels.astype(np.int32), fs


def epoch_bandpower(values: np.ndarray, fs: float, bands: Sequence[tuple[float, float]]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"epoch_bandpower expects shape (epochs, samples), got {values.shape}.")
    n_samples = values.shape[1]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / float(fs))
    centered = values - np.mean(values, axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    outputs = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            outputs.append(np.sum(spectrum[:, mask], axis=1))
        else:
            outputs.append(np.zeros(values.shape[0], dtype=np.float32))
    return np.stack(outputs, axis=1).astype(np.float32)


def direct_epoch_concepts_np(x_raw: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Compute concept primitives directly per 30-s epoch.

    These values are intentionally not row-normalized within an epoch. The
    recording-wise percentile scaling later preserves dominance patterns like
    high Wake alpha/beta/EMG and high N3 delta.
    """
    x_raw = sanitize_signal_epochs(x_raw)
    n_epochs, _, n_channels = x_raw.shape
    eeg = x_raw[:, :, 0] if n_channels > 0 else np.zeros((n_epochs, x_raw.shape[1]), dtype=np.float32)
    eog = x_raw[:, :, 1] if n_channels > 1 else np.zeros((n_epochs, x_raw.shape[1]), dtype=np.float32)
    emg = x_raw[:, :, 2] if n_channels > 2 else np.zeros((n_epochs, x_raw.shape[1]), dtype=np.float32)

    eeg_bp = epoch_bandpower(eeg, fs, [(0.5, 4), (4, 8), (8, 12), (12, 16), (16, 30)])
    eeg_rel = eeg_bp / np.maximum(np.sum(eeg_bp, axis=1, keepdims=True), 1e-8)

    eog_bp = epoch_bandpower(eog, fs, [(0.1, 1), (1, 5), (5, 15)])
    eog_total = np.maximum(np.sum(eog_bp, axis=1), 1e-8)
    eog_slow = eog_bp[:, 0] / eog_total
    eog_rapid = eog_bp[:, 1] / eog_total
    if eog.shape[1] > 1:
        eog_rapid = eog_rapid + 0.25 * np.std(np.diff(eog, axis=1), axis=1)

    emg_rms = np.sqrt(np.mean(emg**2, axis=1) + 1e-8)
    emg_p95 = np.percentile(np.abs(emg), 95, axis=1)
    emg_active = emg_rms + 0.25 * emg_p95
    emg_atonia = -emg_active

    return np.stack(
        [
            eeg_rel[:, 0],
            eeg_rel[:, 1],
            eeg_rel[:, 2],
            eeg_rel[:, 3],
            eeg_rel[:, 4],
            eog_slow,
            eog_rapid,
            emg_atonia,
            emg_active,
        ],
        axis=1,
    ).astype(np.float32)


def fit_segments_to_count(
    segments: Sequence[tuple[int, int]],
    total_length: int,
    target_segments: int,
) -> list[tuple[int, int]]:
    segments = [
        (max(0, int(start_idx)), min(int(total_length), int(end_idx)))
        for start_idx, end_idx in segments
        if int(end_idx) > int(start_idx)
    ]
    segments = sorted(segments, key=lambda item: item[0])
    target_segments = max(1, int(target_segments))
    total_length = max(1, int(total_length))

    if not segments:
        boundaries = np.linspace(0, total_length, target_segments + 1, dtype=int)
        return [(int(boundaries[idx]), int(boundaries[idx + 1])) for idx in range(target_segments)]

    if segments[0][0] > 0:
        segments[0] = (0, segments[0][1])
    if segments[-1][1] < total_length:
        segments[-1] = (segments[-1][0], total_length)

    while len(segments) > target_segments:
        lengths = np.asarray([end_idx - start_idx for start_idx, end_idx in segments])
        merge_idx = int(np.argmin(lengths))
        if merge_idx == 0:
            segments[1] = (segments[0][0], segments[1][1])
            del segments[0]
        elif merge_idx == len(segments) - 1:
            segments[-2] = (segments[-2][0], segments[-1][1])
            del segments[-1]
        else:
            left_len = segments[merge_idx - 1][1] - segments[merge_idx - 1][0]
            right_len = segments[merge_idx + 1][1] - segments[merge_idx + 1][0]
            if left_len <= right_len:
                segments[merge_idx - 1] = (segments[merge_idx - 1][0], segments[merge_idx][1])
            else:
                segments[merge_idx + 1] = (segments[merge_idx][0], segments[merge_idx + 1][1])
            del segments[merge_idx]

    while len(segments) < target_segments:
        lengths = np.asarray([end_idx - start_idx for start_idx, end_idx in segments])
        split_idx = int(np.argmax(lengths))
        start_idx, end_idx = segments[split_idx]
        if end_idx - start_idx <= 1:
            break
        mid_idx = (start_idx + end_idx) // 2
        segments[split_idx : split_idx + 1] = [(start_idx, mid_idx), (mid_idx, end_idx)]

    if len(segments) != target_segments:
        boundaries = np.linspace(0, total_length, target_segments + 1, dtype=int)
        segments = [(int(boundaries[idx]), int(boundaries[idx + 1])) for idx in range(target_segments)]

    return segments


def semantic_segmentation_np(
    instance_multimodal: np.ndarray,
    target_segments: int = 30,
    min_len: int = 50,
) -> list[tuple[int, int]]:
    instance_multimodal = sanitize_signal_epochs(np.asarray(instance_multimodal)[np.newaxis, ...])[0]
    total_length = int(instance_multimodal.shape[0])
    num_channels = int(instance_multimodal.shape[1])
    window_size = min(100, max(1, total_length))
    smoothed_energies = []
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)

    for channel_idx in range(num_channels):
        channel_signal = instance_multimodal[:, channel_idx]
        energy = channel_signal**2
        smoothed = np.convolve(energy, kernel, mode="same")
        min_val, max_val = float(np.min(smoothed)), float(np.max(smoothed))
        if max_val - min_val > 1e-6:
            smoothed = (smoothed - min_val) / (max_val - min_val)
        smoothed_energies.append(smoothed)

    weights = np.asarray([0.5, 0.25, 0.25], dtype=np.float32)
    if num_channels > len(weights):
        weights = np.pad(weights, (0, num_channels - len(weights)), constant_values=0.0)
    weights = weights[:num_channels]
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones(num_channels, dtype=np.float32) / max(num_channels, 1)
    total_energy = np.zeros(total_length, dtype=np.float32)
    for channel_idx in range(num_channels):
        total_energy += smoothed_energies[channel_idx] * weights[channel_idx]

    from scipy import signal  # noqa: WPS433

    valleys, _ = signal.find_peaks(-total_energy, distance=int(min_len), prominence=0.005)
    segments: list[tuple[int, int]] = []
    start_idx = 0
    for valley_idx in valleys:
        if int(valley_idx) - start_idx >= min_len:
            segments.append((start_idx, int(valley_idx)))
            start_idx = int(valley_idx)
    if start_idx < total_length:
        segments.append((start_idx, total_length))

    if len(segments) < int(target_segments) // 2:
        boundaries = np.linspace(0, total_length, int(target_segments) + 1, dtype=int)
        segments = [(int(boundaries[idx]), int(boundaries[idx + 1])) for idx in range(int(target_segments))]

    return fit_segments_to_count(segments, total_length, int(target_segments))


def extract_segment_features_np(
    signal_multimodal: np.ndarray,
    segments: Sequence[tuple[int, int]],
    max_segments: int = 30,
    fs: float = 100.0,
    include_spectral: bool = True,
) -> np.ndarray:
    signal_multimodal = sanitize_signal_epochs(np.asarray(signal_multimodal)[np.newaxis, ...])[0]
    features: list[np.ndarray] = []
    for start_idx, end_idx in segments:
        seg_len = int(end_idx) - int(start_idx)
        if seg_len <= 0:
            continue
        seg_data = signal_multimodal[int(start_idx) : int(end_idx), :]
        mean_val = np.mean(seg_data, axis=0)
        std_val = np.std(seg_data, axis=0)
        max_val = np.max(seg_data, axis=0)
        energy = np.sum(seg_data**2, axis=0) / float(seg_len)
        feat_parts = [np.asarray([seg_len], dtype=np.float32), mean_val, std_val, max_val, energy]

        if include_spectral and seg_len >= 8:
            freqs = np.fft.rfftfreq(seg_len, d=1.0 / float(fs))
            spectrum = np.abs(np.fft.rfft(seg_data, axis=0)) ** 2
            total_power = np.sum(spectrum, axis=0) + 1e-8
            band_features = []
            for low, high in [(0.5, 4), (4, 8), (8, 12), (12, 16), (16, 30)]:
                mask = (freqs >= low) & (freqs < high)
                if np.any(mask):
                    band_power = np.sum(spectrum[mask], axis=0) / total_power
                else:
                    band_power = np.zeros(seg_data.shape[1], dtype=np.float32)
                band_features.append(band_power)
            feat_parts.append(np.concatenate(band_features))
        elif include_spectral:
            feat_parts.append(np.zeros(seg_data.shape[1] * 5, dtype=np.float32))

        features.append(np.concatenate(feat_parts).astype(np.float32))

    channel_count = signal_multimodal.shape[1] if signal_multimodal.ndim == 2 else 1
    num_features = (
        features[0].shape[0]
        if features
        else 1 + channel_count * (4 + (5 if include_spectral else 0))
    )
    feature_array = np.asarray(features, dtype=np.float32) if features else np.zeros((0, num_features), dtype=np.float32)
    if feature_array.shape[0] < max_segments:
        pad = np.zeros((max_segments - feature_array.shape[0], num_features), dtype=np.float32)
        feature_array = np.vstack((feature_array, pad))
    else:
        feature_array = feature_array[:max_segments]
    return feature_array.astype(np.float32)


def scale_rows_to_unit(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    row_max = np.max(values, axis=1, keepdims=True)
    return (values / np.maximum(row_max, 1e-8)).astype(np.float32)


def segment_feature(segment_features: np.ndarray, feature_idx: int) -> np.ndarray:
    if feature_idx >= segment_features.shape[-1]:
        return np.zeros(segment_features.shape[:2], dtype=np.float32)
    return segment_features[:, :, feature_idx].astype(np.float32)


def clinical_concept_targets_from_segment_features_np(
    segment_features: np.ndarray,
    concept_version: str = "v2",
) -> np.ndarray:
    segment_features = np.asarray(segment_features, dtype=np.float32)
    if segment_features.ndim == 2:
        segment_features = segment_features[np.newaxis, :, :]
    if segment_features.ndim != 3:
        raise ValueError("segment_features must have shape (samples, segments, features).")

    eeg_delta = segment_feature(segment_features, 13)
    eog_delta = segment_feature(segment_features, 14)
    eeg_theta = segment_feature(segment_features, 16)
    eog_theta = segment_feature(segment_features, 17)
    eeg_alpha = segment_feature(segment_features, 19)
    eeg_sigma = segment_feature(segment_features, 22)
    eeg_beta = segment_feature(segment_features, 25)
    eog_beta = segment_feature(segment_features, 26)
    emg_beta = segment_feature(segment_features, 27)
    eog_energy = segment_feature(segment_features, 11)
    emg_energy = segment_feature(segment_features, 12)

    if str(concept_version).lower() == "v1":
        eog_slow = 0.70 * eog_delta + 0.30 * eog_theta
        eog_rapid = 0.55 * eog_energy + 0.30 * eog_theta + 0.15 * eog_beta
        emg_activation = 0.70 * emg_energy + 0.30 * emg_beta
        emg_low_tone = 1.0 / (1.0 + np.maximum(emg_activation, 0.0))
        concept_segments = np.stack(
            [
                eeg_delta,
                eeg_theta,
                eeg_alpha,
                eeg_sigma,
                eeg_beta,
                eog_slow,
                eog_rapid,
                emg_low_tone,
                emg_activation,
            ],
            axis=-1,
        )
        return np.clip(scale_rows_to_unit(np.mean(np.clip(concept_segments, 0.0, None), axis=1)), 0.0, 1.0)

    eps = 1e-6
    eeg_total = eeg_delta + eeg_theta + eeg_alpha + eeg_sigma + eeg_beta + eps
    slow_wave_dominance = eeg_delta / eeg_total
    theta_light_sleep = eeg_theta / eeg_total
    alpha_wake = eeg_alpha / eeg_total
    sigma_relative = eeg_sigma / eeg_total
    sigma_median = np.median(eeg_sigma, axis=1, keepdims=True)
    sigma_p90 = np.percentile(eeg_sigma, 90, axis=1, keepdims=True)
    sigma_burstiness = (sigma_p90 - sigma_median) / (sigma_median + eps)
    sigma_spindle = sigma_relative * (1.0 + 0.5 * sigma_burstiness)
    beta_arousal = eeg_beta / eeg_total
    eog_slow = 0.70 * eog_delta + 0.30 * eog_theta

    if eog_energy.shape[1] > 1:
        eog_variation = np.abs(np.diff(eog_energy, axis=1))
        eog_variation = np.pad(eog_variation, ((0, 0), (0, 1)), mode="edge")
        eog_rapid = 0.60 * eog_variation + 0.25 * eog_energy + 0.15 * eog_beta
    else:
        eog_rapid = 0.70 * eog_energy + 0.30 * eog_beta

    emg_activation = 0.70 * emg_energy + 0.30 * emg_beta
    emg_center = np.median(emg_activation, axis=1, keepdims=True)
    emg_iqr = np.percentile(emg_activation, 75, axis=1, keepdims=True) - np.percentile(
        emg_activation,
        25,
        axis=1,
        keepdims=True,
    )
    emg_z = np.clip((emg_activation - emg_center) / (emg_iqr + eps), -60.0, 60.0)
    emg_atonia = 1.0 / (1.0 + np.exp(emg_z))
    emg_active = 1.0 / (1.0 + np.exp(-emg_z))

    concept_segments = np.stack(
        [
            slow_wave_dominance,
            theta_light_sleep,
            alpha_wake,
            sigma_spindle,
            beta_arousal,
            eog_slow,
            eog_rapid,
            emg_atonia,
            emg_active,
        ],
        axis=-1,
    )
    concept_profile = np.mean(np.clip(concept_segments, 0.0, None), axis=1)
    return np.clip(scale_rows_to_unit(concept_profile), 0.0, 1.0).astype(np.float32)


def compute_concepts_from_split(
    split_json: Path,
    npz_dir_override: str | None,
    split: str,
    expected_counts: np.ndarray | None,
    concept_version: str,
    concept_extractor: str,
    preprocess: bool,
    filter_profile: str,
    normalization_mode: str,
    return_payload: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
    split_data = load_json(split_json)
    files = split_files_from_json(split_data, split)
    npz_dir = resolve_npz_dir(split_data, npz_dir_override)

    concepts_by_file: list[np.ndarray] = []
    labels_by_file: list[np.ndarray] = []
    counts: list[int] = []
    for idx, filename in enumerate(files, start=1):
        x_raw, _labels, fs = load_npz_recording(
            npz_dir / filename,
            preprocess=preprocess,
            filter_profile=filter_profile,
            normalization_mode=normalization_mode,
        )
        if str(concept_extractor) == "direct_epoch":
            file_concepts = direct_epoch_concepts_np(x_raw, fs=fs)
        else:
            segment_features = []
            for instance in x_raw:
                segments = semantic_segmentation_np(instance, target_segments=30)
                segment_features.append(
                    extract_segment_features_np(
                        instance,
                        segments,
                        max_segments=30,
                        fs=fs,
                        include_spectral=True,
                    )
                )
            segment_features = np.asarray(segment_features, dtype=np.float32)
            file_concepts = clinical_concept_targets_from_segment_features_np(
                segment_features,
                concept_version=concept_version,
            )
        concepts_by_file.append(np.asarray(file_concepts, dtype=np.float32))
        labels_by_file.append(np.asarray(_labels, dtype=np.int32).reshape(-1))
        counts.append(int(file_concepts.shape[0]))
        print(f"Computed concepts {idx}/{len(files)}: {filename} epochs={counts[-1]}")

    if expected_counts is not None:
        expected = [int(item) for item in np.asarray(expected_counts).reshape(-1)]
        if expected != counts:
            raise ValueError(
                "split-json recording counts do not match prediction counts: "
                f"expected={expected[:8]}... computed={counts[:8]}..."
            )
    concepts = np.concatenate(concepts_by_file, axis=0) if concepts_by_file else np.zeros((0, 9), dtype=np.float32)
    if not return_payload:
        return concepts
    labels = np.concatenate(labels_by_file, axis=0) if labels_by_file else np.zeros((0,), dtype=np.int32)
    counts_array = np.asarray(counts, dtype=np.int32)
    recording_id, epoch_index = recording_axes_from_counts(split, counts_array, int(concepts.shape[0]))
    return {
        "concepts": concepts,
        "labels": labels.astype(np.int32),
        "counts": counts_array,
        "recording_id": recording_id,
        "epoch_index": epoch_index,
    }


def stage_prototypes_from_concepts(
    concepts: np.ndarray,
    labels: np.ndarray,
    recording_id: np.ndarray,
    adaptive_concepts: bool,
    num_classes: int = 5,
) -> np.ndarray:
    concepts = np.asarray(concepts, dtype=np.float32)
    labels = labels_to_indices(labels)
    scaled = adaptive_percentile_scale(concepts, recording_id) if adaptive_concepts else np.clip(concepts, 0.0, 1.0)
    prototypes = np.zeros((num_classes, scaled.shape[1]), dtype=np.float32)
    global_center = np.mean(scaled, axis=0) if scaled.size else np.zeros((scaled.shape[1],), dtype=np.float32)
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if np.any(mask):
            prototypes[class_idx] = np.mean(scaled[mask], axis=0)
        else:
            prototypes[class_idx] = global_center
    return prototypes.astype(np.float32)


def load_or_compute_concepts(args: argparse.Namespace, prediction: Mapping[str, object]) -> tuple[np.ndarray | None, str]:
    concepts = prediction.get("concepts")
    if concepts is not None:
        return np.asarray(concepts, dtype=np.float32), str(prediction.get("concept_key") or "prediction_npz")

    if args.concept_cache:
        cache_path = Path(args.concept_cache).expanduser()
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=True) as data:
                key = "concepts" if "concepts" in data.files else data.files[0]
                return np.asarray(data[key], dtype=np.float32), f"concept_cache:{cache_path}"

    if args.split_json:
        concepts = compute_concepts_from_split(
            Path(args.split_json).expanduser().resolve(),
            args.npz_dir,
            args.split,
            np.asarray(prediction.get("counts"), dtype=np.int32) if prediction.get("counts") is not None else None,
            args.concept_version,
            args.concept_extractor,
            preprocess=not bool(args.no_preprocess),
            filter_profile=args.filter_profile,
            normalization_mode=args.normalization_mode,
        )
        if args.concept_cache:
            cache_path = Path(args.concept_cache).expanduser()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                concepts=concepts,
                concept_extractor=np.asarray(args.concept_extractor),
                concept_version=np.asarray(args.concept_version),
            )
            print(f"Saved concept cache: {cache_path}")
        return concepts, f"computed:{args.split_json}"

    return None, "missing"


def load_or_compute_stage_prototypes(
    args: argparse.Namespace,
    adaptive_concepts: bool,
    num_classes: int,
) -> tuple[np.ndarray | None, str]:
    if not args.calibrate_prototypes:
        return None, "disabled"
    if args.prototype_cache:
        cache_path = Path(args.prototype_cache).expanduser()
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=True) as data:
                if "stage_prototypes" not in data.files:
                    raise KeyError(f"{cache_path} does not contain stage_prototypes.")
                return np.asarray(data["stage_prototypes"], dtype=np.float32), f"prototype_cache:{cache_path}"
    if not args.split_json:
        raise ValueError("--calibrate-prototypes requires --split-json unless --prototype-cache exists.")
    payload = compute_concepts_from_split(
        Path(args.split_json).expanduser().resolve(),
        args.npz_dir,
        args.prototype_split,
        expected_counts=None,
        concept_version=args.concept_version,
        concept_extractor=args.concept_extractor,
        preprocess=not bool(args.no_preprocess),
        filter_profile=args.filter_profile,
        normalization_mode=args.normalization_mode,
        return_payload=True,
    )
    assert isinstance(payload, dict)
    prototypes = stage_prototypes_from_concepts(
        payload["concepts"],
        payload["labels"],
        payload["recording_id"],
        adaptive_concepts=adaptive_concepts,
        num_classes=num_classes,
    )
    if args.prototype_cache:
        cache_path = Path(args.prototype_cache).expanduser()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            stage_prototypes=prototypes,
            prototype_split=np.asarray(args.prototype_split),
            concept_version=np.asarray(args.concept_version),
            concept_extractor=np.asarray(args.concept_extractor),
        )
        print(f"Saved prototype cache: {cache_path}")
    return prototypes, f"computed:{args.prototype_split}:{args.split_json}"


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_name(class_labels: Sequence[str], idx: int) -> str:
    if 0 <= int(idx) < len(class_labels):
        return str(class_labels[int(idx)])
    return str(idx)


def top_conflict_rows(
    y_true: np.ndarray,
    probs: np.ndarray,
    model_pred: np.ndarray,
    rule_result: Mapping[str, np.ndarray],
    recording_id: np.ndarray,
    epoch_index: np.ndarray,
    class_labels: Sequence[str],
    top_n: int,
    high_conflict: np.ndarray,
    risk: Mapping[str, np.ndarray] | None = None,
) -> list[dict[str, object]]:
    p_rule = rule_result["p_rule"]
    rule_pred = rule_result["rule_pred"]
    rule_conf = rule_result["rule_conf"]
    model_conf = np.max(probs, axis=1)
    disagreement = model_pred != rule_pred
    rows = np.flatnonzero(disagreement)
    if rows.size == 0:
        return []
    if risk is not None:
        score = np.asarray(risk["risk_score"], dtype=np.float32)[rows]
        model_margin = np.asarray(risk["model_margin"], dtype=np.float32)
        rule_margin = np.asarray(risk["rule_margin"], dtype=np.float32)
        prob_l1 = np.asarray(risk["prob_l1"], dtype=np.float32)
        prob_js = np.asarray(risk["prob_js"], dtype=np.float32)
    else:
        score = model_conf[rows] * rule_conf[rows] + high_conflict[rows].astype(np.float32)
        model_margin = probability_margin(probs)
        rule_margin = probability_margin(p_rule)
        prob_l1 = (0.5 * np.sum(np.abs(normalize_probabilities(probs) - normalize_probabilities(p_rule)), axis=1)).astype(np.float32)
        prob_js = probability_js_distance(probs, p_rule)
    order = rows[np.argsort(-score)[: max(0, int(top_n))]]
    output: list[dict[str, object]] = []
    for idx in order:
        row = {
            "row": int(idx),
            "recording_id": str(recording_id[idx]),
            "epoch_index": int(epoch_index[idx]),
            "true": label_name(class_labels, int(y_true[idx])),
            "model_pred": label_name(class_labels, int(model_pred[idx])),
            "rule_pred": label_name(class_labels, int(rule_pred[idx])),
            "model_conf": float(model_conf[idx]),
            "model_margin": float(model_margin[idx]),
            "rule_conf": float(rule_conf[idx]),
            "rule_margin": float(rule_margin[idx]),
            "prob_l1": float(prob_l1[idx]),
            "prob_js": float(prob_js[idx]),
            "risk_score": float(np.asarray(risk["risk_score"], dtype=np.float32)[idx]) if risk is not None else float(score[np.where(rows == idx)[0][0]]),
            "model_correct": bool(model_pred[idx] == y_true[idx]),
            "rule_correct": bool(rule_pred[idx] == y_true[idx]),
            "high_conflict": bool(high_conflict[idx]),
            "transition_context": float(rule_result["transition_context"][idx]),
            "n1_residual_hint": float(rule_result["n1_residual_hint"][idx]),
        }
        for class_idx, class_name in enumerate(class_labels):
            row[f"model_p_{class_name}"] = float(probs[idx, class_idx])
            row[f"rule_p_{class_name}"] = float(p_rule[idx, class_idx])
        output.append(row)
    return output


def compact_counter(values: np.ndarray, class_labels: Sequence[str]) -> dict[str, int]:
    counter = Counter(int(item) for item in values.reshape(-1))
    return {label_name(class_labels, key): int(value) for key, value in sorted(counter.items())}


def main() -> None:
    args = parse_args()
    pred_path = Path(args.prediction_npz).expanduser().resolve()
    prediction = read_prediction_file(pred_path, args.split, probs_key=args.probs_key, pred_key=args.pred_key)
    y_true = np.asarray(prediction["y_true"], dtype=np.int32)
    probs = np.asarray(prediction["probs"], dtype=np.float32)
    model_pred = np.asarray(prediction["model_pred"], dtype=np.int32)
    recording_id = np.asarray(prediction["recording_id"]).reshape(-1)
    epoch_index = np.asarray(prediction["epoch_index"], dtype=np.int32).reshape(-1)
    class_labels = tuple(str(item) for item in prediction["class_labels"])

    concepts, concept_source = load_or_compute_concepts(args, prediction)
    stage_prototypes, prototype_source = load_or_compute_stage_prototypes(
        args,
        adaptive_concepts=not bool(args.no_adaptive_concepts),
        num_classes=len(class_labels),
    )
    if concepts is None:
        if not args.allow_probability_fallback:
            raise SystemExit(
                "No clinical concepts found. Re-run with --split-json/--npz-dir, "
                "pass --concept-cache, or use --allow-probability-fallback for a weaker smoke-test analysis."
            )
        print("WARNING: using probability fallback because no clinical concepts were available.")
        rule_result = compute_probability_fallback_rule(
            probs,
            recording_id,
            rule_temperature=args.rule_temperature,
            n1_window=args.n1_window,
            transition_radius=args.transition_radius,
        )
        concept_source = "probability_fallback"
    else:
        if concepts.shape[0] != y_true.shape[0]:
            raise ValueError(f"concepts rows={concepts.shape[0]} do not match predictions rows={y_true.shape[0]}.")
        rule_result = compute_rule_teacher(
            concepts,
            recording_id,
            adaptive_concepts=not bool(args.no_adaptive_concepts),
            rule_temperature=args.rule_temperature,
            n1_window=args.n1_window,
            transition_radius=args.transition_radius,
            stage_prototypes=stage_prototypes,
            prototype_weight=args.prototype_weight,
        )

    risk = compute_disagreement_risk(
        probs,
        rule_result["p_rule"],
        model_pred,
        rule_result["rule_pred"],
        rule_result["rule_conf"],
    )
    summary = summarize_disagreement(
        y_true,
        probs,
        model_pred,
        rule_result["rule_pred"],
        rule_result["rule_conf"],
        class_labels,
        model_conf_threshold=args.model_conf_threshold,
        rule_conf_threshold=args.rule_conf_threshold,
        risk=risk,
    )
    model_conf = risk["model_conf"]
    disagreement = model_pred != rule_result["rule_pred"]
    high_conflict = disagreement & (model_conf >= args.model_conf_threshold) & (
        rule_result["rule_conf"] >= args.rule_conf_threshold
    )

    summary.update(
        {
            "dataset": args.dataset,
            "split": args.split,
            "prediction_npz": str(pred_path),
            "label_key": str(prediction["label_key"]),
            "score_key": str(prediction["score_key"]),
            "pred_key": str(prediction["pred_key"]),
            "concept_source": concept_source,
            "concept_extractor": args.concept_extractor,
            "adaptive_concepts": not bool(args.no_adaptive_concepts),
            "prototype_source": prototype_source,
            "prototype_weight": float(args.prototype_weight if stage_prototypes is not None else 0.0),
            "rule_temperature": float(args.rule_temperature),
            "n1_window": int(args.n1_window),
            "transition_radius": int(args.transition_radius),
            "model_pred_distribution": compact_counter(model_pred, class_labels),
            "rule_pred_distribution": compact_counter(rule_result["rule_pred"], class_labels),
        }
    )

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "diagnostics" / "b2_rule_teacher" / pred_path.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_dir / "per_class.csv", summary["per_class"])
    write_csv(out_dir / "focus_pairs.csv", summary["focus_pairs"])
    write_csv(
        out_dir / "top_conflicts.csv",
        top_conflict_rows(
            y_true,
            probs,
            model_pred,
            rule_result,
            recording_id,
            epoch_index,
            class_labels,
            args.top_examples,
            high_conflict,
            risk,
        ),
    )
    if args.save_npz:
        np.savez_compressed(
            out_dir / "rule_teacher_outputs.npz",
            p_rule=rule_result["p_rule"].astype(np.float32),
            rule_scores=rule_result["rule_scores"].astype(np.float32),
            clinical_rule_scores=rule_result["clinical_rule_scores"].astype(np.float32),
            prototype_scores=rule_result["prototype_scores"].astype(np.float32),
            rule_pred=rule_result["rule_pred"].astype(np.int16),
            rule_conf=rule_result["rule_conf"].astype(np.float32),
            model_conf=model_conf.astype(np.float32),
            model_margin=risk["model_margin"].astype(np.float32),
            model_uncertainty=risk["model_uncertainty"].astype(np.float32),
            rule_margin=risk["rule_margin"].astype(np.float32),
            prob_l1=risk["prob_l1"].astype(np.float32),
            prob_js=risk["prob_js"].astype(np.float32),
            risk_score=risk["risk_score"].astype(np.float32),
            disagreement=disagreement.astype(bool),
            high_conflict=high_conflict.astype(bool),
            base_stage=rule_result["base_stage"].astype(np.int16),
            base_conf=rule_result["base_conf"].astype(np.float32),
            residual=rule_result["residual"].astype(np.float32),
            transition_context=rule_result["transition_context"].astype(np.float32),
            n1_residual_hint=rule_result["n1_residual_hint"].astype(np.float32),
            concepts_scaled=rule_result["concepts_scaled"].astype(np.float32),
        )

    print(f"Saved B2 rule-teacher disagreement summary: {summary_path}")
    print(
        "B2 headline | "
        f"model_acc={summary['model_metrics']['accuracy']:.4f} "
        f"rule_acc={summary['rule_metrics']['accuracy']:.4f} "
        f"disagreement={summary['disagreement_rate']:.4f} "
        f"high_conflict={summary['high_conflict_rate']:.4f} "
        f"err_top10risk={summary.get('model_error_rate_when_top_risk_10pct', 0.0):.4f} "
        f"err_if_disagree={summary['model_error_rate_when_disagree']:.4f} "
        f"err_if_high_conflict={summary['model_error_rate_when_high_conflict']:.4f}"
    )


if __name__ == "__main__":
    main()
