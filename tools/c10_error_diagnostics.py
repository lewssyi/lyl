#!/usr/bin/env python
"""C10 error diagnostics for fixed sleep-stage prediction files.

This script does not modify predictions. It turns a C10/C10a prediction npz
into analysis tables for paper-facing error analysis: focused confusions,
quality strata, transition-boundary behavior, time-position behavior, and
recording-level failure modes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

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
    "main_probs",
    "main_probabilities",
    "prediction_probs",
    "pred_probs",
    "y_probs",
    "y_prob",
    "test_probs_raw",
    "test_probs_calibrated",
)
LOGIT_KEYS = (
    "final_logits",
    "logits",
    "prediction_logits",
    "pred_logits",
    "y_logits",
    "base_logits",
    "main_logits",
)
PRED_KEYS = (
    "y_pred_final",
    "pred_final",
    "final_pred",
    "prediction",
    "pred",
    "predicted_labels",
    "y_pred",
)
COUNT_KEYS = ("file_sample_counts", "recording_sample_counts", "sample_counts")
RECORDING_ID_KEYS = ("recording_id", "recording_ids", "record_ids", "record_id")
EPOCH_INDEX_KEYS = ("epoch_index", "epoch_indices", "epoch_idx")
QUALITY_KEYS = ("quality", "quality_scores", "signal_quality", "epoch_quality")
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


@dataclass(frozen=True)
class SplitPrediction:
    split: str
    path: Path
    y_true: np.ndarray
    probs: np.ndarray
    logits: np.ndarray
    c10a_pred: np.ndarray
    recording_id: np.ndarray
    epoch_index: np.ndarray
    class_labels: tuple[str, ...]
    score_key: str
    score_kind: str
    label_key: str
    pred_key: str | None
    quality: np.ndarray | None = None
    concepts: np.ndarray | None = None


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
    include_unscoped: bool = False,
) -> str | None:
    if explicit:
        explicit_candidates = [explicit]
        if not explicit.startswith(f"{split}_"):
            explicit_candidates.insert(0, f"{split}_{explicit}")
        for key in explicit_candidates:
            if key in data.files:
                return key
        raise KeyError(f"Explicit key {explicit!r} was not found. Available keys: {', '.join(data.files)}")
    for key in split_candidates(split, base_keys, include_unscoped=include_unscoped):
        if key in data.files:
            return key
    return None


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    if values.ndim > 2:
        values = values.reshape(-1, values.shape[-1])
    if values.ndim != 2:
        raise ValueError(f"Probability array must be 2D, got {values.shape}.")
    values = np.clip(values, 0.0, None)
    return values / np.maximum(np.sum(values, axis=1, keepdims=True), 1e-12)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim > 2:
        values = values.reshape(-1, values.shape[-1])
    if values.ndim != 2:
        raise ValueError(f"Logit array must be 2D, got {values.shape}.")
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(np.sum(exp_values, axis=1, keepdims=True), 1e-12)


def logits_from_probabilities(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(normalize_probabilities(probs), 1e-12, 1.0))


def coerce_scores(array: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim > 2:
        values = values.reshape(-1, values.shape[-1])
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {values.shape}.")
    return values


def coerce_labels(array: np.ndarray, n_classes: int) -> np.ndarray:
    labels = np.asarray(array)
    if labels.ndim >= 2 and labels.shape[-1] == n_classes:
        labels = np.argmax(labels, axis=-1)
    return np.asarray(labels, dtype=np.int64).reshape(-1)


def read_class_labels(data: np.lib.npyio.NpzFile, n_classes: int) -> tuple[str, ...]:
    if "class_labels" in data.files:
        labels = tuple(str(item) for item in np.asarray(data["class_labels"]).tolist())
        if len(labels) == n_classes:
            return labels
    if n_classes == len(DEFAULT_CLASS_LABELS):
        return DEFAULT_CLASS_LABELS
    return tuple(f"class_{idx}" for idx in range(n_classes))


def read_counts(
    data: np.lib.npyio.NpzFile,
    split: str,
    n_rows: int,
    include_unscoped: bool,
) -> np.ndarray | None:
    key = find_key(data, split, COUNT_KEYS, include_unscoped=include_unscoped)
    if key is None:
        return None
    counts = np.asarray(data[key], dtype=np.int64).reshape(-1)
    counts = counts[counts > 0]
    if int(np.sum(counts)) != int(n_rows):
        raise ValueError(f"{split} {key!r} sums to {int(np.sum(counts))}, expected {n_rows}.")
    return counts


def recording_axes_from_counts(split: str, counts: np.ndarray | None, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    if counts is None:
        return np.asarray([f"{split}_recording_000"] * n_rows, dtype=object), np.arange(n_rows, dtype=np.int64)
    recording_ids: list[str] = []
    epoch_indices: list[int] = []
    for rec_idx, count in enumerate(counts):
        recording_ids.extend([str(rec_idx)] * int(count))
        epoch_indices.extend(range(int(count)))
    return np.asarray(recording_ids, dtype=object), np.asarray(epoch_indices, dtype=np.int64)


def read_recording_axes(
    data: np.lib.npyio.NpzFile,
    split: str,
    n_rows: int,
    counts: np.ndarray | None,
    include_unscoped: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rec_key = find_key(data, split, RECORDING_ID_KEYS, include_unscoped=include_unscoped)
    idx_key = find_key(data, split, EPOCH_INDEX_KEYS, include_unscoped=include_unscoped)
    if rec_key is None or idx_key is None:
        return recording_axes_from_counts(split, counts, n_rows)
    recording_id = np.asarray(data[rec_key]).reshape(-1)
    epoch_index = np.asarray(data[idx_key], dtype=np.int64).reshape(-1)
    if recording_id.shape[0] != n_rows or epoch_index.shape[0] != n_rows:
        raise ValueError(
            f"{split} recording axes must have {n_rows} rows; got "
            f"{recording_id.shape[0]} recording ids and {epoch_index.shape[0]} epoch indices."
        )
    return recording_id.astype(object), epoch_index


def coerce_optional_matrix(array: np.ndarray, n_rows: int, max_cols: int | None, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    if values.shape[0] != n_rows:
        raise ValueError(f"{name} has {values.shape[0]} rows, expected {n_rows}.")
    if max_cols is not None and values.shape[1] > max_cols:
        values = values[:, :max_cols]
    return values


def coerce_concepts(array: np.ndarray, n_rows: int, name: str, n_concepts: int = 9) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 2:
        if values.shape[0] != n_rows:
            raise ValueError(f"{name} has {values.shape[0]} rows, expected {n_rows}.")
        return values[:, :n_concepts]
    if values.ndim == 3:
        if values.shape[0] != n_rows:
            raise ValueError(f"{name} has {values.shape[0]} rows, expected {n_rows}.")
        if values.shape[-1] >= n_concepts:
            return np.mean(values[..., :n_concepts], axis=1)
        if values.shape[1] >= n_concepts:
            return np.mean(values[:, :n_concepts, :], axis=2)
    raise ValueError(f"{name} cannot be coerced to concept matrix; got shape {values.shape}.")


def load_split_prediction(path: Path, split: str, args: argparse.Namespace) -> SplitPrediction:
    if not path.exists():
        raise FileNotFoundError(f"{split} prediction npz does not exist: {path}")
    combined_path = Path(args.prediction_npz or "").expanduser().resolve() if args.prediction_npz else None
    include_unscoped = split == "test" or path != combined_path
    with np.load(path, allow_pickle=True) as data:
        label_key = find_key(data, split, LABEL_KEYS, args.label_key, include_unscoped=include_unscoped)
        logit_key = find_key(data, split, LOGIT_KEYS, args.logit_key, include_unscoped=include_unscoped)
        prob_key = find_key(data, split, PROB_KEYS, args.prob_key, include_unscoped=include_unscoped)
        if label_key is None:
            raise KeyError(f"{path} is missing {split} labels. Available keys: {', '.join(data.files)}")
        if logit_key is None and prob_key is None:
            raise KeyError(f"{path} is missing {split} logits/probabilities. Available keys: {', '.join(data.files)}")

        if logit_key is not None:
            logits = coerce_scores(np.asarray(data[logit_key]), logit_key)
            probs = softmax(logits) if prob_key is None else normalize_probabilities(np.asarray(data[prob_key]))
            score_key = logit_key
            score_kind = "logits"
        else:
            probs = normalize_probabilities(np.asarray(data[prob_key]))
            logits = logits_from_probabilities(probs)
            score_key = str(prob_key)
            score_kind = "probabilities_as_log"

        n_classes = int(probs.shape[1])
        y_true = coerce_labels(np.asarray(data[label_key]), n_classes)
        if y_true.shape[0] != probs.shape[0]:
            raise ValueError(f"{path} {split}: labels have {y_true.shape[0]} rows, scores have {probs.shape[0]} rows.")

        counts = read_counts(data, split, len(y_true), include_unscoped=include_unscoped)
        recording_id, epoch_index = read_recording_axes(data, split, len(y_true), counts, include_unscoped)
        class_labels = read_class_labels(data, n_classes)

        pred_key = find_key(data, split, PRED_KEYS, args.pred_key, include_unscoped=include_unscoped)
        if args.baseline_pred_source == "argmax" or pred_key is None:
            if args.baseline_pred_source == "saved" and pred_key is None:
                raise KeyError(f"{path} {split}: --baseline-pred-source saved requested, but no saved prediction key found.")
            c10a_pred = np.argmax(probs, axis=1).astype(np.int64)
            pred_key_used = None
        else:
            c10a_pred = coerce_labels(np.asarray(data[pred_key]), n_classes)
            pred_key_used = pred_key
            if c10a_pred.shape[0] != y_true.shape[0]:
                raise ValueError(f"{path} {split}: saved prediction {pred_key!r} has wrong length.")

        quality = None
        if not args.no_quality:
            quality_key = find_key(data, split, QUALITY_KEYS, include_unscoped=include_unscoped)
            if quality_key is not None:
                quality = coerce_optional_matrix(np.asarray(data[quality_key]), len(y_true), None, quality_key)

        concepts = None
        if not args.no_concepts:
            concept_key = find_key(data, split, CONCEPT_KEYS, include_unscoped=include_unscoped)
            if concept_key is not None:
                concepts = coerce_concepts(np.asarray(data[concept_key]), len(y_true), concept_key)

        return SplitPrediction(
            split=split,
            path=path,
            y_true=y_true,
            probs=probs,
            logits=logits,
            c10a_pred=c10a_pred,
            recording_id=recording_id,
            epoch_index=epoch_index,
            class_labels=class_labels,
            score_key=score_key,
            score_kind=score_kind,
            label_key=label_key,
            pred_key=pred_key_used,
            quality=quality,
            concepts=concepts,
        )


def stage_index(class_labels: Sequence[str], label: str, fallback: int) -> int:
    for idx, class_label in enumerate(class_labels):
        if str(class_label).upper() == label.upper():
            return idx
    return fallback


def top_two(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(probs, axis=1)
    top1 = order[:, -1].astype(np.int64)
    top2 = order[:, -2].astype(np.int64)
    return top1, top2, probs[np.arange(len(probs)), top1], probs[np.arange(len(probs)), top2]


def entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def normalized_time_position(recording_id: np.ndarray, epoch_index: np.ndarray) -> np.ndarray:
    pos = np.zeros(len(epoch_index), dtype=np.float64)
    for rec in dict.fromkeys(recording_id.tolist()):
        rows = np.flatnonzero(recording_id == rec)
        if len(rows) <= 1:
            pos[rows] = 0.0
            continue
        local = epoch_index[rows].astype(np.float64)
        local = local - float(np.min(local))
        denom = float(np.max(local))
        pos[rows] = local / denom if denom > 0 else np.linspace(0.0, 1.0, len(rows))
    return pos.reshape(-1, 1)


def selection_score(metrics: Mapping[str, float], mode: str) -> float:
    if mode == "accuracy":
        return float(metrics["accuracy"])
    if mode == "macro_f1":
        return float(metrics["macro_f1"])
    if mode == "accuracy_macro":
        return 0.5 * float(metrics["accuracy"]) + 0.5 * float(metrics["macro_f1"])
    if mode == "n1_f1":
        return float(metrics["n1_f1"])
    if mode == "kappa":
        return float(metrics["kappa"])
    raise ValueError(f"Unsupported selection metric: {mode}")


def metric_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[str],
    selection_metric: str,
) -> dict[str, float]:
    n1 = stage_index(class_labels, "N1", 1 if len(class_labels) > 1 else 0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n1_f1": float(f1_score(y_true == n1, y_pred == n1, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    if not np.isfinite(metrics["kappa"]):
        metrics["kappa"] = 0.0
    metrics["score"] = selection_score(metrics, selection_metric)
    return metrics


def focus_confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[str],
) -> dict[str, int]:
    n1 = stage_index(class_labels, "N1", 1)
    n2 = stage_index(class_labels, "N2", 2)
    n3 = stage_index(class_labels, "N3", 3)
    rem = stage_index(class_labels, "REM", 4)
    return {
        "n1_n2_confusion": int(np.sum(((y_true == n1) & (y_pred == n2)) | ((y_true == n2) & (y_pred == n1)))),
        "n2_n3_confusion": int(np.sum(((y_true == n2) & (y_pred == n3)) | ((y_true == n3) & (y_pred == n2)))),
        "rem_to_n1_n2_confusion": int(np.sum((y_true == rem) & ((y_pred == n1) | (y_pred == n2)))),
    }


def per_class_rows(
    dataset: str,
    model_name: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[str],
) -> list[dict[str, object]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_labels)),
        zero_division=0,
    )
    return [
        {
            "dataset": dataset,
            "model": model_name,
            "split": split,
            "class_index": idx,
            "class_label": label,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, label in enumerate(class_labels)
    ]


def confusion_matrix_rows(
    dataset: str,
    model_name: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[str],
) -> list[dict[str, object]]:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_labels)))
    rows = []
    for true_idx, true_label in enumerate(class_labels):
        row: dict[str, object] = {
            "dataset": dataset,
            "model": model_name,
            "split": split,
            "true_index": true_idx,
            "true_label": true_label,
        }
        for pred_idx, pred_label in enumerate(class_labels):
            row[f"pred_{pred_label}"] = int(cm[true_idx, pred_idx])
        rows.append(row)
    return rows


def csv_value(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8f}"
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def safe_dataset_name(dataset: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(dataset).strip().lower())
    return cleaned or "dataset"


CLASS_PAIR_LABELS = (
    ("N1", "N2"),
    ("N2", "N3"),
    ("REM", "N1"),
    ("REM", "N2"),
    ("W", "N1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate C10/C10a error diagnostics from prediction npz files."
    )
    parser.add_argument("--prediction-npz", default=None)
    parser.add_argument("--train-prediction-npz", default=None)
    parser.add_argument("--val-prediction-npz", default=None)
    parser.add_argument("--test-prediction-npz", default=None)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--output-dir", default="outputs/diagnostics/c10")
    parser.add_argument(
        "--splits",
        default="test",
        help="Comma-separated splits to analyze. Use train,val,test for split npz files.",
    )
    parser.add_argument(
        "--baseline-pred-source",
        choices=("auto", "argmax", "saved"),
        default="auto",
        help="Use saved C10 final predictions when available, otherwise argmax unless set explicitly.",
    )
    parser.add_argument("--prob-key", default=None)
    parser.add_argument("--logit-key", default=None)
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--pred-key", default=None)
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--no-concepts", action="store_true")
    parser.add_argument("--quality-low-threshold", type=float, default=0.70)
    parser.add_argument("--quality-severe-threshold", type=float, default=0.40)
    parser.add_argument("--quality-bins", type=int, default=4)
    parser.add_argument("--confidence-bins", type=int, default=5)
    parser.add_argument("--time-bins", type=int, default=5)
    parser.add_argument("--transition-radii", default="1,2,5")
    parser.add_argument("--top-confusions", type=int, default=20)
    parser.add_argument("--top-errors", type=int, default=200)
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "macro_f1", "accuracy_macro", "n1_f1", "kappa"),
        default="accuracy_macro",
    )
    return parser.parse_args()


def split_names(raw: str) -> tuple[str, ...]:
    names = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    valid = {"train", "val", "test"}
    if not names or any(name not in valid for name in names):
        raise ValueError("--splits must contain only train, val, and/or test.")
    return names


def split_path(args: argparse.Namespace, split: str) -> Path:
    explicit = {
        "train": args.train_prediction_npz,
        "val": args.val_prediction_npz,
        "test": args.test_prediction_npz,
    }[split]
    raw = explicit or args.prediction_npz
    if not raw:
        raise ValueError(f"No prediction npz was provided for split {split!r}.")
    return Path(raw).expanduser().resolve()


def parse_radii(raw: str) -> tuple[int, ...]:
    radii = tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))
    if not radii or any(radius < 0 for radius in radii):
        raise ValueError("--transition-radii must contain non-negative integers.")
    return radii


def label_lookup(class_labels: Sequence[str]) -> dict[str, int]:
    return {str(label).upper(): idx for idx, label in enumerate(class_labels)}


def label_name(class_labels: Sequence[str], idx: int) -> str:
    if 0 <= int(idx) < len(class_labels):
        return str(class_labels[int(idx)])
    return f"class_{idx}"


def safe_float(value: float | int | np.generic | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def safe_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return safe_float(np.nanmean(values))


def quality_vector(split: SplitPrediction) -> np.ndarray | None:
    if split.quality is None:
        return None
    quality = np.asarray(split.quality, dtype=np.float64)
    if quality.ndim == 1:
        return quality
    return np.nanmean(quality, axis=1)


def probability_summary(split: SplitPrediction) -> dict[str, np.ndarray]:
    top1, top2, top1_prob, top2_prob = top_two(split.probs)
    return {
        "top1": top1,
        "top2": top2,
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
        "confidence": top1_prob,
        "margin": top1_prob - top2_prob,
        "entropy": entropy(split.probs),
    }


def metric_row(
    dataset: str,
    split: SplitPrediction,
    subset: str,
    mask: np.ndarray,
    selection_metric: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    mask = np.asarray(mask, dtype=bool)
    row: dict[str, object] = {
        "dataset": dataset,
        "split": split.split,
        "subset": subset,
        "n": int(np.sum(mask)),
        "fraction": float(np.mean(mask)) if len(mask) else 0.0,
    }
    if extra:
        row.update(extra)
    if int(np.sum(mask)) == 0:
        row.update({"accuracy": None, "macro_f1": None, "n1_f1": None, "kappa": None, "score": None})
        return row
    metrics = metric_bundle(
        split.y_true[mask],
        split.c10a_pred[mask],
        split.class_labels,
        selection_metric,
    )
    row.update(metrics)
    focus = focus_confusion_counts(split.y_true[mask], split.c10a_pred[mask], split.class_labels)
    row.update(focus)
    return row


def transition_distances(split: SplitPrediction) -> tuple[np.ndarray, np.ndarray]:
    n = len(split.y_true)
    distances = np.full(n, np.inf, dtype=np.float64)
    transition_event = np.zeros(n, dtype=bool)
    for rec in dict.fromkeys(split.recording_id.tolist()):
        rows = np.flatnonzero(split.recording_id == rec)
        if len(rows) == 0:
            continue
        order = rows[np.argsort(split.epoch_index[rows], kind="stable")]
        labels = split.y_true[order]
        starts = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        if len(starts) == 0:
            distances[order] = np.inf
            continue
        event_positions = np.unique(np.concatenate([starts, np.maximum(starts - 1, 0)]))
        local_event = np.zeros(len(order), dtype=bool)
        local_event[event_positions] = True
        transition_event[order[local_event]] = True

        local_distance = np.full(len(order), np.inf, dtype=np.float64)
        last = -10**9
        for idx in range(len(order)):
            if local_event[idx]:
                last = idx
            local_distance[idx] = min(local_distance[idx], idx - last)
        last = 10**9
        for idx in range(len(order) - 1, -1, -1):
            if local_event[idx]:
                last = idx
            local_distance[idx] = min(local_distance[idx], last - idx)
        distances[order] = local_distance
    return distances, transition_event


def transition_pair_rows(
    dataset: str,
    split: SplitPrediction,
    distances: np.ndarray,
    radius: int,
    selection_metric: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rec in dict.fromkeys(split.recording_id.tolist()):
        rec_rows = np.flatnonzero(split.recording_id == rec)
        order = rec_rows[np.argsort(split.epoch_index[rec_rows], kind="stable")]
        labels = split.y_true[order]
        for local_idx in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
            left = int(labels[local_idx - 1])
            right = int(labels[local_idx])
            lo = max(0, local_idx - int(radius))
            hi = min(len(order), local_idx + int(radius) + 1)
            global_rows = order[lo:hi]
            mask = np.zeros(len(split.y_true), dtype=bool)
            mask[global_rows] = True
            pair = f"{label_name(split.class_labels, left)}->{label_name(split.class_labels, right)}"
            row = metric_row(
                dataset,
                split,
                f"transition_{pair}_pm{radius}",
                mask,
                selection_metric,
                {
                    "transition_pair": pair,
                    "radius": int(radius),
                    "recording_id": str(rec),
                    "transition_epoch_index": int(split.epoch_index[order[local_idx]]),
                    "nearest_transition_distance_mean": safe_mean(distances[mask]),
                },
            )
            rows.append(row)
    return rows


def aggregate_transition_pair_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["split"]),
            str(row["transition_pair"]),
            int(row["radius"]),
        )
        current = grouped.setdefault(
            key,
            {
                "dataset": key[0],
                "split": key[1],
                "transition_pair": key[2],
                "radius": key[3],
                "transition_events": 0,
                "n": 0,
                "correct": 0,
                "n1_n2_confusion": 0,
                "n2_n3_confusion": 0,
                "rem_to_n1_n2_confusion": 0,
            },
        )
        n = int(row.get("n", 0) or 0)
        acc = row.get("accuracy")
        current["transition_events"] = int(current["transition_events"]) + 1
        current["n"] = int(current["n"]) + n
        if acc is not None:
            current["correct"] = int(current["correct"]) + int(round(float(acc) * n))
        for name in ("n1_n2_confusion", "n2_n3_confusion", "rem_to_n1_n2_confusion"):
            current[name] = int(current[name]) + int(row.get(name, 0) or 0)
    output = []
    for row in grouped.values():
        n = int(row["n"])
        row["accuracy"] = float(row["correct"] / n) if n else None
        output.append(row)
    return sorted(output, key=lambda item: (-int(item["transition_events"]), str(item["transition_pair"])))


def build_quality_rows(
    dataset: str,
    split: SplitPrediction,
    quality: np.ndarray | None,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    if quality is None:
        return []
    rows = [
        metric_row(dataset, split, "quality_all", np.ones(len(quality), dtype=bool), args.selection_metric),
        metric_row(
            dataset,
            split,
            f"quality_severe_lt_{args.quality_severe_threshold:g}",
            quality < float(args.quality_severe_threshold),
            args.selection_metric,
            {"quality_min": None, "quality_max": float(args.quality_severe_threshold)},
        ),
        metric_row(
            dataset,
            split,
            f"quality_low_lt_{args.quality_low_threshold:g}",
            quality < float(args.quality_low_threshold),
            args.selection_metric,
            {"quality_min": None, "quality_max": float(args.quality_low_threshold)},
        ),
        metric_row(
            dataset,
            split,
            f"quality_high_ge_{args.quality_low_threshold:g}",
            quality >= float(args.quality_low_threshold),
            args.selection_metric,
            {"quality_min": float(args.quality_low_threshold), "quality_max": None},
        ),
    ]
    rows.extend(
        quantile_bin_rows(
            dataset,
            split,
            quality,
            "quality_quantile",
            max(1, int(args.quality_bins)),
            args.selection_metric,
        )
    )
    return rows


def quantile_bin_rows(
    dataset: str,
    split: SplitPrediction,
    values: np.ndarray,
    prefix: str,
    n_bins: int,
    selection_metric: str,
) -> list[dict[str, object]]:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return []
    quantiles = np.linspace(0.0, 1.0, max(2, int(n_bins) + 1))
    edges = np.unique(np.nanquantile(values[finite], quantiles))
    if len(edges) <= 1:
        return [
            metric_row(
                dataset,
                split,
                f"{prefix}_all",
                finite,
                selection_metric,
                {"bin_min": safe_float(edges[0]), "bin_max": safe_float(edges[0])},
            )
        ]
    rows: list[dict[str, object]] = []
    for idx in range(len(edges) - 1):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx == 0:
            mask = finite & (values >= lo) & (values <= hi)
        else:
            mask = finite & (values > lo) & (values <= hi)
        rows.append(
            metric_row(
                dataset,
                split,
                f"{prefix}_{idx + 1:02d}",
                mask,
                selection_metric,
                {"bin_index": idx + 1, "bin_min": lo, "bin_max": hi},
            )
        )
    return rows


def build_transition_rows(
    dataset: str,
    split: SplitPrediction,
    distances: np.ndarray,
    radii: Sequence[int],
    selection_metric: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    finite = np.isfinite(distances)
    no_transition_recording = ~finite
    if np.any(no_transition_recording):
        rows.append(
            metric_row(
                dataset,
                split,
                "recordings_without_transitions",
                no_transition_recording,
                selection_metric,
            )
        )
    for radius in radii:
        near = finite & (distances <= int(radius))
        stable = finite & (distances > int(radius))
        rows.append(metric_row(dataset, split, f"transition_boundary_pm{radius}", near, selection_metric, {"radius": int(radius)}))
        rows.append(metric_row(dataset, split, f"stable_epochs_gt_pm{radius}", stable, selection_metric, {"radius": int(radius)}))
    buckets = (
        ("dist_0", finite & (distances == 0)),
        ("dist_1", finite & (distances == 1)),
        ("dist_2", finite & (distances == 2)),
        ("dist_3_5", finite & (distances >= 3) & (distances <= 5)),
        ("dist_6_10", finite & (distances >= 6) & (distances <= 10)),
        ("dist_gt_10", finite & (distances > 10)),
    )
    for name, mask in buckets:
        rows.append(metric_row(dataset, split, name, mask, selection_metric))
    return rows


def build_time_rows(
    dataset: str,
    split: SplitPrediction,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    time_pos = normalized_time_position(split.recording_id, split.epoch_index).reshape(-1)
    return quantile_bin_rows(
        dataset,
        split,
        time_pos,
        "recording_time",
        max(1, int(args.time_bins)),
        args.selection_metric,
    )


def build_confidence_rows(
    dataset: str,
    split: SplitPrediction,
    prob_stats: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows = quantile_bin_rows(
        dataset,
        split,
        np.asarray(prob_stats["confidence"]),
        "confidence_quantile",
        max(1, int(args.confidence_bins)),
        args.selection_metric,
    )
    rows.extend(
        quantile_bin_rows(
            dataset,
            split,
            np.asarray(prob_stats["margin"]),
            "margin_quantile",
            max(1, int(args.confidence_bins)),
            args.selection_metric,
        )
    )
    return rows


def top_confusion_rows(
    dataset: str,
    split: SplitPrediction,
    prob_stats: Mapping[str, np.ndarray],
    quality: np.ndarray | None,
    distances: np.ndarray,
    limit: int,
) -> list[dict[str, object]]:
    wrong = split.y_true != split.c10a_pred
    counter: Counter[tuple[int, int]] = Counter(
        (int(true_idx), int(pred_idx))
        for true_idx, pred_idx in zip(split.y_true[wrong], split.c10a_pred[wrong])
    )
    true_support = Counter(int(item) for item in split.y_true)
    rows: list[dict[str, object]] = []
    for (true_idx, pred_idx), count in counter.most_common(int(limit)):
        mask = wrong & (split.y_true == true_idx) & (split.c10a_pred == pred_idx)
        row = {
            "dataset": dataset,
            "split": split.split,
            "true_index": true_idx,
            "true_label": label_name(split.class_labels, true_idx),
            "pred_index": pred_idx,
            "pred_label": label_name(split.class_labels, pred_idx),
            "confusion_pair": f"{label_name(split.class_labels, true_idx)}->{label_name(split.class_labels, pred_idx)}",
            "count": int(count),
            "fraction_of_all_epochs": float(count / max(len(split.y_true), 1)),
            "fraction_of_true_class": float(count / max(true_support[true_idx], 1)),
            "mean_confidence": safe_mean(prob_stats["confidence"][mask]),
            "mean_margin": safe_mean(prob_stats["margin"][mask]),
            "mean_entropy": safe_mean(prob_stats["entropy"][mask]),
            "mean_transition_distance": safe_mean(distances[mask][np.isfinite(distances[mask])]),
        }
        if quality is not None:
            row["mean_quality"] = safe_mean(quality[mask])
            row["low_quality_fraction"] = safe_mean((quality[mask] < 0.70).astype(np.float64))
        rows.append(row)
    return rows


def focused_confusion_rows(
    dataset: str,
    split: SplitPrediction,
    prob_stats: Mapping[str, np.ndarray],
    quality: np.ndarray | None,
    distances: np.ndarray,
) -> list[dict[str, object]]:
    lookup = label_lookup(split.class_labels)
    rows: list[dict[str, object]] = []
    for true_label, pred_label in CLASS_PAIR_LABELS:
        if true_label.upper() not in lookup or pred_label.upper() not in lookup:
            continue
        true_idx = lookup[true_label.upper()]
        pred_idx = lookup[pred_label.upper()]
        mask = (split.y_true == true_idx) & (split.c10a_pred == pred_idx)
        row = {
            "dataset": dataset,
            "split": split.split,
            "direction": f"{true_label}->{pred_label}",
            "count": int(np.sum(mask)),
            "mean_confidence": safe_mean(prob_stats["confidence"][mask]),
            "mean_margin": safe_mean(prob_stats["margin"][mask]),
            "mean_entropy": safe_mean(prob_stats["entropy"][mask]),
            "mean_transition_distance": safe_mean(distances[mask][np.isfinite(distances[mask])]),
        }
        if quality is not None:
            row["mean_quality"] = safe_mean(quality[mask])
            row["low_quality_fraction"] = safe_mean((quality[mask] < 0.70).astype(np.float64))
        rows.append(row)
    return rows


def recording_summary_rows(
    dataset: str,
    split: SplitPrediction,
    quality: np.ndarray | None,
    distances: np.ndarray,
    prob_stats: Mapping[str, np.ndarray],
    selection_metric: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rec in dict.fromkeys(split.recording_id.tolist()):
        mask = np.asarray(split.recording_id == rec)
        metrics = metric_bundle(
            split.y_true[mask],
            split.c10a_pred[mask],
            split.class_labels,
            selection_metric,
        )
        wrong = mask & (split.y_true != split.c10a_pred)
        focus = focus_confusion_counts(split.y_true[mask], split.c10a_pred[mask], split.class_labels)
        row: dict[str, object] = {
            "dataset": dataset,
            "split": split.split,
            "recording_id": str(rec),
            "n": int(np.sum(mask)),
            "error_count": int(np.sum(wrong)),
            "error_rate": float(np.mean(split.y_true[mask] != split.c10a_pred[mask])),
            "mean_confidence": safe_mean(prob_stats["confidence"][mask]),
            "mean_entropy": safe_mean(prob_stats["entropy"][mask]),
            "transition_fraction_pm2": safe_mean((distances[mask] <= 2).astype(np.float64)),
            **metrics,
            **focus,
        }
        if quality is not None:
            row["mean_quality"] = safe_mean(quality[mask])
            row["low_quality_fraction"] = safe_mean((quality[mask] < 0.70).astype(np.float64))
        rows.append(row)
    return sorted(rows, key=lambda row: (-float(row["error_rate"]), -int(row["n"])))


def top_error_epoch_rows(
    dataset: str,
    split: SplitPrediction,
    prob_stats: Mapping[str, np.ndarray],
    quality: np.ndarray | None,
    distances: np.ndarray,
    limit: int,
) -> list[dict[str, object]]:
    wrong_rows = np.flatnonzero(split.y_true != split.c10a_pred)
    if len(wrong_rows) == 0:
        return []
    order = wrong_rows[np.argsort(prob_stats["confidence"][wrong_rows])[::-1]]
    time_pos = normalized_time_position(split.recording_id, split.epoch_index).reshape(-1)
    rows: list[dict[str, object]] = []
    for row_idx in order[: int(limit)]:
        row: dict[str, object] = {
            "dataset": dataset,
            "split": split.split,
            "recording_id": str(split.recording_id[row_idx]),
            "epoch_index": int(split.epoch_index[row_idx]),
            "true_label": label_name(split.class_labels, int(split.y_true[row_idx])),
            "pred_label": label_name(split.class_labels, int(split.c10a_pred[row_idx])),
            "top2_label": label_name(split.class_labels, int(prob_stats["top2"][row_idx])),
            "confidence": float(prob_stats["confidence"][row_idx]),
            "top2_prob": float(prob_stats["top2_prob"][row_idx]),
            "margin": float(prob_stats["margin"][row_idx]),
            "entropy": float(prob_stats["entropy"][row_idx]),
            "recording_time_position": float(time_pos[row_idx]),
            "transition_distance": safe_float(distances[row_idx]),
        }
        if quality is not None:
            row["quality"] = safe_float(quality[row_idx])
        rows.append(row)
    return rows


def concept_rows(
    dataset: str,
    split: SplitPrediction,
) -> list[dict[str, object]]:
    if split.concepts is None:
        return []
    concept_names = (
        "EEG_Delta",
        "EEG_Theta",
        "EEG_Alpha",
        "EEG_Sigma",
        "EEG_Beta",
        "EOG_Slow",
        "EOG_Rapid",
        "EMG_Low_Tone",
        "EMG_Activation",
    )
    concepts = np.asarray(split.concepts, dtype=np.float64)
    correct = split.y_true == split.c10a_pred
    rows: list[dict[str, object]] = []
    for idx in range(concepts.shape[1]):
        name = concept_names[idx] if idx < len(concept_names) else f"concept_{idx}"
        correct_mean = safe_mean(concepts[correct, idx])
        wrong_mean = safe_mean(concepts[~correct, idx])
        rows.append(
            {
                "dataset": dataset,
                "split": split.split,
                "concept": name,
                "correct_mean": correct_mean,
                "wrong_mean": wrong_mean,
                "wrong_minus_correct": None
                if correct_mean is None or wrong_mean is None
                else float(wrong_mean - correct_mean),
            }
        )
    return rows


def markdown_report(
    dataset: str,
    split: SplitPrediction,
    metrics: Mapping[str, float],
    focus: Mapping[str, int],
    top_confusions: Sequence[Mapping[str, object]],
    quality_rows: Sequence[Mapping[str, object]],
    transition_rows: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        f"# C10 Diagnostics: {dataset} / {split.split}",
        "",
        "## Main Metrics",
        "",
        f"- accuracy: {metrics['accuracy']:.4f}",
        f"- macro-F1: {metrics['macro_f1']:.4f}",
        f"- N1-F1: {metrics['n1_f1']:.4f}",
        f"- kappa: {metrics['kappa']:.4f}",
        "",
        "## Focused Confusions",
        "",
        f"- N1/N2 confusion: {focus['n1_n2_confusion']}",
        f"- N2/N3 confusion: {focus['n2_n3_confusion']}",
        f"- REM->N1/N2 confusion: {focus['rem_to_n1_n2_confusion']}",
        "",
        "## Top Confusions",
        "",
    ]
    for row in top_confusions[:5]:
        lines.append(
            f"- {row['confusion_pair']}: {row['count']} "
            f"({100.0 * float(row['fraction_of_all_epochs']):.2f}% of epochs)"
        )
    if quality_rows:
        low_rows = [row for row in quality_rows if str(row.get("subset", "")).startswith("quality_low_lt")]
        high_rows = [row for row in quality_rows if str(row.get("subset", "")).startswith("quality_high_ge")]
        if low_rows or high_rows:
            lines.extend(["", "## Quality Pattern", ""])
        for row in low_rows[:1] + high_rows[:1]:
            acc = row.get("accuracy")
            lines.append(
                f"- {row['subset']}: n={row['n']}, "
                f"accuracy={'NA' if acc is None else f'{float(acc):.4f}'}"
            )
    boundary_rows = [row for row in transition_rows if str(row.get("subset", "")).startswith("transition_boundary_pm2")]
    stable_rows = [row for row in transition_rows if str(row.get("subset", "")).startswith("stable_epochs_gt_pm2")]
    if boundary_rows or stable_rows:
        lines.extend(["", "## Transition Pattern", ""])
        for row in boundary_rows[:1] + stable_rows[:1]:
            acc = row.get("accuracy")
            lines.append(
                f"- {row['subset']}: n={row['n']}, "
                f"accuracy={'NA' if acc is None else f'{float(acc):.4f}'}"
            )
    lines.append("")
    return "\n".join(lines)


def analyze_split(args: argparse.Namespace, split: SplitPrediction, output_dir: Path) -> dict[str, object]:
    dataset_name = safe_dataset_name(args.dataset)
    prob_stats = probability_summary(split)
    quality = quality_vector(split)
    distances, transition_event = transition_distances(split)
    radii = parse_radii(args.transition_radii)

    metrics = metric_bundle(split.y_true, split.c10a_pred, split.class_labels, args.selection_metric)
    focus = focus_confusion_counts(split.y_true, split.c10a_pred, split.class_labels)
    report_dict = classification_report(
        split.y_true,
        split.c10a_pred,
        labels=np.arange(len(split.class_labels)),
        target_names=list(split.class_labels),
        output_dict=True,
        zero_division=0,
    )

    metrics_rows = [
        {
            "dataset": args.dataset,
            "split": split.split,
            "model": "C10a",
            **metrics,
            **focus,
            "n": int(len(split.y_true)),
            "score_key": split.score_key,
            "score_kind": split.score_kind,
            "pred_key": split.pred_key or "argmax",
        }
    ]
    per_class = per_class_rows(args.dataset, "C10a", split.split, split.y_true, split.c10a_pred, split.class_labels)
    confusion = confusion_matrix_rows(args.dataset, "C10a", split.split, split.y_true, split.c10a_pred, split.class_labels)
    top_confusions = top_confusion_rows(args.dataset, split, prob_stats, quality, distances, args.top_confusions)
    focused = focused_confusion_rows(args.dataset, split, prob_stats, quality, distances)
    quality_rows = build_quality_rows(args.dataset, split, quality, args)
    transition_rows = build_transition_rows(args.dataset, split, distances, radii, args.selection_metric)
    transition_pairs = []
    if 2 in radii:
        transition_pairs = transition_pair_rows(args.dataset, split, distances, 2, args.selection_metric)
    elif radii:
        transition_pairs = transition_pair_rows(args.dataset, split, distances, radii[0], args.selection_metric)
    transition_pair_summary = aggregate_transition_pair_rows(transition_pairs)
    time_rows = build_time_rows(args.dataset, split, args)
    confidence_rows = build_confidence_rows(args.dataset, split, prob_stats, args)
    recording_rows = recording_summary_rows(args.dataset, split, quality, distances, prob_stats, args.selection_metric)
    top_errors = top_error_epoch_rows(args.dataset, split, prob_stats, quality, distances, args.top_errors)
    concepts = concept_rows(args.dataset, split)

    prefix = f"{dataset_name}_c10_{split.split}"
    paths = {
        "metrics": output_dir / f"{prefix}_metrics.csv",
        "per_class": output_dir / f"{prefix}_per_class.csv",
        "confusion_matrix": output_dir / f"{prefix}_confusion_matrix.csv",
        "top_confusions": output_dir / f"{prefix}_top_confusions.csv",
        "focused_confusions": output_dir / f"{prefix}_focused_confusions.csv",
        "quality_strata": output_dir / f"{prefix}_quality_strata.csv",
        "transition_context": output_dir / f"{prefix}_transition_context.csv",
        "transition_pair_summary": output_dir / f"{prefix}_transition_pair_summary.csv",
        "time_position": output_dir / f"{prefix}_time_position.csv",
        "confidence_bins": output_dir / f"{prefix}_confidence_bins.csv",
        "recording_summary": output_dir / f"{prefix}_recording_summary.csv",
        "top_error_epochs": output_dir / f"{prefix}_top_error_epochs.csv",
        "concept_error_gap": output_dir / f"{prefix}_concept_error_gap.csv",
        "summary": output_dir / f"{prefix}_summary.json",
        "report": output_dir / f"{prefix}_report.md",
    }

    for key, rows in (
        ("metrics", metrics_rows),
        ("per_class", per_class),
        ("confusion_matrix", confusion),
        ("top_confusions", top_confusions),
        ("focused_confusions", focused),
        ("quality_strata", quality_rows),
        ("transition_context", transition_rows),
        ("transition_pair_summary", transition_pair_summary),
        ("time_position", time_rows),
        ("confidence_bins", confidence_rows),
        ("recording_summary", recording_rows),
        ("top_error_epochs", top_errors),
        ("concept_error_gap", concepts),
    ):
        if rows:
            write_csv(paths[key], rows)

    summary = {
        "dataset": args.dataset,
        "split": split.split,
        "path": str(split.path),
        "fields": {
            "label_key": split.label_key,
            "score_key": split.score_key,
            "score_kind": split.score_kind,
            "pred_key": split.pred_key or "argmax",
            "has_quality": quality is not None,
            "has_concepts": split.concepts is not None,
        },
        "class_labels": list(split.class_labels),
        "metrics": metrics,
        "focus_confusions": focus,
        "n_transition_event_epochs": int(np.sum(transition_event)),
        "transition_event_epoch_fraction": float(np.mean(transition_event)) if len(transition_event) else 0.0,
        "top_confusions": list(top_confusions[:10]),
        "classification_report": report_dict,
        "outputs": {key: str(path) for key, path in paths.items() if path.exists() or key in {"summary", "report"}},
    }
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    paths["report"].write_text(
        markdown_report(args.dataset, split, metrics, focus, top_confusions, quality_rows, transition_rows),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split_name in split_names(args.splits):
        path = split_path(args, split_name)
        split = load_split_prediction(path, split_name, args)
        summaries.append(analyze_split(args, split, output_dir))

    if len(summaries) > 1:
        combined = {
            "dataset": args.dataset,
            "splits": [summary["split"] for summary in summaries],
            "metrics": {summary["split"]: summary["metrics"] for summary in summaries},
            "focus_confusions": {summary["split"]: summary["focus_confusions"] for summary in summaries},
            "outputs": {summary["split"]: summary["outputs"] for summary in summaries},
        }
        combined_path = output_dir / f"{safe_dataset_name(args.dataset)}_c10_diagnostics_summary.json"
        combined_path.write_text(json.dumps(json_ready(combined), indent=2), encoding="utf-8")

    print("C10 diagnostics complete")
    for summary in summaries:
        metrics = summary["metrics"]
        print(
            f"  {summary['split']}: "
            f"accuracy={metrics['accuracy']:.4f} "
            f"macro-F1={metrics['macro_f1']:.4f} "
            f"N1-F1={metrics['n1_f1']:.4f} "
            f"kappa={metrics['kappa']:.4f}"
        )
    print(f"  output_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
