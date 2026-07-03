#!/usr/bin/env python
"""Build train-split hard-example weights from split prediction probabilities.

The intended C15/B4 flow is:
1. Run A1 zero with LFTS_SAVE_SPLIT_PREDICTION_PROBS=1.
2. Mine validation confusion pairs and train uncertainty/error regions.
3. Save per-train-epoch weights for LFTS_RISK_WEIGHT_PATH.

Optional B2 rule-teacher risk NPZ files can be blended in through
--b2-risk-npz. This keeps B2 as a risk detector, not as a post-hoc corrector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_LABELS = ("W", "N1", "N2", "N3", "REM")
FOCUS_PAIRS = {(0, 1), (1, 2), (1, 4), (0, 4), (2, 4)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create hard-example train sample weights.")
    parser.add_argument("--prediction-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--b2-risk-npz", action="append", default=[])
    parser.add_argument("--min-pair-rate", type=float, default=0.04)
    parser.add_argument("--min-pair-count", type=int, default=20)
    parser.add_argument("--margin-threshold", type=float, default=0.18)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--max-weight", type=float, default=2.75)
    parser.add_argument("--min-weight", type=float, default=0.50)
    parser.add_argument("--error-weight", type=float, default=0.30)
    parser.add_argument("--uncertainty-weight", type=float, default=0.30)
    parser.add_argument("--pair-weight", type=float, default=0.22)
    parser.add_argument("--boundary-weight", type=float, default=0.10)
    parser.add_argument("--b2-weight", type=float, default=0.20)
    parser.add_argument("--class-weight-scale", type=float, default=0.35)
    return parser.parse_args()


def first_key(data: np.lib.npyio.NpzFile, keys: Iterable[str]) -> str | None:
    for key in keys:
        if key and key in data.files:
            return key
    return None


def labels_to_indices(values: np.ndarray, n_classes: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim >= 2 and arr.shape[-1] == n_classes:
        arr = np.argmax(arr, axis=-1)
    return np.asarray(arr, dtype=np.int64).reshape(-1)


def normalize_probs(values: np.ndarray) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float64)
    if probs.ndim > 2:
        probs = probs.reshape(-1, probs.shape[-1])
    probs = np.clip(probs, 0.0, None)
    return probs / np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-12)


def split_payload(data: np.lib.npyio.NpzFile, split: str) -> dict[str, np.ndarray]:
    prob_key = first_key(
        data,
        [
            f"{split}_probs_calibrated",
            f"{split}_probs_raw",
            f"{split}_probs",
            f"{split}_probabilities",
        ],
    )
    label_key = first_key(data, [f"{split}_y_true", f"{split}_labels", "y_true" if split == "test" else ""])
    pred_key = first_key(data, [f"{split}_pred", f"{split}_y_pred_final", f"{split}_pred_argmax"])
    count_key = first_key(data, [f"{split}_file_sample_counts", f"{split}_recording_sample_counts"])
    if not prob_key or not label_key:
        raise KeyError(f"Missing {split} labels/probabilities in split prediction file.")
    probs = normalize_probs(np.asarray(data[prob_key]))
    labels = labels_to_indices(np.asarray(data[label_key]), probs.shape[1])
    pred = labels_to_indices(np.asarray(data[pred_key]), probs.shape[1]) if pred_key else np.argmax(probs, axis=1)
    counts = np.asarray(data[count_key], dtype=np.int64).reshape(-1) if count_key else np.asarray([len(labels)])
    if probs.shape[0] != labels.shape[0] or pred.shape[0] != labels.shape[0]:
        raise ValueError(f"{split} arrays have inconsistent lengths.")
    if int(np.sum(counts)) != int(labels.shape[0]):
        raise ValueError(f"{split} file counts sum {int(np.sum(counts))}, expected {labels.shape[0]}.")
    return {"probs": probs, "labels": labels, "pred": pred, "counts": counts, "prob_key": prob_key}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < n_classes and 0 <= int(p) < n_classes:
            cm[int(t), int(p)] += 1
    return cm


def mine_risk_pairs(cm: np.ndarray, min_rate: float, min_count: int) -> list[tuple[int, int, float, int]]:
    pairs: list[tuple[int, int, float, int]] = []
    row_totals = np.maximum(cm.sum(axis=1), 1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            count = int(cm[i, j])
            rate = float(count / row_totals[i])
            unordered = tuple(sorted((i, j)))
            if (rate >= min_rate and count >= min_count) or unordered in FOCUS_PAIRS:
                pairs.append((i, j, rate, count))
    pairs.sort(key=lambda item: (item[2], item[3]), reverse=True)
    return pairs


def normalize01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    lo = float(np.percentile(arr, 1)) if arr.size else 0.0
    hi = float(np.percentile(arr, 99)) if arr.size else 1.0
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def boundary_mask(labels: np.ndarray, counts: np.ndarray, radius: int) -> np.ndarray:
    mask = np.zeros(labels.shape[0], dtype=bool)
    start = 0
    radius = max(0, int(radius))
    for count in counts:
        end = start + int(count)
        local = labels[start:end]
        if local.size > 1:
            changes = np.flatnonzero(local[1:] != local[:-1])
            for change in changes:
                lo = max(0, int(change) - radius + 1)
                hi = min(local.size, int(change) + radius + 2)
                mask[start + lo : start + hi] = True
        start = end
    return mask


def pair_ambiguity(labels: np.ndarray, probs: np.ndarray, risk_pairs: list[tuple[int, int, float, int]]) -> np.ndarray:
    if not risk_pairs:
        return np.zeros(labels.shape[0], dtype=np.float64)
    pair_strength: dict[tuple[int, int], float] = {}
    for i, j, rate, count in risk_pairs:
        pair_strength[(int(i), int(j))] = max(pair_strength.get((int(i), int(j)), 0.0), float(rate))
    score = np.zeros(labels.shape[0], dtype=np.float64)
    for idx, label in enumerate(labels):
        label = int(label)
        candidates = [
            (other, strength)
            for (src, other), strength in pair_strength.items()
            if src == label and 0 <= other < probs.shape[1]
        ]
        if not candidates:
            continue
        # Risk is high when probability leaks into the historically confused
        # opposite class, not merely because the pair exists for this stage.
        score[idx] = max(float(probs[idx, other]) * max(strength, 1e-6) for other, strength in candidates)
    return normalize01(score)


def load_b2_risk(paths: list[str], expected_len: int) -> np.ndarray | None:
    scores = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as data:
            key = first_key(data, ["risk_score", "train_risk_score", "score"])
            if key is None:
                raise KeyError(f"No risk_score in {path}")
            score = np.asarray(data[key], dtype=np.float64).reshape(-1)
        if score.shape[0] != expected_len:
            raise ValueError(f"B2 risk length {score.shape[0]} from {path}, expected {expected_len}")
        scores.append(normalize01(score))
    if not scores:
        return None
    return np.mean(scores, axis=0)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = Path(args.prediction_npz).expanduser().resolve()

    with np.load(pred_path, allow_pickle=True) as data:
        train = split_payload(data, "train")
        val = split_payload(data, "val")
        labels = tuple(str(item) for item in np.asarray(data["class_labels"]).reshape(-1)) if "class_labels" in data.files else DEFAULT_LABELS

    n_classes = train["probs"].shape[1]
    labels = labels[:n_classes]
    val_cm = confusion_matrix(val["labels"], val["pred"], n_classes)
    risk_pairs = mine_risk_pairs(val_cm, args.min_pair_rate, args.min_pair_count)

    val_error_by_class = []
    for cls in range(n_classes):
        rows = val["labels"] == cls
        val_error_by_class.append(float(np.mean(val["pred"][rows] != cls)) if np.any(rows) else 0.0)
    val_error_by_class = np.asarray(val_error_by_class, dtype=np.float64)
    denom = float(np.mean(val_error_by_class[val_error_by_class > 0])) if np.any(val_error_by_class > 0) else 1.0
    class_weights = 1.0 + float(args.class_weight_scale) * (val_error_by_class / max(denom, 1e-6))
    class_weights = np.clip(class_weights, 0.85, 1.75)

    probs = train["probs"]
    y = train["labels"]
    pred = train["pred"]
    p_true = probs[np.arange(len(y)), np.clip(y, 0, n_classes - 1)]
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    error_score = (pred != y).astype(np.float64)
    uncertainty_score = np.maximum(1.0 - p_true, np.clip((args.margin_threshold - margins) / max(args.margin_threshold, 1e-6), 0.0, 1.0))
    pair_score = pair_ambiguity(y, probs, risk_pairs)
    transition_score = boundary_mask(y, train["counts"], args.boundary_radius).astype(np.float64)
    b2_score = load_b2_risk(args.b2_risk_npz, len(y))

    components = (
        args.error_weight * error_score
        + args.uncertainty_weight * normalize01(uncertainty_score)
        + args.pair_weight * pair_score
        + args.boundary_weight * transition_score
    )
    total_weight = args.error_weight + args.uncertainty_weight + args.pair_weight + args.boundary_weight
    if b2_score is not None:
        components = components + args.b2_weight * b2_score
        total_weight += args.b2_weight
    risk_score = np.clip(components / max(total_weight, 1e-6), 0.0, 1.0)

    sample_weights = 1.0 + (float(args.max_weight) - 1.0) * risk_score
    sample_weights = sample_weights * class_weights[np.clip(y, 0, n_classes - 1)]
    sample_weights = np.clip(sample_weights, args.min_weight, args.max_weight)
    sample_weights = sample_weights / max(float(np.mean(sample_weights)), 1e-8)

    npz_path = out_dir / "hard_example_train_weights.npz"
    np.savez_compressed(
        npz_path,
        train_sample_weights=sample_weights.astype(np.float32),
        risk_score=risk_score.astype(np.float32),
        train_y_true=y.astype(np.int16),
        train_pred=pred.astype(np.int16),
        class_weights=class_weights.astype(np.float32),
        val_confusion_matrix=val_cm.astype(np.int64),
        risk_pairs=np.asarray(risk_pairs, dtype=np.float32),
        class_labels=np.asarray(labels, dtype=object),
        prediction_npz=np.asarray(str(pred_path)),
    )

    summary = {
        "prediction_npz": str(pred_path),
        "output_npz": str(npz_path),
        "train_rows": int(len(y)),
        "val_rows": int(len(val["labels"])),
        "prob_key": {"train": train["prob_key"], "val": val["prob_key"]},
        "class_labels": list(labels),
        "val_error_by_class": {labels[i]: float(val_error_by_class[i]) for i in range(n_classes)},
        "class_weights": {labels[i]: float(class_weights[i]) for i in range(n_classes)},
        "class_weights_env": ",".join(f"{float(v):.4f}" for v in class_weights),
        "risk_pairs": [
            {"true": labels[i], "pred": labels[j], "rate": float(rate), "count": int(count)}
            for i, j, rate, count in risk_pairs
        ],
        "weight_summary": {
            "min": float(np.min(sample_weights)),
            "mean": float(np.mean(sample_weights)),
            "max": float(np.max(sample_weights)),
            "p90": float(np.percentile(sample_weights, 90)),
            "top10_mean": float(np.mean(np.sort(sample_weights)[-max(1, len(sample_weights) // 10):])),
        },
        "component_means": {
            "error": float(np.mean(error_score)),
            "uncertainty": float(np.mean(uncertainty_score)),
            "pair": float(np.mean(pair_score)),
            "boundary": float(np.mean(transition_score)),
            "b2": None if b2_score is None else float(np.mean(b2_score)),
        },
    }
    summary_path = out_dir / "hard_example_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
