#!/usr/bin/env python
"""B3 risk-aware post-hoc corrector.

The corrector is intentionally lightweight and leakage-aware:

1. Fit a small numpy softmax model on train split features.
2. Select blend/gate/Viterbi hyperparameters on val only.
3. Report the selected configuration on test.

It is meant to test whether B2 risk signals can raise full-coverage accuracy,
not to replace the main sequence model with a hand-written rule system.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.b2_physiology_rule_teacher_disagreement import (  # noqa: E402
    DEFAULT_CLASS_LABELS,
    compute_concepts_from_split,
    compute_disagreement_risk,
    compute_rule_teacher,
    labels_to_indices,
    metric_bundle,
    normalize_probabilities,
    probability_margin,
    softmax,
    stage_prototypes_from_concepts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate B3 risk-aware post-hoc corrector.")
    parser.add_argument("--prediction-npz", required=True)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--npz-dir", default=None)
    parser.add_argument("--concept-cache-dir", default=None)
    parser.add_argument("--concept-extractor", choices=("direct_epoch", "segment"), default="direct_epoch")
    parser.add_argument("--concept-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--filter-profile", default="clinical_canonical")
    parser.add_argument("--normalization-mode", default="recording_robust_zscore")
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--no-rule-features", action="store_true")
    parser.add_argument("--prototype-source", choices=("train", "val"), default="train")
    parser.add_argument("--prototype-weight", type=float, default=0.90)
    parser.add_argument("--rule-temperature", type=float, default=0.75)
    parser.add_argument("--n1-window", type=int, default=20)
    parser.add_argument("--transition-radius", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--l2", type=float, default=2e-3)
    parser.add_argument("--class-weight-power", type=float, default=0.35)
    parser.add_argument("--risk-sample-weight", type=float, default=1.0)
    parser.add_argument("--error-sample-weight", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-npz", action="store_true", default=True)
    parser.add_argument("--no-save-npz", action="store_false", dest="save_npz")
    return parser.parse_args()


def split_key(split: str, key: str) -> str:
    return f"{split}_{key}"


def first_existing(data: np.lib.npyio.NpzFile, candidates: Sequence[str]) -> str | None:
    for key in candidates:
        if key in data.files:
            return key
    return None


def load_split_predictions(path: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        y_key = first_existing(data, [split_key(split, "y_true"), "y_true" if split == "test" else ""])
        if not y_key:
            raise KeyError(f"No labels found for split={split!r} in {path}.")
        prob_key = first_existing(
            data,
            [
                split_key(split, "probs_calibrated"),
                split_key(split, "probs_raw"),
                split_key(split, "probs"),
                "test_probs_calibrated" if split == "test" else "",
                "test_probs_raw" if split == "test" else "",
            ],
        )
        if not prob_key:
            raise KeyError(f"No probabilities found for split={split!r} in {path}.")
        logit_key = first_existing(data, [split_key(split, "final_logits"), split_key(split, "logits")])
        pred_key = first_existing(
            data,
            [
                split_key(split, "pred"),
                split_key(split, "y_pred_final"),
                split_key(split, "pred_final"),
                "y_pred_final" if split == "test" else "",
            ],
        )
        count_key = first_existing(data, [split_key(split, "file_sample_counts"), split_key(split, "recording_sample_counts")])
        rec_key = first_existing(data, [split_key(split, "recording_id"), split_key(split, "recording_ids")])
        epoch_key = first_existing(data, [split_key(split, "epoch_index"), split_key(split, "epoch_indices")])
        quality_key = first_existing(data, [split_key(split, "quality")])
        labels_key = first_existing(data, ["class_labels"])

        y_true = labels_to_indices(np.asarray(data[y_key]))
        probs = normalize_probabilities(np.asarray(data[prob_key], dtype=np.float32))
        logits = np.asarray(data[logit_key], dtype=np.float32) if logit_key else np.log(np.clip(probs, 1e-8, 1.0))
        pred = labels_to_indices(np.asarray(data[pred_key])) if pred_key else np.argmax(probs, axis=1).astype(np.int32)
        if pred.shape[0] != y_true.shape[0] or probs.shape[0] != y_true.shape[0]:
            raise ValueError(f"Split {split} arrays have inconsistent lengths.")

        if rec_key and epoch_key:
            recording_id = np.asarray(data[rec_key]).reshape(-1)
            epoch_index = np.asarray(data[epoch_key]).reshape(-1).astype(np.int32)
        else:
            counts = np.asarray(data[count_key], dtype=np.int32).reshape(-1) if count_key else np.asarray([len(y_true)])
            recording_id = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
            epoch_index = np.concatenate([np.arange(int(count), dtype=np.int32) for count in counts])
        counts = np.asarray(data[count_key], dtype=np.int32).reshape(-1) if count_key else None
        quality = np.asarray(data[quality_key], dtype=np.float32).reshape(-1) if quality_key else np.ones(len(y_true), dtype=np.float32)
        class_labels = tuple(str(item) for item in np.asarray(data[labels_key]).reshape(-1)) if labels_key else DEFAULT_CLASS_LABELS

    return {
        "y_true": y_true.astype(np.int32),
        "probs": probs.astype(np.float32),
        "logits": logits.astype(np.float32),
        "pred": pred.astype(np.int32),
        "recording_id": recording_id,
        "epoch_index": epoch_index.astype(np.int32),
        "counts": counts,
        "quality": quality.astype(np.float32),
        "class_labels": np.asarray(class_labels, dtype=object),
    }


def load_or_compute_concepts_for_split(args: argparse.Namespace, split: str, expected_counts: np.ndarray | None) -> np.ndarray | None:
    if args.no_rule_features or not args.split_json:
        return None
    cache_path: Path | None = None
    if args.concept_cache_dir:
        cache_dir = Path(args.concept_cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{args.dataset}_{split}_{args.concept_extractor}_{args.concept_version}_concepts.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=True) as data:
                key = "concepts" if "concepts" in data.files else data.files[0]
                return np.asarray(data[key], dtype=np.float32)
    concepts = compute_concepts_from_split(
        Path(args.split_json).expanduser().resolve(),
        args.npz_dir,
        split,
        expected_counts,
        args.concept_version,
        args.concept_extractor,
        preprocess=not bool(args.no_preprocess),
        filter_profile=args.filter_profile,
        normalization_mode=args.normalization_mode,
    )
    if cache_path is not None:
        np.savez_compressed(cache_path, concepts=np.asarray(concepts, dtype=np.float32))
        print(f"Saved B3 concept cache: {cache_path}")
    return np.asarray(concepts, dtype=np.float32)


def sequence_context(probs: np.ndarray, recording_id: np.ndarray) -> np.ndarray:
    probs = normalize_probabilities(probs)
    n_rows, n_classes = probs.shape
    prev1 = np.zeros_like(probs)
    next1 = np.zeros_like(probs)
    prev2 = np.zeros_like(probs)
    next2 = np.zeros_like(probs)
    mean3 = np.zeros_like(probs)
    mean5 = np.zeros_like(probs)
    for rec in np.unique(recording_id):
        idx = np.flatnonzero(recording_id == rec)
        if idx.size == 0:
            continue
        rec_probs = probs[idx]
        prev1[idx] = np.vstack([rec_probs[:1], rec_probs[:-1]])
        next1[idx] = np.vstack([rec_probs[1:], rec_probs[-1:]])
        prev2[idx] = np.vstack([rec_probs[:1], rec_probs[:1], rec_probs[:-2]])[: idx.size]
        next2[idx] = np.vstack([rec_probs[2:], rec_probs[-1:], rec_probs[-1:]])[: idx.size]
        for local_pos in range(idx.size):
            lo3 = max(0, local_pos - 1)
            hi3 = min(idx.size, local_pos + 2)
            lo5 = max(0, local_pos - 2)
            hi5 = min(idx.size, local_pos + 3)
            mean3[idx[local_pos]] = np.mean(rec_probs[lo3:hi3], axis=0)
            mean5[idx[local_pos]] = np.mean(rec_probs[lo5:hi5], axis=0)
    return np.concatenate(
        [
            prev1,
            next1,
            prev2,
            next2,
            mean3,
            mean5,
            probs - prev1,
            probs - next1,
        ],
        axis=1,
    ).astype(np.float32)


def entropy(probs: np.ndarray) -> np.ndarray:
    probs = normalize_probabilities(probs)
    return (-np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1) / np.log(probs.shape[1])).astype(np.float32)


def one_hot(values: np.ndarray, n_classes: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int32).reshape(-1)
    out = np.zeros((values.shape[0], n_classes), dtype=np.float32)
    mask = (values >= 0) & (values < n_classes)
    out[np.flatnonzero(mask), values[mask]] = 1.0
    return out


def build_feature_matrix(
    split_data: Mapping[str, np.ndarray],
    rule_result: Mapping[str, np.ndarray] | None,
    n_classes: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    probs = normalize_probabilities(np.asarray(split_data["probs"], dtype=np.float32))
    logits = np.asarray(split_data["logits"], dtype=np.float32)
    model_pred = np.asarray(split_data["pred"], dtype=np.int32)
    recording_id = np.asarray(split_data["recording_id"]).reshape(-1)
    model_margin = probability_margin(probs)
    model_conf = np.max(probs, axis=1).astype(np.float32)
    base_features = [
        probs,
        np.log(np.clip(probs, 1e-8, 1.0)),
        logits,
        sequence_context(probs, recording_id),
        one_hot(model_pred, n_classes),
        model_conf[:, None],
        model_margin[:, None],
        entropy(probs)[:, None],
        np.asarray(split_data["quality"], dtype=np.float32).reshape(-1, 1),
    ]
    aux: dict[str, np.ndarray] = {
        "model_conf": model_conf,
        "model_margin": model_margin,
        "risk_score": np.zeros(probs.shape[0], dtype=np.float32),
    }
    if rule_result is not None:
        p_rule = normalize_probabilities(rule_result["p_rule"])
        risk = compute_disagreement_risk(
            probs,
            p_rule,
            model_pred,
            np.asarray(rule_result["rule_pred"], dtype=np.int32),
            np.asarray(rule_result["rule_conf"], dtype=np.float32),
        )
        base_features.extend(
            [
                p_rule,
                np.log(np.clip(p_rule, 1e-8, 1.0)),
                np.asarray(rule_result["concepts_scaled"], dtype=np.float32),
                np.asarray(rule_result["rule_scores"], dtype=np.float32),
                np.asarray(rule_result["clinical_rule_scores"], dtype=np.float32),
                one_hot(np.asarray(rule_result["rule_pred"], dtype=np.int32), n_classes),
                np.asarray(rule_result["rule_conf"], dtype=np.float32).reshape(-1, 1),
                np.asarray(rule_result["base_conf"], dtype=np.float32).reshape(-1, 1),
                np.asarray(rule_result["residual"], dtype=np.float32).reshape(-1, 1),
                np.asarray(rule_result["transition_context"], dtype=np.float32).reshape(-1, 1),
                np.asarray(rule_result["n1_residual_hint"], dtype=np.float32).reshape(-1, 1),
                risk["risk_score"].reshape(-1, 1),
                risk["model_uncertainty"].reshape(-1, 1),
                risk["rule_margin"].reshape(-1, 1),
                risk["prob_l1"].reshape(-1, 1),
                risk["prob_js"].reshape(-1, 1),
            ]
        )
        aux.update(risk)
    return np.nan_to_num(np.concatenate(base_features, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), aux


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def standardize_transform(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - center) / scale).astype(np.float32)


def class_weights(y_true: np.ndarray, n_classes: int, power: float) -> np.ndarray:
    counts = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    weights = (np.sum(counts) / np.maximum(counts * n_classes, 1.0)) ** float(power)
    return np.clip(weights, 0.25, 4.0).astype(np.float32)


def train_softmax_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    train_base_pred: np.ndarray,
    train_risk_score: np.ndarray,
    args: argparse.Namespace,
    n_classes: int,
) -> dict[str, np.ndarray | float | int | list[dict[str, float]]]:
    rng = np.random.default_rng(int(args.seed))
    n_features = x_train.shape[1]
    weights = rng.normal(0.0, 0.01, size=(n_features, n_classes)).astype(np.float32)
    bias = np.zeros(n_classes, dtype=np.float32)
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = np.zeros_like(bias)
    v_b = np.zeros_like(bias)
    onehot = one_hot(y_train, n_classes)
    cw = class_weights(y_train, n_classes, args.class_weight_power)
    sample_weight = cw[y_train].astype(np.float32)
    sample_weight *= 1.0 + float(args.risk_sample_weight) * np.asarray(train_risk_score, dtype=np.float32)
    sample_weight *= 1.0 + float(args.error_sample_weight) * (np.asarray(train_base_pred) != y_train).astype(np.float32)
    sample_weight = sample_weight / max(float(np.mean(sample_weight)), 1e-6)
    history: list[dict[str, float]] = []
    best = {
        "weights": weights.copy(),
        "bias": bias.copy(),
        "val_acc": 0.0,
        "epoch": 0,
    }
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    for epoch in range(1, int(args.epochs) + 1):
        order = rng.permutation(x_train.shape[0])
        batch_size = max(256, int(args.batch_size))
        for start_idx in range(0, x_train.shape[0], batch_size):
            step += 1
            batch_idx = order[start_idx : start_idx + batch_size]
            xb = x_train[batch_idx]
            yb = onehot[batch_idx]
            wb = sample_weight[batch_idx]
            logits = xb @ weights + bias
            pred_probs = softmax(logits, temperature=1.0)
            err = (pred_probs - yb) * wb[:, None]
            grad_w = (xb.T @ err) / max(xb.shape[0], 1) + float(args.l2) * weights
            grad_b = np.mean(err, axis=0)
            m_w = beta1 * m_w + (1.0 - beta1) * grad_w
            v_w = beta2 * v_w + (1.0 - beta2) * (grad_w**2)
            m_b = beta1 * m_b + (1.0 - beta1) * grad_b
            v_b = beta2 * v_b + (1.0 - beta2) * (grad_b**2)
            m_w_hat = m_w / (1.0 - beta1**step)
            v_w_hat = v_w / (1.0 - beta2**step)
            m_b_hat = m_b / (1.0 - beta1**step)
            v_b_hat = v_b / (1.0 - beta2**step)
            weights -= float(args.lr) * m_w_hat / (np.sqrt(v_w_hat) + eps)
            bias -= float(args.lr) * m_b_hat / (np.sqrt(v_b_hat) + eps)

        if epoch == 1 or epoch % max(1, int(args.log_every)) == 0 or epoch == int(args.epochs):
            pred_probs = softmax(x_train @ weights + bias, temperature=1.0)
            train_acc = float(np.mean(np.argmax(pred_probs, axis=1) == y_train))
            val_probs = softmax(x_val @ weights + bias)
            val_acc = float(np.mean(np.argmax(val_probs, axis=1) == y_val))
            history.append({"epoch": float(epoch), "train_acc": train_acc, "val_acc": val_acc})
            print(f"B3 train epoch={epoch} train_acc={train_acc:.4f} val_meta_acc={val_acc:.4f}")
            if val_acc >= float(best["val_acc"]):
                best = {
                    "weights": weights.copy(),
                    "bias": bias.copy(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                }
    best["history"] = history
    return best


def transition_log_matrix(y_true: np.ndarray, recording_id: np.ndarray, n_classes: int, smoothing: float = 0.5) -> np.ndarray:
    counts = np.full((n_classes, n_classes), float(smoothing), dtype=np.float64)
    for rec in np.unique(recording_id):
        idx = np.flatnonzero(recording_id == rec)
        rec_y = y_true[idx].astype(np.int32)
        for left, right in zip(rec_y[:-1], rec_y[1:]):
            if 0 <= left < n_classes and 0 <= right < n_classes:
                counts[left, right] += 1.0
    probs = counts / np.maximum(np.sum(counts, axis=1, keepdims=True), 1e-12)
    return np.log(np.clip(probs, 1e-8, 1.0)).astype(np.float32)


def viterbi_decode_recording(emission_log: np.ndarray, transition_log: np.ndarray, transition_weight: float) -> np.ndarray:
    n_steps, n_classes = emission_log.shape
    if n_steps == 0:
        return np.zeros(0, dtype=np.int32)
    dp = np.zeros((n_steps, n_classes), dtype=np.float32)
    back = np.zeros((n_steps, n_classes), dtype=np.int16)
    dp[0] = emission_log[0]
    weighted_transition = float(transition_weight) * transition_log
    for t in range(1, n_steps):
        scores = dp[t - 1][:, None] + weighted_transition
        back[t] = np.argmax(scores, axis=0).astype(np.int16)
        dp[t] = np.max(scores, axis=0) + emission_log[t]
    out = np.zeros(n_steps, dtype=np.int32)
    out[-1] = int(np.argmax(dp[-1]))
    for t in range(n_steps - 2, -1, -1):
        out[t] = int(back[t + 1, out[t + 1]])
    return out


def viterbi_decode(
    probs: np.ndarray,
    recording_id: np.ndarray,
    transition_log: np.ndarray,
    transition_weight: float,
) -> np.ndarray:
    if float(transition_weight) <= 0.0:
        return np.argmax(probs, axis=1).astype(np.int32)
    preds = np.zeros(probs.shape[0], dtype=np.int32)
    emissions = np.log(np.clip(normalize_probabilities(probs), 1e-8, 1.0)).astype(np.float32)
    for rec in np.unique(recording_id):
        idx = np.flatnonzero(recording_id == rec)
        preds[idx] = viterbi_decode_recording(emissions[idx], transition_log, float(transition_weight))
    return preds


def blend_probs(base_probs: np.ndarray, meta_probs: np.ndarray, alpha: float, gate: np.ndarray) -> np.ndarray:
    gate = np.asarray(gate, dtype=np.float32).reshape(-1, 1)
    mixed = (1.0 - float(alpha) * gate) * base_probs + float(alpha) * gate * meta_probs
    return normalize_probabilities(mixed).astype(np.float32)


def candidate_gates(aux: Mapping[str, np.ndarray], base_pred: np.ndarray, rule_pred: np.ndarray | None) -> list[dict[str, object]]:
    margin = np.asarray(aux["model_margin"], dtype=np.float32)
    risk_score = np.asarray(aux.get("risk_score", np.zeros_like(margin)), dtype=np.float32)
    gates: list[dict[str, object]] = [{"name": "all", "gate": np.ones_like(margin, dtype=np.float32), "threshold": None}]
    for q in (5, 10, 15, 20, 30, 40, 50):
        threshold = float(np.percentile(margin, q))
        gates.append(
            {
                "name": f"margin_le_p{q}",
                "gate": (margin <= threshold).astype(np.float32),
                "threshold": threshold,
            }
        )
    if np.max(risk_score) > 0.0:
        for q in (50, 60, 70, 80, 90, 95):
            threshold = float(np.percentile(risk_score, q))
            gates.append(
                {
                    "name": f"risk_ge_p{q}",
                    "gate": (risk_score >= threshold).astype(np.float32),
                    "threshold": threshold,
                }
            )
    if rule_pred is not None:
        gates.append(
            {
                "name": "model_rule_disagree",
                "gate": (np.asarray(base_pred).reshape(-1) != np.asarray(rule_pred).reshape(-1)).astype(np.float32),
                "threshold": None,
            }
        )
    return gates


def apply_gate_by_spec(aux: Mapping[str, np.ndarray], base_pred: np.ndarray, rule_pred: np.ndarray | None, spec: Mapping[str, object]) -> np.ndarray:
    name = str(spec.get("name", spec.get("gate_name", "all")))
    margin = np.asarray(aux["model_margin"], dtype=np.float32)
    risk_score = np.asarray(aux.get("risk_score", np.zeros_like(margin)), dtype=np.float32)
    threshold = spec.get("threshold")
    if name == "all":
        return np.ones_like(margin, dtype=np.float32)
    if name.startswith("margin_le"):
        return (margin <= float(threshold)).astype(np.float32)
    if name.startswith("risk_ge"):
        return (risk_score >= float(threshold)).astype(np.float32)
    if name == "model_rule_disagree" and rule_pred is not None:
        return (np.asarray(base_pred).reshape(-1) != np.asarray(rule_pred).reshape(-1)).astype(np.float32)
    return np.zeros_like(margin, dtype=np.float32)


def evaluate_grid(
    y_val: np.ndarray,
    base_probs: np.ndarray,
    meta_probs: np.ndarray,
    aux_val: Mapping[str, np.ndarray],
    base_pred: np.ndarray,
    val_recording_id: np.ndarray,
    transition_log: np.ndarray,
    rule_pred: np.ndarray | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]
    transition_weights = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.75]
    rows: list[dict[str, object]] = []
    best: dict[str, object] = {"accuracy": -1.0}
    no_transition_candidates: list[tuple[float, dict[str, object], float]] = []
    for gate_spec in candidate_gates(aux_val, base_pred, rule_pred):
        gate = np.asarray(gate_spec["gate"], dtype=np.float32)
        for alpha in alphas:
            probs = blend_probs(base_probs, meta_probs, alpha, gate)
            pred = np.argmax(probs, axis=1).astype(np.int32)
            acc = float(np.mean(pred == y_val))
            row = {
                "accuracy": acc,
                "alpha": float(alpha),
                "gate_name": str(gate_spec["name"]),
                "threshold": gate_spec.get("threshold"),
                "gate_rate": float(np.mean(gate)),
                "transition_weight": 0.0,
            }
            rows.append(row)
            no_transition_candidates.append((acc, dict(gate_spec), float(alpha)))
            if acc > float(best["accuracy"]):
                best = dict(row)
    top_candidates = sorted(no_transition_candidates, key=lambda item: item[0], reverse=True)[:10]
    for _acc, gate_spec, alpha in top_candidates:
        gate = apply_gate_by_spec(aux_val, base_pred, rule_pred, gate_spec)
        probs = blend_probs(base_probs, meta_probs, alpha, gate)
        for transition_weight in transition_weights[1:]:
            pred = viterbi_decode(probs, val_recording_id, transition_log, transition_weight)
            acc = float(np.mean(pred == y_val))
            row = {
                "accuracy": acc,
                "alpha": float(alpha),
                "gate_name": str(gate_spec["name"]),
                "threshold": gate_spec.get("threshold"),
                "gate_rate": float(np.mean(gate)),
                "transition_weight": float(transition_weight),
            }
            rows.append(row)
            if acc > float(best["accuracy"]):
                best = dict(row)
    rows = sorted(rows, key=lambda item: float(item["accuracy"]), reverse=True)
    return best, rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_split(y_true: np.ndarray, pred: np.ndarray, class_labels: Sequence[str]) -> dict[str, object]:
    return metric_bundle(y_true, pred, class_labels)


def main() -> None:
    args = parse_args()
    pred_path = Path(args.prediction_npz).expanduser().resolve()
    train = load_split_predictions(pred_path, args.train_split)
    val = load_split_predictions(pred_path, args.val_split)
    test = load_split_predictions(pred_path, args.test_split)
    class_labels = tuple(str(item) for item in np.asarray(train["class_labels"]).reshape(-1))
    n_classes = len(class_labels)

    concepts: dict[str, np.ndarray | None] = {}
    for split_name, split_data in ((args.train_split, train), (args.val_split, val), (args.test_split, test)):
        concepts[split_name] = load_or_compute_concepts_for_split(args, split_name, split_data.get("counts"))
        if concepts[split_name] is not None and concepts[split_name].shape[0] != split_data["y_true"].shape[0]:
            raise ValueError(f"Concept count mismatch for {split_name}.")

    stage_prototypes = None
    if concepts.get(args.prototype_source) is not None:
        source = train if args.prototype_source == args.train_split else val
        source_concepts = concepts[args.prototype_source]
        assert source_concepts is not None
        stage_prototypes = stage_prototypes_from_concepts(
            source_concepts,
            source["y_true"],
            source["recording_id"],
            adaptive_concepts=True,
            num_classes=n_classes,
        )

    rule_results: dict[str, Mapping[str, np.ndarray] | None] = {}
    for split_name, split_data in ((args.train_split, train), (args.val_split, val), (args.test_split, test)):
        split_concepts = concepts.get(split_name)
        if split_concepts is None:
            rule_results[split_name] = None
            continue
        rule_results[split_name] = compute_rule_teacher(
            split_concepts,
            split_data["recording_id"],
            adaptive_concepts=True,
            rule_temperature=args.rule_temperature,
            n1_window=args.n1_window,
            transition_radius=args.transition_radius,
            stage_prototypes=stage_prototypes,
            prototype_weight=args.prototype_weight,
        )

    x_train_raw, aux_train = build_feature_matrix(train, rule_results[args.train_split], n_classes)
    x_val_raw, aux_val = build_feature_matrix(val, rule_results[args.val_split], n_classes)
    x_test_raw, aux_test = build_feature_matrix(test, rule_results[args.test_split], n_classes)
    center, scale = standardize_fit(x_train_raw)
    x_train = standardize_transform(x_train_raw, center, scale)
    x_val = standardize_transform(x_val_raw, center, scale)
    x_test = standardize_transform(x_test_raw, center, scale)

    model = train_softmax_model(
        x_train,
        train["y_true"],
        x_val,
        val["y_true"],
        train["pred"],
        aux_train["risk_score"],
        args,
        n_classes,
    )
    weights = np.asarray(model["weights"], dtype=np.float32)
    bias = np.asarray(model["bias"], dtype=np.float32)
    train_meta_probs = softmax(x_train @ weights + bias)
    val_meta_probs = softmax(x_val @ weights + bias)
    test_meta_probs = softmax(x_test @ weights + bias)

    transition_log = transition_log_matrix(train["y_true"], train["recording_id"], n_classes)
    val_rule_pred = (
        np.asarray(rule_results[args.val_split]["rule_pred"], dtype=np.int32)
        if rule_results[args.val_split] is not None
        else None
    )
    best, grid_rows = evaluate_grid(
        val["y_true"],
        val["probs"],
        val_meta_probs,
        aux_val,
        val["pred"],
        val["recording_id"],
        transition_log,
        val_rule_pred,
    )

    test_rule_pred = (
        np.asarray(rule_results[args.test_split]["rule_pred"], dtype=np.int32)
        if rule_results[args.test_split] is not None
        else None
    )
    test_gate = apply_gate_by_spec(aux_test, test["pred"], test_rule_pred, best)
    test_probs_b3 = blend_probs(test["probs"], test_meta_probs, float(best["alpha"]), test_gate)
    test_pred_b3 = viterbi_decode(test_probs_b3, test["recording_id"], transition_log, float(best["transition_weight"]))
    val_gate = apply_gate_by_spec(aux_val, val["pred"], val_rule_pred, best)
    val_probs_b3 = blend_probs(val["probs"], val_meta_probs, float(best["alpha"]), val_gate)
    val_pred_b3 = viterbi_decode(val_probs_b3, val["recording_id"], transition_log, float(best["transition_weight"]))

    test_base_pred = np.asarray(test["pred"], dtype=np.int32)
    val_base_pred = np.asarray(val["pred"], dtype=np.int32)
    summary = {
        "dataset": args.dataset,
        "prediction_npz": str(pred_path),
        "class_labels": list(class_labels),
        "n_features": int(x_train.shape[1]),
        "train_n": int(len(train["y_true"])),
        "val_n": int(len(val["y_true"])),
        "test_n": int(len(test["y_true"])),
        "base_val_accuracy": float(np.mean(val_base_pred == val["y_true"])),
        "base_test_accuracy": float(np.mean(test_base_pred == test["y_true"])),
        "meta_val_accuracy": float(np.mean(np.argmax(val_meta_probs, axis=1) == val["y_true"])),
        "meta_test_accuracy": float(np.mean(np.argmax(test_meta_probs, axis=1) == test["y_true"])),
        "b3_val_accuracy": float(np.mean(val_pred_b3 == val["y_true"])),
        "b3_test_accuracy": float(np.mean(test_pred_b3 == test["y_true"])),
        "test_accuracy_delta": float(np.mean(test_pred_b3 == test["y_true"]) - np.mean(test_base_pred == test["y_true"])),
        "selected": best,
        "train_meta_best_epoch": int(model["epoch"]),
        "train_meta_best_val_acc": float(model["val_acc"]),
        "base_val_metrics": summarize_split(val["y_true"], val_base_pred, class_labels),
        "b3_val_metrics": summarize_split(val["y_true"], val_pred_b3, class_labels),
        "base_test_metrics": summarize_split(test["y_true"], test_base_pred, class_labels),
        "b3_test_metrics": summarize_split(test["y_true"], test_pred_b3, class_labels),
        "rule_features_enabled": bool(rule_results[args.test_split] is not None),
        "prototype_source": args.prototype_source if stage_prototypes is not None else "none",
        "history": model["history"],
    }

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "diagnostics" / "b3_corrector" / pred_path.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_dir / "grid_search.csv", grid_rows)
    if args.save_npz:
        np.savez_compressed(
            out_dir / "b3_outputs.npz",
            val_probs_b3=val_probs_b3.astype(np.float32),
            test_probs_b3=test_probs_b3.astype(np.float32),
            val_pred_b3=val_pred_b3.astype(np.int16),
            test_pred_b3=test_pred_b3.astype(np.int16),
            val_meta_probs=val_meta_probs.astype(np.float32),
            test_meta_probs=test_meta_probs.astype(np.float32),
            test_gate=test_gate.astype(np.float32),
            center=center.astype(np.float32),
            scale=scale.astype(np.float32),
            weights=weights.astype(np.float32),
            bias=bias.astype(np.float32),
        )
    print(f"Saved B3 summary: {out_dir / 'summary.json'}")
    print(
        "B3 headline | "
        f"base_val={summary['base_val_accuracy']:.4f} "
        f"b3_val={summary['b3_val_accuracy']:.4f} "
        f"base_test={summary['base_test_accuracy']:.4f} "
        f"b3_test={summary['b3_test_accuracy']:.4f} "
        f"delta={summary['test_accuracy_delta']:.4f} "
        f"selected={summary['selected']}"
    )


if __name__ == "__main__":
    main()
