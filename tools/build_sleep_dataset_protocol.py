#!/usr/bin/env python
"""Build dataset-specific sleep-staging protocol artifacts.

This tool is intentionally separate from run_main.py, src/data_processing.py,
and src/model_training.py. It reads an NPZ epoch directory and writes a
dataset-specific protocol folder containing:

* manifest.csv: one row per recording/subject profile.
* split.json: a fixed subject-level train/val/test split.
* protocol_report.json: split diagnostics and recommended protocol metadata.
* epoch_metadata/*.npz: optional per-recording quality weights and masks.

Expected NPZ keys match the training pipeline: x, y, and optional fs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_processing import (  # noqa: E402
    compute_epoch_channel_quality_scores,
    compute_epoch_quality_scores,
    infer_subject_id_from_npz,
)


CLASS_NAMES = ("W", "N1", "N2", "N3", "REM")
TRANSITION_NAMES = (
    "W_N1",
    "N1_N2",
    "N2_N3",
    "N2_REM",
    "W_REM",
    "N1_REM",
    "other",
)


@dataclass
class RecordingProfile:
    dataset: str
    file: str
    subject: str
    epochs: int
    fs: float
    sleep_period_start: int
    sleep_period_end: int
    sleep_period_epochs: int
    pre_sleep_w_epochs: int
    post_sleep_w_epochs: int
    transition_epochs: int
    transition_ratio: float
    mean_q: float
    min_q: float
    low_q_ratio: float
    severe_q_ratio: float
    flagged_ratio: float
    quality_weight_mean: float
    quality_weight_min: float
    channel_quality_mean: float
    channel_quality_min: float
    W: int
    N1: int
    N2: int
    N3: int
    REM: int
    W_ratio: float
    N1_ratio: float
    N2_ratio: float
    N3_ratio: float
    REM_ratio: float
    W_N1: int
    N1_N2: int
    N2_N3: int
    N2_REM: int
    W_REM: int
    N1_REM: int
    other: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed subject-level split and quality protocol artifacts for sleep NPZ data."
    )
    parser.add_argument("--dataset-name", required=True, help="Name used in output files, e.g. ISRUC.")
    parser.add_argument("--npz-dir", required=True, help="Directory containing per-recording .npz files.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "dataset_protocols"),
        help="Root output directory. A dataset-name subfolder is created inside it.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--search-iters",
        type=int,
        default=6000,
        help="Random candidate subject splits to score.",
    )
    parser.add_argument(
        "--sleep-margin-minutes",
        type=float,
        default=30.0,
        help="Keep this margin before first non-W and after last non-W for sleep-period masks.",
    )
    parser.add_argument(
        "--epoch-seconds",
        type=float,
        default=30.0,
        help="Epoch duration used for sleep-period margin conversion.",
    )
    parser.add_argument(
        "--save-epoch-metadata",
        action="store_true",
        help="Write per-recording quality weights and masks to epoch_metadata/*.npz.",
    )
    parser.add_argument(
        "--quality-low-threshold",
        type=float,
        default=0.70,
        help="Q below this threshold is treated as low quality.",
    )
    parser.add_argument(
        "--quality-severe-threshold",
        type=float,
        default=0.40,
        help="Q below this threshold is treated as severe quality.",
    )
    return parser.parse_args()


def load_labels_and_signal(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    with np.load(npz_path, mmap_mode="r") as data:
        if "x" not in data.files or "y" not in data.files:
            raise KeyError(f"{npz_path} must contain keys 'x' and 'y'.")
        labels = np.asarray(data["y"], dtype=np.int32).reshape(-1)
        signal = np.asarray(data["x"], dtype=np.float32)
        fs = float(data["fs"]) if "fs" in data.files else 100.0
    if signal.ndim == 2:
        signal = signal[..., np.newaxis]
    if signal.ndim != 3:
        raise ValueError(f"{npz_path} x must have shape (epochs, samples, channels), got {signal.shape}.")
    if signal.shape[0] != labels.shape[0]:
        raise ValueError(f"{npz_path} has {signal.shape[0]} x epochs but {labels.shape[0]} labels.")
    return labels, signal, fs


def transition_category(a: int, b: int) -> str:
    pair = frozenset((int(a), int(b)))
    if pair == frozenset((0, 1)):
        return "W_N1"
    if pair == frozenset((1, 2)):
        return "N1_N2"
    if pair == frozenset((2, 3)):
        return "N2_N3"
    if pair == frozenset((2, 4)):
        return "N2_REM"
    if pair == frozenset((0, 4)):
        return "W_REM"
    if pair == frozenset((1, 4)):
        return "N1_REM"
    return "other"


def transition_counts(labels: np.ndarray) -> Dict[str, int]:
    counts = {name: 0 for name in TRANSITION_NAMES}
    if labels.size < 2:
        return counts
    change_idx = np.where(labels[1:] != labels[:-1])[0]
    for idx in change_idx:
        counts[transition_category(int(labels[idx]), int(labels[idx + 1]))] += 1
    return counts


def transition_epoch_mask(labels: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = np.zeros(labels.shape[0], dtype=bool)
    if labels.size < 2:
        return mask
    change_positions = np.where(labels[1:] != labels[:-1])[0]
    for pos in change_positions:
        start = max(0, int(pos) - radius + 1)
        end = min(labels.shape[0], int(pos) + radius + 2)
        mask[start:end] = True
    return mask


def sleep_period_bounds(labels: np.ndarray, margin_epochs: int) -> Tuple[int, int]:
    non_w = np.where(labels != 0)[0]
    if non_w.size == 0:
        return 0, int(labels.size)
    start = max(0, int(non_w[0]) - margin_epochs)
    end = min(int(labels.size), int(non_w[-1]) + margin_epochs + 1)
    return start, end


def quality_to_training_weight(quality: np.ndarray, low_threshold: float, severe_threshold: float) -> np.ndarray:
    quality = np.asarray(quality, dtype=np.float32)
    weights = np.ones_like(quality, dtype=np.float32)
    low_mask = quality < float(low_threshold)
    severe_mask = quality < float(severe_threshold)
    if low_threshold > severe_threshold:
        span = max(float(low_threshold) - float(severe_threshold), 1e-6)
        weights[low_mask] = 0.35 + 0.45 * np.clip(
            (quality[low_mask] - float(severe_threshold)) / span,
            0.0,
            1.0,
        )
    weights[severe_mask] = 0.15
    return np.clip(weights, 0.10, 1.0).astype(np.float32)


def profile_recording(
    dataset_name: str,
    npz_path: Path,
    margin_epochs: int,
    low_q: float,
    severe_q: float,
    save_epoch_metadata_dir: Optional[Path] = None,
) -> RecordingProfile:
    labels, signal, fs = load_labels_and_signal(npz_path)
    quality, flags, quality_report = compute_epoch_quality_scores(
        signal,
        fs=fs,
        return_flags=True,
        return_report=True,
    )
    channel_quality = compute_epoch_channel_quality_scores(signal, fs=fs)
    q_weight = quality_to_training_weight(quality, low_q, severe_q)
    start, end = sleep_period_bounds(labels, margin_epochs=margin_epochs)
    sleep_mask = np.zeros(labels.shape[0], dtype=bool)
    sleep_mask[start:end] = True
    boundary_mask = transition_epoch_mask(labels, radius=1)

    if save_epoch_metadata_dir is not None:
        save_epoch_metadata_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_epoch_metadata_dir / f"{npz_path.stem}_epoch_protocol.npz",
            quality=quality.astype(np.float32),
            quality_weight=q_weight.astype(np.float32),
            channel_quality=channel_quality.astype(np.float32),
            sleep_period_mask=sleep_mask,
            boundary_mask=boundary_mask,
            labels=labels.astype(np.int32),
        )

    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.int64)
    total = int(labels.shape[0])
    ratios = counts / max(total, 1)
    trans_counts = transition_counts(labels)
    transition_events = int(sum(trans_counts.values()))
    channel_quality_flat = np.asarray(channel_quality, dtype=np.float32).reshape(-1)

    return RecordingProfile(
        dataset=dataset_name,
        file=npz_path.name,
        subject=infer_subject_id_from_npz(npz_path.name),
        epochs=total,
        fs=float(fs),
        sleep_period_start=int(start),
        sleep_period_end=int(end),
        sleep_period_epochs=int(end - start),
        pre_sleep_w_epochs=int(start),
        post_sleep_w_epochs=int(total - end),
        transition_epochs=int(np.sum(boundary_mask)),
        transition_ratio=float(np.mean(boundary_mask)) if total else 0.0,
        mean_q=float(np.mean(quality)) if total else 1.0,
        min_q=float(np.min(quality)) if total else 1.0,
        low_q_ratio=float(np.mean(quality < low_q)) if total else 0.0,
        severe_q_ratio=float(np.mean(quality < severe_q)) if total else 0.0,
        flagged_ratio=float(quality_report["flagged_ratio"]),
        quality_weight_mean=float(np.mean(q_weight)) if total else 1.0,
        quality_weight_min=float(np.min(q_weight)) if total else 1.0,
        channel_quality_mean=float(np.mean(channel_quality_flat)) if channel_quality_flat.size else 1.0,
        channel_quality_min=float(np.min(channel_quality_flat)) if channel_quality_flat.size else 1.0,
        W=int(counts[0]),
        N1=int(counts[1]),
        N2=int(counts[2]),
        N3=int(counts[3]),
        REM=int(counts[4]),
        W_ratio=float(ratios[0]),
        N1_ratio=float(ratios[1]),
        N2_ratio=float(ratios[2]),
        N3_ratio=float(ratios[3]),
        REM_ratio=float(ratios[4]),
        W_N1=int(trans_counts["W_N1"]),
        N1_N2=int(trans_counts["N1_N2"]),
        N2_N3=int(trans_counts["N2_N3"]),
        N2_REM=int(trans_counts["N2_REM"]),
        W_REM=int(trans_counts["W_REM"]),
        N1_REM=int(trans_counts["N1_REM"]),
        other=int(trans_counts["other"]),
    )


def write_manifest(rows: Sequence[RecordingProfile], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def aggregate_profiles(rows: Sequence[RecordingProfile]) -> Dict[str, Any]:
    epochs = int(sum(row.epochs for row in rows))
    class_counts = np.array(
        [
            sum(row.W for row in rows),
            sum(row.N1 for row in rows),
            sum(row.N2 for row in rows),
            sum(row.N3 for row in rows),
            sum(row.REM for row in rows),
        ],
        dtype=np.float64,
    )
    transition_counts_vec = np.array(
        [sum(getattr(row, name) for row in rows) for name in TRANSITION_NAMES],
        dtype=np.float64,
    )
    transition_epoch_count = int(sum(row.transition_epochs for row in rows))
    return {
        "subjects": int(len({row.subject for row in rows})),
        "files": int(len(rows)),
        "epochs": epochs,
        "class_counts": {name: int(class_counts[idx]) for idx, name in enumerate(CLASS_NAMES)},
        "class_ratio": (class_counts / max(float(np.sum(class_counts)), 1.0)).round(6).tolist(),
        "transition_counts": {
            name: int(transition_counts_vec[idx]) for idx, name in enumerate(TRANSITION_NAMES)
        },
        "transition_ratio": float(transition_epoch_count / max(epochs, 1)),
        "mean_q": weighted_mean((row.mean_q, row.epochs) for row in rows),
        "low_q_ratio": weighted_mean((row.low_q_ratio, row.epochs) for row in rows),
        "severe_q_ratio": weighted_mean((row.severe_q_ratio, row.epochs) for row in rows),
        "flagged_ratio": weighted_mean((row.flagged_ratio, row.epochs) for row in rows),
        "sleep_period_ratio": float(sum(row.sleep_period_epochs for row in rows) / max(epochs, 1)),
    }


def weighted_mean(values_and_weights: Iterable[Tuple[float, int]]) -> float:
    total = 0.0
    weight = 0.0
    for value, item_weight in values_and_weights:
        total += float(value) * float(item_weight)
        weight += float(item_weight)
    return float(total / max(weight, 1.0))


def group_by_subject(rows: Sequence[RecordingProfile]) -> Dict[str, List[RecordingProfile]]:
    grouped: Dict[str, List[RecordingProfile]] = defaultdict(list)
    for row in rows:
        grouped[row.subject].append(row)
    return dict(sorted(grouped.items()))


def subject_vectors(grouped: Mapping[str, Sequence[RecordingProfile]]) -> Dict[str, Dict[str, Any]]:
    vectors: Dict[str, Dict[str, Any]] = {}
    for subject, rows in grouped.items():
        summary = aggregate_profiles(rows)
        class_counts = np.array(
            [summary["class_counts"][name] for name in CLASS_NAMES],
            dtype=np.float64,
        )
        transition_counts_vec = np.array(
            [summary["transition_counts"][name] for name in TRANSITION_NAMES],
            dtype=np.float64,
        )
        vectors[subject] = {
            "class_counts": class_counts,
            "transition_counts": transition_counts_vec,
            "epochs": int(summary["epochs"]),
            "mean_q_sum": float(summary["mean_q"]) * int(summary["epochs"]),
            "low_q_sum": float(summary["low_q_ratio"]) * int(summary["epochs"]),
            "severe_q_sum": float(summary["severe_q_ratio"]) * int(summary["epochs"]),
            "flagged_sum": float(summary["flagged_ratio"]) * int(summary["epochs"]),
            "transition_epoch_sum": float(summary["transition_ratio"]) * int(summary["epochs"]),
        }
    return vectors


def split_subject_counts(n_subjects: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, int]:
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    ratios = ratios / np.maximum(np.sum(ratios), 1e-6)
    raw = ratios * n_subjects
    counts = np.floor(raw).astype(int)
    while np.sum(counts) < n_subjects:
        idx = int(np.argmax(raw - counts))
        counts[idx] += 1
    for idx in range(3):
        if counts[idx] == 0 and n_subjects >= 3:
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[idx] += 1
    return {"train": int(counts[0]), "val": int(counts[1]), "test": int(counts[2])}


def summarize_subject_split(
    split_subjects: Mapping[str, Sequence[str]],
    grouped: Mapping[str, Sequence[RecordingProfile]],
) -> Dict[str, Dict[str, object]]:
    return {
        split: aggregate_profiles([row for subject in subjects for row in grouped[subject]])
        for split, subjects in split_subjects.items()
    }


def score_split(
    split_subjects: Mapping[str, Sequence[str]],
    vectors: Mapping[str, Mapping[str, Any]],
    target_epoch_ratios: Mapping[str, float],
    global_class_ratio: np.ndarray,
    global_transition_ratio: np.ndarray,
    global_quality: Mapping[str, float],
) -> float:
    total_epochs = sum(int(item["epochs"]) for item in vectors.values())
    score = 0.0
    for split, subjects in split_subjects.items():
        if not subjects:
            return float("inf")
        class_counts = np.sum([vectors[sub]["class_counts"] for sub in subjects], axis=0)
        trans_counts = np.sum([vectors[sub]["transition_counts"] for sub in subjects], axis=0)
        epochs = int(sum(int(vectors[sub]["epochs"]) for sub in subjects))
        class_ratio = class_counts / max(float(np.sum(class_counts)), 1.0)
        trans_ratio = trans_counts / max(float(np.sum(trans_counts)), 1.0)
        mean_q = sum(float(vectors[sub]["mean_q_sum"]) for sub in subjects) / max(epochs, 1)
        low_q = sum(float(vectors[sub]["low_q_sum"]) for sub in subjects) / max(epochs, 1)
        severe_q = sum(float(vectors[sub]["severe_q_sum"]) for sub in subjects) / max(epochs, 1)
        flagged = sum(float(vectors[sub]["flagged_sum"]) for sub in subjects) / max(epochs, 1)
        transition_epoch_ratio = (
            sum(float(vectors[sub]["transition_epoch_sum"]) for sub in subjects) / max(epochs, 1)
        )

        score += 3.0 * float(np.sum(np.abs(class_ratio - global_class_ratio)))
        score += 1.2 * abs(float(class_ratio[0] - global_class_ratio[0]))
        score += 1.5 * abs(float(class_ratio[1] - global_class_ratio[1]))
        score += 0.8 * float(np.sum(np.abs(trans_ratio - global_transition_ratio)))
        score += 1.0 * abs((epochs / max(total_epochs, 1)) - float(target_epoch_ratios[split]))
        score += 0.8 * abs(mean_q - float(global_quality["mean_q"]))
        score += 1.2 * abs(low_q - float(global_quality["low_q_ratio"]))
        score += 1.2 * abs(severe_q - float(global_quality["severe_q_ratio"]))
        score += 0.5 * abs(flagged - float(global_quality["flagged_ratio"]))
        score += 0.8 * abs(transition_epoch_ratio - float(global_quality["transition_ratio"]))
    return float(score)


def optimize_split(
    rows: Sequence[RecordingProfile],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    search_iters: int,
) -> Tuple[Dict[str, List[str]], float, Dict[str, Dict[str, object]]]:
    grouped = group_by_subject(rows)
    subjects = list(grouped.keys())
    if len(subjects) < 3:
        raise ValueError("Need at least 3 subjects for train/val/test splitting.")
    counts = split_subject_counts(len(subjects), train_ratio, val_ratio, test_ratio)
    target_epoch_ratios = {
        "train": train_ratio / (train_ratio + val_ratio + test_ratio),
        "val": val_ratio / (train_ratio + val_ratio + test_ratio),
        "test": test_ratio / (train_ratio + val_ratio + test_ratio),
    }
    vectors = subject_vectors(grouped)
    global_summary = aggregate_profiles(rows)
    global_class_counts = np.array(
        [global_summary["class_counts"][name] for name in CLASS_NAMES],
        dtype=np.float64,
    )
    global_transition_counts = np.array(
        [global_summary["transition_counts"][name] for name in TRANSITION_NAMES],
        dtype=np.float64,
    )
    global_class_ratio = global_class_counts / max(float(np.sum(global_class_counts)), 1.0)
    global_transition_ratio = global_transition_counts / max(float(np.sum(global_transition_counts)), 1.0)
    global_quality = {
        "mean_q": float(global_summary["mean_q"]),
        "low_q_ratio": float(global_summary["low_q_ratio"]),
        "severe_q_ratio": float(global_summary["severe_q_ratio"]),
        "flagged_ratio": float(global_summary["flagged_ratio"]),
        "transition_ratio": float(global_summary["transition_ratio"]),
    }

    rng = random.Random(seed)
    best_split: Optional[Dict[str, List[str]]] = None
    best_score = float("inf")
    ordered_subjects = sorted(subjects, key=lambda sub: int(vectors[sub]["epochs"]), reverse=True)

    candidate_orders: List[List[str]] = [ordered_subjects, sorted(subjects)]
    for _ in range(max(1, int(search_iters))):
        shuffled = list(subjects)
        rng.shuffle(shuffled)
        candidate_orders.append(shuffled)

    for order in candidate_orders:
        split = {
            "train": sorted(order[: counts["train"]]),
            "val": sorted(order[counts["train"]: counts["train"] + counts["val"]]),
            "test": sorted(order[counts["train"] + counts["val"]:]),
        }
        value = score_split(
            split,
            vectors,
            target_epoch_ratios,
            global_class_ratio,
            global_transition_ratio,
            global_quality,
        )
        if value < best_score:
            best_score = value
            best_split = split

    assert best_split is not None
    return best_split, best_score, summarize_subject_split(best_split, grouped)


def files_for_split(split_subjects: Mapping[str, Sequence[str]], grouped: Mapping[str, Sequence[RecordingProfile]]):
    return {
        split: sorted(row.file for subject in subjects for row in grouped[subject])
        for split, subjects in split_subjects.items()
    }


def make_report(
    dataset_name: str,
    npz_dir: Path,
    rows: Sequence[RecordingProfile],
    split_subjects: Mapping[str, Sequence[str]],
    split_score: float,
    split_summary: Mapping[str, Mapping[str, object]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    grouped = group_by_subject(rows)
    split_files = files_for_split(split_subjects, grouped)
    global_summary = aggregate_profiles(rows)
    extreme_subjects = []
    for subject, subject_rows in grouped.items():
        summary = aggregate_profiles(subject_rows)
        class_ratio = summary["class_ratio"]
        extreme_subjects.append(
            {
                "subject": subject,
                "files": [row.file for row in subject_rows],
                "epochs": summary["epochs"],
                "W_ratio": class_ratio[0],
                "N1_ratio": class_ratio[1],
                "REM_ratio": class_ratio[4],
                "low_q_ratio": summary["low_q_ratio"],
                "severe_q_ratio": summary["severe_q_ratio"],
                "transition_ratio": summary["transition_ratio"],
            }
        )
    extreme_subjects = sorted(
        extreme_subjects,
        key=lambda item: (
            abs(float(item["W_ratio"]) - global_summary["class_ratio"][0])
            + abs(float(item["N1_ratio"]) - global_summary["class_ratio"][1])
            + float(item["low_q_ratio"])
            + float(item["severe_q_ratio"])
        ),
        reverse=True,
    )[:20]

    return {
        "dataset": dataset_name,
        "npz_dir": str(npz_dir),
        "class_names": list(CLASS_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "settings": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "search_iters": args.search_iters,
            "sleep_margin_minutes": args.sleep_margin_minutes,
            "epoch_seconds": args.epoch_seconds,
            "quality_low_threshold": args.quality_low_threshold,
            "quality_severe_threshold": args.quality_severe_threshold,
        },
        "global_summary": global_summary,
        "split_score": split_score,
        "split_subjects": split_subjects,
        "split_files": split_files,
        "split_summary": split_summary,
        "extreme_subjects_for_review": extreme_subjects,
        "recommended_training_notes": [
            "Use split_files as fixed subject-level train/val/test membership.",
            "Use sleep_period_mask for a secondary sleep-period protocol; keep full-night protocol for primary clinical reporting.",
            "Use quality_weight as optional sample-weight multiplier; avoid deleting epochs unless nonfinite/flat across all channels.",
            "For ISRUC-like data, consider narrowing transition boundary weighting if boundary_epoch ratio exceeds roughly 35%.",
        ],
    }


def print_summary(report: Mapping[str, object]) -> None:
    print(f"\nDataset protocol built: {report['dataset']}")
    print(f"Source: {report['npz_dir']}")
    global_summary = report["global_summary"]
    print(
        "Global: "
        f"subjects={global_summary['subjects']} files={global_summary['files']} epochs={global_summary['epochs']} "
        f"class_ratio={global_summary['class_ratio']} mean_Q={global_summary['mean_q']:.3f} "
        f"low_Q={100 * global_summary['low_q_ratio']:.1f}% severe_Q={100 * global_summary['severe_q_ratio']:.1f}%"
    )
    for split, summary in report["split_summary"].items():
        print(
            f"{split}: subjects={summary['subjects']} files={summary['files']} epochs={summary['epochs']} "
            f"class_ratio={summary['class_ratio']} mean_Q={summary['mean_q']:.3f} "
            f"low_Q={100 * summary['low_q_ratio']:.1f}% severe_Q={100 * summary['severe_q_ratio']:.1f}%"
        )
    print(f"Split score: {report['split_score']:.6f}")


def main() -> None:
    args = parse_args()
    ratios_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(ratios_sum, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        raise ValueError("train/val/test ratios should sum to 1.0.")

    dataset_name = str(args.dataset_name).strip()
    npz_dir = Path(args.npz_dir).resolve()
    output_dir = Path(args.output_dir).resolve() / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_metadata_dir = output_dir / "epoch_metadata" if args.save_epoch_metadata else None

    files = sorted(npz_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}.")

    margin_epochs = int(round(float(args.sleep_margin_minutes) * 60.0 / float(args.epoch_seconds)))
    rows: List[RecordingProfile] = []
    print(f"Scanning {len(files)} files from {npz_dir}...")
    for idx, npz_path in enumerate(files, start=1):
        rows.append(
            profile_recording(
                dataset_name,
                npz_path,
                margin_epochs=margin_epochs,
                low_q=args.quality_low_threshold,
                severe_q=args.quality_severe_threshold,
                save_epoch_metadata_dir=epoch_metadata_dir,
            )
        )
        if idx % 10 == 0 or idx == len(files):
            print(f"  profiled {idx}/{len(files)} files")

    grouped = group_by_subject(rows)
    split_subjects, split_score, split_summary = optimize_split(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        search_iters=args.search_iters,
    )
    report = make_report(dataset_name, npz_dir, rows, split_subjects, split_score, split_summary, args)

    write_manifest(rows, output_dir / "manifest.csv")
    with (output_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset_name,
                "source_npz_dir": str(npz_dir),
                "split_subjects": split_subjects,
                "split_files": files_for_split(split_subjects, grouped),
                "split_summary": split_summary,
                "split_score": split_score,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    with (output_dir / "protocol_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print_summary(report)
    print(f"\nWrote: {output_dir / 'manifest.csv'}")
    print(f"Wrote: {output_dir / 'split.json'}")
    print(f"Wrote: {output_dir / 'protocol_report.json'}")
    if epoch_metadata_dir is not None:
        print(f"Wrote per-recording epoch metadata under: {epoch_metadata_dir}")


if __name__ == "__main__":
    main()
