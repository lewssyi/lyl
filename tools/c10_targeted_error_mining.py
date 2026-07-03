#!/usr/bin/env python
"""Targeted C10 error mining for deciding whether another training change is worthwhile.

The script reads a C10/C10a prediction NPZ and ranks error modes that could
plausibly support a training-side improvement: focused sleep-stage confusions,
recording-level concentration, transition context, and low-margin/top-2
recoverability. It does not train or modify predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.c10_error_diagnostics import (  # noqa: E402
    entropy,
    json_ready,
    load_split_prediction,
    metric_bundle,
    normalized_time_position,
    safe_dataset_name,
    stage_index,
    top_two,
    transition_distances,
    write_csv,
)


DEFAULT_FOCUS_PAIRS = "N1/N2,N2/N3,REM/N1,REM/N2,N2/REM,W/N1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine C10 prediction errors for actionable targets: focus-pair "
            "confusions, recording failures, transition context, and top-2 "
            "recoverability."
        )
    )
    parser.add_argument("--prediction-npz", default=None, help="Combined prediction NPZ, usually containing test arrays.")
    parser.add_argument("--split-prediction-npz", default=None, help="Split-specific prediction NPZ.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--output-dir", default="outputs/diagnostics/c10_targeted_error_mining")
    parser.add_argument("--baseline-pred-source", choices=("auto", "argmax", "saved"), default="auto")
    parser.add_argument("--prob-key", default=None)
    parser.add_argument("--logit-key", default=None)
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--pred-key", default=None)
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--no-concepts", action="store_true")
    parser.add_argument("--selection-metric", choices=("accuracy", "macro_f1", "accuracy_macro", "n1_f1", "kappa"), default="accuracy_macro")
    parser.add_argument("--focus-pairs", default=DEFAULT_FOCUS_PAIRS)
    parser.add_argument("--top-recordings", type=int, default=20)
    parser.add_argument("--top-confusions", type=int, default=20)
    parser.add_argument("--low-margin", type=float, default=0.15)
    parser.add_argument("--transition-radius", type=int, default=2)
    return parser.parse_args()


def prediction_path(args: argparse.Namespace) -> Path:
    raw = args.split_prediction_npz or args.prediction_npz
    if raw is None:
        raise ValueError("Provide --prediction-npz or --split-prediction-npz.")
    return Path(raw).expanduser()


def parse_focus_pairs(raw: str, class_labels: Sequence[str]) -> list[tuple[int, int, str]]:
    lookup = {str(label).upper(): idx for idx, label in enumerate(class_labels)}
    pairs: list[tuple[int, int, str]] = []
    for token in str(raw).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "/" in token:
            left, right = token.split("/", 1)
        elif "-" in token:
            left, right = token.split("-", 1)
        else:
            raise ValueError(f"Focus pair {token!r} must look like N1/N2.")
        left_key = left.strip().upper()
        right_key = right.strip().upper()
        if left_key not in lookup or right_key not in lookup:
            raise ValueError(f"Unknown focus pair {token!r}; class labels are {tuple(class_labels)}.")
        a = lookup[left_key]
        b = lookup[right_key]
        pairs.append((a, b, f"{class_labels[a]}/{class_labels[b]}"))
    return pairs


def label_name(class_labels: Sequence[str], idx: int) -> str:
    if 0 <= int(idx) < len(class_labels):
        return str(class_labels[int(idx)])
    return f"class_{idx}"


def safe_div(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if float(den) else 0.0


def safe_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def safe_median(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values))


def quality_vector(split) -> np.ndarray | None:
    if split.quality is None:
        return None
    values = np.asarray(split.quality, dtype=np.float64)
    if values.ndim == 1:
        return values
    return np.mean(values, axis=1)


def confusion_key(class_labels: Sequence[str], y_true: int, y_pred: int) -> str:
    return f"{label_name(class_labels, y_true)}->{label_name(class_labels, y_pred)}"


def count_focus_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[str],
    mask: np.ndarray,
) -> dict[str, int]:
    pairs = parse_focus_pairs(DEFAULT_FOCUS_PAIRS, class_labels)
    output: dict[str, int] = {}
    for a, b, name in pairs:
        pair_mask = mask & (((y_true == a) & (y_pred == b)) | ((y_true == b) & (y_pred == a)))
        output[f"{name}_errors"] = int(np.sum(pair_mask))
    return output


def summarize_mask(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    class_labels: Sequence[str],
    margins: np.ndarray,
    confidence: np.ndarray,
    true_in_top2: np.ndarray,
    distances: np.ndarray,
    quality: np.ndarray | None,
    transition_radius: int,
    low_margin: float,
) -> dict[str, object]:
    n = int(np.sum(mask))
    if n == 0:
        return {
            "n": 0,
            "errors": 0,
            "accuracy": None,
            "error_rate": None,
            "top2_recoverable_errors": 0,
            "top2_recoverable_error_fraction": None,
            "low_margin_error_fraction": None,
            "transition_error_fraction": None,
            "mean_confidence_wrong": None,
            "median_margin_wrong": None,
            "mean_quality_wrong": None,
        }
    wrong = mask & (y_true != y_pred)
    errors = int(np.sum(wrong))
    recoverable = int(np.sum(wrong & true_in_top2))
    transition_wrong = wrong & (distances <= int(transition_radius))
    low_margin_wrong = wrong & (margins <= float(low_margin))
    quality_wrong = None if quality is None else quality[wrong]
    return {
        "n": n,
        "errors": errors,
        "accuracy": float(1.0 - safe_div(errors, n)),
        "error_rate": safe_div(errors, n),
        "top2_recoverable_errors": recoverable,
        "top2_recoverable_error_fraction": safe_div(recoverable, errors),
        "low_margin_error_fraction": safe_div(int(np.sum(low_margin_wrong)), errors),
        "transition_error_fraction": safe_div(int(np.sum(transition_wrong)), errors),
        "mean_confidence_wrong": safe_mean(confidence[wrong]),
        "median_margin_wrong": safe_median(margins[wrong]),
        "mean_quality_wrong": safe_mean(quality_wrong) if quality_wrong is not None else None,
    }


def focus_pair_rows(args, split, margins, confidence, true_in_top2, distances, quality) -> list[dict[str, object]]:
    y_true = split.y_true
    y_pred = split.c10a_pred
    wrong = y_true != y_pred
    total_errors = int(np.sum(wrong))
    rows: list[dict[str, object]] = []
    for a, b, pair_name in parse_focus_pairs(args.focus_pairs, split.class_labels):
        true_pair = (y_true == a) | (y_true == b)
        pred_pair = (y_pred == a) | (y_pred == b)
        pair_error = wrong & true_pair & pred_pair
        a_to_b = (y_true == a) & (y_pred == b)
        b_to_a = (y_true == b) & (y_pred == a)
        row = {
            "dataset": args.dataset,
            "split": split.split,
            "pair": pair_name,
            "true_pair_n": int(np.sum(true_pair)),
            "pred_pair_n": int(np.sum(pred_pair)),
            "pair_errors": int(np.sum(pair_error)),
            "fraction_of_all_errors": safe_div(int(np.sum(pair_error)), total_errors),
            "upper_bound_accuracy_gain_if_fixed": safe_div(int(np.sum(pair_error)), len(y_true)),
            "directed_a_to_b": int(np.sum(a_to_b)),
            "directed_b_to_a": int(np.sum(b_to_a)),
        }
        row.update(
            summarize_mask(
                y_true,
                y_pred,
                pair_error,
                split.class_labels,
                margins,
                confidence,
                true_in_top2,
                distances,
                quality,
                args.transition_radius,
                args.low_margin,
            )
        )
        rows.append(row)
    return sorted(rows, key=lambda item: (-int(item["pair_errors"]), str(item["pair"])))


def directed_confusion_rows(args, split, margins, confidence, true_in_top2, distances, quality) -> list[dict[str, object]]:
    y_true = split.y_true
    y_pred = split.c10a_pred
    wrong = y_true != y_pred
    counts: Counter[tuple[int, int]] = Counter(zip(y_true[wrong].tolist(), y_pred[wrong].tolist()))
    rows: list[dict[str, object]] = []
    total_errors = int(np.sum(wrong))
    for (true_label, pred_label), count in counts.most_common(int(args.top_confusions)):
        mask = (y_true == true_label) & (y_pred == pred_label)
        row = {
            "dataset": args.dataset,
            "split": split.split,
            "confusion": confusion_key(split.class_labels, true_label, pred_label),
            "true_label": label_name(split.class_labels, true_label),
            "pred_label": label_name(split.class_labels, pred_label),
            "count": int(count),
            "fraction_of_all_errors": safe_div(int(count), total_errors),
            "upper_bound_accuracy_gain_if_fixed": safe_div(int(count), len(y_true)),
        }
        row.update(
            summarize_mask(
                y_true,
                y_pred,
                mask,
                split.class_labels,
                margins,
                confidence,
                true_in_top2,
                distances,
                quality,
                args.transition_radius,
                args.low_margin,
            )
        )
        rows.append(row)
    return rows


def recording_rows(args, split, margins, confidence, true_in_top2, distances, quality) -> list[dict[str, object]]:
    y_true = split.y_true
    y_pred = split.c10a_pred
    wrong = y_true != y_pred
    total_errors = int(np.sum(wrong))
    rows: list[dict[str, object]] = []
    for rec in dict.fromkeys(split.recording_id.tolist()):
        mask = split.recording_id == rec
        rec_wrong = mask & wrong
        top_pair = ""
        top_pair_count = 0
        if np.any(rec_wrong):
            local_counts: Counter[str] = Counter(
                confusion_key(split.class_labels, int(t), int(p))
                for t, p in zip(y_true[rec_wrong].tolist(), y_pred[rec_wrong].tolist())
            )
            top_pair, top_pair_count = local_counts.most_common(1)[0]
        row = {
            "dataset": args.dataset,
            "split": split.split,
            "recording_id": str(rec),
            "fraction_of_all_errors": safe_div(int(np.sum(rec_wrong)), total_errors),
            "top_confusion": top_pair,
            "top_confusion_count": int(top_pair_count),
        }
        row.update(
            summarize_mask(
                y_true,
                y_pred,
                mask,
                split.class_labels,
                margins,
                confidence,
                true_in_top2,
                distances,
                quality,
                args.transition_radius,
                args.low_margin,
            )
        )
        row.update(count_focus_confusions(y_true, y_pred, split.class_labels, mask))
        rows.append(row)
    return sorted(rows, key=lambda item: (-int(item["errors"]), float(item["accuracy"] or 1.0), str(item["recording_id"])))[: int(args.top_recordings)]


def transition_rows(args, split, margins, confidence, true_in_top2, distances, quality) -> list[dict[str, object]]:
    y_true = split.y_true
    y_pred = split.c10a_pred
    bins = [
        ("transition_pm1", distances <= 1),
        ("transition_pm2", distances <= 2),
        ("near_transition_3_to_5", (distances > 2) & (distances <= 5)),
        ("stable_gt5", distances > 5),
        ("stable_or_no_transition", np.isinf(distances) | (distances > 5)),
    ]
    rows = []
    for name, mask in bins:
        row = {"dataset": args.dataset, "split": split.split, "subset": name}
        row.update(
            summarize_mask(
                y_true,
                y_pred,
                mask,
                split.class_labels,
                margins,
                confidence,
                true_in_top2,
                distances,
                quality,
                args.transition_radius,
                args.low_margin,
            )
        )
        row.update(count_focus_confusions(y_true, y_pred, split.class_labels, mask))
        rows.append(row)
    return rows


def margin_rows(args, split, margins, confidence, true_in_top2, distances, quality) -> list[dict[str, object]]:
    y_true = split.y_true
    y_pred = split.c10a_pred
    bins = [
        ("margin_le_0.05", margins <= 0.05),
        ("margin_0.05_to_0.10", (margins > 0.05) & (margins <= 0.10)),
        ("margin_0.10_to_0.20", (margins > 0.10) & (margins <= 0.20)),
        ("margin_gt_0.20", margins > 0.20),
        ("confidence_lt_0.50", confidence < 0.50),
        ("confidence_0.50_to_0.70", (confidence >= 0.50) & (confidence < 0.70)),
        ("confidence_0.70_to_0.90", (confidence >= 0.70) & (confidence < 0.90)),
        ("confidence_ge_0.90", confidence >= 0.90),
    ]
    rows = []
    for name, mask in bins:
        row = {"dataset": args.dataset, "split": split.split, "subset": name}
        row.update(
            summarize_mask(
                y_true,
                y_pred,
                mask,
                split.class_labels,
                margins,
                confidence,
                true_in_top2,
                distances,
                quality,
                args.transition_radius,
                args.low_margin,
            )
        )
        rows.append(row)
    return rows


def build_recommendations(
    args,
    split,
    metrics: Mapping[str, float],
    focus_rows: Sequence[Mapping[str, object]],
    recording_summary: Sequence[Mapping[str, object]],
    directed_rows: Sequence[Mapping[str, object]],
    transition_summary: Sequence[Mapping[str, object]],
    margins: np.ndarray,
    true_in_top2: np.ndarray,
) -> tuple[list[str], str]:
    wrong = split.y_true != split.c10a_pred
    total_errors = int(np.sum(wrong))
    top2_recoverable = int(np.sum(wrong & true_in_top2))
    low_margin_errors = int(np.sum(wrong & (margins <= float(args.low_margin))))
    top_recording_error_fraction = sum(float(row.get("fraction_of_all_errors") or 0.0) for row in recording_summary[:5])
    transition_pm2 = next((row for row in transition_summary if row["subset"] == "transition_pm2"), None)
    stable = next((row for row in transition_summary if row["subset"] == "stable_or_no_transition"), None)
    best_pair = focus_rows[0] if focus_rows else None

    findings = [
        f"Overall C10 metrics: accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f}, "
        f"N1-F1={metrics['n1_f1']:.4f}, kappa={metrics['kappa']:.4f}.",
        f"Top-2 recoverability: {top2_recoverable}/{total_errors} errors "
        f"({100.0 * safe_div(top2_recoverable, total_errors):.1f}%) have the true class inside the probability top-2.",
        f"Low-margin errors: {low_margin_errors}/{total_errors} errors "
        f"({100.0 * safe_div(low_margin_errors, total_errors):.1f}%) have top-2 margin <= {args.low_margin:.2f}.",
        f"Top 5 recordings contribute {100.0 * top_recording_error_fraction:.1f}% of all errors.",
    ]
    if best_pair is not None:
        findings.append(
            f"Largest focus pair is {best_pair['pair']} with {best_pair['pair_errors']} errors "
            f"({100.0 * float(best_pair['fraction_of_all_errors']):.1f}% of all errors)."
        )
    if transition_pm2 is not None and stable is not None:
        findings.append(
            f"Transition +/-2 error rate is {100.0 * float(transition_pm2['error_rate'] or 0.0):.1f}%; "
            f"stable/no-transition error rate is {100.0 * float(stable['error_rate'] or 0.0):.1f}%."
        )

    recommendations: list[str] = []
    if best_pair and float(best_pair.get("top2_recoverable_error_fraction") or 0.0) >= 0.45 and int(best_pair["pair_errors"]) >= max(30, int(0.10 * total_errors)):
        recommendations.append(
            f"Try a narrow pair-aware training change for {best_pair['pair']}: hard-example sampling or margin loss, "
            "not a global boundary boost."
        )
    if top_recording_error_fraction >= 0.35:
        recommendations.append(
            "Errors are recording-concentrated. Inspect the top failing recordings first; a recording/domain robustness change "
            "is more promising than another class-level tweak."
        )
    if transition_pm2 is not None and stable is not None:
        transition_error = float(transition_pm2["error_rate"] or 0.0)
        stable_error = float(stable["error_rate"] or 0.0)
        if transition_error >= stable_error + 0.12:
            recommendations.append(
                "Transition errors are sharply elevated. If changing training, target only high-risk transition pairs rather than all boundaries."
            )
    if not recommendations:
        recommendations.append(
            "No strong structured target was detected. Keep C10 as the main model unless the detailed CSVs show a clinically meaningful pair to pursue."
        )

    lines = [
        f"# Targeted C10 Error Mining: {args.dataset} / {split.split}",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- {item}" for item in findings)
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in recommendations)
    lines.extend(["", "## Top Directed Confusions", ""])
    for row in directed_rows[:8]:
        lines.append(
            f"- {row['confusion']}: {row['count']} errors, "
            f"top2-recoverable={100.0 * float(row.get('top2_recoverable_error_fraction') or 0.0):.1f}%"
        )
    lines.extend(["", "## Top Recordings", ""])
    for row in recording_summary[:8]:
        lines.append(
            f"- {row['recording_id']}: errors={row['errors']}, accuracy={float(row['accuracy'] or 0.0):.4f}, "
            f"top={row['top_confusion']}"
        )
    lines.append("")
    return recommendations, "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = prediction_path(args)
    split = load_split_prediction(path, args.split, args)
    y_true = split.y_true
    y_pred = split.c10a_pred
    top1, top2, top1_prob, top2_prob = top_two(split.probs)
    confidence = np.asarray(split.probs[np.arange(len(split.probs)), y_pred], dtype=np.float64)
    probability_top1_confidence = np.asarray(top1_prob, dtype=np.float64)
    margins = np.asarray(top1_prob - top2_prob, dtype=np.float64)
    true_in_top2 = (y_true == top1) | (y_true == top2)
    quality = quality_vector(split)
    distances, transition_event = transition_distances(split)
    time_position = normalized_time_position(split.recording_id, split.epoch_index).reshape(-1)

    metrics = metric_bundle(y_true, y_pred, split.class_labels, args.selection_metric)
    focus = focus_pair_rows(args, split, margins, confidence, true_in_top2, distances, quality)
    directed = directed_confusion_rows(args, split, margins, confidence, true_in_top2, distances, quality)
    recordings = recording_rows(args, split, margins, confidence, true_in_top2, distances, quality)
    transition = transition_rows(args, split, margins, confidence, true_in_top2, distances, quality)
    margin = margin_rows(args, split, margins, confidence, true_in_top2, distances, quality)
    recommendations, markdown = build_recommendations(
        args,
        split,
        metrics,
        focus,
        recordings,
        directed,
        transition,
        margins,
        true_in_top2,
    )

    prefix = f"{safe_dataset_name(args.dataset)}_c10_targeted_{split.split}"
    paths = {
        "focus_pairs": out_dir / f"{prefix}_focus_pairs.csv",
        "directed_confusions": out_dir / f"{prefix}_directed_confusions.csv",
        "recording_failures": out_dir / f"{prefix}_recording_failures.csv",
        "transition_bins": out_dir / f"{prefix}_transition_bins.csv",
        "margin_confidence_bins": out_dir / f"{prefix}_margin_confidence_bins.csv",
        "summary": out_dir / f"{prefix}_summary.json",
        "recommendations": out_dir / f"{prefix}_recommendations.md",
    }
    write_csv(paths["focus_pairs"], focus)
    write_csv(paths["directed_confusions"], directed)
    write_csv(paths["recording_failures"], recordings)
    write_csv(paths["transition_bins"], transition)
    write_csv(paths["margin_confidence_bins"], margin)
    summary = {
        "dataset": args.dataset,
        "split": split.split,
        "path": str(path),
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
        "n": int(len(y_true)),
        "errors": int(np.sum(y_true != y_pred)),
        "probability_context": {
            "true_in_probability_top2_error_fraction": safe_div(int(np.sum((y_true != y_pred) & true_in_top2)), int(np.sum(y_true != y_pred))),
            "low_margin_error_fraction": safe_div(int(np.sum((y_true != y_pred) & (margins <= float(args.low_margin)))), int(np.sum(y_true != y_pred))),
            "mean_saved_pred_confidence": safe_mean(confidence),
            "mean_probability_top1_confidence": safe_mean(probability_top1_confidence),
            "mean_entropy": safe_mean(entropy(split.probs)),
        },
        "transition_event_epoch_fraction": float(np.mean(transition_event)) if len(transition_event) else 0.0,
        "time_position_error_mean": safe_mean(time_position[y_true != y_pred]),
        "focus_pairs": focus,
        "top_directed_confusions": directed[:10],
        "top_recordings": recordings[:10],
        "recommendations": recommendations,
        "outputs": {name: str(path_value) for name, path_value in paths.items()},
    }
    paths["summary"].write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    paths["recommendations"].write_text(markdown, encoding="utf-8")

    print("C10 targeted error mining complete")
    print(
        f"  {split.split}: accuracy={metrics['accuracy']:.4f} "
        f"macro-F1={metrics['macro_f1']:.4f} N1-F1={metrics['n1_f1']:.4f} "
        f"errors={int(np.sum(y_true != y_pred))}/{len(y_true)}"
    )
    print(f"  output_dir={out_dir.resolve()}")


if __name__ == "__main__":
    main()
