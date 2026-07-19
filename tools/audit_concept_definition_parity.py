#!/usr/bin/env python
"""Audit distinct raw-physiology and segment-proxy concept definitions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clinical_concepts import (  # noqa: E402
    concept_names,
    validate_raw_segment_concept_parity,
)
from src.lime_explanation import semantic_segmentation  # noqa: E402


CACHE = (
    PROJECT_ROOT
    / "outputs/sequence_lime_paper_evaluation/isruc_seed42/cache/test_inputs_segments.npz"
)
PREDICTIONS = (
    PROJECT_ROOT
    / "outputs/predictions/result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42_split_predictions.npz"
)
DEFAULT_OUT = PROJECT_ROOT / "outputs/concept_definition_parity"


def choose_indices(labels: np.ndarray, per_stage: int) -> list[int]:
    selected = []
    for stage in range(5):
        indices = np.flatnonzero(labels == stage)
        if len(indices):
            positions = np.linspace(0, len(indices) - 1, min(per_stage, len(indices)), dtype=int)
            selected.extend(indices[positions].tolist())
    return sorted(set(selected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-stage", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not CACHE.exists() or not PREDICTIONS.exists():
        raise FileNotFoundError("Run the canonical parity diagnostic to build the input cache first.")
    cache = np.load(CACHE)
    prediction = np.load(PREDICTIONS)
    labels = prediction["test_y_true"].astype(np.int16)
    rows = []
    for index in choose_indices(labels, max(1, args.per_stage)):
        waveform = cache["x_raw"][index]
        features = cache["x_seg"][index]
        segments = semantic_segmentation(waveform, target_segments=30)
        report = validate_raw_segment_concept_parity(waveform, features, segments)
        for concept_index, name in enumerate(concept_names()):
            rows.append(
                {
                    "global_epoch_index": int(index),
                    "true_stage": int(labels[index]),
                    "concept": name,
                    "segment_definition_version": report["segment_definition_version"],
                    "raw_definition_version": report["raw_definition_version"],
                    "mathematically_equivalent": False,
                    "pearson": report["per_concept_pearson"][concept_index],
                    "sample_mean_absolute_difference": report["mean_absolute_difference"],
                }
            )
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "concept_definition_parity.csv", index=False)
    finite = pd.to_numeric(frame["pearson"], errors="coerce")
    summary = {
        "status": "PASS_DISTINCT_VERSIONING",
        "sample_count": int(frame["global_epoch_index"].nunique()),
        "row_count": int(len(frame)),
        "segment_definition_version": "segment_proxy_v1",
        "raw_definition_version": "raw_physiology_v1",
        "mathematically_equivalent": False,
        "mean_available_pearson": float(finite.mean()) if finite.notna().any() else None,
        "mean_sample_absolute_difference": float(frame["sample_mean_absolute_difference"].mean()),
        "interpretation": (
            "These are related evidence views with distinct estimands. Neither may be "
            "reported as the other or as a single nine-dimensional clinical ground truth."
        ),
    }
    (args.output_dir / "concept_definition_parity.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
