#!/usr/bin/env python
"""Analyze a locked blinded clinical-expert rating table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_COLUMNS = (
    "physiological_plausibility",
    "temporal_localization",
    "concept_evidence_coherence",
    "clinical_usefulness",
)


def icc_2_1(frame: pd.DataFrame, value: str) -> float | None:
    matrix = frame.pivot(index="item_id", columns="reviewer_id", values=value).dropna()
    n, k = matrix.shape
    if n < 2 or k < 2:
        return None
    values = matrix.to_numpy(dtype=float)
    grand = values.mean()
    item_mean, reviewer_mean = values.mean(axis=1), values.mean(axis=0)
    ms_item = k * np.sum((item_mean - grand) ** 2) / (n - 1)
    ms_reviewer = n * np.sum((reviewer_mean - grand) ** 2) / (k - 1)
    residual = np.sum((values - item_mean[:, None] - reviewer_mean[None, :] + grand) ** 2)
    ms_error = residual / ((n - 1) * (k - 1))
    denominator = ms_item + (k - 1) * ms_error + k * (ms_reviewer - ms_error) / n
    return float((ms_item - ms_error) / denominator) if abs(denominator) > 1e-12 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ratings = pd.read_csv(args.ratings)
    key = pd.read_csv(args.private_key)
    required = {"reviewer_id", "item_id", *SCORE_COLUMNS, "reviewer_stage", "artifact_flag"}
    missing = sorted(required - set(ratings.columns))
    if missing:
        raise ValueError(f"Missing rating columns: {missing}")
    if ratings.duplicated(["reviewer_id", "item_id"]).any():
        raise ValueError("Duplicate reviewer/item ratings are not allowed")
    merged = ratings.merge(key, on="item_id", how="left", validate="many_to_one")
    for column in SCORE_COLUMNS:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
        invalid = merged[column].dropna().loc[lambda values: ~values.between(1, 5)]
        if len(invalid):
            raise ValueError(f"{column} must be in [1,5]")
    summary = {
        "status": "completed expert ratings analysis; interpretation requires clinical team",
        "reviewer_count": int(merged["reviewer_id"].nunique()),
        "item_count": int(merged["item_id"].nunique()),
        "rating_rows": int(len(merged)),
        "missing_score_fraction": float(merged[list(SCORE_COLUMNS)].isna().mean().mean()),
        "scores": {},
    }
    for column in SCORE_COLUMNS:
        values = merged[column].dropna()
        summary["scores"][column] = {
            "mean": float(values.mean()) if len(values) else None,
            "median": float(values.median()) if len(values) else None,
            "iqr": float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else None,
            "icc_2_1": icc_2_1(merged, column),
        }
    stage_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
    reviewer_stage = merged["reviewer_stage"].map(stage_map)
    summary["reviewer_true_stage_agreement"] = float(
        np.mean(reviewer_stage.dropna().astype(int) == merged.loc[reviewer_stage.notna(), "true_stage"].astype(int))
    ) if reviewer_stage.notna().any() else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "ratings_with_private_strata.csv", index=False)
    (args.output_dir / "expert_rating_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
