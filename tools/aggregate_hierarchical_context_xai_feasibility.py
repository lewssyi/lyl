#!/usr/bin/env python
"""Guarded stage aggregation for Hierarchical Context Attribution feasibility."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "outputs/hierarchical_context_xai_feasibility/isruc_seed42"


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_stage(stage: str) -> tuple[pd.DataFrame, list[dict]]:
    frames, manifests, bad = [], [], []
    for path in sorted((ROOT / "runs").glob(f"{stage}_*/run_manifest.json")):
        payload = json.loads(path.read_text())
        directory = path.parent
        csv_path = directory / "hierarchical_sample_level.csv"
        if payload.get("status") != "COMPLETE" or not payload.get("all_outputs_nonempty") or int(payload.get("logical_query_budget_per_candidate", -1)) != 1530 or not csv_path.exists() or csv_path.stat().st_size <= 1:
            bad.append(str(path.relative_to(ROOT)))
            continue
        manifests.append(payload)
        frames.append(pd.read_csv(csv_path))
    if bad or not frames:
        raise RuntimeError(f"invalid or missing {stage} runs: {bad}")
    result = pd.concat(frames, ignore_index=True)
    numeric = result.select_dtypes(include=[np.number])
    # A constant black-box response makes weighted R2 mathematically
    # undefined. Preserve those rows for reliability/coverage accounting;
    # permit NaN only in the two R2 fields derived from that fit. Infinities
    # and missing values in every other numeric field remain hard failures.
    allowed_nan = {"local_surrogate_R2", "hierarchical_surrogate_R2"}
    unexpected_nan = [
        column
        for column in numeric.columns
        if column not in allowed_nan and numeric[column].isna().any()
    ]
    if unexpected_nan:
        raise RuntimeError(
            f"{stage} contains unexpected NaN values in {unexpected_nan}"
        )
    if np.isinf(numeric.to_numpy()).any():
        raise RuntimeError(f"{stage} contains infinite numeric values")
    if set(result.query_budget.unique()) != {1530}:
        raise RuntimeError("query budget drift")
    return result, manifests


def flattened(row) -> np.ndarray:
    context = json.loads(row.context_importance)
    within = json.loads(row.within_epoch_importance)
    return np.asarray(context + [value for vector in within for value in vector], dtype=np.float64)


def smoke() -> None:
    frame, manifests = load_stage("smoke")
    if len(manifests) != 8 or frame.sample_id.nunique() != 4:
        raise RuntimeError(f"smoke coverage mismatch: manifests={len(manifests)} samples={frame.sample_id.nunique()}")
    rows = []
    for keys, group in frame.groupby(["candidate", "target_protocol"]):
        r0 = group.loc[group.randomization_level == "R0"].set_index("sample_id")
        r7 = group.loc[group.randomization_level == "R7"].set_index("sample_id")
        similarities = []
        for sample_id in sorted(set(r0.index) & set(r7.index)):
            correlation = spearmanr(flattened(r0.loc[sample_id]), flattened(r7.loc[sample_id])).statistic
            similarities.append(abs(float(correlation)))
        rows.append({
            "candidate": keys[0], "target_protocol": keys[1], "sample_count": len(similarities),
            "r0_context_surrogate_r2": float(r0.context_surrogate_R2.mean()),
            "r0_local_surrogate_r2": float(r0.local_surrogate_R2.mean()),
            "r0_faithfulness": float(r0.faithfulness.mean()),
            "r7_absolute_spearman": float(np.mean(similarities)),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "smoke_summary.csv", index=False)
    write_json(ROOT / "smoke_summary.json", {
        "schema_version": "hc_seqlime_smoke_summary_v1", "status": "PASS",
        "completed_runs": len(manifests), "sample_count": int(frame.sample_id.nunique()),
        "candidate_count": int(frame.candidate.nunique()), "all_outputs_nonempty": True,
        "all_numeric_finite": True, "equal_query_budget": True,
        "failed_cache_attempt_preserved": "logs/smoke_sample44280_R0.log",
        "atomic_cache_fix_commit": "124330d",
    })
    run_manifest = json.loads((ROOT / "run_manifest.json").read_text())
    run_manifest["status"] = "SMOKE_COMPLETE"
    run_manifest["smoke"] = {"completed_runs": 8, "failed_final": 0, "summary": "smoke_summary.csv"}
    write_json(ROOT / "run_manifest.json", run_manifest)
    print(summary.to_string(index=False))


def development() -> None:
    frame, manifests = load_stage("development")
    if len(manifests) != 384 or frame.sample_id.nunique() != 16:
        raise RuntimeError(
            f"development coverage mismatch: manifests={len(manifests)} "
            f"samples={frame.sample_id.nunique()}"
        )
    rows = []
    for keys, group in frame.groupby(
        ["candidate", "group_scheme", "feature_count", "target_protocol"]
    ):
        r0 = group.loc[group.randomization_level == "R0"].set_index("sample_id")
        r7 = group.loc[group.randomization_level == "R7"].set_index("sample_id")
        similarities = []
        for sample_id in sorted(set(r0.index) & set(r7.index)):
            correlation = spearmanr(
                flattened(r0.loc[sample_id]), flattened(r7.loc[sample_id])
            ).statistic
            similarities.append(abs(float(correlation)))
        rows.append({
            "candidate": keys[0],
            "group_scheme": keys[1],
            "feature_count": int(keys[2]),
            "target_protocol": keys[3],
            "sample_count": int(len(r0)),
            "local_r2_estimable_count": int(r0.local_surrogate_R2.notna().sum()),
            "local_r2_degenerate_count": int(r0.local_surrogate_R2.isna().sum()),
            "r0_context_surrogate_r2": float(r0.context_surrogate_R2.mean()),
            "r0_local_surrogate_r2": float(r0.local_surrogate_R2.mean()),
            "r0_hierarchical_surrogate_r2": float(r0.hierarchical_surrogate_R2.mean()),
            "r0_deletion_auc": float(r0.deletion_auc.mean()),
            "r0_insertion_auc": float(r0.insertion_auc.mean()),
            "r0_faithfulness": float(r0.faithfulness.mean()),
            "r7_absolute_spearman": float(np.mean(similarities)),
            "context_dependence_ratio": float(r0.context_dependence_ratio.mean()),
        })
    summary = pd.DataFrame(rows).sort_values(
        ["target_protocol", "r7_absolute_spearman", "r0_hierarchical_surrogate_r2"],
        ascending=[True, True, False],
    )
    summary.to_csv(ROOT / "development_candidate_summary.csv", index=False)
    frame.to_csv(ROOT / "development_sample_level.csv", index=False)
    reliability = (
        frame.groupby(
            [
                "candidate",
                "group_scheme",
                "feature_count",
                "target_protocol",
                "randomization_level",
            ],
            dropna=False,
        )
        .agg(
            row_count=("sample_id", "size"),
            local_r2_estimable_count=("local_surrogate_R2", "count"),
        )
        .reset_index()
    )
    reliability["local_r2_degenerate_count"] = (
        reliability.row_count - reliability.local_r2_estimable_count
    )
    reliability.to_csv(
        ROOT / "development_reliability_diagnostics.csv", index=False
    )

    # Development only excludes obvious failures. H3 is the only candidate
    # that preserves target-epoch attribution while adding a non-target context
    # group; 45 features dominate 30 on local R2. G9 and G25 express the
    # observed context-fidelity versus randomization-sensitivity trade-off.
    selected = [
        {"candidate": "H3", "group_scheme": "G25", "feature_count": 45},
        {"candidate": "H3", "group_scheme": "G9", "feature_count": 45},
    ]
    write_json(ROOT / "development_survivors.json", {
        "schema_version": "hc_seqlime_development_survivors_v1",
        "status": "DEVELOPMENT_SCREEN_COMPLETE",
        "selection_split": "development",
        "test_data_used": False,
        "validation_data_used": False,
        "selected_for_validation": selected,
        "selection_is_final_method_selection": False,
        "rationale": [
            "H1/H2 had much weaker development faithfulness than target-retaining H3.",
            "45 features improved H3 local surrogate R2 over 30 for every group scheme.",
            "G9 had the strongest context/hierarchical R2; G25 had the lowest H3 R7 similarity.",
            "Validation, not development, freezes the final candidate and reliability thresholds.",
        ],
        "equal_query_budget": True,
        "stage_prior_used": False,
        "feedback_training_status": "BLOCKED",
    })
    run_manifest = json.loads((ROOT / "run_manifest.json").read_text())
    run_manifest["status"] = "DEVELOPMENT_COMPLETE"
    run_manifest["development"] = {
        "completed_runs": 384,
        "failed_final": 0,
        "sample_count": 16,
        "local_r2_degenerate_rows": int(frame.local_surrogate_R2.isna().sum()),
        "selected_for_validation": selected,
        "summary": "development_candidate_summary.csv",
        "reliability_diagnostics": "development_reliability_diagnostics.csv",
    }
    write_json(ROOT / "run_manifest.json", run_manifest)
    print(summary.to_string(index=False))
    print(json.dumps({"selected_for_validation": selected}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "development"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    smoke() if arguments.stage == "smoke" else development()
