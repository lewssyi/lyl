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
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError(f"{stage} contains non-finite numeric values")
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke",), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    smoke()
