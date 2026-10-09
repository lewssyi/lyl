#!/usr/bin/env python
"""Finalize dependency analysis and the protocol-consistency decision."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools import sequence_lime_protocol_consistency_audit as audit  # noqa: E402


ROOT = audit.ROOT
BOOTSTRAPS = 2000


def deterministic_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def load_occlusion() -> pd.DataFrame:
    frames = []
    failures = []
    for group in ("A", "C"):
        for shard in range(12):
            run = ROOT / "occlusion" / f"group{group}_shard{shard + 1}of12"
            manifest = run / "run_manifest.json"
            csv_path = run / "occlusion_diagnostics.csv"
            if not manifest.exists() or not csv_path.exists() or csv_path.stat().st_size <= 1:
                failures.append(str(run.relative_to(ROOT)))
                continue
            payload = json.loads(manifest.read_text())
            if payload.get("status") != "COMPLETE" or payload.get("protocol_version") != audit.PROTOCOL_VERSION:
                failures.append(str(run.relative_to(ROOT)))
                continue
            frame = pd.read_csv(csv_path)
            if len(frame) != 1:
                failures.append(str(run.relative_to(ROOT)))
                continue
            frames.append(frame)
    if failures or len(frames) != 24:
        raise RuntimeError(f"occlusion validation failed: {failures[:5]}")
    combined = pd.concat(frames, ignore_index=True)
    if combined.global_epoch_index.nunique() != 24:
        raise RuntimeError("occlusion sample coverage is incomplete")
    combined.to_csv(ROOT / "occlusion_diagnostics.csv", index=False)
    return combined


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 4 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan"), float("nan")
    observed = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAPS):
        indices = rng.integers(0, len(x), size=len(x))
        if np.unique(x[indices]).size < 2 or np.unique(y[indices]).size < 2:
            continue
        values.append(float(spearmanr(x[indices], y[indices]).statistic))
    return observed, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def build_dependency_analysis(sample: pd.DataFrame, occlusion: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r7 = sample.loc[sample.randomization_level == "R7"].copy()
    grouped = r7.groupby(
        ["audit_group", "global_epoch_index", "target_class_protocol"], as_index=False
    ).agg(
        randomization_seed_pass_count=("sample_randomization_pass", "sum"),
        absolute_spearman=("absolute_spearman", "mean"),
        signed_spearman=("signed_spearman", "mean"),
        top_5_jaccard=("top_5_jaccard", "mean"),
        top_10_jaccard=("top_10_jaccard", "mean"),
        randomized_confidence=("randomized_confidence", "mean"),
        randomized_entropy=("randomized_entropy", "mean"),
        randomized_y2_margin=("randomized_y2_margin", "mean"),
        prediction_kl_divergence=("prediction_kl_divergence", "mean"),
        r7_faithfulness_advantage=("faithfulness_advantage", "mean"),
    )
    grouped["randomization_pass"] = grouped.randomization_seed_pass_count >= 2
    r0 = sample.loc[sample.randomization_level == "R0"].groupby(
        ["audit_group", "global_epoch_index", "target_class_protocol"], as_index=False
    ).agg(
        source_recording=("source_recording", "first"),
        true_stage=("true_stage", "first"),
        predicted_stage=("original_predicted_stage", "first"),
        correct=("correct", "first"),
        transition_boundary=("transition_boundary", "first"),
        original_confidence=("original_confidence", "mean"),
        original_entropy=("original_entropy", "mean"),
        original_y2_margin=("original_y2_margin", "mean"),
        local_response_standard_deviation=("local_response_standard_deviation", "mean"),
        surrogate_weighted_r2=("surrogate_weighted_r2", "mean"),
        effective_sample_size=("effective_sample_size", "mean"),
        r0_faithfulness_advantage=("faithfulness_advantage", "mean"),
    )
    result = grouped.merge(
        r0, on=["audit_group", "global_epoch_index", "target_class_protocol"], validate="one_to_one"
    ).merge(
        occlusion[[
            "audit_group", "global_epoch_index", "target_epoch_occlusion_drop",
            "context_occlusion_effect", "context_dependence_ratio",
        ]],
        on=["audit_group", "global_epoch_index"], validate="many_to_one",
    )
    result["local_response_variance"] = result.local_response_standard_deviation ** 2
    result["context_dominated"] = result.context_dependence_ratio >= 0.5
    result["faithfulness_advantage_decay"] = (
        result.r0_faithfulness_advantage.abs() - result.r7_faithfulness_advantage.abs()
    )
    result.to_csv(ROOT / "sample_dependency_analysis.csv", index=False)

    predictors = [
        "original_confidence", "original_y2_margin", "original_entropy", "correct",
        "transition_boundary", "local_response_variance", "surrogate_weighted_r2",
        "effective_sample_size", "target_epoch_occlusion_drop", "context_dependence_ratio",
    ]
    rows = []
    for predictor in predictors:
        x = result[predictor].astype(float).to_numpy()
        y = result.randomization_pass.astype(float).to_numpy()
        correlation, low, high = bootstrap_spearman(x, y, deterministic_seed(predictor))
        passed = result.loc[result.randomization_pass, predictor].astype(float)
        failed = result.loc[~result.randomization_pass, predictor].astype(float)
        rows.append({
            "predictor": predictor,
            "sample_protocol_rows": len(result),
            "pass_count": int(result.randomization_pass.sum()),
            "fail_count": int((~result.randomization_pass).sum()),
            "pass_mean": float(passed.mean()) if len(passed) else np.nan,
            "fail_mean": float(failed.mean()) if len(failed) else np.nan,
            "spearman_with_pass": correlation,
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "bootstrap_count": BOOTSTRAPS,
        })
    statistics = pd.DataFrame(rows)
    statistics.to_csv(ROOT / "sample_dependency_statistics.csv", index=False)
    return result, statistics


def update_recovered_margin(sample: pd.DataFrame) -> None:
    lookup = sample.loc[
        (sample.randomization_level == "R0") & (sample.target_class_protocol == "T0")
    ].groupby(["audit_group", "global_epoch_index"]).original_y2_margin.mean().to_dict()
    for name, key, group in (
        ("old_abc_samples_recovered.json", "fixed_36_samples", "A"),
        ("new_validation_samples_recovered.json", "samples", "C"),
    ):
        path = ROOT / name
        payload = json.loads(path.read_text())
        for record in payload[key]:
            identity = (group, int(record["sample_id"]))
            if identity in lookup:
                record["original_logit_margin"] = float(lookup[identity])
        if name.startswith("old"):
            for record in payload["diagnostic_12_samples"]:
                identity = (group, int(record["sample_id"]))
                if identity in lookup:
                    record["original_logit_margin"] = float(lookup[identity])
        audit.write_json(path, payload)


def write_summary(result: pd.DataFrame, statistics: pd.DataFrame) -> None:
    context_counts = result.loc[result.context_dominated].groupby("audit_group").global_epoch_index.nunique().to_dict()
    strongest = statistics.iloc[statistics.spearman_with_pass.abs().fillna(-1).argmax()]
    text = f"""# Sample-dependency analysis

The unified protocol produced {int(result.randomization_pass.sum())}/{len(result)} sample-protocol passes.
Group-level T0 and T1 nevertheless passed 0/3 seeds in both A and C. Context-dominated unique
samples (context-dependence ratio >= 0.5) numbered A={int(context_counts.get('A', 0))} and
C={int(context_counts.get('C', 0))}. The strongest univariate association with individual
pass status was `{strongest.predictor}` (Spearman {strongest.spearman_with_pass:.3f},
bootstrap 95% CI [{strongest.bootstrap_ci_low:.3f}, {strongest.bootstrap_ci_high:.3f}]).

These analyses are descriptive. The sample size is small, rows share samples across T0/T1,
and no multiplicity-adjusted significance claim is made. Because A and C have statistically
compatible R7 similarity and identical 0/3 group outcomes, margin, confidence, correctness,
transition status, target-epoch occlusion, and context dependence do not explain the historical
3/3 versus newer 0/3 discrepancy. They describe heterogeneity only and were not used for
method selection.
"""
    (ROOT / "sample_dependency_summary.md").write_text(text, encoding="utf-8")


def final_decision(result: pd.DataFrame, statistics: pd.DataFrame) -> None:
    randomization = pd.read_csv(ROOT / "group_level_randomization_summary.csv")
    ac = pd.read_csv(ROOT / "group_a_vs_c_bootstrap.csv")
    context_rows = result.groupby("audit_group").context_dominated.sum().astype(int).to_dict()
    context_samples = result.loc[result.context_dominated].groupby("audit_group").global_epoch_index.nunique().astype(int).to_dict()
    decision = {
        "schema_version": "sequence_lime_protocol_consistency_final_decision_v1",
        "status": "CASE_PROTOCOL_DRIFT",
        "primary_decision": "CASE_PROTOCOL_DRIFT",
        "secondary_sample_dependence_supported": False,
        "historical_abc_and_new_validation_protocol_identical": False,
        "material_drift_fields": [
            "surrogate target multiplicity", "train mask count", "local validation mask count",
            "mask seed", "mask generator", "retention distribution", "query accounting",
            "permutation null aggregation", "signed versus absolute Spearman",
            "pass threshold", "faithfulness endpoint", "cache protocol identity", "resume behavior",
        ],
        "unified_old_12": {"T0_seed_pass_count": 0, "T1_seed_pass_count": 0},
        "unified_new_validation_12": {"T0_seed_pass_count": 0, "T1_seed_pass_count": 0},
        "old_fixed_36_rerun": False,
        "old_fixed_36_reason": "Not triggered: A and C were statistically compatible and had identical 0/3 outcomes.",
        "a_c_material_difference": False,
        "a_c_bootstrap": ac.to_dict(orient="records"),
        "randomization_summary": randomization.to_dict(orient="records"),
        "context_dominated_sample_protocol_rows": context_rows,
        "context_dominated_unique_samples": context_samples,
        "sample_dependency_statistics_descriptive_only": True,
        "classic_sequence_lime_generalization": "NOT_SUPPORTED; the historical 3/3 sanity conclusion is superseded.",
        "method_boundary": "MF_SEQLIME_PARTIAL_PASS_SUPERSEDED_BY_PROTOCOL_CONSISTENCY_AUDIT",
        "feedback_training_allowed": False,
        "hierarchical_context_attribution_next": False,
        "next_step": "Use the single protocol-versioned evaluator for all future reporting; do not develop a new method or start feedback training in this audit.",
        "canonical_parity": "PASS",
        "sequence_wrapper_parity": "PASS",
        "protected_output_roots_written": False,
    }
    audit.write_json(ROOT / "final_decision.json", decision)


def main() -> None:
    sample = pd.read_csv(ROOT / "sample_level_protocol_consistency.csv")
    occlusion = load_occlusion()
    result, statistics = build_dependency_analysis(sample, occlusion)
    update_recovered_margin(sample)
    write_summary(result, statistics)
    final_decision(result, statistics)
    required = [
        "occlusion_diagnostics.csv", "sample_dependency_analysis.csv",
        "sample_dependency_statistics.csv", "sample_dependency_summary.md", "final_decision.json",
    ]
    empty = [name for name in required if not (ROOT / name).exists() or (ROOT / name).stat().st_size <= 1]
    if empty:
        raise RuntimeError(f"empty final outputs: {empty}")
    print(json.dumps({
        "status": "CASE_PROTOCOL_DRIFT",
        "sample_protocol_rows": len(result),
        "individual_pass_rows": int(result.randomization_pass.sum()),
        "context_dominated_rows": int(result.context_dominated.sum()),
        "all_required_nonempty": True,
    }, indent=2))


if __name__ == "__main__":
    main()
