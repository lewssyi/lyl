#!/usr/bin/env python
"""Guarded validation aggregation and hard-gate decision for HC-SeqLIME."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools.hierarchical_context_xai_feasibility import (  # noqa: E402
    AUTHORITY,
    ROOT,
    authority_guard,
    write_json,
)


EXPECTED_RUNS = 1080
NULL_PERMUTATIONS = 1000
LEVELS = ("R0", "R3", "R5", "R7")
PROTOCOLS = ("T0", "T1")
SEEDS = (3101, 3102, 3103)


def method_name(frame: pd.DataFrame) -> pd.Series:
    return np.where(
        frame.candidate.eq("B0_CLASSIC_90"),
        "B0_CLASSIC_90",
        frame.candidate
        + "_"
        + frame.group_scheme
        + "_F"
        + frame.feature_count.astype(str),
    )


def required_paths(directory: Path, classic: bool) -> tuple[Path, ...]:
    names = (
        (
            "classic_sample_level.csv",
            "classic_attribution.csv",
            "runtime_query_summary.csv",
        )
        if classic
        else (
            "hierarchical_sample_level.csv",
            "context_group_importance.csv",
            "within_epoch_attribution.csv",
            "runtime_query_summary.csv",
        )
    )
    return tuple(directory / name for name in names)


def load_validation() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]
]:
    sample_frames: list[pd.DataFrame] = []
    context_frames: list[pd.DataFrame] = []
    within_frames: list[pd.DataFrame] = []
    runtime_frames: list[pd.DataFrame] = []
    manifests: list[dict] = []
    bad: list[str] = []
    for manifest_path in sorted(
        (ROOT / "runs").glob("validation_*/run_manifest.json")
    ):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            directory = manifest_path.parent
            classic = manifest.get("candidate") == "B0_CLASSIC_90"
            required = required_paths(directory, classic)
            valid = (
                manifest.get("status") == "COMPLETE"
                and manifest.get("all_outputs_nonempty") is True
                and int(
                    manifest.get("logical_query_budget_per_candidate", -1)
                )
                == 1530
                and all(
                    path.exists() and path.stat().st_size > 1
                    for path in required
                )
            )
            if not valid:
                bad.append(str(manifest_path.relative_to(ROOT)))
                continue
            sample_name = (
                "classic_sample_level.csv"
                if classic
                else "hierarchical_sample_level.csv"
            )
            sample = pd.read_csv(directory / sample_name)
            sample["run_id"] = manifest["run_id"]
            sample_frames.append(sample)
            if not classic:
                context_frames.append(
                    pd.read_csv(directory / "context_group_importance.csv")
                )
                within_frames.append(
                    pd.read_csv(directory / "within_epoch_attribution.csv")
                )
            runtime = pd.read_csv(directory / "runtime_query_summary.csv")
            runtime["run_id"] = manifest["run_id"]
            runtime["kind"] = "classic" if classic else "hierarchical"
            runtime_frames.append(runtime)
            manifests.append(manifest)
        except Exception as error:
            bad.append(
                f"{manifest_path.relative_to(ROOT)}:{type(error).__name__}"
            )
    if bad or len(manifests) != EXPECTED_RUNS:
        raise RuntimeError(
            f"validation coverage invalid: runs={len(manifests)}, bad={bad[:10]}"
        )
    samples = pd.concat(sample_frames, ignore_index=True)
    samples["method"] = method_name(samples)
    contexts = pd.concat(context_frames, ignore_index=True)
    within = pd.concat(within_frames, ignore_index=True)
    runtime = pd.concat(runtime_frames, ignore_index=True)
    if len(samples) != 2160:
        raise RuntimeError(f"sample row coverage drift: {len(samples)}")
    expected_methods = {
        "B0_CLASSIC_90": 720,
        "H3_G25_F45": 720,
        "H3_G9_F45": 720,
    }
    if samples.method.value_counts().to_dict() != expected_methods:
        raise RuntimeError("method coverage drift")
    if set(samples.query_budget.unique()) != {1530}:
        raise RuntimeError("query budget drift")
    if samples.stage_prior_used.astype(bool).any():
        raise RuntimeError("stage prior detected")
    return samples, contexts, within, runtime, manifests


def attribution_vector(row: pd.Series) -> np.ndarray:
    within = json.loads(row.within_epoch_importance)
    if row.method == "B0_CLASSIC_90":
        return np.asarray(within, dtype=np.float64)
    context = json.loads(row.context_importance)
    flattened_within = [value for vector in within for value in vector]
    return np.asarray(context + flattened_within, dtype=np.float64)


def normalized_ranks(vector: np.ndarray) -> np.ndarray | None:
    ranks = rankdata(np.asarray(vector, dtype=np.float64), method="average")
    ranks -= ranks.mean()
    norm = float(np.linalg.norm(ranks))
    return None if norm <= 1e-15 else ranks / norm


def absolute_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = normalized_ranks(left)
    right_ranks = normalized_ranks(right)
    if left_ranks is None or right_ranks is None:
        return 1.0
    return abs(float(np.dot(left_ranks, right_ranks)))


def stable_seed(text: str) -> int:
    return int.from_bytes(
        hashlib.sha256(text.encode("utf-8")).digest()[:8], "little"
    )


def permutation_null(
    pairs: Iterable[tuple[np.ndarray, np.ndarray]], key: str
) -> tuple[np.ndarray, int]:
    prepared = []
    degenerate = 0
    for left, right in pairs:
        left_ranks = normalized_ranks(left)
        right_ranks = normalized_ranks(right)
        if left_ranks is None or right_ranks is None:
            degenerate += 1
            continue
        prepared.append((left_ranks, right_ranks))
    if not prepared:
        return np.zeros(NULL_PERMUTATIONS, dtype=np.float64), degenerate
    rng = np.random.default_rng(stable_seed(key))
    null = np.zeros(NULL_PERMUTATIONS, dtype=np.float64)
    for permutation_index in range(NULL_PERMUTATIONS):
        values = [
            abs(float(np.dot(left, right[rng.permutation(len(right))])))
            for left, right in prepared
        ]
        null[permutation_index] = float(np.mean(values))
    return null, degenerate


def stability_frame(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    original = samples.loc[samples.randomization_level.eq("R0")]
    for keys, group in original.groupby(
        ["method", "target_protocol", "sample_id"], sort=True
    ):
        by_seed = group.set_index("randomization_seed")
        values = []
        for left_seed, right_seed in (
            (3101, 3102),
            (3101, 3103),
            (3102, 3103),
        ):
            values.append(
                absolute_spearman(
                    attribution_vector(by_seed.loc[left_seed]),
                    attribution_vector(by_seed.loc[right_seed]),
                )
            )
        rows.append(
            {
                "method": keys[0],
                "target_protocol": keys[1],
                "sample_id": int(keys[2]),
                "mask_seed_stability": float(np.mean(values)),
            }
        )
    return pd.DataFrame(rows)


def randomization_frame(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    relevant = samples.loc[
        samples.randomization_level.isin(("R0", "R7"))
    ]
    for keys, group in relevant.groupby(
        ["method", "target_protocol", "sample_id", "randomization_seed"],
        sort=True,
    ):
        levels = group.set_index("randomization_level")
        r0 = levels.loc["R0"]
        r7 = levels.loc["R7"]
        pair_rows.append(
            {
                "method": keys[0],
                "target_protocol": keys[1],
                "sample_id": int(keys[2]),
                "randomization_seed": int(keys[3]),
                "R7_similarity": absolute_spearman(
                    attribution_vector(r0), attribution_vector(r7)
                ),
                "R0_faithfulness": float(r0.faithfulness),
                "R7_faithfulness": float(r7.faithfulness),
                "_r0_vector": attribution_vector(r0),
                "_r7_vector": attribution_vector(r7),
            }
        )
    pairs = pd.DataFrame(pair_rows)
    summaries = []
    for keys, group in pairs.groupby(
        ["method", "target_protocol", "randomization_seed"], sort=True
    ):
        null, degenerate = permutation_null(
            zip(group._r0_vector, group._r7_vector),
            f"{keys[0]}|{keys[1]}|{keys[2]}",
        )
        observed = float(group.R7_similarity.mean())
        null_q95 = float(np.quantile(null, 0.95))
        r0_faith = float(group.R0_faithfulness.mean())
        r7_faith = float(group.R7_faithfulness.mean())
        faith_ratio = r7_faith / max(abs(r0_faith), 1e-8)
        near_null = observed <= null_q95
        faith_decay = bool(
            r7_faith < r0_faith and faith_ratio <= 0.5
        )
        summaries.append(
            {
                "method": keys[0],
                "target_protocol": keys[1],
                "randomization_seed": int(keys[2]),
                "sample_count": int(len(group)),
                "R7_similarity_mean": observed,
                "permutation_null_mean": float(null.mean()),
                "permutation_null_q95": null_q95,
                "permutation_p_upper": float(
                    (1 + np.sum(null >= observed))
                    / (NULL_PERMUTATIONS + 1)
                ),
                "degenerate_pair_count": int(degenerate),
                "R0_faithfulness_mean": r0_faith,
                "R7_faithfulness_mean": r7_faith,
                "R7_to_R0_faithfulness_ratio": faith_ratio,
                "R7_near_null": bool(near_null),
                "faithfulness_clearly_decayed": faith_decay,
                "seed_protocol_pass": bool(near_null and faith_decay),
                "null_permutations": NULL_PERMUTATIONS,
            }
        )
    pairs = pairs.drop(columns=["_r0_vector", "_r7_vector"])
    return pairs, pd.DataFrame(summaries)


def reliability_outputs(
    samples: pd.DataFrame,
    stability: pd.DataFrame,
    randomization_pairs: pd.DataFrame,
    randomization_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    original = (
        samples.loc[samples.randomization_level.eq("R0")]
        .groupby(["method", "target_protocol", "sample_id"], as_index=False)
        .agg(
            source_recording=("source_recording", "first"),
            true_stage=("true_stage", "first"),
            predicted_stage=("predicted_stage", "first"),
            correct=("correct", "first"),
            transition_boundary=("transition_boundary", "first"),
            confidence=("confidence", "first"),
            hierarchical_surrogate_R2=("hierarchical_surrogate_R2", "mean"),
            local_surrogate_R2=("local_surrogate_R2", "mean"),
            context_surrogate_R2=("context_surrogate_R2", "mean"),
            response_variance=("response_variance", "mean"),
            effective_sample_size=("effective_sample_size", "mean"),
            context_dependence_ratio=("context_dependence_ratio", "mean"),
            target_epoch_drop=("target_epoch_drop", "mean"),
        )
        .merge(
            stability,
            on=["method", "target_protocol", "sample_id"],
            how="left",
            validate="one_to_one",
        )
    )
    random_sample = (
        randomization_pairs.groupby(
            ["method", "target_protocol", "sample_id"], as_index=False
        )
        .agg(R7_similarity=("R7_similarity", "mean"))
    )
    original = original.merge(
        random_sample,
        on=["method", "target_protocol", "sample_id"],
        how="left",
        validate="one_to_one",
    )
    hc = original.loc[original.method.ne("B0_CLASSIC_90")]
    thresholds = {
        "response_variance_low_q10": float(
            hc.response_variance.quantile(0.10)
        ),
        "hierarchical_surrogate_r2_low_q25": float(
            hc.hierarchical_surrogate_R2.quantile(0.25)
        ),
        "mask_seed_stability_low_q25": float(
            hc.mask_seed_stability.quantile(0.25)
        ),
        "context_dominated_ratio": 0.5,
        "confidence_high_median": float(original.confidence.median()),
        "R7_similarity_q95_by_method_protocol": {
            f"{keys[0]}|{keys[1]}": float(group.permutation_null_q95.mean())
            for keys, group in randomization_summary.groupby(
                ["method", "target_protocol"]
            )
        },
    }
    q95_lookup = thresholds["R7_similarity_q95_by_method_protocol"]

    def classify(row: pd.Series) -> str:
        if row.response_variance <= thresholds["response_variance_low_q10"]:
            return "LOW_MODEL_RESPONSE"
        if (
            not np.isfinite(row.hierarchical_surrogate_R2)
            or row.hierarchical_surrogate_R2
            < thresholds["hierarchical_surrogate_r2_low_q25"]
        ):
            return "SURROGATE_UNRELIABLE"
        key = f"{row.method}|{row.target_protocol}"
        if (
            row.mask_seed_stability
            < thresholds["mask_seed_stability_low_q25"]
            or row.R7_similarity > q95_lookup[key]
        ):
            return "RANDOMIZATION_UNRELIABLE"
        if (
            np.isfinite(row.context_dependence_ratio)
            and row.context_dependence_ratio
            >= thresholds["context_dominated_ratio"]
        ):
            return "CONTEXT_DOMINATED"
        return "LOCALLY_EXPLAINABLE"

    original["reliability_state"] = original.apply(classify, axis=1)
    original["reliable"] = original.reliability_state.isin(
        ("LOCALLY_EXPLAINABLE", "CONTEXT_DOMINATED")
    )
    coverage = (
        original.groupby(
            ["method", "target_protocol", "reliability_state"],
            as_index=False,
        )
        .agg(sample_count=("sample_id", "size"))
    )
    totals = original.groupby(
        ["method", "target_protocol"]
    ).sample_id.transform("size")
    original["coverage_weight"] = 1.0 / totals
    coverage["coverage_ratio"] = coverage.apply(
        lambda row: float(row.sample_count)
        / float(
            len(
                original.loc[
                    original.method.eq(row.method)
                    & original.target_protocol.eq(row.target_protocol)
                ]
            )
        ),
        axis=1,
    )
    reliability_summary = (
        original.groupby(["method", "target_protocol"], as_index=False)
        .agg(
            total_samples=("sample_id", "size"),
            reliable_samples=("reliable", "sum"),
        )
    )
    reliability_summary["reliable_coverage"] = (
        reliability_summary.reliable_samples
        / reliability_summary.total_samples
    )
    reliability_summary["rejection_rate"] = (
        1.0 - reliability_summary.reliable_coverage
    )
    coverage = coverage.merge(
        reliability_summary,
        on=["method", "target_protocol"],
        how="left",
    )

    confidence_high = (
        original.confidence >= thresholds["confidence_high_median"]
    )
    context_high = original.context_dependence_ratio >= 0.5
    stratum_specs = {
        "true_stage": original.true_stage.astype(str),
        "correctness": np.where(original.correct, "correct", "incorrect"),
        "transition": np.where(
            original.transition_boundary, "transition", "stable"
        ),
        "confidence": np.where(confidence_high, "high", "low"),
        "context_dependence": np.where(
            context_high.fillna(False), "high", "low_or_not_applicable"
        ),
    }
    stratified_rows = []
    for stratum, values in stratum_specs.items():
        temporary = original.assign(stratum_value=values)
        grouped = temporary.groupby(
            ["method", "target_protocol", "stratum_value"], as_index=False
        ).agg(
            sample_count=("sample_id", "size"),
            reliable_count=("reliable", "sum"),
        )
        grouped["stratum"] = stratum
        grouped["reliable_coverage"] = (
            grouped.reliable_count / grouped.sample_count
        )
        grouped["rejection_rate"] = 1.0 - grouped.reliable_coverage
        stratified_rows.append(grouped)
    stratified = pd.concat(stratified_rows, ignore_index=True)
    return original, coverage, stratified, thresholds


def main() -> None:
    authority = authority_guard()
    samples, contexts, within, runtime, manifests = load_validation()
    stability = stability_frame(samples)
    random_pairs, random_summary = randomization_frame(samples)
    (
        reliability_sample,
        reliability_coverage,
        stratified_coverage,
        thresholds,
    ) = reliability_outputs(
        samples, stability, random_pairs, random_summary
    )

    original = samples.loc[samples.randomization_level.eq("R0")]
    fidelity = (
        original.groupby(["method", "target_protocol"], as_index=False)
        .agg(
            sample_seed_rows=("sample_id", "size"),
            local_surrogate_R2=("local_surrogate_R2", "mean"),
            context_surrogate_R2=("context_surrogate_R2", "mean"),
            hierarchical_surrogate_R2=("hierarchical_surrogate_R2", "mean"),
        )
    )
    faithfulness = (
        samples.groupby(
            [
                "method",
                "target_protocol",
                "randomization_level",
                "randomization_seed",
            ],
            as_index=False,
        )
        .agg(
            sample_count=("sample_id", "size"),
            deletion_auc=("deletion_auc", "mean"),
            insertion_auc=("insertion_auc", "mean"),
            faithfulness=("faithfulness", "mean"),
        )
    )
    stability_summary = (
        stability.groupby(["method", "target_protocol"], as_index=False)
        .agg(
            sample_count=("sample_id", "size"),
            mask_seed_stability_mean=("mask_seed_stability", "mean"),
            mask_seed_stability_median=("mask_seed_stability", "median"),
        )
    )

    baseline_metrics = original.loc[
        original.method.eq("B0_CLASSIC_90")
    ]
    baseline_fidelity = float(
        baseline_metrics.hierarchical_surrogate_R2.mean()
    )
    baseline_deletion = float(baseline_metrics.deletion_auc.mean())
    baseline_insertion = float(baseline_metrics.insertion_auc.mean())
    baseline_stability = float(
        stability.loc[
            stability.method.eq("B0_CLASSIC_90"),
            "mask_seed_stability",
        ].mean()
    )
    selection_rows = []
    for method in ("H3_G25_F45", "H3_G9_F45"):
        candidate_original = original.loc[original.method.eq(method)]
        candidate_fidelity = fidelity.loc[fidelity.method.eq(method)]
        best_fidelity = float(
            np.nanmax(
                [
                    candidate_fidelity.context_surrogate_R2.mean(),
                    candidate_fidelity.hierarchical_surrogate_R2.mean(),
                ]
            )
        )
        deletion = float(candidate_original.deletion_auc.mean())
        insertion = float(candidate_original.insertion_auc.mean())
        stability_mean = float(
            stability.loc[
                stability.method.eq(method), "mask_seed_stability"
            ].mean()
        )
        random = random_summary.loc[random_summary.method.eq(method)]
        seed_pass_counts = (
            random.groupby("target_protocol").seed_protocol_pass.sum().to_dict()
        )
        near_null_counts = (
            random.groupby("target_protocol").R7_near_null.sum().to_dict()
        )
        decay_counts = (
            random.groupby(
                "target_protocol"
            ).faithfulness_clearly_decayed.sum().to_dict()
        )
        method_rows = samples.loc[samples.method.eq(method)]
        gate_values = {
            "gate_1_fidelity_above_classic": best_fidelity
            > baseline_fidelity,
            "gate_2_faithfulness_endpoint": (
                deletion <= baseline_deletion
                and insertion >= baseline_insertion
                and (
                    deletion < baseline_deletion
                    or insertion > baseline_insertion
                )
            ),
            "gate_3_stability_noninferior": stability_mean
            >= baseline_stability,
            "gate_4_T0_at_least_2_of_3": int(
                seed_pass_counts.get("T0", 0)
            )
            >= 2,
            "gate_5_T1_at_least_2_of_3": int(
                seed_pass_counts.get("T1", 0)
            )
            >= 2,
            "gate_6_R7_near_permutation_null": all(
                int(near_null_counts.get(protocol, 0)) >= 2
                for protocol in PROTOCOLS
            ),
            "gate_7_faithfulness_decays_after_randomization": all(
                int(decay_counts.get(protocol, 0)) >= 2
                for protocol in PROTOCOLS
            ),
            "gate_8_equal_query_budget": set(
                method_rows.query_budget.unique()
            )
            == {1530},
            "gate_9_no_stage_prior": not method_rows.stage_prior_used.astype(
                bool
            ).any(),
            "gate_10_complete_coverage_reporting": (
                len(reliability_sample.loc[reliability_sample.method.eq(method)])
                == 60
                and not stratified_coverage.loc[
                    stratified_coverage.method.eq(method)
                ].empty
            ),
        }
        row = {
            "method": method,
            "classic_fidelity": baseline_fidelity,
            "candidate_best_context_or_hierarchical_fidelity": best_fidelity,
            "classic_deletion_auc": baseline_deletion,
            "candidate_deletion_auc": deletion,
            "classic_insertion_auc": baseline_insertion,
            "candidate_insertion_auc": insertion,
            "classic_mask_seed_stability": baseline_stability,
            "candidate_mask_seed_stability": stability_mean,
            "T0_seed_protocol_pass_count": int(
                seed_pass_counts.get("T0", 0)
            ),
            "T1_seed_protocol_pass_count": int(
                seed_pass_counts.get("T1", 0)
            ),
            **gate_values,
        }
        row["all_10_validation_gates_pass"] = all(gate_values.values())
        selection_rows.append(row)
    selection = pd.DataFrame(selection_rows)
    qualified = selection.loc[
        selection.all_10_validation_gates_pass, "method"
    ].tolist()
    if len(qualified) > 1:
        raise RuntimeError("validation did not freeze a unique candidate")
    fixed_test_allowed = len(qualified) == 1
    status = (
        "VALIDATION_QUALIFIED_HIERARCHICAL_CANDIDATE"
        if fixed_test_allowed
        else "VALIDATION_NO_QUALIFIED_HIERARCHICAL_CANDIDATE"
    )

    context_dependency = samples.loc[
        samples.method.ne("B0_CLASSIC_90"),
        [
            "method",
            "data_split",
            "source_recording",
            "sample_id",
            "true_stage",
            "predicted_stage",
            "correct",
            "transition_boundary",
            "confidence",
            "target_protocol",
            "randomization_level",
            "randomization_seed",
            "target_epoch_drop",
            "context_dependence_ratio",
            "response_variance",
            "effective_sample_size",
        ],
    ]
    context_dependency.to_csv(
        ROOT / "context_dependency_sample_level.csv", index=False
    )
    contexts.to_csv(ROOT / "context_group_importance.csv", index=False)
    within.to_csv(ROOT / "within_epoch_attribution.csv", index=False)
    fidelity.to_csv(ROOT / "hierarchical_surrogate_fidelity.csv", index=False)
    faithfulness.to_csv(ROOT / "faithfulness_summary.csv", index=False)
    stability_summary.to_csv(ROOT / "stability_summary.csv", index=False)
    random_summary.to_csv(ROOT / "randomization_summary.csv", index=False)
    reliability_sample.to_csv(
        ROOT / "reliability_sample_level.csv", index=False
    )
    reliability_coverage.to_csv(
        ROOT / "reliability_coverage.csv", index=False
    )
    stratified_coverage.to_csv(
        ROOT / "stratified_coverage.csv", index=False
    )
    selection.to_csv(
        ROOT / "validation_candidate_selection.csv", index=False
    )
    runtime.to_csv(ROOT / "runtime_query_summary.csv", index=False)

    frozen = {
        "schema_version": "hc_seqlime_frozen_validation_selection_v1",
        "status": status,
        "selection_split": "validation",
        "validation_run_count": EXPECTED_RUNS,
        "validation_sample_count": 30,
        "selected_candidate": qualified[0] if qualified else None,
        "fixed_test_allowed": fixed_test_allowed,
        "test_execution_count": 0,
        "test_data_used_for_selection": False,
        "all_required_outputs_nonempty": True,
        "hard_gate_count": 10,
        "all_hard_gates_required": True,
        "candidate_gate_records": json.loads(
            selection.to_json(orient="records")
        ),
        "reliability_thresholds_frozen_on_validation": thresholds,
        "randomization_null_permutations": NULL_PERMUTATIONS,
        "feedback_training_status": "BLOCKED",
        "scientific_status": "MODEL_FAITHFULNESS_NOT_ESTABLISHED",
    }
    write_json(ROOT / "frozen_validation_selection.json", frozen)
    decision = {
        "schema_version": "hc_seqlime_final_scientific_decision_v1",
        "status": status,
        "paper_level_decision": (
            "PENDING_FIXED_TEST"
            if fixed_test_allowed
            else "HC_SEQLIME_FAIL"
        ),
        "fixed_test_executed": False,
        "fixed_test_execution_count": 0,
        "authoritative_scientific_status": authority[
            "scientific_status"
        ],
        "feedback_training_status": authority["feedback_training_status"],
        "interpretation": (
            "A hierarchical candidate passed all validation gates; the frozen "
            "test may run exactly once."
            if fixed_test_allowed
            else "No hierarchical candidate passed all ten validation gates; "
            "the frozen test remains blocked."
        ),
    }
    write_json(ROOT / "final_scientific_decision.json", decision)
    run_manifest = json.loads(
        (ROOT / "run_manifest.json").read_text(encoding="utf-8")
    )
    run_manifest["status"] = status
    run_manifest["validation"] = {
        "completed_runs": EXPECTED_RUNS,
        "failed_final": 0,
        "sample_count": 30,
        "candidate_count": 2,
        "classic_baseline_included": True,
        "all_required_outputs_nonempty": True,
        "selection": "frozen_validation_selection.json",
    }
    run_manifest["test_eligible"] = fixed_test_allowed
    run_manifest["test_execution_count"] = 0
    write_json(ROOT / "run_manifest.json", run_manifest)

    required_final = (
        "context_dependency_sample_level.csv",
        "context_group_importance.csv",
        "within_epoch_attribution.csv",
        "hierarchical_surrogate_fidelity.csv",
        "faithfulness_summary.csv",
        "stability_summary.csv",
        "randomization_summary.csv",
        "reliability_coverage.csv",
        "stratified_coverage.csv",
        "validation_candidate_selection.csv",
        "frozen_validation_selection.json",
        "final_scientific_decision.json",
        "runtime_query_summary.csv",
    )
    empty = [
        name
        for name in required_final
        if not (ROOT / name).exists() or (ROOT / name).stat().st_size <= 1
    ]
    if empty:
        raise RuntimeError(f"empty required validation outputs: {empty}")
    print(selection.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
