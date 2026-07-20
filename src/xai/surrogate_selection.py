"""Validation-only selection utilities for surrogate-improvement candidates."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SELECTION_WEIGHTS = {
    "fidelity": 0.35,
    "faithfulness": 0.30,
    "stability": 0.15,
    "randomization": 0.20,
}
REQUIRED_SUMMARY_COLUMNS = (
    "candidate",
    "validation_weighted_r2",
    "validation_mae",
    "validation_rmse",
    "deletion_auc",
    "insertion_auc",
    "faithfulness_advantage_r0",
    "faithfulness_advantage_r7",
    "stability_spearman",
    "r7_similarity",
    "r7_null_q95",
    "r7_t0_seed_pass_count",
    "r7_t1_seed_pass_count",
    "query_count",
)


def _oriented_minmax(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    low, high = float(numeric.min()), float(numeric.max())
    if high - low <= 1e-12:
        result = pd.Series(np.ones(len(numeric)), index=numeric.index, dtype=float)
    else:
        result = (numeric - low) / (high - low)
    return result if higher_is_better else 1.0 - result


def score_and_gate_candidates(
    summary: pd.DataFrame,
    baseline_candidate: str = "C0_CLASSIC_Y2",
    tolerance: float = 1e-8,
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    """Apply all hard gates, then rank only eligible validation candidates.

    This function is intentionally split-agnostic; callers must provide the
    validation summary and the returned decision always records that test data
    were not used.  The classic baseline remains a comparator, not a candidate.
    """
    missing = [column for column in REQUIRED_SUMMARY_COLUMNS if column not in summary]
    if missing:
        raise ValueError(f"candidate summary missing columns: {missing}")
    table = summary.copy().reset_index(drop=True)
    if table["candidate"].duplicated().any():
        raise ValueError("candidate summary must have one row per candidate")
    baseline_rows = table.loc[table.candidate == baseline_candidate]
    if len(baseline_rows) != 1:
        raise ValueError("exactly one classic baseline row is required")
    baseline = baseline_rows.iloc[0]

    table["gate_r2"] = table.validation_weighted_r2 >= float(baseline.validation_weighted_r2) - tolerance
    table["gate_mae"] = table.validation_mae <= float(baseline.validation_mae) + tolerance
    table["gate_rmse"] = table.validation_rmse <= float(baseline.validation_rmse) + tolerance
    deletion_better = table.deletion_auc < float(baseline.deletion_auc) - tolerance
    deletion_not_worse = table.deletion_auc <= float(baseline.deletion_auc) + tolerance
    insertion_better = table.insertion_auc > float(baseline.insertion_auc) + tolerance
    insertion_not_worse = table.insertion_auc >= float(baseline.insertion_auc) - tolerance
    table["gate_faithfulness_pair"] = (
        (deletion_better & insertion_not_worse)
        | (insertion_better & deletion_not_worse)
    )
    table["gate_t0_randomization"] = table.r7_t0_seed_pass_count.astype(int) >= 2
    table["gate_t1_randomization"] = table.r7_t1_seed_pass_count.astype(int) >= 2
    table["gate_r7_near_null"] = table.r7_similarity.abs() <= table.r7_null_q95.abs() + tolerance
    decay_limit = np.maximum(0.5 * table.faithfulness_advantage_r0.abs(), 0.01)
    table["gate_faithfulness_decay"] = table.faithfulness_advantage_r7.abs() <= decay_limit + tolerance
    table["gate_equal_budget"] = table.query_count.astype(int) == int(baseline.query_count)
    gate_columns = [
        "gate_r2", "gate_mae", "gate_rmse", "gate_faithfulness_pair",
        "gate_t0_randomization", "gate_t1_randomization", "gate_r7_near_null",
        "gate_faithfulness_decay", "gate_equal_budget",
    ]
    table["hard_gate_pass"] = table[gate_columns].all(axis=1)

    fidelity = (
        _oriented_minmax(table.validation_weighted_r2)
        + _oriented_minmax(table.validation_mae, higher_is_better=False)
        + _oriented_minmax(table.validation_rmse, higher_is_better=False)
    ) / 3.0
    faithfulness = (
        _oriented_minmax(table.deletion_auc, higher_is_better=False)
        + _oriented_minmax(table.insertion_auc)
    ) / 2.0
    stability = _oriented_minmax(table.stability_spearman)
    randomization = _oriented_minmax(table.r7_similarity.abs(), higher_is_better=False)
    table["fidelity_score"] = fidelity
    table["faithfulness_score"] = faithfulness
    table["stability_score"] = stability
    table["randomization_score"] = randomization
    table["validation_multiobjective_score"] = (
        SELECTION_WEIGHTS["fidelity"] * fidelity
        + SELECTION_WEIGHTS["faithfulness"] * faithfulness
        + SELECTION_WEIGHTS["stability"] * stability
        + SELECTION_WEIGHTS["randomization"] * randomization
    )
    table["selected"] = False
    eligible = table.loc[(table.candidate != baseline_candidate) & table.hard_gate_pass]
    if eligible.empty:
        decision = {
            "schema_version": "mf_surrogate_validation_decision_v1",
            "status": "VALIDATION_NO_QUALIFIED_CANDIDATE",
            "selection_split": "validation",
            "test_data_used": False,
            "test_evaluation_allowed": False,
            "selected_candidate": None,
            "selection_weights": dict(SELECTION_WEIGHTS),
        }
    else:
        selected_index = eligible.sort_values(
            ["validation_multiobjective_score", "candidate"],
            ascending=[False, True],
        ).index[0]
        table.loc[selected_index, "selected"] = True
        selected = table.loc[selected_index]
        decision = {
            "schema_version": "mf_surrogate_validation_decision_v1",
            "status": "VALIDATION_QUALIFIED_CANDIDATE",
            "selection_split": "validation",
            "test_data_used": False,
            "test_evaluation_allowed": True,
            "selected_candidate": str(selected.candidate),
            "selected_validation_score": float(selected.validation_multiobjective_score),
            "selection_weights": dict(SELECTION_WEIGHTS),
        }
    return table, decision


def permutation_similarity_null(
    first: Sequence[float],
    second: Sequence[float],
    seed: int,
    permutations: int = 1000,
) -> Mapping[str, object]:
    """Return an auditable absolute-Spearman permutation null distribution."""
    first_array = np.asarray(first, dtype=np.float64).ravel()
    second_array = np.asarray(second, dtype=np.float64).ravel()
    if first_array.shape != second_array.shape or first_array.size < 3:
        raise ValueError("equal vectors of length at least three are required")
    count = int(permutations)
    if count < 1:
        raise ValueError("permutations must be positive")
    rng = np.random.default_rng(int(seed))
    observed = float(abs(spearmanr(first_array, second_array).statistic))
    null = np.empty(count, dtype=np.float64)
    for index in range(count):
        null[index] = abs(float(spearmanr(first_array, rng.permutation(second_array)).statistic))
    return {
        "schema_version": "absolute_spearman_permutation_null_v1",
        "seed": int(seed),
        "permutations": count,
        "observed": observed,
        "null_mean": float(np.mean(null)),
        "null_q95": float(np.quantile(null, 0.95)),
        "p_value": float((1 + np.sum(null >= observed)) / (count + 1)),
        "null_values": null,
    }
