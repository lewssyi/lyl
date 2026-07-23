"""Validation-frozen reliability status and candidate-selection rules."""
from __future__ import annotations

from typing import Mapping, Sequence


REQUIRED_GATES = (
    "surrogate_r2_better",
    "faithfulness_gate",
    "stability_noninferior",
    "t0_pass",
    "t1_pass",
    "r7_null_compatible",
    "randomization_faithfulness_decay",
    "equal_query_budget",
    "no_stage_prior",
    "complete_coverage",
)


def reliability_status(metrics: Mapping[str, float], thresholds: Mapping[str, float]) -> str:
    """Assign one status without using a true label or filtering any sample."""
    if float(metrics["response_variance"]) < float(thresholds["minimum_response_variance"]):
        return "LOW_MODEL_RESPONSE"
    if float(metrics["r7_similarity"]) > float(thresholds["maximum_r7_similarity"]):
        return "RANDOMIZATION_UNRELIABLE"
    if min(float(metrics["local_surrogate_r2"]), float(metrics["context_surrogate_r2"])) < float(thresholds["minimum_surrogate_r2"]):
        return "SURROGATE_UNRELIABLE"
    if float(metrics["context_dependence_ratio"]) >= float(thresholds["context_dominated_ratio"]):
        return "CONTEXT_DOMINATED"
    return "LOCALLY_EXPLAINABLE"


def freeze_validation_selection(records: Sequence[Mapping[str, object]], max_candidates: int = 2) -> dict:
    if any(str(row.get("selection_split")) != "validation" for row in records):
        raise RuntimeError("candidate selection must use validation only")
    qualified = []
    for row in records:
        missing = [gate for gate in REQUIRED_GATES if gate not in row]
        if missing:
            raise ValueError(f"missing validation gates: {missing}")
        if all(bool(row[gate]) for gate in REQUIRED_GATES):
            qualified.append(row)
    qualified.sort(key=lambda row: (-float(row.get("validation_score", 0.0)), str(row["candidate"])))
    chosen = [str(row["candidate"]) for row in qualified[: int(max_candidates)]]
    return {
        "status": "VALIDATION_QUALIFIED" if chosen else "VALIDATION_NO_QUALIFIED_HIERARCHICAL_CANDIDATE",
        "selected_candidates": chosen,
        "fixed_test_allowed": bool(chosen),
        "test_data_used": False,
        "required_gates": list(REQUIRED_GATES),
    }


def coverage_counts(statuses: Sequence[str]) -> dict:
    allowed = ("LOCALLY_EXPLAINABLE", "CONTEXT_DOMINATED", "LOW_MODEL_RESPONSE", "SURROGATE_UNRELIABLE", "RANDOMIZATION_UNRELIABLE")
    counts = {status: int(sum(value == status for value in statuses)) for status in allowed}
    counts["total"] = int(len(statuses))
    counts["reliable"] = counts["LOCALLY_EXPLAINABLE"] + counts["CONTEXT_DOMINATED"]
    counts["rejection"] = counts["total"] - counts["reliable"]
    if sum(counts[status] for status in allowed) != counts["total"]:
        raise ValueError("coverage must include every sample")
    return counts
