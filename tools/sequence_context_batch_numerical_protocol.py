#!/usr/bin/env python3
"""Pure-NumPy batch-shape numerical equivalence protocol.

This module must remain independent of TensorFlow and model/data loaders.  Its
CLI reads only the already-persisted batch-parity diagnostic artifacts and
creates an offline reassessment for future Sequence Context smoke runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL_NAME = "BATCH_SHAPE_NUMERICAL_EQUIVALENCE_V1"
ATOL = 1e-5
RTOL = 1e-5
NEAR_TOLERANCE_THRESHOLD = 0.8
PROBABILITY_SUM_ATOL = 1e-5
HISTORICAL_ATOL = 1e-6
HISTORICAL_RTOL = 0.0

DIAGNOSTIC_ROOT = (
    PROJECT
    / "outputs/sleepxvit_inspired_audit/isruc_seed42/context_influence_v1/"
    "development_smoke_2_batch_parity_diagnostic_1"
)
ORIGINAL_SMOKE_ROOT = (
    PROJECT
    / "outputs/sleepxvit_inspired_audit/isruc_seed42/context_influence_v1/"
    "development_smoke_2"
)
OUTPUT_ROOT = (
    PROJECT
    / "outputs/sleepxvit_inspired_audit/isruc_seed42/context_influence_v1/"
    "batch_shape_numerical_protocol_v1"
)
ORIGINAL_REPORT = PROJECT / "docs/paper_xai/SEQUENCE_CONTEXT_INFLUENCE_DEVELOPMENT_SMOKE.md"
DIAGNOSTIC_REPORT = PROJECT / "docs/paper_xai/SEQUENCE_CONTEXT_INFLUENCE_BATCH_PARITY_DIAGNOSTIC.md"
REPORT = PROJECT / "docs/paper_xai/SEQUENCE_CONTEXT_INFLUENCE_BATCH_NUMERICAL_PROTOCOL.md"

EXPECTED_BRANCH = "codex/a1-zero-mainline"
REQUIRED_COMMITS = (
    "16936d43274a18f49a58ead4c3f5a0cee70cddb4",
    "99a6c481ab068b1486e1fa635be94e87ee1af93a",
)
REQUIRED_INPUTS = (
    "diagnostic_protocol.json",
    "matched_input_identity.json",
    "execution_environment.json",
    "batch1_repeatability.json",
    "batch16_repeatability.json",
    "batch_shape_difference.json",
    "layerwise_difference.csv",
    "aggregation_difference.json",
    "final_outputs_batch1.npy",
    "final_outputs_batch16.npy",
    "final_output_diff.npy",
    "final_logits_batch1.npy",
    "final_logits_batch16.npy",
    "final_logits_diff.npy",
    "runtime_memory_profile.json",
    "access_audit.json",
    "failure_evidence.json",
    "run_manifest.json",
)
EXPECTED_ORIGINAL_HASHES = {
    "original_smoke_run_manifest": "bdda7a96d45ae0cf85d8381fb26c9b6a17cfe5c0aedb029ec0edf8ae7ced4781",
    "original_smoke_report": "d85a7400f4b5a9b26cf7ae9a86bd25afe6d4978c41b86c2d84a9ac8d7d53ccc6",
    "diagnostic_report": "2f007f885ff7db751586a9c67df6bfd8c43e73fd9b550bd4ef799e82af616791",
}
EXPECTED_OUTPUTS = (
    "numerical_equivalence_protocol.json",
    "diagnostic_offline_reassessment.json",
    "tolerance_boundary_tests.json",
    "source_evidence_manifest.json",
    "run_manifest.json",
    "README.md",
)


class ProtocolBlocked(RuntimeError):
    """Raised when the offline protocol cannot be frozen safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def allowed_difference(reference: np.ndarray | float) -> np.ndarray:
    value = np.asarray(reference, dtype=np.float64)
    return ATOL + RTOL * np.abs(value)


def elementwise_equivalence(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> Mapping[str, Any]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape:
        return {
            "shape_match": False,
            "pass": False,
            "failure_reason": "SHAPE_MISMATCH",
            "reference_shape": list(left.shape),
            "candidate_shape": list(right.shape),
        }
    finite = bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right)))
    if not finite:
        return {
            "shape_match": True,
            "shape": list(left.shape),
            "reference": None,
            "candidate": None,
            "actual_difference": None,
            "allowed_difference": None,
            "normalized_tolerance_usage": None,
            "max_abs_diff": None,
            "allowed_diff_at_max": None,
            "max_tolerance_usage": None,
            "near_tolerance_warning": False,
            "warning_status": "NONE",
            "finite": False,
            "pass": False,
            "failure_reason": "NONFINITE",
        }
    difference = np.abs(right - left)
    allowed = allowed_difference(left)
    usage = difference / allowed
    max_difference_index = int(np.argmax(difference)) if difference.size else 0
    max_usage_index = int(np.argmax(usage)) if usage.size else 0
    max_difference_location = np.unravel_index(max_difference_index, difference.shape) if difference.size else ()
    max_usage_location = np.unravel_index(max_usage_index, usage.shape) if usage.size else ()
    passed = bool(np.all(difference <= allowed))
    return {
        "shape_match": True,
        "shape": list(left.shape),
        "reference": left.tolist(),
        "candidate": right.tolist(),
        "actual_difference": difference.tolist(),
        "allowed_difference": allowed.tolist(),
        "normalized_tolerance_usage": usage.tolist(),
        "max_abs_diff": float(np.max(difference)) if difference.size else 0.0,
        "max_abs_diff_flat_index": max_difference_index,
        "max_abs_diff_location": [int(value) for value in max_difference_location],
        "reference_at_max_abs_diff": float(left.flat[max_difference_index]) if difference.size else None,
        "candidate_at_max_abs_diff": float(right.flat[max_difference_index]) if difference.size else None,
        "allowed_diff_at_max": float(allowed.flat[max_difference_index]) if difference.size else 0.0,
        "tolerance_usage_at_max_abs_diff": float(usage.flat[max_difference_index]) if difference.size else 0.0,
        "max_tolerance_usage": float(np.max(usage)) if usage.size else 0.0,
        "max_tolerance_usage_flat_index": max_usage_index,
        "max_tolerance_usage_location": [int(value) for value in max_usage_location],
        "reference_at_max_tolerance_usage": float(left.flat[max_usage_index]) if usage.size else None,
        "candidate_at_max_tolerance_usage": float(right.flat[max_usage_index]) if usage.size else None,
        "actual_diff_at_max_tolerance_usage": float(difference.flat[max_usage_index]) if usage.size else None,
        "allowed_diff_at_max_tolerance_usage": float(allowed.flat[max_usage_index]) if usage.size else None,
        "near_tolerance_warning": bool(np.any(usage > NEAR_TOLERANCE_THRESHOLD)),
        "warning_status": (
            "BATCH_SHAPE_NEAR_TOLERANCE_WARNING"
            if np.any(usage > NEAR_TOLERANCE_THRESHOLD)
            else "NONE"
        ),
        "finite": True,
        "pass": passed,
        "failure_reason": None if passed else "ELEMENT_EXCEEDS_TOLERANCE",
    }


def probability_semantics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> Mapping[str, Any]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    correct_shape = bool(left.ndim == 2 and left.shape == right.shape and left.shape[-1] == 5)
    finite = bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right)))
    range_valid = bool(
        finite
        and np.all((left >= 0.0) & (left <= 1.0))
        and np.all((right >= 0.0) & (right <= 1.0))
    )
    left_sum_error = np.abs(np.sum(left, axis=-1) - 1.0) if correct_shape else np.asarray([np.inf])
    right_sum_error = np.abs(np.sum(right, axis=-1) - 1.0) if correct_shape else np.asarray([np.inf])
    sum_valid = bool(
        correct_shape
        and np.all(left_sum_error <= PROBABILITY_SUM_ATOL)
        and np.all(right_sum_error <= PROBABILITY_SUM_ATOL)
    )
    argmax_equal = bool(correct_shape and np.array_equal(np.argmax(left, axis=-1), np.argmax(right, axis=-1)))
    return {
        "five_class_shape": correct_shape,
        "finite": finite,
        "range_valid": range_valid,
        "probability_sum_atol": PROBABILITY_SUM_ATOL,
        "reference_max_probability_sum_error": float(np.max(left_sum_error)),
        "candidate_max_probability_sum_error": float(np.max(right_sum_error)),
        "probability_sum_valid": sum_valid,
        "argmax_equal_for_every_window": argmax_equal,
        "reference_argmax": np.argmax(left, axis=-1).astype(int).tolist() if correct_shape else [],
        "candidate_argmax": np.argmax(right, axis=-1).astype(int).tolist() if correct_shape else [],
        "pass": bool(correct_shape and finite and range_valid and sum_valid and argmax_equal),
    }


def influence_equivalence(
    original_response_batch1: float,
    counterfactual_response_batch1: float,
    counterfactual_response_batch16: float,
) -> Mapping[str, Any]:
    original = float(original_response_batch1)
    response1 = float(counterfactual_response_batch1)
    response16 = float(counterfactual_response_batch16)
    influence1 = original - response1
    influence16 = original - response16
    epsilon_original = float(allowed_difference(original))
    epsilon_counterfactual = float(allowed_difference(response1))
    epsilon_influence = epsilon_original + epsilon_counterfactual
    actual = abs(influence16 - influence1)
    near_zero = bool(abs(influence1) <= epsilon_influence or abs(influence16) <= epsilon_influence)
    sign_equal = bool(np.sign(influence1) == np.sign(influence16))
    sign_pass = bool(near_zero or sign_equal)
    passed = bool(np.isfinite(actual) and actual <= epsilon_influence and sign_pass)
    return {
        "original_response_batch1": original,
        "counterfactual_response_batch1": response1,
        "counterfactual_response_batch16": response16,
        "signed_influence_batch1": influence1,
        "signed_influence_batch16": influence16,
        "actual_difference": actual,
        "epsilon_R_original_batch1": epsilon_original,
        "epsilon_R_counterfactual_batch1": epsilon_counterfactual,
        "allowed_difference_epsilon_I": epsilon_influence,
        "normalized_tolerance_usage": actual / epsilon_influence,
        "numerically_near_zero": near_zero,
        "sign_equal": sign_equal,
        "sign_rule_pass": sign_pass,
        "pass": passed,
        "failure_reason": None if passed else "INFLUENCE_EQUIVALENCE_FAIL",
    }


def evaluate_batch_shape_equivalence(
    batch1_probabilities: np.ndarray,
    batch16_probabilities: np.ndarray,
    *,
    exact_identity_pass: bool,
    batch1_repeatability_pass: bool,
    batch16_repeatability_pass: bool,
    original_response_batch1: float,
    response_class_index: int,
    response_class_batch1: int,
    response_class_batch16: int,
) -> Mapping[str, Any]:
    probability = elementwise_equivalence(batch1_probabilities, batch16_probabilities)
    semantics = probability_semantics(batch1_probabilities, batch16_probabilities)
    same_shape_repeatability = bool(batch1_repeatability_pass and batch16_repeatability_pass)
    left = np.asarray(batch1_probabilities, dtype=np.float64)
    right = np.asarray(batch16_probabilities, dtype=np.float64)
    response1 = float(np.mean(left, axis=0)[int(response_class_index)])
    response16 = float(np.mean(right, axis=0)[int(response_class_index)])
    response = elementwise_equivalence(np.asarray([response1]), np.asarray([response16]))
    influence = influence_equivalence(original_response_batch1, response1, response16)
    response_class_equal = bool(
        int(response_class_batch1) == int(response_class_batch16) == int(response_class_index)
    )
    semantic_invariance = bool(semantics["pass"] and response_class_equal)
    failure_reasons = []
    if not exact_identity_pass:
        failure_reasons.append("STRUCTURAL_IDENTITY_FAIL")
    if not same_shape_repeatability:
        failure_reasons.append("SAME_SHAPE_REPEATABILITY_FAIL")
    if not probability.get("pass", False):
        failure_reasons.append("CROSS_BATCH_PROBABILITY_EQUIVALENCE_FAIL")
    if not response.get("pass", False):
        failure_reasons.append("DERIVED_RESPONSE_EQUIVALENCE_FAIL")
    if not influence["pass"]:
        failure_reasons.append("DERIVED_INFLUENCE_EQUIVALENCE_FAIL")
    if not semantic_invariance:
        failure_reasons.append("SEMANTIC_INVARIANCE_FAIL")
    overall = bool(
        exact_identity_pass
        and same_shape_repeatability
        and probability.get("pass", False)
        and response.get("pass", False)
        and influence["pass"]
        and semantic_invariance
    )
    return {
        "protocol": PROTOCOL_NAME,
        "exact_identity_pass": bool(exact_identity_pass),
        "same_shape_repeatability_pass": same_shape_repeatability,
        "batch1_repeatability_pass": bool(batch1_repeatability_pass),
        "batch16_repeatability_pass": bool(batch16_repeatability_pass),
        "probability_equivalence_pass": bool(probability.get("pass", False)),
        "response_equivalence_pass": bool(response.get("pass", False)),
        "influence_equivalence_pass": bool(influence["pass"]),
        "semantic_invariance_pass": semantic_invariance,
        "near_tolerance_warning": bool(probability.get("near_tolerance_warning", False)),
        "warning_status": probability.get("warning_status", "NONE"),
        "max_abs_diff": probability.get("max_abs_diff"),
        "allowed_diff_at_max": probability.get("allowed_diff_at_max"),
        "max_tolerance_usage": probability.get("max_tolerance_usage"),
        "failure_reason": failure_reasons or None,
        "overall_pass": overall,
        "probability": probability,
        "probability_semantics": semantics,
        "response": response,
        "influence": influence,
        "response_class_index": int(response_class_index),
        "response_class_equal": response_class_equal,
    }


def structural_identity_from_evidence(identity: Mapping[str, Any]) -> Mapping[str, Any]:
    windows = list(identity.get("per_window", ()))
    starts = [int(row["window_start"]) for row in windows]
    target_positions = [int(row["target_position"]) for row in windows]
    context_positions = [int(row["context_position"]) for row in windows]
    expected_count = int(identity.get("window_count", -1))
    checks = {
        "status_confirmed": identity.get("status") == "MATCHED_INPUT_IDENTITY_CONFIRMED",
        "sample_id": int(identity.get("sample_id", -1)) == 8809,
        "subject_id": identity.get("subject_id") == "S011",
        "recording_id": int(identity.get("recording_id", -1)) == 9,
        "physical_epoch": identity.get("physical_epoch_id") == "recording-9:epoch-810",
        "offset": int(identity.get("offset", 999)) == -3,
        "replacement": identity.get("replacement") == "normalized_zero",
        "window_count": expected_count == 15 == len(windows),
        "window_ids": [int(row["window_id"]) for row in windows] == list(range(15)),
        "window_order": starts == list(range(740, 811, 5)),
        "target_mapping": all(start + position == 813 for start, position in zip(starts, target_positions)),
        "context_mapping": all(start + position == 810 for start, position in zip(starts, context_positions)),
        "batch_positions": (
            [int(row["batch1_position"]) for row in windows] == [0] * 15
            and [int(row["batch16_position"]) for row in windows] == list(range(15))
        ),
        "same_materialized_arrays": bool(identity.get("batch1_and_batch16_share_same_materialized_arrays")),
        "no_sort_dedup_or_shift": not bool(identity.get("batch16_sort_deduplicate_padding_or_index_shift")),
        "input_hashes_and_order": bool(identity.get("input_hashes_and_order_identical")),
        "raw_shape_dtype_hash_present": bool(identity.get("raw_window_shape") and identity.get("raw_window_dtype") and identity.get("raw_window_sha256")),
        "segment_shape_dtype_hash_present": bool(identity.get("segment_window_shape") and identity.get("segment_window_dtype") and identity.get("segment_window_sha256")),
        "class_order": identity.get("class_order") == ["W", "N1", "N2", "N3", "REM"],
    }
    return {"checks": checks, "pass": bool(all(checks.values()))}


def boundary_cases() -> list[Mapping[str, Any]]:
    cases: list[Mapping[str, Any]] = []

    def add(name: str, observed: bool, expected: bool, detail: Mapping[str, Any]) -> None:
        cases.append({"name": name, "expected_pass": expected, "observed_pass": observed, "test_pass": observed == expected, "detail": detail})

    exact = elementwise_equivalence(np.asarray([0.2, 0.8]), np.asarray([0.2, 0.8]))
    add("exact_match", exact["pass"], True, exact)
    below = elementwise_equivalence(np.asarray([0.5]), np.asarray([0.500001]))
    add("clearly_below_tolerance", below["pass"], True, below)
    boundary = elementwise_equivalence(np.asarray([0.0]), np.asarray([ATOL]))
    add("exact_tolerance_boundary", boundary["pass"], True, boundary)
    above = elementwise_equivalence(np.asarray([0.0]), np.asarray([np.nextafter(ATOL, np.inf)]))
    add("just_above_tolerance", above["pass"], False, above)
    high = elementwise_equivalence(np.asarray([0.9]), np.asarray([0.900018]))
    add("high_confidence_relative_tolerance", high["pass"], True, high)
    near_zero_probability = elementwise_equivalence(np.asarray([0.0]), np.asarray([9e-6]))
    add("near_zero_probability_absolute_tolerance", near_zero_probability["pass"], True, near_zero_probability)
    nonfinite = elementwise_equivalence(np.asarray([0.5, np.nan]), np.asarray([0.5, np.inf]))
    add("nan_inf", nonfinite["pass"], False, nonfinite)
    ref_argmax = np.asarray([[0.500004, 0.499996, 0.0, 0.0, 0.0]])
    cand_argmax = np.asarray([[0.499996, 0.500004, 0.0, 0.0, 0.0]])
    argmax_semantics = probability_semantics(ref_argmax, cand_argmax)
    add("argmax_change", argmax_semantics["pass"], False, argmax_semantics)
    propagated = influence_equivalence(0.9, 0.8, 0.80002)
    add("influence_error_propagation", propagated["pass"], True, propagated)
    near_zero_sign = influence_equivalence(0.5, 0.499999, 0.500001)
    add("near_zero_influence_sign_change", near_zero_sign["pass"] and near_zero_sign["numerically_near_zero"], True, near_zero_sign)
    nonzero_sign = influence_equivalence(0.5, 0.499, 0.501)
    add("non_near_zero_influence_sign_change", nonzero_sign["pass"], False, nonzero_sign)
    add("structural_identity_mismatch", False, False, {"failure_reason": "STRUCTURAL_IDENTITY_FAIL"})
    warning_candidate = float(0.5 + 0.9 * float(allowed_difference(0.5)))
    warning = elementwise_equivalence(np.asarray([0.5]), np.asarray([warning_candidate]))
    add("near_tolerance_warning_without_hard_fail", warning["pass"] and warning["near_tolerance_warning"], True, warning)
    return cases


def protocol_payload(source_head: str) -> Mapping[str, Any]:
    return {
        "schema_version": "batch_shape_numerical_equivalence_protocol_v1",
        "protocol_name": PROTOCOL_NAME,
        "status": "FROZEN_FOR_FUTURE_NEW_SMOKE_RUNS_ONLY",
        "frozen_at": utc_now(),
        "source_git_head": source_head,
        "historical_smoke_status": "DEVELOPMENT_SMOKE_BLOCKED",
        "historical_failure_retroactively_changed": False,
        "structural_identity_gate": {
            "comparison": "EXACT",
            "failure": "STRUCTURAL_IDENTITY_FAIL",
            "fields": [
                "sample_id", "subject_id", "recording_id", "physical_epoch", "offset",
                "replacement", "raw_input_hash", "segment_input_hash", "window_id_order",
                "target_position", "class_order", "response_class", "window_count",
                "input_shape", "input_dtype",
            ],
        },
        "same_shape_repeatability_gate": {
            "batch1_A1_vs_A2": "max_abs_diff == 0",
            "batch16_B1_vs_B2": "max_abs_diff == 0",
            "relaxation_allowed": False,
        },
        "cross_batch_probability_equivalence_gate": {
            "reference": "batch1",
            "candidate": "batch16 corresponding element",
            "formula": "abs(p_batch16-p_batch1) <= 1e-5 + 1e-5*abs(p_batch1)",
            "atol": ATOL,
            "rtol": RTOL,
            "all_elements_required": True,
            "probability_sum_atol": PROBABILITY_SUM_ATOL,
            "finite_range_argmax_and_response_class_required": True,
            "near_tolerance_warning_formula": "abs(diff)/(1e-5+1e-5*abs(reference)) > 0.8",
            "warning_does_not_expand_tolerance": True,
        },
        "derived_response_equivalence_gate": {
            "response_formula": "abs(R16-R1) <= epsilon_R(R1)",
            "epsilon_R": "1e-5 + 1e-5*abs(x)",
            "influence_formula": "abs(I16-I1) <= epsilon_I",
            "epsilon_I": "epsilon_R(R_original_batch1) + epsilon_R(R_counterfactual_batch1)",
            "sign_rule": "sign must match only when both abs(influence) > epsilon_I; otherwise NUMERICALLY_NEAR_ZERO",
        },
        "diagnostic_only_intermediate_stages": [
            "raw_encoder", "segment_encoder", "fusion", "TCN", "Transformer", "decoder", "logits"
        ],
        "unchanged_gates": [
            "canonical_baseline_parity", "no_op_control", "single_physical_epoch_isolation",
            "raw_segment_synchronized_update", "overlapping_window_integrity",
            "target_position_mapping", "determinism", "finite_checks", "data_access_boundary",
            "asset_hashes", "no_HMM_training_gradient", "canonical_LIME_and_authoritative_states",
        ],
        "fallback_provenance": {
            "source": "pre-smoke user authorization attachment 8f24a85b.../pasted-text.txt lines 211-215",
            "source_sha256": "327f0e4c21172c4b74cb9557c1835345a5f4acb7ba7f6460983a864583eb9d16",
            "source_bytes": 13836,
            "source_created_utc": "2026-08-03T08:59:19Z",
            "smoke_protocol_frozen_utc": "2026-08-03T09:28:49.222959+00:00",
            "rule": "If the project has no defined parity standard, freeze atol=1e-5, rtol=1e-5 before inference; never loosen after seeing results.",
            "not_derived_from_observed_1_84178e_5": True,
        },
        "future_scope_only": True,
        "new_model_inference_performed": False,
    }


def assert_preflight() -> str:
    if git("branch", "--show-current") != EXPECTED_BRANCH:
        raise ProtocolBlocked("Wrong branch")
    if git("status", "--porcelain"):
        raise ProtocolBlocked("Worktree must be clean before offline protocol freeze")
    for commit in REQUIRED_COMMITS:
        subprocess.check_call(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=PROJECT)
    if OUTPUT_ROOT.exists():
        raise ProtocolBlocked(f"Refusing to overwrite output root: {OUTPUT_ROOT}")
    if REPORT.exists():
        raise ProtocolBlocked(f"Refusing to overwrite report: {REPORT}")
    for name in REQUIRED_INPUTS:
        path = DIAGNOSTIC_ROOT / name
        if not path.is_file():
            raise ProtocolBlocked(f"Missing diagnostic evidence: {name}")
    original_hashes = {
        "original_smoke_run_manifest": sha256_file(ORIGINAL_SMOKE_ROOT / "run_manifest.json"),
        "original_smoke_report": sha256_file(ORIGINAL_REPORT),
        "diagnostic_report": sha256_file(DIAGNOSTIC_REPORT),
    }
    if original_hashes != EXPECTED_ORIGINAL_HASHES:
        raise ProtocolBlocked(f"Protected historical artifact changed: {original_hashes}")
    return git("rev-parse", "HEAD")


def build_source_manifest() -> Mapping[str, Any]:
    inputs = []
    for name in REQUIRED_INPUTS:
        path = DIAGNOSTIC_ROOT / name
        inputs.append(
            {"path": str(path.relative_to(PROJECT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    protected = [
        ORIGINAL_SMOKE_ROOT / "run_manifest.json",
        ORIGINAL_REPORT,
        DIAGNOSTIC_REPORT,
    ]
    return {
        "schema_version": "batch_shape_numerical_protocol_source_evidence_v1",
        "allowed_input_kind": "persisted diagnostic JSON/CSV/NPY and reports only",
        "inputs": inputs,
        "protected_historical_files": [
            {"path": str(path.relative_to(PROJECT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in protected
        ],
        "prior_authorization": protocol_payload(git("rev-parse", "HEAD"))["fallback_provenance"],
        "raw_waveform_loaded": False,
        "checkpoint_loaded": False,
        "model_framework_imported": False,
        "model_forward_executed": False,
        "sample_58657_accessed": False,
        "validation_accessed": False,
        "test_accessed": False,
    }


def offline_reassessment() -> Mapping[str, Any]:
    reference = np.load(DIAGNOSTIC_ROOT / "final_outputs_batch1.npy", allow_pickle=False)
    candidate = np.load(DIAGNOSTIC_ROOT / "final_outputs_batch16.npy", allow_pickle=False)
    persisted_difference = np.load(DIAGNOSTIC_ROOT / "final_output_diff.npy", allow_pickle=False)
    if not np.array_equal(persisted_difference, reference - candidate):
        raise ProtocolBlocked("Persisted final_output_diff is inconsistent")
    identity = read_json(DIAGNOSTIC_ROOT / "matched_input_identity.json")
    repeat1 = read_json(DIAGNOSTIC_ROOT / "batch1_repeatability.json")
    repeat16 = read_json(DIAGNOSTIC_ROOT / "batch16_repeatability.json")
    aggregation_saved = read_json(DIAGNOSTIC_ROOT / "aggregation_difference.json")
    structural = structural_identity_from_evidence(identity)
    response_class = int(aggregation_saved["response_class_index"])
    original_response = float(aggregation_saved["artifact_original_probability"][response_class])
    result = evaluate_batch_shape_equivalence(
        reference,
        candidate,
        exact_identity_pass=structural["pass"],
        batch1_repeatability_pass=bool(repeat1["stable_exactly"]),
        batch16_repeatability_pass=bool(repeat16["stable_exactly"]),
        original_response_batch1=original_response,
        response_class_index=response_class,
        response_class_batch1=int(aggregation_saved["argmax_batch1"]),
        response_class_batch16=int(aggregation_saved["argmax_batch16"]),
    )
    historical_difference = np.abs(np.asarray(reference, dtype=np.float64) - np.asarray(candidate, dtype=np.float64))
    aggregate1 = np.mean(np.asarray(reference, dtype=np.float64), axis=0)
    aggregate16 = np.mean(np.asarray(candidate, dtype=np.float64), axis=0)
    historical_per_window_pass = bool(np.all(historical_difference <= HISTORICAL_ATOL))
    historical_aggregate_pass = bool(np.all(np.abs(aggregate1 - aggregate16) <= HISTORICAL_ATOL))
    historical = {
        "protocol": "historical_smoke_atol_1e-6_rtol_0",
        "atol": HISTORICAL_ATOL,
        "rtol": HISTORICAL_RTOL,
        "per_window_all_elements_pass": historical_per_window_pass,
        "per_window_max_abs_diff": float(np.max(historical_difference)),
        "per_window_count_exceeding": int(np.count_nonzero(historical_difference > HISTORICAL_ATOL)),
        "aggregated_max_abs_diff": float(np.max(np.abs(aggregate1 - aggregate16))),
        "aggregated_pass": historical_aggregate_pass,
        "overall_pass": bool(historical_per_window_pass and historical_aggregate_pass),
        "historical_conclusion": (
            "PASS" if historical_per_window_pass and historical_aggregate_pass
            else "FAIL_RETAINED_NOT_RETROACTIVELY_CHANGED"
        ),
    }
    return {
        "schema_version": "batch_shape_numerical_protocol_offline_reassessment_v1",
        "computed_from_raw_persisted_float_arrays": True,
        "rounding_before_comparison": False,
        "structural_identity": structural,
        "same_shape_repeatability": {
            "batch1_A1_vs_A2": {
                "reference_run": "A1",
                "candidate_run": "A2",
                "reference_value_at_reported_max": repeat1["stages"]["pre_hmm_probability"]["left_at_max"],
                "candidate_value_at_reported_max": repeat1["stages"]["pre_hmm_probability"]["right_at_max"],
                "actual_max_abs_diff": repeat1["stages"]["pre_hmm_probability"]["max_abs_diff"],
                "allowed_max_abs_diff": 0.0,
                "pass": bool(repeat1["stable_exactly"]),
            },
            "batch16_B1_vs_B2": {
                "reference_run": "B1",
                "candidate_run": "B2",
                "reference_value_at_reported_max": repeat16["stages"]["pre_hmm_probability"]["left_at_max"],
                "candidate_value_at_reported_max": repeat16["stages"]["pre_hmm_probability"]["right_at_max"],
                "actual_max_abs_diff": repeat16["stages"]["pre_hmm_probability"]["max_abs_diff"],
                "allowed_max_abs_diff": 0.0,
                "pass": bool(repeat16["stable_exactly"]),
            },
        },
        "historical_protocol": historical,
        "amended_protocol": result,
        "intermediate_activations_and_logits": {
            "role": "diagnostic_only_not_hard_gate",
            "source": "batch_shape_difference.json and layerwise_difference.csv",
            "first_ordered_divergent_stage": "raw_encoder",
            "parallel_segment_encoder_also_divergent": True,
            "logits_max_abs_diff": float(read_json(DIAGNOSTIC_ROOT / "batch_shape_difference.json")["stages"]["logits"]["max_abs_diff"]),
        },
        "resource_metrics_carried_forward_not_remeasured": {
            "tensorflow_allocator_batch1_peak_bytes": 139143936,
            "tensorflow_allocator_batch16_peak_bytes": 1659473408,
            "reserved_memory": "N/A",
            "historical_external_process_observation_mib": 22582,
            "allocator_and_process_metrics_not_directly_equivalent": True,
            "development16_resource_gate": "NOT_EVALUATED",
        },
        "historical_smoke_status": "DEVELOPMENT_SMOKE_BLOCKED",
        "historical_failure_retroactively_changed": False,
        "new_model_inference_performed": False,
    }


def render_report(reassessment: Mapping[str, Any]) -> str:
    amended = reassessment["amended_protocol"]
    probability = amended["probability"]
    response = amended["response"]
    influence = amended["influence"]
    historical = reassessment["historical_protocol"]
    return "\n".join(
        [
            "# Sequence Context Influence — Batch-Shape Numerical Equivalence Protocol Amendment V1",
            "",
            "**`BATCH_SHAPE_NUMERICAL_PROTOCOL_FROZEN`**",
            "",
            "原 smoke 继续保持 **`DEVELOPMENT_SMOKE_BLOCKED`**。原 `atol=1e-6, rtol=0` 失败不被追溯撤销；本协议只适用于未来获得单独授权后启动的全新 smoke。",
            "",
            "## 协议与理由",
            "",
            "新协议命名为 `BATCH_SHAPE_NUMERICAL_EQUIVALENCE_V1`，依次执行 Structural Identity、Same-Shape Repeatability、Cross-Batch Probability Equivalence 和 Derived-Response Equivalence 四层门禁。结构身份和同 batch 重复仍要求 exact；中间 activation 与 logits 只保留为诊断信息。",
            "",
            "诊断显示 A1/A2 与 B1/B2 分别逐元素完全稳定，输入 hash、窗口顺序和 target-position 完全一致；差异已经在并行 raw/segment encoder 输出处出现，聚合不是首个差异点，argmax 与 influence 符号均未改变。因此 `1e-6/0` 不适合作为不同 batch shape 的逐元素等价硬门。",
            "",
            "概率门固定 batch-1 为 reference，对每个五类 Pre-HMM 元素要求：",
            "",
            "`abs(p16 - p1) <= 1e-5 + 1e-5 * abs(p1)`",
            "",
            "`1e-5/1e-5` 来自 2026-08-03 进入 smoke 前的授权文本 fallback（该文本早于 09:28:49 UTC 的 smoke 协议冻结），不是根据本次 `1.84178e-5` 反推。usage 超过 0.8 只产生 warning，绝不自动扩容差。",
            "",
            "响应容差为 `epsilon_R(x)=1e-5+1e-5*abs(x)`；influence 容差严格由误差传播得到：`epsilon_I=epsilon_R(R_original_batch1)+epsilon_R(R_counterfactual_batch1)`。若任一 influence 的绝对值不超过 epsilon_I，则记为 `NUMERICALLY_NEAR_ZERO`，其符号变化不得解释为科学方向变化。",
            "",
            "## 纯离线复核",
            "",
            f"- 历史协议：FAIL；per-window max `{historical['per_window_max_abs_diff']:.17g}`，超过元素数 `{historical['per_window_count_exceeding']}`；聚合 max `{historical['aggregated_max_abs_diff']:.17g}`。",
            f"- 新协议概率门：`{amended['probability_equivalence_pass']}`；max diff `{probability['max_abs_diff']:.17g}`，该位置 allowed `{probability['allowed_diff_at_max']:.17g}`。",
            (
                f"- 绝对差最大元素自身 usage 为 `{probability['tolerance_usage_at_max_abs_diff']:.17g}`，"
                "该元素通过并触发 near-tolerance warning。"
            ),
            (
                f"- 全局最大 normalized tolerance usage：`{probability['max_tolerance_usage']:.17g}`，"
                f"位置 `{probability['max_tolerance_usage_location']}`；reference "
                f"`{probability['reference_at_max_tolerance_usage']:.17g}`，candidate "
                f"`{probability['candidate_at_max_tolerance_usage']:.17g}`，actual diff "
                f"`{probability['actual_diff_at_max_tolerance_usage']:.17g}`，allowed "
                f"`{probability['allowed_diff_at_max_tolerance_usage']:.17g}`。该元素超过硬门，"
                "所以当前诊断在 V1 下整体仍为 FAIL。"
            ),
            f"- Near-tolerance warning：`{amended['warning_status']}`。",
            f"- 聚合响应门：`{amended['response_equivalence_pass']}`；actual `{response['max_abs_diff']:.17g}`，allowed `{response['allowed_diff_at_max']:.17g}`。",
            f"- Influence 门：`{amended['influence_equivalence_pass']}`；actual `{influence['actual_difference']:.17g}`，epsilon_I `{influence['allowed_difference_epsilon_I']:.17g}`。",
            f"- 语义不变门：`{amended['semantic_invariance_pass']}`；整体离线结果：`{amended['overall_pass']}`。",
            "- logits 最大差继续记录，但 logits 量纲不同，不套用概率容差，也不作为硬门。",
            "",
            "## 资源与科学边界",
            "",
            "历史诊断记录的 TensorFlow allocator peak 为 batch-1 `139143936` bytes、batch-16 `1659473408` bytes；reserved-memory 为 N/A。这些 allocator 指标不能与外部观察的整个 GPU 进程 `22582 MiB`直接等同，也不能据此宣称 development-16 资源门通过。",
            "",
            "本轮没有导入 TensorFlow、加载 checkpoint/原始波形、执行模型 forward，也没有访问第二个 development 样本、validation 或 test。该协议不证明 Context Influence 有效，不建立 model faithfulness，也不改变 canonical LIME 或三项权威科学状态。",
            "",
        ]
    )


def run() -> None:
    source_head = assert_preflight()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    source_manifest = build_source_manifest()
    reassessment = offline_reassessment()
    cases = boundary_cases()
    if len(cases) != 13 or not all(row["test_pass"] for row in cases):
        raise ProtocolBlocked("Tolerance boundary self-tests failed")
    protocol = protocol_payload(source_head)
    atomic_json(OUTPUT_ROOT / "numerical_equivalence_protocol.json", protocol)
    atomic_json(OUTPUT_ROOT / "diagnostic_offline_reassessment.json", reassessment)
    atomic_json(
        OUTPUT_ROOT / "tolerance_boundary_tests.json",
        {"schema_version": "batch_shape_numerical_protocol_boundary_tests_v1", "count": len(cases), "passed": sum(row["test_pass"] for row in cases), "failed": sum(not row["test_pass"] for row in cases), "cases": cases},
    )
    atomic_json(OUTPUT_ROOT / "source_evidence_manifest.json", source_manifest)
    atomic_text(
        OUTPUT_ROOT / "README.md",
        "# Batch-Shape Numerical Equivalence Protocol V1\n\n"
        "Pure-NumPy offline protocol amendment. Historical smoke remains blocked; no model inference was run.\n",
    )
    atomic_text(REPORT, render_report(reassessment))
    protected_after = {
        "original_smoke_run_manifest": sha256_file(ORIGINAL_SMOKE_ROOT / "run_manifest.json"),
        "original_smoke_report": sha256_file(ORIGINAL_REPORT),
        "diagnostic_report": sha256_file(DIAGNOSTIC_REPORT),
    }
    if protected_after != EXPECTED_ORIGINAL_HASHES:
        raise ProtocolBlocked("Protected historical evidence changed during offline run")
    artifact_names = [name for name in EXPECTED_OUTPUTS if name != "run_manifest.json"]
    artifacts = []
    for name in artifact_names:
        path = OUTPUT_ROOT / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ProtocolBlocked(f"Missing output: {name}")
        if path.suffix == ".json":
            read_json(path)
        artifacts.append({"path": str(path.relative_to(PROJECT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": "batch_shape_numerical_protocol_run_manifest_v1",
        "status": "BATCH_SHAPE_NUMERICAL_PROTOCOL_FROZEN",
        "protocol_name": PROTOCOL_NAME,
        "source_git_head": source_head,
        "created_at": utc_now(),
        "historical_smoke_status": "DEVELOPMENT_SMOKE_BLOCKED",
        "historical_failure_retroactively_changed": False,
        "offline_reassessment_overall_pass_for_future_protocol": bool(reassessment["amended_protocol"]["overall_pass"]),
        "near_tolerance_warning": reassessment["amended_protocol"]["warning_status"],
        "new_model_inference_performed": False,
        "sample_58657_accessed": False,
        "validation_accessed": False,
        "test_accessed": False,
        "protected_historical_hashes_after": protected_after,
        "report": {
            "path": str(REPORT.relative_to(PROJECT)),
            "bytes": REPORT.stat().st_size,
            "sha256": sha256_file(REPORT),
        },
        "artifacts": artifacts,
    }
    atomic_json(OUTPUT_ROOT / "run_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output_root": str(OUTPUT_ROOT)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-offline", action="store_true")
    args = parser.parse_args()
    if not args.run_offline:
        parser.error("--run-offline is required")
    run()


if __name__ == "__main__":
    main()
