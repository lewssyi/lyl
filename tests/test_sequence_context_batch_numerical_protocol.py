import importlib.util
import json
import pathlib
import sys
import unittest

import numpy as np


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "tools"
    / "sequence_context_batch_numerical_protocol.py"
)
SPEC = importlib.util.spec_from_file_location("batch_numerical_protocol", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BatchNumericalProtocolTests(unittest.TestCase):
    def test_module_has_no_tensorflow_dependency(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("import tensorflow", source)

    def test_exact_match(self):
        result = MODULE.elementwise_equivalence(np.asarray([0.2, 0.8]), np.asarray([0.2, 0.8]))
        self.assertTrue(result["pass"])
        self.assertEqual(result["max_abs_diff"], 0.0)

    def test_clearly_below_tolerance(self):
        self.assertTrue(MODULE.elementwise_equivalence(np.asarray([0.5]), np.asarray([0.500001]))["pass"])

    def test_exact_boundary(self):
        self.assertTrue(MODULE.elementwise_equivalence(np.asarray([0.0]), np.asarray([MODULE.ATOL]))["pass"])

    def test_just_above_boundary(self):
        candidate = np.nextafter(MODULE.ATOL, np.inf)
        self.assertFalse(MODULE.elementwise_equivalence(np.asarray([0.0]), np.asarray([candidate]))["pass"])

    def test_high_confidence_relative_tolerance(self):
        result = MODULE.elementwise_equivalence(np.asarray([0.9]), np.asarray([0.900018]))
        self.assertTrue(result["pass"])
        self.assertGreater(result["allowed_diff_at_max"], MODULE.ATOL)

    def test_near_zero_probability_absolute_tolerance(self):
        self.assertTrue(MODULE.elementwise_equivalence(np.asarray([0.0]), np.asarray([9e-6]))["pass"])

    def test_nan_and_inf_fail(self):
        result = MODULE.elementwise_equivalence(np.asarray([0.5, np.nan]), np.asarray([0.5, np.inf]))
        self.assertFalse(result["pass"])
        self.assertFalse(result["finite"])

    def test_argmax_change_fails_semantics(self):
        reference = np.asarray([[0.500004, 0.499996, 0.0, 0.0, 0.0]])
        candidate = np.asarray([[0.499996, 0.500004, 0.0, 0.0, 0.0]])
        self.assertFalse(MODULE.probability_semantics(reference, candidate)["pass"])

    def test_influence_tolerance_is_response_error_propagation(self):
        result = MODULE.influence_equivalence(0.9, 0.8, 0.80002)
        expected = float(MODULE.allowed_difference(0.9) + MODULE.allowed_difference(0.8))
        self.assertEqual(result["allowed_difference_epsilon_I"], expected)
        self.assertTrue(result["pass"])

    def test_near_zero_influence_sign_change_is_not_failure(self):
        result = MODULE.influence_equivalence(0.5, 0.499999, 0.500001)
        self.assertTrue(result["numerically_near_zero"])
        self.assertTrue(result["pass"])

    def test_non_near_zero_influence_sign_change_fails(self):
        result = MODULE.influence_equivalence(0.5, 0.499, 0.501)
        self.assertFalse(result["numerically_near_zero"])
        self.assertFalse(result["pass"])

    def test_structural_mismatch_blocks_overall(self):
        probabilities = np.asarray([[0.7, 0.1, 0.1, 0.05, 0.05]])
        result = MODULE.evaluate_batch_shape_equivalence(
            probabilities,
            probabilities,
            exact_identity_pass=False,
            batch1_repeatability_pass=True,
            batch16_repeatability_pass=True,
            original_response_batch1=0.8,
            response_class_index=0,
            response_class_batch1=0,
            response_class_batch16=0,
        )
        self.assertFalse(result["overall_pass"])
        self.assertIn("STRUCTURAL_IDENTITY_FAIL", result["failure_reason"])

    def test_warning_does_not_fail_gate(self):
        reference = np.asarray([0.5])
        candidate = np.asarray([0.5 + 0.9 * float(MODULE.allowed_difference(0.5))])
        result = MODULE.elementwise_equivalence(reference, candidate)
        self.assertTrue(result["pass"])
        self.assertTrue(result["near_tolerance_warning"])

    def test_all_thirteen_embedded_boundary_cases_pass(self):
        cases = MODULE.boundary_cases()
        self.assertEqual(len(cases), 13)
        self.assertTrue(all(row["test_pass"] for row in cases))
        json.dumps(cases, allow_nan=False)

    def test_comparator_returns_all_frozen_gate_fields(self):
        probabilities = np.asarray([[0.7, 0.1, 0.1, 0.05, 0.05]])
        result = MODULE.evaluate_batch_shape_equivalence(
            probabilities,
            probabilities,
            exact_identity_pass=True,
            batch1_repeatability_pass=True,
            batch16_repeatability_pass=True,
            original_response_batch1=0.8,
            response_class_index=0,
            response_class_batch1=0,
            response_class_batch16=0,
        )
        required = {
            "exact_identity_pass",
            "same_shape_repeatability_pass",
            "probability_equivalence_pass",
            "response_equivalence_pass",
            "influence_equivalence_pass",
            "semantic_invariance_pass",
            "near_tolerance_warning",
            "max_abs_diff",
            "allowed_diff_at_max",
            "max_tolerance_usage",
            "failure_reason",
        }
        self.assertTrue(required.issubset(result))


if __name__ == "__main__":
    unittest.main()
