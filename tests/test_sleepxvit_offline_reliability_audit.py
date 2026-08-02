import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "sleepxvit_offline_reliability_audit.py"
SPEC = importlib.util.spec_from_file_location("sleepxvit_offline_reliability_audit", MODULE_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class SleepXViTOfflineReliabilityAuditTests(unittest.TestCase):
    def test_normalized_probabilities_are_not_softmaxed_again(self):
        probabilities = np.array(
            [[0.8, 0.1, 0.05, 0.03, 0.02], [0.1, 0.7, 0.1, 0.05, 0.05]],
            dtype=float,
        )
        check = AUDIT.validate_probability_array(probabilities, expected_rows=2)
        self.assertTrue(check["pass"])
        self.assertEqual(check["probability_semantics"], "NORMALIZED_PROBABILITY")
        self.assertFalse(check["softmax_required"])

    def test_logits_are_rejected_as_probabilities(self):
        logits = np.array([[2.0, -1.0], [0.5, 0.1]], dtype=float)
        check = AUDIT.validate_probability_array(logits, expected_rows=2)
        self.assertFalse(check["pass"])
        self.assertEqual(check["probability_semantics"], "UNCONFIRMED")

    def test_equal_mass_bins_preserve_all_rows(self):
        confidence = np.linspace(0.1, 0.9, 31)
        correct = confidence > 0.5
        frame, ece, mce = AUDIT.calibration_bins_equal_mass(confidence, correct, n_bins=15)
        self.assertEqual(len(frame), 15)
        self.assertEqual(int(frame["n"].sum()), 31)
        self.assertGreaterEqual(ece, 0.0)
        self.assertGreaterEqual(mce, ece)

    def test_equal_width_bins_cover_boundary_values_once(self):
        confidence = np.array([0.0, 0.5, 1.0])
        correct = np.array([0, 1, 1])
        frame, _, _ = AUDIT.calibration_bins_equal_width(confidence, correct, n_bins=2)
        self.assertEqual(frame["n"].tolist(), [1, 2])

    def test_risk_coverage_discrete_aurc_definition(self):
        confidence = np.array([0.9, 0.8, 0.7])
        correct = np.array([1, 0, 1], dtype=bool)
        frame, aurc = AUDIT.risk_coverage_curve(confidence, correct)
        expected_risk = np.array([0.0, 0.5, 1.0 - 2.0 / 3.0])
        np.testing.assert_allclose(frame["error_risk"], expected_risk)
        self.assertAlmostEqual(aurc, float(expected_risk.mean()))

    def test_subject_mapping_is_deterministic(self):
        recording = np.array([0, 0, 1, 1, 1])
        mapped = AUDIT.materialize_subject_ids(recording, ["S016", "S017"])
        self.assertEqual(mapped.tolist(), ["S016", "S016", "S017", "S017", "S017"])

    def test_physical_epoch_duplicates_fail_axis_check(self):
        result = AUDIT.validate_split_axes(
            "validation",
            labels=np.array([0, 1]),
            recording_id=np.array([0, 0]),
            epoch_index=np.array([0, 0]),
            file_counts=np.array([2]),
            subject_ids=np.array(["S016", "S016"]),
        )
        self.assertFalse(result["physical_epoch_keys_unique"])
        self.assertFalse(result["pass"])

    def test_multiclass_brier_uses_sum_over_classes(self):
        probabilities = np.array([[1.0, 0.0], [0.5, 0.5]])
        labels = np.array([0, 1])
        _, brier = AUDIT.per_sample_losses(probabilities, labels)
        np.testing.assert_allclose(brier, [0.0, 0.5])

    def test_top10_overlap_treats_seed_rows_as_pairwise_repeats(self):
        first = np.arange(90, dtype=float)
        second = first.copy()
        self.assertEqual(AUDIT.top_k_overlap(first, second, 10), 1.0)
        self.assertAlmostEqual(AUDIT.cosine_similarity(first, second), 1.0)


if __name__ == "__main__":
    unittest.main()
