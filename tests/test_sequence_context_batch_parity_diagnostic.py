import importlib.util
import pathlib
import sys
import tempfile
import unittest

import numpy as np


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "tools"
    / "sequence_context_batch_parity_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("context_batch_diag", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SequenceContextBatchParityDiagnosticTests(unittest.TestCase):
    def test_scope_is_exactly_the_failed_unit(self):
        self.assertEqual(MODULE.SAMPLE_ID, 8809)
        self.assertEqual(MODULE.SUBJECT_ID, "S011")
        self.assertEqual(MODULE.OFFSET, -3)
        self.assertEqual(MODULE.REPLACEMENT, "normalized_zero")
        self.assertEqual(MODULE.ATOL, 1e-6)
        self.assertEqual(MODULE.RTOL, 0.0)

    def test_difference_stats_preserves_exact_values_and_threshold_counts(self):
        left = np.asarray([[0.0, 2e-7], [0.0, 2e-5]], dtype=np.float32)
        right = np.zeros_like(left)
        stats = MODULE.difference_stats(left, right)
        self.assertEqual(stats["shape"], [2, 2])
        self.assertEqual(stats["count_gt_1e_7"], 2)
        self.assertEqual(stats["count_gt_1e_6"], 1)
        self.assertEqual(stats["count_gt_1e_5"], 1)
        self.assertFalse(stats["exact_equal"])

    def test_root_cause_input_identity_has_priority(self):
        stable = {"exact_equal": True}
        batch = {"max_abs_diff": 2e-6}
        aggregation = {"probability_stats": {"max_abs_diff": 0.0}}
        self.assertEqual(
            MODULE.classify_root_cause(False, stable, stable, batch, aggregation),
            "INPUT_OR_MAPPING_MISMATCH",
        )

    def test_root_cause_runtime_nondeterminism_precedes_batch_shape(self):
        unstable = {"exact_equal": False}
        stable = {"exact_equal": True}
        batch = {"max_abs_diff": 2e-6}
        aggregation = {"probability_stats": {"max_abs_diff": 0.0}}
        self.assertEqual(
            MODULE.classify_root_cause(True, unstable, stable, batch, aggregation),
            "RUNTIME_NONDETERMINISM",
        )

    def test_root_cause_batch_shape_drift(self):
        stable = {"exact_equal": True}
        batch = {"max_abs_diff": 2e-6}
        aggregation = {"probability_stats": {"max_abs_diff": 0.0}}
        self.assertEqual(
            MODULE.classify_root_cause(True, stable, stable, batch, aggregation),
            "BATCH_SHAPE_NUMERICAL_DRIFT",
        )

    def test_atomic_npy_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "value.npy"
            expected = np.arange(10, dtype=np.float32)
            MODULE.atomic_npy(path, expected)
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), expected)

    def test_required_outputs_include_full_arrays_and_manifest(self):
        self.assertIn("final_outputs_batch1.npy", MODULE.REQUIRED_OUTPUTS)
        self.assertIn("final_outputs_batch16.npy", MODULE.REQUIRED_OUTPUTS)
        self.assertIn("final_output_diff.npy", MODULE.REQUIRED_OUTPUTS)
        self.assertIn("run_manifest.json", MODULE.REQUIRED_OUTPUTS)


if __name__ == "__main__":
    unittest.main()
