import sys
import unittest

import numpy as np

from tools import context_influence_paper_convergence as module


class StubEvaluator:
    @staticmethod
    def compiled_perturbation_output(model, raw_batch, segment_batch):
        del model, segment_batch
        output = np.zeros((len(raw_batch), module.frozen.SEQ_LEN, 5), dtype=np.float32)
        output[..., 0] = 0.70
        output[..., 1] = 0.10
        output[..., 2] = 0.08
        output[..., 3] = 0.07
        output[..., 4] = 0.05
        return output


class ContextInfluencePaperConvergenceTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.zeros((80, 2, 1), dtype=np.float32)
        self.segment = np.zeros((80, 1, 1), dtype=np.float32)

    def test_module_does_not_import_tensorflow(self):
        self.assertNotIn("tensorflow", sys.modules)

    def test_batch_one_is_accepted(self):
        self.assertEqual(module.require_batch_one(1), 1)

    def test_noncanonical_batch_fails_fast(self):
        with self.assertRaises(module.PaperSmokeError):
            module.require_batch_one(16)

    def test_forward_shape_requires_leading_one(self):
        raw = np.zeros((2, 75, 2, 1), dtype=np.float32)
        segment = np.zeros((2, 75, 1, 1), dtype=np.float32)
        with self.assertRaises(module.PaperSmokeError):
            module.validate_forward_shapes(raw, segment)

    def test_forward_shape_requires_75_epochs(self):
        raw = np.zeros((1, 74, 2, 1), dtype=np.float32)
        segment = np.zeros((1, 74, 1, 1), dtype=np.float32)
        with self.assertRaises(module.PaperSmokeError):
            module.validate_forward_shapes(raw, segment)

    def test_profile_uses_only_batch_one_shapes(self):
        profile = module.infer_profile(
            StubEvaluator(), object(), self.raw, self.segment, 40, 37, None, None
        )
        self.assertTrue(profile["forward_shapes"])
        self.assertTrue(all(row["raw"][0] == 1 for row in profile["forward_shapes"]))
        self.assertEqual(profile["window_vectors"].shape[1], 5)

    def test_repeat_exact_is_strict(self):
        profile = module.infer_profile(
            StubEvaluator(), object(), self.raw, self.segment, 40, 37, None, None
        )
        stable, maximum = module.exact_profile_repeat(profile, profile)
        self.assertTrue(stable)
        self.assertEqual(maximum, 0.0)

    def test_repeat_change_is_detected(self):
        left = {"window_vectors": np.zeros((1, 5), dtype=np.float32)}
        right = {"window_vectors": np.ones((1, 5), dtype=np.float32)}
        stable, maximum = module.exact_profile_repeat(left, right)
        self.assertFalse(stable)
        self.assertEqual(maximum, 1.0)

    def test_structural_target_positions_are_exact(self):
        original = module.infer_profile(
            StubEvaluator(), object(), self.raw, self.segment, 40, 37, None, None
        )
        changed = module.infer_profile(
            StubEvaluator(), object(), self.raw, self.segment, 40, 43,
            np.ones_like(self.raw[0]), np.ones_like(self.segment[0]),
        )
        audit = module.structural_window_check(original, changed, 40, 43)
        self.assertTrue(audit["pass"])

    def test_report_preserves_claim_boundaries_and_historical_fail(self):
        result = {
            "status": "PAPER_IMPLEMENTATION_READY",
            "samples": [],
            "gates": {key: True for key in (
                "1_program_completed", "2_baseline_response_recovered",
                "3_noop_no_material_influence", "4_single_physical_epoch_only",
                "5_raw_segment_synchronized", "6_overlapping_windows_complete",
                "7_target_position_correct", "8_outputs_finite",
                "9_same_configuration_stable", "10_result_file_readable",
            )},
        }
        report = module.render_report(result)
        self.assertIn("BATCH_SHAPE_NUMERICAL_EQUIVALENCE_V1 = FAIL", report)
        self.assertIn("Context-Preserving Classic Sequence-LIME", report)
        self.assertIn("不能声明", report)


if __name__ == "__main__":
    unittest.main()
