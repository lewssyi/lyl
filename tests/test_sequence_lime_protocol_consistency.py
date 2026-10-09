from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from src.xai.model_faithful_sequence_lime import resolve_target_protocols
from tools import sequence_lime_protocol_consistency_audit as audit


class ProtocolConsistencyPureTests(unittest.TestCase):
    def test_query_budget_is_exact(self):
        self.assertEqual(audit.NEIGHBORHOOD_MASKS + audit.FAITHFULNESS_MASKS, 1530)

    def test_mask_split_is_exact(self):
        self.assertEqual(audit.TRAIN_MASKS + audit.VALIDATION_MASKS, 1506)

    def test_same_sample_two_entry_points_make_identical_masks(self):
        first = audit.unified_masks(3101, 284)
        second = audit.unified_masks(3101, 284)
        np.testing.assert_array_equal(first, second)

    def test_masks_change_with_seed(self):
        first = audit.unified_masks(3101, 284)
        second = audit.unified_masks(3102, 284)
        self.assertFalse(np.array_equal(first, second))

    def test_original_and_zero_masks_are_included(self):
        masks = audit.unified_masks(3101, 284)
        np.testing.assert_array_equal(masks[0], np.ones(90, dtype=np.int32))
        np.testing.assert_array_equal(masks[1], np.zeros(90, dtype=np.int32))

    def test_masks_are_unique(self):
        masks = audit.unified_masks(3101, 284)
        self.assertEqual(len(masks), len({row.tobytes() for row in masks}))

    def test_mask_hash_is_reproducible(self):
        masks = audit.unified_masks(3101, 284)
        self.assertEqual(audit.sha256_bytes(masks), audit.sha256_bytes(masks.copy()))

    def test_t0_uses_original_predicted_class(self):
        protocols = resolve_target_protocols(np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.8, 0.1]))
        self.assertEqual(protocols["T0"], 0)

    def test_t1_uses_randomized_predicted_class(self):
        protocols = resolve_target_protocols(np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.8, 0.1]))
        self.assertEqual(protocols["T1"], 1)

    def test_y2_margin_definition(self):
        observed = audit.y2_margin([0.6, 0.3, 0.1], 0)
        self.assertAlmostEqual(observed, np.log(0.6) - np.log(0.3))

    def test_entropy_is_finite(self):
        self.assertTrue(np.isfinite(audit.entropy([1.0, 0.0, 0.0])))

    def test_kl_self_is_zero(self):
        self.assertAlmostEqual(audit.kl_divergence([0.5, 0.3, 0.2], [0.5, 0.3, 0.2]), 0.0)

    def test_cache_key_is_reproducible(self):
        first = audit.cache_protocol_identity("A", "R7", 3101, 284)
        second = audit.cache_protocol_identity("A", "R7", 3101, 284)
        self.assertEqual(first, second)

    def test_different_groups_cannot_share_cache(self):
        first = audit.cache_protocol_identity("A", "R7", 3101, 284)
        second = audit.cache_protocol_identity("C", "R7", 3101, 284)
        self.assertNotEqual(first, second)

    def test_different_randomization_seeds_cannot_share_cache(self):
        first = audit.cache_protocol_identity("A", "R7", 3101, 284)
        second = audit.cache_protocol_identity("A", "R7", 3102, 284)
        self.assertNotEqual(first, second)

    def test_unified_fit_uses_frozen_grids(self):
        masks = np.ones((audit.NEIGHBORHOOD_MASKS, 90), dtype=np.int32)
        vectors = np.tile(np.array([[0.6, 0.3, 0.1]], dtype=np.float32), (len(masks), 1))
        fake = mock.Mock()
        fake.coef_ = np.arange(90, dtype=np.float32)
        fidelity = {
            "validation_weighted_r2": 0.0,
            "validation_weighted_mae": 0.0,
            "validation_weighted_rmse": 0.0,
            "effective_sample_size": 1.0,
            "selected_ridge_alpha": 0.1,
            "selected_kernel_width": 0.25,
        }
        with mock.patch.object(audit.evaluator, "fit_surrogate", return_value=(fake, fidelity, None)) as fit:
            audit.fit_y2(masks, vectors, 0)
        self.assertEqual(tuple(fit.call_args.kwargs["alpha_grid"]), audit.ALPHA_GRID)
        self.assertEqual(tuple(fit.call_args.kwargs["kernel_width_grid"]), audit.KERNEL_GRID)


if __name__ == "__main__":
    unittest.main()
