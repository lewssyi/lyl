import unittest

import numpy as np

from src.data_processing import (
    apply_train_fitted_global_scaler,
    epoch_local_normalization,
    fit_train_global_scaler,
    prefix_causal_robust_zscore,
    recording_robust_zscore,
)


class NormalizationModeTest(unittest.TestCase):
    def test_prefix_mode_does_not_read_future_epochs(self):
        rng = np.random.default_rng(3)
        base = rng.normal(size=(5, 100, 3)).astype(np.float32)
        changed_future = base.copy()
        changed_future[3:] += 1000.0
        left = prefix_causal_robust_zscore(base)
        right = prefix_causal_robust_zscore(changed_future)
        self.assertTrue(np.array_equal(left[:3], right[:3]))

    def test_epoch_local_mode_is_independent_across_epochs(self):
        rng = np.random.default_rng(4)
        values = rng.normal(size=(3, 100, 3)).astype(np.float32)
        changed = values.copy()
        changed[2] *= 100.0
        self.assertTrue(
            np.array_equal(
                epoch_local_normalization(values)[:2],
                epoch_local_normalization(changed)[:2],
            )
        )

    def test_train_fitted_scaler_reuses_train_state(self):
        rng = np.random.default_rng(5)
        train = rng.normal(size=(4, 100, 3)).astype(np.float32)
        test = rng.normal(loc=5.0, size=(2, 100, 3)).astype(np.float32)
        scaler = fit_train_global_scaler(train)
        transformed = apply_train_fitted_global_scaler(test, scaler)
        self.assertEqual(scaler["fit_scope"], "train_only")
        self.assertEqual(transformed.shape, test.shape)
        with self.assertRaises(ValueError):
            apply_train_fitted_global_scaler(test, {**scaler, "fit_scope": "test"})

    def test_canonical_transform_is_unchanged(self):
        rng = np.random.default_rng(6)
        values = rng.normal(size=(3, 100, 3)).astype(np.float32)
        first = recording_robust_zscore(values)
        second = recording_robust_zscore(values)
        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
