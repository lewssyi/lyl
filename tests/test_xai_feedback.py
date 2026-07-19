import unittest

import numpy as np
import tensorflow as tf

from src.model_training import make_xai_alignment_loss, scheduled_xai_loss_weight


class XAIFeedbackTest(unittest.TestCase):
    def test_schedule_has_zero_warmup_and_linear_ramp(self):
        values = [scheduled_xai_loss_weight(epoch, 0.001, 3, 5) for epoch in range(9)]
        self.assertEqual(values[:3], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(values[3], 0.0002)
        self.assertAlmostEqual(values[7], 0.001)
        self.assertAlmostEqual(values[8], 0.001)

    def test_alignment_losses_return_per_epoch_values(self):
        rng = np.random.default_rng(42)
        targets = tf.constant(rng.random((2, 4, 30, 3), dtype=np.float32))
        logits = tf.constant(rng.normal(size=(2, 4, 30, 3)).astype(np.float32))
        for name in (
            "normalized_mse",
            "kl_divergence",
            "js_divergence",
            "rank_correlation_surrogate",
            "top_k_bce",
        ):
            values = make_xai_alignment_loss(name)(targets, logits)
            self.assertEqual(tuple(values.shape), (2, 4), name)
            self.assertTrue(np.all(np.isfinite(values.numpy())), name)


if __name__ == "__main__":
    unittest.main()
