import unittest

import numpy as np

from src.lime_explanation import run_multimodal_lime


class _DeterministicModel:
    def predict(self, inputs, verbose=0):
        raw, segment = [np.asarray(value, dtype=np.float32) for value in inputs]
        base = np.mean(raw, axis=(1, 2)) + 0.01 * np.mean(segment, axis=(1, 2))
        logits = np.stack([base, -base, 0.5 * base, base + 0.1, -base + 0.2], axis=1)
        logits -= np.max(logits, axis=1, keepdims=True)
        values = np.exp(logits)
        return values / np.sum(values, axis=1, keepdims=True)


class LimeReproducibilityTest(unittest.TestCase):
    def test_same_seed_reproduces_masks_and_coefficients(self):
        rng = np.random.default_rng(9)
        waveform = rng.normal(size=(300, 3)).astype(np.float32)
        boundaries = np.linspace(0, 300, 7, dtype=int)
        segments = list(zip(boundaries[:-1], boundaries[1:]))
        kwargs = dict(
            model=_DeterministicModel(),
            instance_multimodal=waveform,
            segments=segments,
            num_perturbations=48,
            perturbation_mode="gaussian_noise",
            prediction_batch_size=12,
        )
        first = run_multimodal_lime(**kwargs, seed=123)
        second = run_multimodal_lime(**kwargs, seed=123)
        third = run_multimodal_lime(**kwargs, seed=124)
        self.assertEqual(
            first["perturbation_mask_sha256"], second["perturbation_mask_sha256"]
        )
        self.assertTrue(
            np.array_equal(first["segment_importance"], second["segment_importance"])
        )
        self.assertNotEqual(
            first["perturbation_mask_sha256"], third["perturbation_mask_sha256"]
        )


if __name__ == "__main__":
    unittest.main()
