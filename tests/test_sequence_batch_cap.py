import unittest

import numpy as np

from tools.sequence_lime_paper_evaluation import sequence_target_probabilities


class RecordingModel:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, inputs, training=False):
        batch, time_steps = inputs["raw_input"].shape[:2]
        self.batch_sizes.append(int(batch))
        output = np.zeros((batch, time_steps, 5), dtype=np.float32)
        output[..., 2] = 0.75
        return output


class SequenceBatchCapTest(unittest.TestCase):
    def test_perturbation_chunk_does_not_multiply_formal_batch(self):
        model = RecordingModel()
        x_raw = np.zeros((80, 20, 3), dtype=np.float32)
        x_seg = np.zeros((80, 30, 28), dtype=np.float32)
        perturbed_epochs = [np.zeros((20, 3), dtype=np.float32) for _ in range(5)]
        perturbed_features = [np.zeros((30, 28), dtype=np.float32) for _ in range(5)]
        probabilities = sequence_target_probabilities(
            model,
            x_raw,
            x_seg,
            np.asarray([80]),
            global_epoch=40,
            target_class=2,
            perturbed_epochs=perturbed_epochs,
            perturbed_features=perturbed_features,
            batch_windows=2,
        )
        self.assertEqual(probabilities.shape, (5,))
        self.assertTrue(np.allclose(probabilities, 0.75))
        self.assertLessEqual(max(model.batch_sizes), 2)


if __name__ == "__main__":
    unittest.main()
