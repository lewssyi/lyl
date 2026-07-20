import unittest

import numpy as np
import tensorflow as tf

from tools.sequence_lime_paper_evaluation import (
    compiled_perturbation_output,
    infer_precomputed_epochs_for_target,
    sequence_target_probabilities,
)


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
    def test_compiled_keras_runner_is_reused_across_batch_sizes(self):
        raw_input = tf.keras.Input(shape=(75, 20, 3), name="raw_input")
        segment_input = tf.keras.Input(shape=(75, 30, 28), name="segment_input")
        logits = tf.keras.layers.Lambda(
            lambda tensors: tf.zeros(
                (tf.shape(tensors[0])[0], tf.shape(tensors[0])[1], 5),
                dtype=tf.float32,
            )
        )([raw_input, segment_input])
        model = tf.keras.Model(
            inputs={"raw_input": raw_input, "segment_input": segment_input},
            outputs=logits,
        )
        raw = np.zeros((8, 75, 20, 3), dtype=np.float32)
        segment = np.zeros((8, 75, 30, 28), dtype=np.float32)

        first = compiled_perturbation_output(model, raw, segment)
        second = compiled_perturbation_output(model, raw[:3], segment[:3])

        self.assertEqual(first.shape, (8, 75, 5))
        self.assertEqual(second.shape, (3, 75, 5))
        self.assertEqual(len(model._sequence_lime_perturbation_runners), 1)

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

    def test_large_requested_batch_is_memory_capped(self):
        model = RecordingModel()
        x_raw = np.zeros((80, 20, 3), dtype=np.float32)
        x_seg = np.zeros((80, 30, 28), dtype=np.float32)
        perturbed_epochs = [np.zeros((20, 3), dtype=np.float32) for _ in range(12)]
        perturbed_features = [
            np.zeros((30, 28), dtype=np.float32) for _ in range(12)
        ]
        probabilities = sequence_target_probabilities(
            model,
            x_raw,
            x_seg,
            np.asarray([80]),
            global_epoch=40,
            target_class=2,
            perturbed_epochs=perturbed_epochs,
            perturbed_features=perturbed_features,
            batch_windows=16,
        )
        self.assertEqual(probabilities.shape, (12,))
        self.assertTrue(np.allclose(probabilities, 0.75))
        self.assertEqual(max(model.batch_sizes), 8)

    def test_precomputed_inference_streams_only_capped_microbatches(self):
        model = RecordingModel()
        x_raw = np.zeros((80, 20, 3), dtype=np.float32)
        x_seg = np.zeros((80, 30, 28), dtype=np.float32)
        perturbed_epochs = [np.zeros((20, 3), dtype=np.float32) for _ in range(12)]
        perturbed_features = [
            np.zeros((30, 28), dtype=np.float32) for _ in range(12)
        ]
        payload = infer_precomputed_epochs_for_target(
            model,
            x_raw,
            x_seg,
            np.asarray([80]),
            global_epoch=40,
            target_class=2,
            perturbed_epochs=perturbed_epochs,
            perturbed_features=perturbed_features,
            batch_windows=16,
        )
        self.assertTrue(np.allclose(payload["target_probabilities"], 0.75))
        self.assertEqual(
            payload["actual_assembled_window_instances"],
            payload["expected_unpadded_window_instances"],
        )
        self.assertEqual(
            payload["actual_inferred_window_instances"],
            payload["actual_assembled_window_instances"],
        )
        self.assertLessEqual(max(model.batch_sizes), 8)


if __name__ == "__main__":
    unittest.main()
