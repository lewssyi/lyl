import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from tools.sequence_lime_paper_evaluation import (
    MASK_CACHE_SCHEMA_VERSION,
    compiled_perturbation_output,
    evaluated_model_identity,
    model_weight_fingerprint,
    normalize_randomization_level,
    randomization_selected_layers,
    read_mask_prediction,
    stable_mask_hash,
)


def tiny_sequence_model(bias: float = 0.0) -> tf.keras.Model:
    raw = tf.keras.Input((75, 4, 3), name="raw_input")
    segment = tf.keras.Input((75, 2, 2), name="segment_input")
    raw_mean = tf.keras.layers.Lambda(
        lambda value: tf.reduce_mean(value, axis=(2, 3), keepdims=False),
        name="raw_mean",
    )(raw)
    segment_zero = tf.keras.layers.Lambda(
        lambda value: 0.0 * tf.reduce_mean(value, axis=(2, 3), keepdims=False),
        name="segment_zero",
    )(segment)
    joined = tf.keras.layers.Add()([raw_mean, segment_zero])
    joined = tf.keras.layers.Reshape((75, 1))(joined)
    output = tf.keras.layers.Dense(
        5,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(bias),
        name="main_output",
    )(joined)
    return tf.keras.Model({"raw_input": raw, "segment_input": segment}, output)


def tiny_layer_stack() -> tf.keras.Model:
    model_input = tf.keras.Input((3,), name="input")
    value = model_input
    value = tf.keras.layers.Dense(3, name="td_epoch_feature_dense")(value)
    value = tf.keras.layers.Dense(3, name="v3_state_local_conv")(value)
    value = tf.keras.layers.Dense(3, name="multi_head_attention")(value)
    value = tf.keras.layers.Dense(3, name="td_temporal_dense")(value)
    value = tf.keras.layers.Dense(5, name="main_output")(value)
    return tf.keras.Model(model_input, value)


class ModelFingerprintTest(unittest.TestCase):
    def test_fingerprint_is_reproducible_for_same_state(self):
        model = tiny_sequence_model(0.0)
        self.assertEqual(model_weight_fingerprint(model), model_weight_fingerprint(model))

    def test_fingerprint_changes_when_weight_changes(self):
        model = tiny_sequence_model(0.0)
        before = model_weight_fingerprint(model)
        model.get_layer("main_output").bias.assign_add(tf.ones((5,)))
        self.assertNotEqual(before, model_weight_fingerprint(model))

    def test_identity_records_full_fingerprint(self):
        model = tiny_sequence_model(0.0)
        identity = evaluated_model_identity(model, "state", method_id="M0", protocol_id="T0")
        self.assertEqual(identity["schema_version"], MASK_CACHE_SCHEMA_VERSION)
        self.assertEqual(identity["model_weight_fingerprint"], model_weight_fingerprint(model))

    def test_identity_changes_with_randomization_seed(self):
        model = tiny_sequence_model(0.0)
        first = evaluated_model_identity(model, "state", randomization_seed=3101)
        second = evaluated_model_identity(model, "state", randomization_seed=3102)
        self.assertNotEqual(first["identity_digest"], second["identity_digest"])

    def test_identity_changes_with_method(self):
        model = tiny_sequence_model(0.0)
        first = evaluated_model_identity(model, "state", method_id="M0")
        second = evaluated_model_identity(model, "state", method_id="M3")
        self.assertNotEqual(first["identity_digest"], second["identity_digest"])

    def test_identity_changes_with_protocol(self):
        model = tiny_sequence_model(0.0)
        first = evaluated_model_identity(model, "state", protocol_id="T0")
        second = evaluated_model_identity(model, "state", protocol_id="T1")
        self.assertNotEqual(first["identity_digest"], second["identity_digest"])


class CacheIsolationTest(unittest.TestCase):
    def _mask_hash(self, digest: str) -> str:
        return stable_mask_hash(
            np.asarray([1, 0, 1]), 4, "zero", "state", "segments", 2, 2, 16, digest
        )

    def test_mask_hash_changes_with_model_identity(self):
        self.assertNotEqual(self._mask_hash("model-a"), self._mask_hash("model-b"))

    def test_mask_hash_is_stable_for_same_identity(self):
        self.assertEqual(self._mask_hash("model-a"), self._mask_hash("model-a"))

    def test_cache_reader_accepts_matching_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_text(json.dumps({"model_identity_digest": "good"}), encoding="utf-8")
            self.assertIsNotNone(read_mask_prediction(path, "good"))

    def test_cache_reader_rejects_mismatched_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_text(json.dumps({"model_identity_digest": "old"}), encoding="utf-8")
            self.assertIsNone(read_mask_prediction(path, "new"))

    def test_cache_reader_rejects_legacy_entry_when_identity_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_text(json.dumps({"target_probability": 0.5}), encoding="utf-8")
            self.assertIsNone(read_mask_prediction(path, "required"))


class RandomizationRoutingTest(unittest.TestCase):
    def test_legacy_final_layer_maps_to_r1(self):
        self.assertEqual(normalize_randomization_level("final_layer"), "R1")

    def test_invalid_level_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_randomization_level("head-ish")

    def test_r1_selects_actual_main_output(self):
        self.assertEqual(
            [layer.name for layer in randomization_selected_layers(tiny_layer_stack(), "R1")],
            ["main_output"],
        )

    def test_r2_selects_temporal_dense_and_head(self):
        self.assertEqual(
            [layer.name for layer in randomization_selected_layers(tiny_layer_stack(), "R2")],
            ["td_temporal_dense", "main_output"],
        )

    def test_r3_is_cumulative_from_transformer(self):
        self.assertEqual(
            [layer.name for layer in randomization_selected_layers(tiny_layer_stack(), "R3")],
            ["multi_head_attention", "td_temporal_dense", "main_output"],
        )

    def test_r4_is_cumulative_from_tcn(self):
        self.assertEqual(
            [layer.name for layer in randomization_selected_layers(tiny_layer_stack(), "R4")],
            ["v3_state_local_conv", "multi_head_attention", "td_temporal_dense", "main_output"],
        )

    def test_r5_is_cumulative_from_fusion(self):
        self.assertEqual(
            [layer.name for layer in randomization_selected_layers(tiny_layer_stack(), "R5")],
            ["td_epoch_feature_dense", "v3_state_local_conv", "multi_head_attention", "td_temporal_dense", "main_output"],
        )


class GraphCaptureTest(unittest.TestCase):
    def test_compiled_runners_are_model_local(self):
        raw = np.zeros((2, 75, 4, 3), dtype=np.float32)
        segment = np.zeros((2, 75, 2, 2), dtype=np.float32)
        first = tiny_sequence_model(1.0)
        second = tiny_sequence_model(2.0)
        first_output = compiled_perturbation_output(first, raw, segment)
        second_output = compiled_perturbation_output(second, raw, segment)
        self.assertTrue(np.allclose(first_output, 1.0))
        self.assertTrue(np.allclose(second_output, 2.0))
        self.assertIsNot(first._sequence_lime_perturbation_runners, second._sequence_lime_perturbation_runners)

    def test_compiled_runner_reads_updated_model_weights(self):
        raw = np.zeros((2, 75, 4, 3), dtype=np.float32)
        segment = np.zeros((2, 75, 2, 2), dtype=np.float32)
        model = tiny_sequence_model(1.0)
        before = compiled_perturbation_output(model, raw, segment)
        model.get_layer("main_output").bias.assign(tf.fill((5,), 3.0))
        after = compiled_perturbation_output(model, raw, segment)
        self.assertTrue(np.allclose(before, 1.0))
        self.assertTrue(np.allclose(after, 3.0))


if __name__ == "__main__":
    unittest.main()
