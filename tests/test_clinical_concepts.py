import unittest

import numpy as np

from src.clinical_concepts import (
    RAW_PHYSIOLOGY_VERSION,
    SEGMENT_PROXY_VERSION,
    assert_label_conditioning_allowed,
    concept_names,
    concepts_from_raw_waveform,
    concepts_from_segment_features,
    validate_raw_segment_concept_parity,
)
from src.lime_explanation import extract_segment_features


class ClinicalConceptDefinitionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.waveform = rng.normal(size=(3000, 3)).astype(np.float32)
        boundaries = np.linspace(0, 3000, 31, dtype=int)
        self.segments = list(zip(boundaries[:-1], boundaries[1:]))
        self.features = extract_segment_features(
            self.waveform, self.segments, max_segments=30
        )

    def test_names_and_shapes_are_canonical(self):
        self.assertEqual(len(concept_names(SEGMENT_PROXY_VERSION)), 9)
        self.assertEqual(
            concept_names(SEGMENT_PROXY_VERSION), concept_names(RAW_PHYSIOLOGY_VERSION)
        )
        proxy = concepts_from_segment_features(self.features)
        raw = concepts_from_raw_waveform(self.waveform, self.segments)
        self.assertEqual(proxy.shape, (30, 9))
        self.assertEqual(raw["concept_matrix"].shape, (9, 30))
        self.assertTrue(np.all(np.isfinite(proxy)))
        self.assertTrue(np.all(np.isfinite(raw["concept_matrix"])))

    def test_raw_and_segment_definitions_are_explicitly_distinct(self):
        report = validate_raw_segment_concept_parity(
            self.waveform, self.features, self.segments
        )
        self.assertFalse(report["mathematically_equivalent"])
        self.assertEqual(report["segment_definition_version"], SEGMENT_PROXY_VERSION)
        self.assertEqual(report["raw_definition_version"], RAW_PHYSIOLOGY_VERSION)

    def test_label_conditioning_is_forbidden_outside_explicit_train_ablation(self):
        with self.assertRaises(ValueError):
            assert_label_conditioning_allowed("test", [1], 0.2, True)
        with self.assertRaises(ValueError):
            assert_label_conditioning_allowed("train", [1], 0.2, False)
        assert_label_conditioning_allowed("test", [1], 0.0, False)


if __name__ == "__main__":
    unittest.main()
