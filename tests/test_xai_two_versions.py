import unittest

import numpy as np

from src.xai.attribution_types import ModelAttributionResult
from src.xai.clinical_concept_annotation import annotate_clinical_concepts
from src.xai.model_faithful_sequence_lime import (
    assert_validation_only_selection,
    compose_diagnostic_methods,
    grouped_temporal_masks,
    positive_rank_score,
    resolve_target_protocols,
    select_validation_candidate,
    transform_surrogate_targets,
)


class ComponentIsolationTest(unittest.TestCase):
    def setUp(self):
        self.components = {
            "classic_score": np.asarray([1.0, 0.0, 0.0]),
            "zero_marginal_score": np.asarray([0.0, 1.0, 0.0]),
            "physiological_marginal_score": np.asarray([0.0, 0.0, 1.0]),
            "predicted_stage_prior_score": np.asarray([0.2, 0.3, 0.5]),
            "stage_independent_concept_score": np.asarray([0.4, 0.4, 0.2]),
            "signal_energy_score": np.asarray([0.1, 0.8, 0.1]),
            "random_score": np.asarray([0.7, 0.1, 0.2]),
        }

    def test_m0_is_pure_classic(self):
        result = compose_diagnostic_methods(**self.components)
        np.testing.assert_array_equal(result["M0"], self.components["classic_score"])

    def test_m3_contains_no_predicted_stage_prior(self):
        first = compose_diagnostic_methods(**self.components)["M3"]
        changed = dict(self.components)
        changed["predicted_stage_prior_score"] = np.asarray([1.0, 0.0, 0.0])
        second = compose_diagnostic_methods(**changed)["M3"]
        np.testing.assert_array_equal(first, second)

    def test_m4_uses_historical_formula(self):
        result = compose_diagnostic_methods(**self.components)
        expected = 0.15 * positive_rank_score(self.components["classic_score"]) + 0.35 * positive_rank_score(self.components["zero_marginal_score"]) + 0.30 * positive_rank_score(self.components["physiological_marginal_score"]) + 0.20 * positive_rank_score(self.components["predicted_stage_prior_score"])
        expected = positive_rank_score(expected)
        np.testing.assert_allclose(result["M4"], expected)

    def test_m5_is_not_reused_as_m6(self):
        result = compose_diagnostic_methods(**self.components)
        self.assertFalse(np.array_equal(result["M5"], result["M6"]))


class TargetProtocolTest(unittest.TestCase):
    def test_t0_and_t1_are_resolved_independently(self):
        result = resolve_target_protocols([0.8, 0.2], [0.1, 0.9])
        self.assertEqual(result, {"T0": 0, "T1": 1})

    def test_probability_target_is_exact(self):
        probabilities = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])
        result = transform_surrogate_targets(probabilities, 1)
        np.testing.assert_allclose(result["Y0"], [0.2, 0.5])

    def test_logit_margin_is_identifiable_from_softmax(self):
        probabilities = np.asarray([[0.7, 0.2, 0.1]])
        result = transform_surrogate_targets(probabilities, 0)
        self.assertAlmostEqual(float(result["Y2"][0]), float(np.log(0.7) - np.log(0.2)), places=6)


class ClinicalAnnotationIsolationTest(unittest.TestCase):
    def test_annotation_does_not_modify_attribution(self):
        attribution = ModelAttributionResult(1, "M0", [0.1, 0.9], 2, "T0", 100)
        before = attribution.model_attribution_score.copy()
        result = annotate_clinical_concepts(
            attribution,
            [0.4, 0.6],
            ["a", "b"],
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        )
        np.testing.assert_array_equal(before, attribution.model_attribution_score)
        self.assertFalse(result.metadata["ranking_modified"])

    def test_concept_only_method_cannot_masquerade_as_attribution(self):
        with self.assertRaises(ValueError):
            ModelAttributionResult(1, "M5", [0.1, 0.9], 2, "T0", 0)

    def test_predicted_stage_is_display_only(self):
        attribution = ModelAttributionResult(1, "M0", [0.1, 0.9], 2, "T0", 100)
        result = annotate_clinical_concepts(
            attribution, [1.0], ["a"], np.ones((2, 1)), metadata={"predicted_stage": "N2"}
        )
        self.assertEqual(result.metadata["predicted_stage_usage"], "display_context_only")


class SelectionGuardTest(unittest.TestCase):
    def test_test_split_is_rejected_for_selection(self):
        with self.assertRaises(RuntimeError):
            assert_validation_only_selection([{"selection_split": "test"}])

    def test_candidate_needs_two_passing_seeds(self):
        row = {
            "candidate": "B1",
            "selection_split": "validation",
            "fidelity_better": True,
            "faithfulness_noninferior": True,
            "stability_noninferior": True,
            "randomization_pass": True,
            "equal_query_budget": True,
            "validation_score": 1.0,
        }
        candidate, _ = select_validation_candidate([{**row, "randomization_seed": 3101}])
        self.assertEqual(candidate, "NO_MF_SEQLIME_CANDIDATE")

    def test_grouped_masks_preserve_contiguous_groups(self):
        masks = grouped_temporal_masks(15, 10, 42, group_size=5)
        self.assertEqual(masks.shape, (10, 15))
        for row in masks[2:]:
            self.assertTrue(all(len(set(row[start:start + 5])) == 1 for start in range(0, 15, 5)))


if __name__ == "__main__":
    unittest.main()
