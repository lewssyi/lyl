from __future__ import annotations

import dataclasses
import json
import pathlib
import unittest

import numpy as np

from src.xai.context_replacement import (
    apply_context_mask_to_window,
    contiguous_context_groups,
    group_offsets,
    neutral_epoch_index,
    replacement_epoch,
    target_group_index,
)
from src.xai.explanation_reliability import coverage_counts, freeze_validation_selection, reliability_status
from src.xai.hierarchical_context_attribution import (
    ContextAttributionResult,
    ExplanationReliabilityResult,
    HierarchicalExplanationResult,
    WithinEpochAttributionResult,
    cache_identity,
    deterministic_masks,
    query_budget,
    selected_epoch_offsets,
)


class HierarchicalContextTests(unittest.TestCase):
    def test_g15_contiguous_complete(self):
        groups = contiguous_context_groups(75, 15)
        self.assertEqual(groups, tuple((i, i + 5) for i in range(0, 75, 5)))

    def test_g9_contiguous_complete(self):
        groups = contiguous_context_groups(75, 9)
        self.assertEqual(sum(stop - start for start, stop in groups), 75)
        self.assertTrue(all(groups[i][1] == groups[i + 1][0] for i in range(8)))

    def test_g25_contiguous_complete(self):
        groups = contiguous_context_groups(75, 25)
        self.assertTrue(all(stop - start == 3 for start, stop in groups))

    def test_target_group_is_unique(self):
        self.assertEqual(target_group_index(contiguous_context_groups(75, 15)), 7)

    def test_group_offsets_centered(self):
        groups = contiguous_context_groups(75, 15)
        self.assertIn(0, group_offsets(groups, 7))

    def test_replacement_changes_only_masked_context(self):
        raw = np.arange(75 * 2, dtype=np.float32).reshape(75, 2, 1)
        seg = np.arange(75, dtype=np.float32).reshape(75, 1, 1)
        groups = contiguous_context_groups(75, 15)
        mask = np.ones(15, dtype=np.int8); mask[0] = 0
        changed_raw, changed_seg = apply_context_mask_to_window(raw, seg, np.arange(75), 37, mask, groups, np.zeros((2, 1)), np.zeros((1, 1)))
        self.assertTrue(np.array_equal(changed_raw[5:], raw[5:]))
        self.assertTrue(np.array_equal(changed_seg[5:], seg[5:]))

    def test_replacement_keeps_target(self):
        raw = np.ones((75, 2, 1), dtype=np.float32)
        seg = np.ones((75, 1, 1), dtype=np.float32)
        groups = contiguous_context_groups(75, 15)
        mask = np.ones(15, dtype=np.int8); mask[target_group_index(groups)] = 0
        changed, _ = apply_context_mask_to_window(raw, seg, np.arange(75), 37, mask, groups, np.zeros((2, 1)), np.zeros((1, 1)))
        self.assertTrue(np.array_equal(changed[37], raw[37]))

    def test_zero_replacement(self):
        raw, seg, index = replacement_epoch(np.ones((3, 2, 1)), np.ones((3, 1, 1)), "zero")
        self.assertEqual(index, None); self.assertFalse(raw.any()); self.assertFalse(seg.any())

    def test_local_mean_replacement(self):
        raw, _, _ = replacement_epoch(np.asarray([[[0.]], [[2.]]]), np.asarray([[[0.]], [[4.]]]), "local_temporal_mean")
        self.assertEqual(float(raw[0, 0]), 1.0)

    def test_neutral_epoch_is_label_free_energy_median(self):
        raw = np.asarray([[[0.]], [[2.]], [[1.]]], dtype=np.float32)
        self.assertEqual(neutral_epoch_index(raw), 2)

    def test_same_seed_same_masks(self):
        self.assertTrue(np.array_equal(deterministic_masks(30, 200, 7), deterministic_masks(30, 200, 7)))

    def test_different_seed_different_masks(self):
        self.assertFalse(np.array_equal(deterministic_masks(30, 200, 7), deterministic_masks(30, 200, 8)))

    def test_masks_unique_and_have_endpoints(self):
        masks = deterministic_masks(15, 306, 9)
        self.assertEqual(len({row.tobytes() for row in masks}), 306)
        self.assertTrue(masks[0].all()); self.assertFalse(masks[1].any())

    def test_all_candidates_equal_query_budget(self):
        self.assertEqual({query_budget(name)["total"] for name in ("H0", "H1", "H2", "H3")}, {1530})

    def test_h0_explains_target(self):
        self.assertEqual(selected_epoch_offsets("H0", [], []), (0,))

    def test_h3_explains_target_and_new_context(self):
        self.assertEqual(selected_epoch_offsets("H3", [1], [2, 10]), (0, -27))

    def test_randomized_level_changes_cache(self):
        left = cache_identity("m", 1, "H3", "R0", 3101, "T0")
        right = cache_identity("m", 1, "H3", "R7", 3101, "T0")
        self.assertNotEqual(left, right)

    def test_t0_t1_cache_separation(self):
        self.assertNotEqual(cache_identity("m", 1, "H0", "R0", 3101, "T0"), cache_identity("m", 1, "H0", "R0", 3101, "T1"))

    def test_model_hash_is_in_cache(self):
        self.assertNotEqual(cache_identity("a", 1, "H0", "R0", 3101, "T0"), cache_identity("b", 1, "H0", "R0", 3101, "T0"))

    def test_validation_selection_rejects_test_rows(self):
        with self.assertRaises(RuntimeError):
            freeze_validation_selection([{"selection_split": "test"}])

    def test_validation_requires_all_ten_gates(self):
        row = {"selection_split": "validation", "candidate": "H3", "validation_score": 1.0}
        for key in ("surrogate_r2_better", "faithfulness_gate", "stability_noninferior", "t0_pass", "t1_pass", "r7_null_compatible", "randomization_faithfulness_decay", "equal_query_budget", "no_stage_prior", "complete_coverage"):
            row[key] = True
        row["t1_pass"] = False
        self.assertFalse(freeze_validation_selection([row])["fixed_test_allowed"])

    def test_coverage_includes_unreliable(self):
        result = coverage_counts(["LOCALLY_EXPLAINABLE", "SURROGATE_UNRELIABLE", "RANDOMIZATION_UNRELIABLE"])
        self.assertEqual(result["total"], 3); self.assertEqual(result["rejection"], 2)

    def test_reliability_status_does_not_require_label(self):
        metrics = {"response_variance": 1.0, "r7_similarity": 0.1, "local_surrogate_r2": 0.8, "context_surrogate_r2": 0.8, "context_dependence_ratio": 0.7}
        thresholds = {"minimum_response_variance": 0.01, "maximum_r7_similarity": 0.2, "minimum_surrogate_r2": 0.5, "context_dominated_ratio": 0.5}
        self.assertEqual(reliability_status(metrics, thresholds), "CONTEXT_DOMINATED")

    def test_result_arrays_are_immutable(self):
        result = ContextAttributionResult(1, "G15", np.ones(15), 7, (2,), 1, 1, 1, 1, 0.5, 306)
        with self.assertRaises(ValueError):
            result.context_group_importance[0] = 3

    def test_hierarchical_result_rejects_budget_drift(self):
        context = ContextAttributionResult(1, "G15", np.ones(15), 7, (2,), 1, 1, 1, 1, 0.5, 306)
        within = WithinEpochAttributionResult(1, (1,), 30, (np.ones(30),), 0.5, 1200, 2, "T0")
        with self.assertRaises(ValueError):
            HierarchicalExplanationResult(1, "H0", context, within, np.ones(15), (np.ones(30),), 1529)

    def test_unreliable_result_cannot_be_hidden(self):
        with self.assertRaises(ValueError):
            ExplanationReliabilityResult(1, "SURROGATE_UNRELIABLE", 0, 0, 0, 0, 1, 0, 0, 0, 1, False)

    def test_feedback_status_remains_blocked(self):
        path = pathlib.Path("outputs/xai_authoritative_status/isruc_seed42/CURRENT_XAI_STATUS.json")
        payload = json.loads(path.read_text())
        self.assertEqual(payload["feedback_training_status"], "BLOCKED")
        self.assertFalse(payload["explanation_guided_training_allowed"])


if __name__ == "__main__":
    unittest.main()
