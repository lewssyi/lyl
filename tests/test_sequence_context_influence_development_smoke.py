import importlib.util
import pathlib
import sys
import unittest

import numpy as np


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "tools"
    / "sequence_context_influence_development_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("context_smoke", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ContextInfluenceDevelopmentSmokeTests(unittest.TestCase):
    def test_offsets_are_frozen_and_exclude_target(self):
        self.assertEqual(MODULE.OFFSETS, (-3, -2, -1, 1, 2, 3))
        self.assertNotIn(0, MODULE.OFFSETS)

    def test_build_windows_matches_stride_last_window_policy(self):
        windows = MODULE.build_windows(83)
        self.assertEqual(windows, [(0, 75), (5, 75), (8, 75)])
        self.assertEqual(MODULE.build_windows(10), [(0, 10)])

    def test_covering_windows_have_unique_starts_and_correct_positions(self):
        specs = MODULE.covering_window_specs(200, 80, 77)
        self.assertEqual(len({row.start for row in specs}), len(specs))
        for row in specs:
            self.assertEqual(row.start + row.target_position, 80)
            if row.context_position is not None:
                self.assertEqual(row.start + row.context_position, 77)

    def test_single_epoch_pair_has_no_leakage(self):
        raw = np.arange(5 * 4 * 2, dtype=np.float32).reshape(5, 4, 2)
        seg = np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2)
        changed_raw, changed_seg, audit = MODULE.apply_single_epoch_pair(
            raw, seg, 2, np.zeros_like(raw[0]), np.zeros_like(seg[0])
        )
        self.assertEqual(audit["changed_epoch_indices"], [2])
        self.assertTrue(audit["other_epochs_unchanged"])
        np.testing.assert_array_equal(changed_raw[:2], raw[:2])
        np.testing.assert_array_equal(changed_seg[3:], seg[3:])

    def test_no_op_pair_has_zero_changes(self):
        raw = np.arange(5 * 4, dtype=np.float32).reshape(5, 4, 1)
        seg = np.arange(5 * 3, dtype=np.float32).reshape(5, 3, 1)
        _, _, audit = MODULE.apply_single_epoch_pair(
            raw, seg, 3, raw[3], seg[3]
        )
        self.assertEqual(audit["changed_epoch_indices"], [])
        self.assertTrue(audit["other_epochs_unchanged"])

    def test_feature_medoid_excludes_audited_neighborhood(self):
        values = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)[:, None, None]
        index, payload = MODULE.deterministic_feature_medoid(values, {1, 2, 3})
        self.assertIn(index, {0, 4})
        self.assertNotIn(index, payload["excluded_indices"])

    def test_full_and_shared_aggregations_are_separate(self):
        specs = [
            MODULE.WindowSpec(0, 75, 10, None),
            MODULE.WindowSpec(5, 75, 5, 3),
        ]
        vectors = np.asarray([[0.8, 0.2], [0.4, 0.6]])
        full, shared = MODULE.aggregate_window_vectors(vectors, specs)
        np.testing.assert_allclose(full, [0.6, 0.4])
        np.testing.assert_allclose(shared, [0.4, 0.6])

    def test_selected_two_use_manifest_order_only(self):
        samples = []
        for index in range(16):
            samples.append(
                {
                    "recording_id": index,
                    "global_epoch_index": index * 100 + 50,
                    "source_recording": f"S{index:03d}.npz",
                }
            )
        payload = {"sample_count": 16, "samples": samples}
        counts = np.full(16, 100, dtype=int)
        split = {
            "split_files": {"train": [row["source_recording"] for row in samples]},
            "split_subjects": {"train": [f"S{index:03d}" for index in range(16)]},
        }
        selected = MODULE.select_first_two(payload, counts, split)
        self.assertEqual([row["sample_id"] for row in selected], [50, 150])

    def test_probability_semantics(self):
        self.assertTrue(MODULE.probability_valid([[0.2] * 5]))
        self.assertFalse(MODULE.probability_valid([[2.0, -1.0, 0, 0, 0]]))


if __name__ == "__main__":
    unittest.main()
