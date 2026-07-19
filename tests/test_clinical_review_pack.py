import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_clinical_expert_review_pack import load_evaluation_targets


class ClinicalReviewPackTest(unittest.TestCase):
    def test_completed_evaluator_coefficients_map_to_segment_by_channel(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            cache_dir = run_dir / "cache"
            cache_dir.mkdir()
            payload = {
                "global_epoch_index": 7,
                "segments_count": 2,
                "perturbation_results": {
                    "physiological": {"coef": [1, 2, 3, 4, 5, 6]}
                },
            }
            (cache_dir / "sample_000007.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            target_map, provenance = load_evaluation_targets([run_dir])
            self.assertEqual(target_map[7].shape, (2, 3))
            np.testing.assert_array_equal(
                target_map[7],
                np.asarray([[1, 3, 5], [2, 4, 6]], dtype=np.float32),
            )
            self.assertEqual(len(provenance), 1)
            self.assertEqual(len(provenance[0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
