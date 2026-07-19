import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs/xai"


class XAIConfigTest(unittest.TestCase):
    def test_profiles_are_versioned_and_resume_isolated(self):
        paths = sorted(CONFIG_DIR.glob("sequence_lime_*.yaml"))
        self.assertEqual(len(paths), 4)
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        self.assertTrue(
            all(item["schema_version"] == "sequence_lime_config_v1" for item in payloads)
        )
        run_ids = [item["run_id"] for item in payloads]
        self.assertEqual(len(run_ids), len(set(run_ids)))

    def test_full_profile_has_all_replacement_estimands(self):
        payload = json.loads(
            (CONFIG_DIR / "sequence_lime_full.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(payload["required_estimands"]),
            {"zero", "physiological", "gaussian_noise", "local_mean"},
        )
        self.assertIn("incorrect", payload["required_populations"])
        self.assertIn("population_random", payload["required_populations"])

    def test_feedback_profile_is_default_off_and_label_free(self):
        payload = json.loads(
            (CONFIG_DIR / "xai_feedback_experiments.yaml").read_text(encoding="utf-8")
        )
        self.assertFalse(payload["default_enabled"])
        self.assertFalse(payload["label_conditioned_targets_allowed"])
        self.assertEqual(payload["teacher_target_split"], "train")
        self.assertEqual(payload["canonical_baseline"], "E0")
        self.assertTrue(
            payload["experiments"]["E7"]["requires_explicit_stage_prior_ablation_flag"]
        )
        self.assertEqual(payload["experiments"]["E1"]["xai_loss_weight"], 0.0)
        gate = payload["reliability_gate"]
        self.assertEqual(gate["validation_weighted_r2_min"], 0.70)
        self.assertEqual(gate["mask_seed_stability_spearman_min"], 0.50)
        self.assertFalse(gate["prediction_correctness_used_by_gate"])
        self.assertFalse(gate["teacher_confidence_used_by_gate"])


if __name__ == "__main__":
    unittest.main()
