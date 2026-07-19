import sys
import unittest
from unittest import mock

from tools.run_main_with_protocol_split import DATASET_PRESETS, parse_args


class MainlineLauncherTest(unittest.TestCase):
    def test_default_is_a1_zero_and_feedback_is_disabled(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "run_main_with_protocol_split.py",
                "--split-json",
                "dataset_protocols/ISRUC/split.json",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.dataset_preset, "A1_concept_film_zero")
        preset = DATASET_PRESETS[args.dataset_preset]
        self.assertEqual(preset["LFTS_CONCEPT_MODULATION_MODE"], "zero")
        self.assertEqual(preset["LFTS_XAI_FEEDBACK_ENABLED"], "0")
        self.assertEqual(preset["LFTS_XAI_FEEDBACK_LOSS_WEIGHT"], "0.0")

    def test_obsolete_experiment_presets_are_not_exposed(self):
        names = " ".join(DATASET_PRESETS).lower()
        for obsolete in (
            "c10",
            "residual",
            "supcon",
            "scr",
            "utf",
            "hard_risk",
            "shuffled",
        ):
            self.assertNotIn(obsolete, names)


if __name__ == "__main__":
    unittest.main()
