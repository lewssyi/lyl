import csv
import json
import unittest
from pathlib import Path

from src.xai.posthoc_freeze import (
    FEEDBACK_BLOCK_MESSAGE,
    assert_feedback_training_disabled,
    assert_offline_target_is_dummy_control,
)


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs"
PAPER = DOC / "paper_xai"
HC = ROOT / "outputs/hierarchical_context_xai_feasibility/isruc_seed42"
PC = ROOT / "outputs/sequence_lime_protocol_consistency_audit/isruc_seed42"


class XAIFinalFreezeTest(unittest.TestCase):
    def test_01_canonical_definition_excludes_priors_and_fusions(self):
        text = (DOC / "XAI_ARCHITECTURE_CLAIM_BOUNDARIES.md").read_text()
        self.assertIn("excludes clinical", text)
        self.assertIn("stage priors", text)
        self.assertIn("fusion", text)

    def test_02_real_feedback_target_is_always_blocked(self):
        with self.assertRaisesRegex(RuntimeError, FEEDBACK_BLOCK_MESSAGE):
            assert_feedback_training_disabled(target_path="real_lime.npz")
        with self.assertRaisesRegex(RuntimeError, FEEDBACK_BLOCK_MESSAGE):
            assert_offline_target_is_dummy_control("classic_sequence_lime")

    def test_03_normal_inference_defaults_to_no_xai_and_calls_guard(self):
        text = (ROOT / "run_main.py").read_text(encoding="utf-8")
        self.assertIn('parse_bool_env("LFTS_XAI_FEEDBACK_ENABLED", False)', text)
        self.assertIn("assert_feedback_training_disabled(", text)

    def test_04_architecture_has_no_feedback_edge(self):
        with (PAPER / "figures/figure_a/figure_data.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        edges = {(r["source"], r["target"]) for r in rows if r["record_type"] == "edge"}
        forbidden = {"loss", "classifier", "attention", "prediction correction",
                     "FiLM", "concept modulation", "training"}
        self.assertFalse(any(target in forbidden for _, target in edges))
        self.assertFalse(any("Sensitivity" in source and target == "Frozen A1-zero Model"
                             for source, target in edges))

    def test_05_mf_and_hc_reports_are_historical(self):
        for name in ("XAI_ABC_AND_TWO_VERSION_REPORT.md",
                     "MF_SEQLIME_METHOD_SPEC.md",
                     "MF_SEQLIME_SURROGATE_IMPROVEMENT_REPORT.md"):
            self.assertIn("STATUS: SUPERSEDED HISTORICAL CANDIDATE",
                          (DOC / name).read_text())
        self.assertIn("STATUS: VALIDATION FAILED HISTORICAL CANDIDATE",
                      (DOC / "HIERARCHICAL_CONTEXT_XAI_FEASIBILITY_REPORT.md").read_text())

    def test_06_current_xai_status_is_unique(self):
        paths = list(ROOT.glob("outputs/**/CURRENT_XAI_STATUS.json"))
        self.assertEqual(len(paths), 1, paths)

    def test_07_authoritative_status_has_all_frozen_keys(self):
        status = json.loads((ROOT / "outputs/xai_authoritative_status/isruc_seed42/"
                             "CURRENT_XAI_STATUS.json").read_text())
        expected = {
            "engineering_status": "IMPLEMENTATION_AND_AUDIT_PIPELINE_VALID",
            "scientific_status": "MODEL_FAITHFULNESS_NOT_ESTABLISHED",
            "canonical_posthoc_method": "CONTEXT_PRESERVING_CLASSIC_SEQUENCE_LIME",
            "method_role": "EXPLORATORY_LOCAL_PERTURBATION_SENSITIVITY",
            "mf_seqlime_status": "SUPERSEDED_HISTORICAL_CANDIDATE",
            "hc_seqlime_status": "VALIDATION_FAILED_HISTORICAL_CANDIDATE",
            "clinical_annotation_status": "IMPLEMENTATION_VALID_EXPERT_VALIDATION_PENDING",
            "feedback_training_status": "PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD",
        }
        for key, value in expected.items():
            self.assertEqual(status[key], value)

    def test_08_tables_do_not_pool_incompatible_protocols(self):
        text = (PAPER / "TABLE_XAI_NEGATIVE_ABLATIONS.md").read_text()
        self.assertIn("Protocol (not pooled)", text)
        self.assertIn("numeric endpoints are not pooled", text)

    def test_09_case_selection_has_four_unique_reproducible_roles(self):
        with (PAPER / "XAI_CASE_STUDY_SELECTION.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({r["sample_id"] for r in rows}), 4)
        self.assertEqual(
            {r["case"] for r in rows},
            {"Relatively reliable", "Randomization-unreliable",
             "Incorrect or transition", "Context-dominated auxiliary flag"},
        )

    def test_10_case_selection_forbids_figure_quality(self):
        text = (DOC / "XAI_CASE_STUDY_SELECTION_PROTOCOL.md").read_text()
        for phrase in ("visual beauty", "physiological expectation", "story support"):
            self.assertIn(phrase, text)
        self.assertIn("figure quality cannot", text)

    def test_11_cca_cannot_modify_sensitivity_or_model(self):
        text = (DOC / "XAI_ARCHITECTURE_CLAIM_BOUNDARIES.md").read_text()
        self.assertIn("cannot alter", text)
        self.assertIn("ranking, prediction, training", text)
        self.assertIn("Derived annotation; expert validation pending.", text)

    def test_12_canonical_probability_parity_passes(self):
        payload = json.loads((PC / "guard/parity/model_prediction_parity.json").read_text())
        self.assertEqual(payload["exact_artifact_parity"]["status"],
                         "EXACT_ARTIFACT_PARITY_PASS")
        self.assertEqual(payload["exact_artifact_parity"]["max_abs_probability_diff"], 0.0)

    def test_13_sequence_wrapper_parity_passes(self):
        payload = json.loads((PC / "guard/parity/model_prediction_parity.json").read_text())
        self.assertEqual(payload["sequence_wrapper_parity"]["status"],
                         "SEQUENCE_WRAPPER_PARITY_PASS")
        self.assertLessEqual(payload["sequence_wrapper_parity"]["overall_max_abs_diff"], 1e-6)

    def test_14_memory_guard_passes(self):
        payload = json.loads((PC / "guard/fixed_graph_memory_1200.json").read_text())
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["forward_count"], 1200)
        self.assertTrue(payload["checkpoint_unchanged"])

    def test_15_all_required_figure_artifacts_are_nonempty(self):
        for letter in "abcde":
            directory = PAPER / f"figures/figure_{letter}"
            for name in ("figure.pdf", "figure.png", "figure_data.csv",
                         "figure_manifest.json"):
                path = directory / name
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_16_all_required_tables_are_nonempty(self):
        for name in (
            "TABLE_XAI_ENGINEERING_VALIDATION.md",
            "TABLE_XAI_SCIENTIFIC_BOUNDARIES.md",
            "TABLE_XAI_RELIABILITY_RESULTS.md",
            "TABLE_XAI_NEGATIVE_ABLATIONS.md",
        ):
            path = PAPER / name
            self.assertGreater(path.stat().st_size, 0, path)

    def test_17_figure_manifests_use_local_sensitivity_role(self):
        for letter in "abcde":
            payload = json.loads(
                (PAPER / f"figures/figure_{letter}/figure_manifest.json").read_text()
            )
            self.assertEqual(payload["method_role"],
                             "EXPLORATORY_LOCAL_PERTURBATION_SENSITIVITY")
            self.assertEqual(payload["scientific_status"],
                             "MODEL_FAITHFULNESS_NOT_ESTABLISHED")

    def test_18_current_paper_docs_have_no_forbidden_method_name(self):
        names = [
            "XAI_METHOD_SECTION.md", "XAI_RESULTS_SECTION.md",
            "XAI_FINAL_POSITIONING.md",
        ]
        for name in names:
            text = (PAPER / name).read_text()
            self.assertNotIn("Validated Model Attribution", text)
            self.assertNotIn("Causal Attribution", text)
            self.assertNotIn("Context-Preserving Model-Faithful", text)

    def test_19_feedback_config_has_only_zero_lambda(self):
        payload = json.loads(
            (ROOT / "configs/xai/xai_feedback_experiments.yaml").read_text()
        )
        self.assertEqual(payload["lambda_sweep"], [0.0])
        self.assertEqual(payload["feedback_training_status"],
                         "PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD")

    def test_20_cca_caption_discloses_pending_validation(self):
        text = (PAPER / "XAI_FIGURE_CAPTIONS.md").read_text()
        self.assertIn("Derived annotation; expert validation pending.", text)


if __name__ == "__main__":
    unittest.main()
