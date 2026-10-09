import csv
import hashlib
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/paper_xai"
OUT = ROOT / "outputs/physiology_aware_display_validation/isruc_seed42"
PACKAGE = PAPER / "expert_evaluation_package"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PhysiologyAwareDisplayValidationTest(unittest.TestCase):
    def test_01_decision_requires_real_blinded_evaluation(self):
        decision = load_json(OUT / "final_decision.json")
        self.assertEqual(
            decision["decision"],
            "PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED",
        )
        self.assertFalse(decision["human_blinded_evaluation_completed"])
        self.assertEqual(decision["evaluator_count"], 0)
        self.assertIsNone(decision["expert_agreement"])

    def test_02_v0_v1_attribution_is_bitwise_identical(self):
        guard = load_json(OUT / "attribution_identity_guard.json")
        self.assertEqual(guard["status"], "PASS")
        self.assertEqual(guard["case_count"], 30)
        self.assertEqual(guard["bitwise_identical_cases"], 30)
        for row in guard["rows"]:
            self.assertTrue(row["bitwise_identical"])
            self.assertEqual(row["source_array_sha256"], row["v0_array_sha256"])
            self.assertEqual(row["v0_array_sha256"], row["v1_array_sha256"])

    def test_03_ranking_and_shared_scale_are_identical(self):
        guard = load_json(OUT / "attribution_identity_guard.json")
        self.assertEqual(guard["ranking_identical_cases"], 30)
        self.assertEqual(guard["shared_scale_cases"], 30)
        self.assertTrue(all(row["ranking_identical"] for row in guard["rows"]))
        self.assertTrue(all(row["shared_color_limit"] > 0 for row in guard["rows"]))

    def test_04_cca_never_uses_stage_or_changes_ranking(self):
        with (OUT / "physiology_metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 30)
        for row in rows:
            self.assertEqual(row["true_stage_used_for_annotation"], "False")
            self.assertEqual(row["predicted_stage_used_for_annotation"], "False")
            self.assertEqual(row["ranking_modified"], "False")
            self.assertEqual(row["attribution_unchanged_exact"], "True")

    def test_05_allocation_is_seeded_balanced_and_reproducible(self):
        allocation = load_json(OUT / "blinded_pair_manifest.json")
        self.assertEqual(allocation["allocation_seed"], 20260726)
        self.assertEqual(allocation["balanced"]["context_in_a"], 15)
        self.assertEqual(allocation["balanced"]["context_in_b"], 15)
        self.assertEqual(len(allocation["pairs"]), 30)
        self.assertEqual(
            len({row["anonymous_case_id"] for row in allocation["pairs"]}), 30
        )

    def test_06_evaluator_package_hides_method_identity(self):
        forbidden = (
            b"Classic",
            b"Enhanced",
            b"New method",
            b"Our method",
            b"V0",
            b"V1",
        )
        for path in PACKAGE.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in {".md", ".json", ".csv"}:
                payload = path.read_bytes()
            elif path.suffix == ".pdf":
                payload = subprocess.run(
                    ["pdftotext", str(path), "-"],
                    check=True,
                    capture_output=True,
                ).stdout
            else:
                continue
            for term in forbidden:
                self.assertNotIn(term, payload, (path, term))

    def test_07_case_pool_covers_prespecified_strata(self):
        selection = load_json(OUT / "case_selection.json")
        cases = selection["cases"]
        self.assertEqual(selection["main_case_count"], 30)
        self.assertEqual({row["true_stage_reporting_only"] for row in cases},
                         {"W", "N1", "N2", "N3", "REM"})
        self.assertEqual({row["correct"] for row in cases}, {True, False})
        self.assertEqual({row["transition"] for row in cases}, {True, False})
        self.assertEqual({row["dominant_channel"] for row in cases},
                         {"EEG", "EOG", "EMG"})
        self.assertTrue(any(row["auxiliary_context_dominated"] for row in cases))

    def test_08_pilot_has_ten_rule_selected_cases(self):
        selection = load_json(OUT / "case_selection.json")
        pilot = [row for row in selection["cases"] if row["pilot_case"]]
        self.assertEqual(selection["pilot_case_count"], 10)
        self.assertEqual(len(pilot), 10)
        protocol = (PAPER / "PHYSIOLOGY_DISPLAY_CASE_SELECTION.md").read_text()
        self.assertIn("maximizes the number of uncovered tokens", protocol)
        self.assertIn("The algorithm never reads a", protocol)
        self.assertIn("rendered image", protocol)

    def test_09_unreliable_cases_are_retained(self):
        cases = load_json(OUT / "case_selection.json")["cases"]
        states = {row["reliability_status"] for row in cases}
        self.assertIn("RANDOMIZATION_UNRELIABLE", states)
        self.assertIn("SURROGATE_UNRELIABLE", states)
        self.assertIn("LOW_MODEL_RESPONSE", states)
        self.assertIn("LOCALLY_EXPLAINABLE", states)

    def test_10_none_annotation_is_valid_and_not_filtered(self):
        protocol = (PAPER / "PHYSIOLOGY_AWARE_DISPLAY_PROTOCOL.md").read_text()
        selection = (PAPER / "PHYSIOLOGY_DISPLAY_CASE_SELECTION.md").read_text()
        self.assertIn("NONE/INSUFFICIENT_EVIDENCE", protocol)
        self.assertIn("unsupported-annotation", selection)

    def test_11_no_gpu_model_or_inference_was_used(self):
        run = load_json(OUT / "run_manifest.json")
        self.assertTrue(run["no_lime_experiment_started"])
        self.assertTrue(run["no_model_loaded"])
        self.assertTrue(run["no_model_inference"])
        self.assertTrue(run["no_gpu_execution"])
        self.assertEqual(run["cuda_visible_devices"], "")
        self.assertFalse(run["test_set_used"] if "test_set_used" in run else False)

    def test_12_cpu_diagnostic_is_excluded(self):
        run = load_json(OUT / "run_manifest.json")
        self.assertFalse(run["cpu_diagnostic_used"])
        for source in run["source_files"]:
            self.assertNotIn("parity_cpu_driver_mismatch_20260724", source["path"])

    def test_13_feedback_remains_permanently_blocked(self):
        run = load_json(OUT / "run_manifest.json")
        self.assertEqual(
            run["feedback_training_status"],
            "PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD",
        )

    def test_14_normal_inference_still_defaults_to_no_xai(self):
        text = (ROOT / "run_main.py").read_text(encoding="utf-8")
        self.assertIn('parse_bool_env("LFTS_XAI_FEEDBACK_ENABLED", False)', text)
        self.assertIn("assert_feedback_training_disabled(", text)

    def test_15_all_frozen_source_hashes_still_match(self):
        run = load_json(OUT / "run_manifest.json")
        self.assertGreater(len(run["source_files"]), 30)
        for source in run["source_files"]:
            path = ROOT / source["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(file_sha(path), source["sha256"], path)

    def test_16_package_manifest_hashes_all_evaluator_files(self):
        manifest = load_json(PACKAGE / "PACKAGE_MANIFEST.json")
        self.assertTrue(manifest["method_identity_hidden"])
        self.assertFalse(manifest["allocation_key_included"])
        self.assertEqual(manifest["case_count"], 30)
        self.assertEqual(len(manifest["pilot_case_ids"]), 10)
        for item in manifest["files"]:
            path = PACKAGE / item["path"]
            self.assertGreater(path.stat().st_size, 0, path)
            self.assertEqual(path.stat().st_size, item["bytes"])
            self.assertEqual(file_sha(path), item["sha256"])

    def test_17_output_manifest_is_complete_and_not_self_referential(self):
        manifest = load_json(OUT / "CURRENT_RESULTS_MANIFEST.json")
        paths = {item["path"] for item in manifest["artifacts"]}
        self.assertNotIn(
            "outputs/physiology_aware_display_validation/isruc_seed42/"
            "CURRENT_RESULTS_MANIFEST.json",
            paths,
        )
        self.assertGreater(len(paths), 190)
        for item in manifest["artifacts"]:
            path = ROOT / item["path"]
            self.assertGreater(path.stat().st_size, 0, path)
            self.assertEqual(file_sha(path), item["sha256"])

    def test_18_rating_and_task_templates_contain_no_fabricated_results(self):
        decision = load_json(OUT / "final_decision.json")
        self.assertFalse(decision["human_blinded_evaluation_completed"])
        with (OUT / "expert_rating_template.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 60)
        self.assertTrue(all(not row["evaluator_id"] for row in rows))
        self.assertTrue(
            all(not row["physiological_plausibility_1_5"] for row in rows)
        )

    def test_19_risk_of_misleading_is_analyzed_separately(self):
        plan = load_json(OUT / "statistical_analysis_plan.json")
        self.assertEqual(
            plan["risk_of_misleading_direction"],
            "lower_is_better_analyzed_separately",
        )
        text = (PAPER / "PHYSIOLOGY_DISPLAY_STATISTICAL_ANALYSIS_PLAN.md").read_text()
        self.assertIn("never summed in", text)
        self.assertIn("same direction", text)

    def test_20_figure_a_retains_native_vector_boundary(self):
        manifest = load_json(PAPER / "figures/figure_a/figure_manifest.json")
        self.assertEqual(manifest["render_backend"], "native_vector_fallback")
        with (PAPER / "figures/figure_a/figure_data.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            payload = stream.read()
        for label in (
            "Frozen Model",
            "Pre-HMM Neural Response",
            "No Gradient",
            "No Training Feedback",
            "Post-hoc Only",
        ):
            self.assertIn(label, payload)

    def test_21_required_outputs_are_nonempty(self):
        paths = [
            OUT / "run_manifest.json",
            OUT / "case_selection.json",
            OUT / "blinded_pair_manifest.json",
            OUT / "attribution_identity_guard.json",
            OUT / "physiology_metrics.csv",
            OUT / "annotation_coverage.csv",
            OUT / "annotation_conflicts.csv",
            OUT / "expert_rating_template.csv",
            OUT / "user_task_template.csv",
            OUT / "statistical_analysis_plan.json",
            OUT / "final_decision.json",
            OUT / "figures/physiology_metric_summary.pdf",
            OUT / "figures/physiology_metric_summary.png",
        ]
        for path in paths:
            self.assertTrue(path.is_file(), path)
            self.assertGreater(path.stat().st_size, 0, path)

    def test_22_display_report_preserves_claim_boundaries(self):
        report = (PAPER / "PHYSIOLOGY_AWARE_DISPLAY_REPORT.md").read_text()
        self.assertIn("NOT_YET_VALIDATED", report)
        self.assertIn("No real blinded evaluator responses", report)
        self.assertIn("MODEL_FAITHFULNESS_NOT_ESTABLISHED", report)
        self.assertIn("Display and evaluation-protocol innovation", report)


if __name__ == "__main__":
    unittest.main()
