import csv
import hashlib
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/paper_xai"
OUTPUT = ROOT / "outputs/physiology_aware_display_validation/isruc_seed42"
PACKAGE = PAPER / "expert_evaluation_package"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PRSeqXAIPositioningTest(unittest.TestCase):
    def test_01_multidimensional_status_is_explicit(self):
        status = json.loads(
            (
                ROOT
                / "outputs/xai_authoritative_status/isruc_seed42/"
                "CURRENT_XAI_STATUS.json"
            ).read_text(encoding="utf-8")
        )
        expected = {
            "engineering_status": "IMPLEMENTATION_AND_AUDIT_PIPELINE_VALID",
            "sequence_adaptation_status":
                "CONTEXT_PRESERVING_SEQUENCE_INFERENCE_ESTABLISHED",
            "physiology_annotation_status":
                "AUTOMATIC_PHYSIOLOGY_CONSISTENCY_SUPPORTED",
            "visualization_status":
                "PHYSIOLOGY_AND_RELIABILITY_ENHANCED_PRESENTATION_IMPLEMENTED",
            "human_interpretability_status": "NOT_EVALUATED",
            "clinical_validation_status": "NOT_VALIDATED",
            "model_faithfulness_status": "NOT_ESTABLISHED",
            "feedback_training_status":
                "PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD",
        }
        for key, value in expected.items():
            self.assertEqual(status[key], value)
        self.assertEqual(
            status["scientific_status"],
            "MODEL_FAITHFULNESS_NOT_ESTABLISHED",
        )
        self.assertEqual(
            status["physiology_aware_visualization_decision"],
            "PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED",
        )

    def test_02_framework_is_not_renamed_attribution(self):
        status = json.loads(
            (
                ROOT
                / "outputs/xai_authoritative_status/isruc_seed42/"
                "CURRENT_XAI_STATUS.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(status["framework_short_name"], "PR-SeqXAI")
        self.assertEqual(
            status["canonical_posthoc_method"],
            "CONTEXT_PRESERVING_CLASSIC_SEQUENCE_LIME",
        )
        self.assertIn(
            "CONTEXT_PRESERVING_CLASSIC_SEQUENCE_LIME",
            status["framework_definition"],
        )

    def test_03_required_narrative_artifacts_are_nonempty(self):
        required = [
            "PR_SEQXAI_STORYLINE.md",
            "PR_SEQXAI_RESULTS_NARRATIVE.md",
            "PR_SEQXAI_VISUALIZATION_CONTRIBUTION.md",
            "PR_SEQXAI_CLAIM_MATRIX.csv",
            "PR_SEQXAI_REVIEWER_QA.md",
            "PR_SEQXAI_FINAL_CLAIM_AUDIT.csv",
            "PR_SEQXAI_FINAL_POSITIONING_REPORT.md",
        ]
        for name in required:
            path = PAPER / name
            self.assertTrue(path.is_file(), path)
            self.assertGreater(path.stat().st_size, 0, path)

    def test_04_claim_matrix_covers_required_boundaries(self):
        with (PAPER / "PR_SEQXAI_CLAIM_MATRIX.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        claims = " ".join(row["claim"].casefold() for row in rows)
        for concept in (
            "sequence adaptation",
            "physiology",
            "annotation coverage",
            "annotation consistency",
            "visualization",
            "reliability",
            "model faithfulness",
            "clinical validity",
            "human usability",
            "causal",
        ):
            self.assertIn(concept, claims)
        for row in rows:
            for field in (
                "claim",
                "evidence",
                "status",
                "allowed_wording",
                "forbidden_wording",
                "paper_location",
            ):
                self.assertTrue(row[field].strip(), (row["claim"], field))

    def test_05_claim_audit_has_no_unresolved_current_claim(self):
        with (PAPER / "PR_SEQXAI_FINAL_CLAIM_AUDIT.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertGreater(len(rows), 0)
        statuses = {row["classification"] for row in rows}
        self.assertTrue(
            statuses <= {
                "CURRENT_ALLOWED",
                "HISTORICAL_WITH_WARNING",
                "PROHIBITED_CURRENT_CLAIM",
            }
        )
        unresolved = [
            row for row in rows
            if row["classification"] == "PROHIBITED_CURRENT_CLAIM"
        ]
        self.assertEqual(unresolved, [])

    def test_06_automatic_results_are_frozen(self):
        expected = {
            "physiology_metrics.csv":
                "b2d96be12bf76f793e58b51abbc2885fdb7adcfb4b753b984b72816a2fdb62f2",
            "attribution_identity_guard.json":
                "56e5ebcd5af70ba400e7018af173407ee26a0ef4cb1607a0a71b060493ad3fdf",
            "blinded_pair_manifest.json":
                "adf59db7c4fa6a36704b150ea73a9ab270b5c76a004732ae35fa6368ff90a697",
            "final_decision.json":
                "799fb9fea453402af7ab5276974e73c1409e94ab89480eae866b4c2c19a86e8e",
        }
        for name, digest in expected.items():
            self.assertEqual(sha256(OUTPUT / name), digest, name)

    def test_07_expert_package_is_untouched(self):
        self.assertEqual(
            sha256(PACKAGE / "PACKAGE_MANIFEST.json"),
            "26f3900ae498d8a3d463d3bc9e0efaaf7b63c0955931b91716037f5891c55fe9",
        )
        files = [path for path in PACKAGE.rglob("*") if path.is_file()]
        self.assertEqual(len(files), 92)
        manifest = json.loads(
            (PACKAGE / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8")
        )
        for item in manifest["files"]:
            path = PACKAGE / item["path"]
            self.assertEqual(path.stat().st_size, item["bytes"])
            self.assertEqual(sha256(path), item["sha256"])

    def test_08_expert_package_git_tree_is_unchanged(self):
        tree = subprocess.check_output(
            [
                "git",
                "write-tree",
            ],
            cwd=ROOT,
            text=True,
        ).strip()
        self.assertTrue(tree)
        package_tree = subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD:docs/paper_xai/expert_evaluation_package",
            ],
            cwd=ROOT,
            text=True,
        ).strip()
        self.assertEqual(
            package_tree,
            "2c9603065c7758fd45f2cc825d0a76fce32feb0c",
        )

    def test_09_no_human_result_is_fabricated(self):
        text = "\n".join(
            (PAPER / name).read_text(encoding="utf-8")
            for name in (
                "PR_SEQXAI_STORYLINE.md",
                "PR_SEQXAI_RESULTS_NARRATIVE.md",
                "PR_SEQXAI_FINAL_POSITIONING_REPORT.md",
            )
        ).casefold()
        self.assertIn("human interpretability", text)
        self.assertIn("not evaluated", text)
        self.assertIn("evaluator count is zero", text)

    def test_10_feedback_and_normal_inference_boundaries_remain(self):
        storyline = (
            PAPER / "PR_SEQXAI_STORYLINE.md"
        ).read_text(encoding="utf-8")
        positioning = (
            PAPER / "PR_SEQXAI_FINAL_POSITIONING_REPORT.md"
        ).read_text(encoding="utf-8")
        self.assertIn("PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD", positioning)
        self.assertIn("strictly post-hoc", storyline)
        self.assertIn("Normal inference", positioning)


if __name__ == "__main__":
    unittest.main()
