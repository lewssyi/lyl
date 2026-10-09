import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from tools.sequence_lime_paper_evaluation import (
    assert_no_test_oracle_fields,
    mode_config,
    run_evaluation,
)


class XAIPaperSafetyGuardTest(unittest.TestCase):
    def test_paper_table_rejects_oracle_fields(self):
        assert_no_test_oracle_fields(pd.DataFrame({"metric": [1.0]}))
        with self.assertRaises(ValueError):
            assert_no_test_oracle_fields(pd.DataFrame({"test_oracle_best": [1.0]}))

    @patch("tools.sequence_lime_paper_evaluation.mark_run_complete")
    @patch("tools.sequence_lime_paper_evaluation.merge_full_evaluation_shards")
    @patch("tools.sequence_lime_paper_evaluation.resolve_mode_config")
    @patch("tools.sequence_lime_paper_evaluation.write_static_audit")
    @patch("tools.sequence_lime_paper_evaluation.build_artifact_inventory")
    @patch("tools.sequence_lime_paper_evaluation.np.load")
    def test_merge_routes_before_loading_data_or_model(
        self,
        load_mock,
        inventory_mock,
        _audit_mock,
        config_mock,
        merge_mock,
        complete_mock,
    ):
        inventory_mock.return_value = {"status": "OK"}
        config = mode_config("full")
        config_mock.return_value = config
        args = SimpleNamespace(
            mode="full",
            merge_shard_run_ids="shard_a,shard_b",
            resolved_run_id="merged_guarded",
        )

        run_evaluation(args)

        merge_mock.assert_called_once_with(args, config, ["shard_a", "shard_b"])
        complete_mock.assert_called_once_with(args)
        load_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
