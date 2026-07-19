import unittest

import pandas as pd

from tools.sequence_lime_paper_evaluation import assert_no_test_oracle_fields


class XAIPaperSafetyGuardTest(unittest.TestCase):
    def test_paper_table_rejects_oracle_fields(self):
        assert_no_test_oracle_fields(pd.DataFrame({"metric": [1.0]}))
        with self.assertRaises(ValueError):
            assert_no_test_oracle_fields(pd.DataFrame({"test_oracle_best": [1.0]}))


if __name__ == "__main__":
    unittest.main()
