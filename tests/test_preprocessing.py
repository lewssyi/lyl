import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_processing import format_for_model, load_npz_data


class PreprocessingTest(unittest.TestCase):
    def test_load_npz_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "eeg_subfolder"
            folder.mkdir()
            np.savez(folder / "test1.npz", x=np.random.rand(10, 3000), y=np.zeros(10))
            np.savez(folder / "test2.npz", x=np.random.rand(5, 3000), y=np.ones(5))
            features, labels = load_npz_data(str(folder))
            self.assertEqual(features.shape, (15, 3000, 1))
            self.assertEqual(labels.shape, (15,))
            self.assertEqual(len(np.unique(labels)), 2)

    def test_format_for_model(self):
        raw = np.random.default_rng(3).random((20, 3000))
        labels = np.asarray([0, 1, 2, 3, 4] * 4)
        features, one_hot = format_for_model(raw, labels, num_classes=5)
        self.assertEqual(features.shape, (20, 3000, 1))
        self.assertEqual(one_hot.shape, (20, 5))
        self.assertTrue(np.all(np.sum(one_hot, axis=1) == 1))

    def test_load_npz_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            load_npz_data("non_existent_path_123")


if __name__ == "__main__":
    unittest.main()
