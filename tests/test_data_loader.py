"""
Unit tests for data_loader module.
Tests file loading, header normalization, and edge case resilience using synthetic fixtures.
"""

import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.data_loader import load_classes, load_edgelist, load_features


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_classes_standard_header(self):
        fp = self.temp_path / "classes_std.csv"
        fp.write_text("txId,class\n100,1\n101,2\n102,unknown\n", encoding="utf-8")
        df = load_classes(fp)
        self.assertEqual(list(df.columns), ["txId", "class"])
        self.assertEqual(len(df), 3)
        self.assertEqual(df["txId"].tolist(), ["100", "101", "102"])
        self.assertEqual(df["class"].tolist(), ["1", "2", "unknown"])

    def test_load_classes_unusual_header_with_whitespace(self):
        fp = self.temp_path / "classes_weird.csv"
        fp.write_text("  tx_id  ,  Class_Label  \n 201 , 1 \n 202 , unknown \n", encoding="utf-8")
        df = load_classes(fp)
        self.assertEqual(list(df.columns), ["txId", "class"])
        self.assertEqual(df["txId"].tolist(), ["201", "202"])
        self.assertEqual(df["class"].tolist(), ["1", "unknown"])

    def test_load_classes_no_header(self):
        fp = self.temp_path / "classes_nohead.csv"
        fp.write_text("301,2\n302,1\n", encoding="utf-8")
        df = load_classes(fp)
        self.assertEqual(list(df.columns), ["txId", "class"])
        self.assertEqual(df["txId"].tolist(), ["301", "302"])

    def test_load_edgelist_standard(self):
        fp = self.temp_path / "edges.csv"
        fp.write_text("txId1,txId2\n100,101\n101,102\n", encoding="utf-8")
        df = load_edgelist(fp)
        self.assertEqual(list(df.columns), ["txId1", "txId2"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df["txId1"].tolist(), ["100", "101"])

    def test_load_features_raw_no_header(self):
        # 1 txId + 1 timestep + 4 sample features = 6 cols
        fp = self.temp_path / "features.csv"
        content = "100,1,0.1,0.2,0.3,0.4\n101,2,-0.5,0.0,1.2,0.8\n"
        fp.write_text(content, encoding="utf-8")
        df = load_features(fp, total_cols=6)
        self.assertEqual(df.shape, (2, 6))
        self.assertEqual(list(df.columns[:2]), ["txId", "time_step"])
        self.assertEqual(df["txId"].tolist(), ["100", "101"])
        self.assertEqual(df["time_step"].tolist(), [1, 2])
        self.assertAlmostEqual(df["feat_1"].iloc[0], 0.1, places=5)

    def test_missing_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            load_classes(self.temp_path / "non_existent.csv")


if __name__ == "__main__":
    unittest.main()
