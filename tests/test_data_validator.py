"""
Unit tests for data_validator module.
Validates structural integrity checks, edge endpoint validation, anomaly detection,
and report generation.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.data_validator import DataValidator


class TestDataValidator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.config = PipelineConfig(project_root=self.temp_path)
        self.config.total_numeric_features = 4
        self.validator = DataValidator(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_data(self):
        df_classes = pd.DataFrame({
            "txId": ["1", "2", "3", "4"],
            "class": ["1", "2", "unknown", "2"]
        })
        df_edgelist = pd.DataFrame({
            "txId1": ["1", "2", "3"],
            "txId2": ["2", "3", "4"]
        })
        df_features = pd.DataFrame({
            "txId": ["1", "2", "3", "4"],
            "time_step": [1, 1, 2, 2],
            "feat_1": [0.1, 0.2, 0.3, 0.4],
            "feat_2": [-0.1, 0.5, 0.2, -0.3],
            "feat_3": [1.0, 2.0, 1.5, 0.8],
            "feat_4": [0.0, -1.0, 0.5, 0.2]
        })
        return df_classes, df_edgelist, df_features

    def test_clean_dataset_passes(self):
        df_classes, df_edgelist, df_features = self._create_synthetic_data()
        summary = self.validator.validate_dataset(df_classes, df_edgelist, df_features)
        self.assertEqual(summary["overall_status"], "PASSED")
        self.assertTrue(summary["node_checks"]["perfect_node_alignment"])
        self.assertEqual(summary["edge_checks"]["invalid_source_nodes_count"], 0)
        self.assertEqual(summary["edge_checks"]["invalid_dest_nodes_count"], 0)
        self.assertEqual(summary["feature_checks"]["total_null_values"], 0)

    def test_detect_duplicate_transaction_ids(self):
        df_classes, df_edgelist, df_features = self._create_synthetic_data()
        # Inject duplicate ID in classes
        df_classes = pd.concat([df_classes, pd.DataFrame({"txId": ["1"], "class": ["1"]})], ignore_index=True)
        summary = self.validator.validate_dataset(df_classes, df_edgelist, df_features)
        self.assertEqual(summary["node_checks"]["duplicate_classes_tx_ids"], 1)

    def test_detect_mismatched_transaction_ids(self):
        df_classes, df_edgelist, df_features = self._create_synthetic_data()
        # Extra ID in classes
        df_classes = pd.concat([df_classes, pd.DataFrame({"txId": ["999"], "class": ["1"]})], ignore_index=True)
        summary = self.validator.validate_dataset(df_classes, df_edgelist, df_features)
        self.assertFalse(summary["node_checks"]["perfect_node_alignment"])
        self.assertEqual(summary["node_checks"]["tx_ids_classes_only_count"], 1)

    def test_detect_invalid_edge_endpoints(self):
        df_classes, df_edgelist, df_features = self._create_synthetic_data()
        # Edge pointing to non-existent node 888
        df_edgelist = pd.concat([df_edgelist, pd.DataFrame({"txId1": ["1"], "txId2": ["888"]})], ignore_index=True)
        summary = self.validator.validate_dataset(df_classes, df_edgelist, df_features)
        self.assertEqual(summary["edge_checks"]["invalid_dest_nodes_count"], 1)
        self.assertEqual(summary["overall_status"], "WARNINGS_OR_FAILURES_DETECTED")

    def test_detect_self_loops_and_duplicates(self):
        df_classes, df_edgelist, df_features = self._create_synthetic_data()
        # Add self-loop 1->1 and duplicate 1->2
        extra_edges = pd.DataFrame({"txId1": ["1", "1"], "txId2": ["1", "2"]})
        df_edgelist = pd.concat([df_edgelist, extra_edges], ignore_index=True)
        summary = self.validator.validate_dataset(df_classes, df_edgelist, df_features)
        self.assertEqual(summary["edge_checks"]["self_loops"], 1)
        self.assertEqual(summary["edge_checks"]["duplicate_edges"], 1)

    def test_detect_nan_and_inf_features(self):
        df_classes, df_edgelist, df_features = self._create_synthetic_data()
        df_features.loc[0, "feat_1"] = np.nan
        df_features.loc[1, "feat_2"] = np.inf
        summary = self.validator.validate_dataset(df_classes, df_edgelist, df_features)
        self.assertEqual(summary["feature_checks"]["total_null_values"], 1)
        self.assertEqual(summary["feature_checks"]["total_infinite_values"], 1)
        self.assertTrue(summary["feature_checks"]["has_missing_or_inf"])

    def test_save_reports(self):
        df_classes, df_edgelist, df_features = self._create_synthetic_data()
        summary = self.validator.validate_dataset(df_classes, df_edgelist, df_features)
        md_path, json_path = self.validator.save_reports(summary)
        self.assertTrue(md_path.exists())
        self.assertTrue(json_path.exists())
        self.assertIn("Elliptic Dataset Validation Report", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
