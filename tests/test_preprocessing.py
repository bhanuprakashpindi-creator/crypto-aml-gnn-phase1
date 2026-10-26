"""
Unit tests for preprocessing module.
Tests alignment, label encoding, feature separation, preventing target leakage into features,
and correlation-based feature reduction.
"""

import unittest
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.preprocessing import DataPreprocessor


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.config = PipelineConfig()
        self.config.num_local_features = 2
        self.config.num_aggregated_features = 3
        self.config.total_numeric_features = 5
        self.config.correlation_threshold = 0.85
        self.config.train_timesteps = (1, 2)
        self.preprocessor = DataPreprocessor(self.config)

    def _create_synthetic_data(self):
        # 4 transactions
        df_classes = pd.DataFrame({
            "txId": ["tx_4", "tx_1", "tx_3", "tx_2"],  # out of order
            "class": ["unknown", "1", "2", "1"]
        })
        # Features with col 1,2 local, 3,4,5 aggregated
        # Make feat_4 perfectly correlated with feat_3 (feat_4 = 2 * feat_3)
        df_features = pd.DataFrame({
            "txId": ["tx_1", "tx_2", "tx_3", "tx_4"],
            "time_step": [1, 2, 1, 3],
            "feat_1": [1.0, 2.0, 3.0, 4.0],
            "feat_2": [0.5, 0.4, 0.3, 0.2],
            "feat_3": [10.0, 20.0, 30.0, 40.0],
            "feat_4": [20.0, 40.0, 60.0, 80.0],  # perfectly correlated with feat_3
            "feat_5": [5.0, 1.0, 8.0, 2.0]
        })
        return df_classes, df_features

    def test_alignment_and_index_mapping(self):
        df_classes, df_features = self._create_synthetic_data()
        df_aligned, tx_to_idx, idx_to_tx = self.preprocessor.align_and_index_nodes(df_classes, df_features)
        
        self.assertEqual(len(df_aligned), 4)
        self.assertEqual(len(tx_to_idx), 4)
        self.assertEqual(len(idx_to_tx), 4)
        for idx, tx in enumerate(idx_to_tx):
            self.assertEqual(tx_to_idx[tx], idx)

    def test_label_encoding(self):
        df_classes, df_features = self._create_synthetic_data()
        df_aligned, _, _ = self.preprocessor.align_and_index_nodes(df_classes, df_features)
        y = self.preprocessor.extract_labels(df_aligned)
        
        # Verify valid label range: 1 = illicit, 0 = licit, -1 = unknown
        unique_labels = set(np.unique(y))
        self.assertTrue(unique_labels.issubset({1, 0, -1}))
        # Verify unknown label is NOT converted to licit
        self.assertIn(-1, unique_labels)

    def test_feature_matrix_integrity_and_isolation(self):
        df_classes, df_features = self._create_synthetic_data()
        df_aligned, _, _ = self.preprocessor.align_and_index_nodes(df_classes, df_features)
        x_all, x_local = self.preprocessor.extract_features(df_aligned)
        
        self.assertEqual(x_all.shape, (4, 5))
        self.assertEqual(x_local.shape, (4, 2))
        self.assertEqual(x_all.dtype, np.float32)
        
        # Verify no string or label columns leaked into numeric feature matrix
        self.assertFalse(np.isnan(x_all).any())

    def test_correlation_pruning_on_train_split(self):
        df_classes, df_features = self._create_synthetic_data()
        df_aligned, _, _ = self.preprocessor.align_and_index_nodes(df_classes, df_features)
        timesteps = self.preprocessor.extract_timesteps(df_aligned)
        x_all, _ = self.preprocessor.extract_features(df_aligned)
        
        x_reduced, retained_idx, dropped_idx = self.preprocessor.compute_reduced_features(x_all, timesteps)
        
        # One of the two perfectly correlated aggregated features (index 2 or 3 in x_all, global idx 3) should be dropped
        self.assertEqual(len(dropped_idx), 1)
        self.assertEqual(x_reduced.shape[1], 4)  # 2 local + 2 retained aggregated = 4


if __name__ == "__main__":
    unittest.main()
