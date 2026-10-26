"""
Unit tests for data_splitter module.
Tests temporal split mask generation, labeled-only filtering, boundary exclusivity,
and temporal leakage detection across splits.
"""

import unittest
import numpy as np

from src.config import PipelineConfig
from src.data_splitter import DataSplitter


class TestDataSplitter(unittest.TestCase):

    def setUp(self):
        self.config = PipelineConfig()
        self.config.train_timesteps = (1, 10)
        self.config.val_timesteps = (11, 15)
        self.config.test_timesteps = (16, 20)
        self.splitter = DataSplitter(self.config)

    def test_temporal_masks_exclusivity_and_coverage(self):
        timesteps = np.array([1, 5, 10, 11, 15, 16, 20], dtype=np.int32)
        y = np.array([1, 0, -1, 1, 0, 1, -1], dtype=np.int64)

        masks, stats = self.splitter.generate_temporal_masks(timesteps, y)

        train_mask = masks["train_mask"]
        val_mask = masks["val_mask"]
        test_mask = masks["test_mask"]

        # Ensure mutually exclusive
        self.assertFalse((train_mask & val_mask).any())
        self.assertFalse((train_mask & test_mask).any())
        self.assertFalse((val_mask & test_mask).any())

        # Ensure all nodes within 1..20 are covered
        self.assertTrue((train_mask | val_mask | test_mask).all())

        # Check counts
        self.assertEqual(stats["train_split"]["total_nodes"], 3)
        self.assertEqual(stats["val_split"]["total_nodes"], 2)
        self.assertEqual(stats["test_split"]["total_nodes"], 2)

    def test_labeled_only_masks(self):
        timesteps = np.array([2, 3, 12, 18], dtype=np.int32)
        y = np.array([1, -1, 0, -1], dtype=np.int64)  # nodes 1 and 3 are unknown

        masks, stats = self.splitter.generate_temporal_masks(timesteps, y)

        train_labeled = masks["train_mask_labeled"]
        # In train (timesteps 2, 3), node 0 is illicit (1), node 1 is unknown (-1)
        self.assertEqual(train_labeled.sum(), 1)
        self.assertTrue(train_labeled[0])
        self.assertFalse(train_labeled[1])

    def test_temporal_leakage_detection(self):
        # 3 nodes: node 0 in train (ts 5), node 1 in val (ts 12), node 2 in test (ts 18)
        timesteps = np.array([5, 12, 18], dtype=np.int32)
        
        # Case A: strictly same timestep edges (no cross-split)
        edge_index_safe = np.array([[0, 1], [0, 1]], dtype=np.int64)
        assessment_safe = self.splitter.analyze_temporal_leakage(edge_index_safe, timesteps)
        self.assertTrue(assessment_safe["is_topologically_leakage_free"])
        self.assertEqual(assessment_safe["total_cross_split_edges"], 0)

        # Case B: cross-split edge from train (node 0) to test (node 2)
        edge_index_leak = np.array([[0], [2]], dtype=np.int64)
        assessment_leak = self.splitter.analyze_temporal_leakage(edge_index_leak, timesteps)
        self.assertFalse(assessment_leak["is_topologically_leakage_free"])
        self.assertEqual(assessment_leak["train_to_test_edges"], 1)


if __name__ == "__main__":
    unittest.main()
