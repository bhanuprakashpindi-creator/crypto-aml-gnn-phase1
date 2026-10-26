"""
Unit tests for graph_builder module.
Tests edge_index construction, node mapping resolution, edge validation,
duplicate and self-loop handling, and PyG Data formatting.
"""

import unittest
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.graph_builder import GraphBuilder, HAS_TORCH_GEOMETRIC


class TestGraphBuilder(unittest.TestCase):

    def setUp(self):
        self.config = PipelineConfig()
        self.builder = GraphBuilder(self.config)
        self.tx_to_idx = {"tx_A": 0, "tx_B": 1, "tx_C": 2}

    def test_build_edge_index_standard(self):
        df_edges = pd.DataFrame({
            "txId1": ["tx_A", "tx_B"],
            "txId2": ["tx_B", "tx_C"]
        })
        edge_index, stats = self.builder.build_edge_index(
            df_edges, self.tx_to_idx, remove_self_loops=False, remove_duplicates=True
        )
        self.assertEqual(edge_index.shape, (2, 2))
        self.assertEqual(edge_index.dtype, np.int64)
        np.testing.assert_array_equal(edge_index, np.array([[0, 1], [1, 2]]))
        self.assertEqual(stats["invalid_edges_dropped"], 0)

    def test_filter_invalid_endpoints(self):
        df_edges = pd.DataFrame({
            "txId1": ["tx_A", "tx_A", "tx_UNKNOWN"],
            "txId2": ["tx_B", "tx_MISSING", "tx_C"]
        })
        edge_index, stats = self.builder.build_edge_index(
            df_edges, self.tx_to_idx, remove_self_loops=False, remove_duplicates=True
        )
        # Only tx_A -> tx_B is valid
        self.assertEqual(edge_index.shape, (2, 1))
        self.assertEqual(stats["invalid_edges_dropped"], 2)

    def test_handle_duplicates_and_self_loops(self):
        df_edges = pd.DataFrame({
            "txId1": ["tx_A", "tx_A", "tx_B"],
            "txId2": ["tx_B", "tx_B", "tx_B"]  # duplicate tx_A->tx_B, self loop tx_B->tx_B
        })
        # Remove self-loops and duplicates
        edge_index, stats = self.builder.build_edge_index(
            df_edges, self.tx_to_idx, remove_self_loops=True, remove_duplicates=True
        )
        self.assertEqual(edge_index.shape, (2, 1))  # only 1 unique valid non-self-loop edge
        self.assertEqual(stats["duplicate_edges_removed"], 1)
        self.assertEqual(stats["self_loops_detected"], 1)

    def test_create_pyg_data_container(self):
        edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
        x = np.ones((3, 4), dtype=np.float32)
        y = np.array([1, 0, -1], dtype=np.int64)
        timesteps = np.array([1, 1, 2], dtype=np.int32)
        train_mask = np.array([True, True, False])

        data = self.builder.create_pyg_data(
            edge_index=edge_index,
            x=x,
            y=y,
            timesteps=timesteps,
            train_mask=train_mask
        )

        if HAS_TORCH_GEOMETRIC:
            self.assertEqual(data.num_nodes, 3)
            self.assertEqual(data.num_edges, 2)
            self.assertEqual(data.x.shape, (3, 4))
        else:
            self.assertIn("edge_index", data)
            self.assertIn("x", data)


if __name__ == "__main__":
    unittest.main()
