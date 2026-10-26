"""
Graph builder module for constructing PyTorch Geometric compatible transaction graphs.
Maps original Bitcoin transaction identifiers to continuous node indices, validates endpoints,
preserves edge directionality, and generates PyG Data objects and NumPy arrays.
"""

from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
import numpy as np
import pandas as pd

try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

from src.config import PipelineConfig, get_default_config


class GraphBuilder:
    """Constructs structured graph representations from transaction edge lists."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_default_config()

    def build_edge_index(
        self,
        df_edgelist: pd.DataFrame,
        tx_to_idx: Dict[str, int],
        remove_self_loops: bool = False,
        remove_duplicates: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Convert transaction edge list (txId1 -> txId2) into a numerical edge_index [2, E].
        
        Args:
            df_edgelist: DataFrame with columns ['txId1', 'txId2']
            tx_to_idx: Mapping from string txId to continuous integer node index [0..N-1]
            remove_self_loops: Whether to prune self-loops (txId1 == txId2)
            remove_duplicates: Whether to prune duplicate directed edges
            
        Returns:
            edge_index: NumPy int64 array of shape [2, E]
            stats: Validation dictionary with counts of edges, self-loops, and dropped items
        """
        src_raw = df_edgelist["txId1"].astype(str).str.strip().tolist()
        dst_raw = df_edgelist["txId2"].astype(str).str.strip().tolist()

        valid_src: List[int] = []
        valid_dst: List[int] = []
        invalid_edges_count = 0
        self_loops_count = 0

        for s, d in zip(src_raw, dst_raw):
            if s not in tx_to_idx or d not in tx_to_idx:
                invalid_edges_count += 1
                continue

            s_idx = tx_to_idx[s]
            d_idx = tx_to_idx[d]

            if s_idx == d_idx:
                self_loops_count += 1
                if remove_self_loops:
                    continue

            valid_src.append(s_idx)
            valid_dst.append(d_idx)

        raw_edge_count = len(valid_src)
        edge_array = np.vstack([valid_src, valid_dst]).astype(np.int64)  # shape [2, E]

        # Handle duplicates if requested
        if remove_duplicates and edge_array.shape[1] > 0:
            # Transpose to [E, 2] for unique row extraction
            edges_t = edge_array.T
            unique_edges = np.unique(edges_t, axis=0)
            duplicates_removed = len(edges_t) - len(unique_edges)
            edge_index = unique_edges.T
        else:
            duplicates_removed = 0
            edge_index = edge_array

        stats = {
            "total_input_edges": len(df_edgelist),
            "valid_edges_retained": int(edge_index.shape[1]),
            "invalid_edges_dropped": invalid_edges_count,
            "self_loops_detected": self_loops_count,
            "self_loops_removed": self_loops_count if remove_self_loops else 0,
            "duplicate_edges_removed": duplicates_removed,
            "is_directed": True
        }

        print(
            f"[GraphBuilder] Constructed edge_index: {edge_index.shape[1]:,} directed edges "
            f"(dropped {invalid_edges_count} invalid, {duplicates_removed} duplicates)."
        )

        return edge_index, stats

    def create_pyg_data(
        self,
        edge_index: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        timesteps: np.ndarray,
        train_mask: Optional[np.ndarray] = None,
        val_mask: Optional[np.ndarray] = None,
        test_mask: Optional[np.ndarray] = None
    ) -> Any:
        """
        Package features, labels, edge_index, and masks into a PyTorch Geometric Data object.
        Falls back to a standard Python dictionary if PyG is not installed.
        """
        if not HAS_TORCH_GEOMETRIC:
            print("[GraphBuilder] PyTorch Geometric not available. Creating dict container.")
            return {
                "edge_index": edge_index,
                "x": x,
                "y": y,
                "timesteps": timesteps,
                "train_mask": train_mask,
                "val_mask": val_mask,
                "test_mask": test_mask
            }

        edge_index_torch = torch.from_numpy(np.array(edge_index, copy=True)).long()
        x_torch = torch.from_numpy(np.array(x, copy=True)).float()
        y_torch = torch.from_numpy(np.array(y, copy=True)).long()
        timesteps_torch = torch.from_numpy(np.array(timesteps, copy=True)).int()

        data = Data(
            x=x_torch,
            edge_index=edge_index_torch,
            y=y_torch,
            timesteps=timesteps_torch
        )

        if train_mask is not None:
            data.train_mask = torch.from_numpy(np.array(train_mask, copy=True)).bool()
        if val_mask is not None:
            data.val_mask = torch.from_numpy(np.array(val_mask, copy=True)).bool()
        if test_mask is not None:
            data.test_mask = torch.from_numpy(np.array(test_mask, copy=True)).bool()

        return data
