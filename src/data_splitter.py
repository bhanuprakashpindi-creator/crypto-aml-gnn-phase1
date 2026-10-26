"""
Data splitter module implementing temporal graph partitioning for the Elliptic dataset.
Generates strict time-based masks for train, validation, and test splits,
and provides leakage analysis across split boundaries.
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np

from src.config import PipelineConfig, get_default_config


class DataSplitter:
    """Manages temporal data splitting and evaluates potential leakage across boundaries."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_default_config()

    def generate_temporal_masks(
        self,
        timesteps: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Generate train, validation, and test boolean masks based on timesteps.
        
        Default Boundaries (Weber et al. / Configurable):
        - Train: timesteps 1..34
        - Val: timesteps 35..41
        - Test: timesteps 42..49
        
        Also generates labeled-only masks (excluding unknown labels, y == -1).
        """
        train_min, train_max = self.config.train_timesteps
        val_min, val_max = self.config.val_timesteps
        test_min, test_max = self.config.test_timesteps

        train_mask = (timesteps >= train_min) & (timesteps <= train_max)
        val_mask = (timesteps >= val_min) & (timesteps <= val_max)
        test_mask = (timesteps >= test_min) & (timesteps <= test_max)

        # Labeled-only masks
        is_labeled = (y != -1)
        train_mask_labeled = train_mask & is_labeled
        val_mask_labeled = val_mask & is_labeled
        test_mask_labeled = test_mask & is_labeled

        masks = {
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "train_mask_labeled": train_mask_labeled,
            "val_mask_labeled": val_mask_labeled,
            "test_mask_labeled": test_mask_labeled
        }

        # Calculate statistics per split
        stats = {
            "total_nodes": int(len(timesteps)),
            "train_split": {
                "timesteps": f"{train_min}..{train_max}",
                "total_nodes": int(train_mask.sum()),
                "labeled_nodes": int(train_mask_labeled.sum()),
                "illicit_count": int(((y == 1) & train_mask).sum()),
                "licit_count": int(((y == 0) & train_mask).sum()),
                "unknown_count": int(((y == -1) & train_mask).sum()),
                "illicit_ratio_labeled": float(
                    ((y == 1) & train_mask).sum() / max(1, train_mask_labeled.sum())
                )
            },
            "val_split": {
                "timesteps": f"{val_min}..{val_max}",
                "total_nodes": int(val_mask.sum()),
                "labeled_nodes": int(val_mask_labeled.sum()),
                "illicit_count": int(((y == 1) & val_mask).sum()),
                "licit_count": int(((y == 0) & val_mask).sum()),
                "unknown_count": int(((y == -1) & val_mask).sum()),
                "illicit_ratio_labeled": float(
                    ((y == 1) & val_mask).sum() / max(1, val_mask_labeled.sum())
                )
            },
            "test_split": {
                "timesteps": f"{test_min}..{test_max}",
                "total_nodes": int(test_mask.sum()),
                "labeled_nodes": int(test_mask_labeled.sum()),
                "illicit_count": int(((y == 1) & test_mask).sum()),
                "licit_count": int(((y == 0) & test_mask).sum()),
                "unknown_count": int(((y == -1) & test_mask).sum()),
                "illicit_ratio_labeled": float(
                    ((y == 1) & test_mask).sum() / max(1, test_mask_labeled.sum())
                )
            }
        }

        print("[DataSplitter] Partitioned dataset into temporal splits:")
        print(
            f"  Train (TS {train_min}..{train_max}): {stats['train_split']['total_nodes']:,} nodes "
            f"({stats['train_split']['labeled_nodes']:,} labeled: "
            f"{stats['train_split']['illicit_count']:,} illicit, {stats['train_split']['licit_count']:,} licit)"
        )
        print(
            f"  Val   (TS {val_min}..{val_max}): {stats['val_split']['total_nodes']:,} nodes "
            f"({stats['val_split']['labeled_nodes']:,} labeled: "
            f"{stats['val_split']['illicit_count']:,} illicit, {stats['val_split']['licit_count']:,} licit)"
        )
        print(
            f"  Test  (TS {test_min}..{test_max}): {stats['test_split']['total_nodes']:,} nodes "
            f"({stats['test_split']['labeled_nodes']:,} labeled: "
            f"{stats['test_split']['illicit_count']:,} illicit, {stats['test_split']['licit_count']:,} licit)"
        )

        return masks, stats

    def analyze_temporal_leakage(
        self,
        edge_index: np.ndarray,
        timesteps: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze whether graph edges cross split boundaries.
        
        Evaluates potential temporal leakage paths where test/validation information
        could propagate into training representations via message passing.
        """
        src = edge_index[0]
        dst = edge_index[1]

        src_ts = timesteps[src]
        dst_ts = timesteps[dst]

        train_min, train_max = self.config.train_timesteps
        val_min, val_max = self.config.val_timesteps
        test_min, test_max = self.config.test_timesteps

        src_is_train = (src_ts >= train_min) & (src_ts <= train_max)
        dst_is_train = (dst_ts >= train_min) & (dst_ts <= train_max)

        src_is_val = (src_ts >= val_min) & (src_ts <= val_max)
        dst_is_val = (dst_ts >= val_min) & (dst_ts <= val_max)

        src_is_test = (src_ts >= test_min) & (src_ts <= test_max)
        dst_is_test = (dst_ts >= test_min) & (dst_ts <= test_max)

        # Cross split edges
        train_to_val_edges = int((src_is_train & dst_is_val).sum())
        val_to_train_edges = int((src_is_val & dst_is_train).sum())
        train_to_test_edges = int((src_is_train & dst_is_test).sum())
        test_to_train_edges = int((src_is_test & dst_is_train).sum())
        val_to_test_edges = int((src_is_val & dst_is_test).sum())
        test_to_val_edges = int((src_is_test & dst_is_val).sum())

        total_cross_split_edges = (
            train_to_val_edges + val_to_train_edges +
            train_to_test_edges + test_to_train_edges +
            val_to_test_edges + test_to_val_edges
        )

        leakage_assessment = {
            "total_edges_evaluated": int(edge_index.shape[1]),
            "train_to_val_edges": train_to_val_edges,
            "val_to_train_edges": val_to_train_edges,
            "train_to_test_edges": train_to_test_edges,
            "test_to_train_edges": test_to_train_edges,
            "val_to_test_edges": val_to_test_edges,
            "test_to_val_edges": test_to_val_edges,
            "total_cross_split_edges": total_cross_split_edges,
            "is_topologically_leakage_free": (total_cross_split_edges == 0),
            "findings": (
                "No edges cross temporal split boundaries. "
                "The dataset consists of completely isolated temporal graph snapshots."
                if total_cross_split_edges == 0 else
                "Warning: Cross-split edges exist. GNN message passing across temporal boundaries must be masked."
            )
        }

        return leakage_assessment
