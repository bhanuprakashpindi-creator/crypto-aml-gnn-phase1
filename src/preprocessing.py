"""
Preprocessing module for the Elliptic Bitcoin dataset.
Aligns nodes by transaction ID, separates identifiers, timesteps, and target labels,
constructs clean numeric feature matrices, and computes leakage-free feature reductions.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
import numpy as np
import pandas as pd

from src.config import PipelineConfig, get_default_config


class DataPreprocessor:
    """Preprocesses raw transaction records into clean, model-ready structures."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_default_config()
        self.tx_to_idx: Dict[str, int] = {}
        self.idx_to_tx: List[str] = []
        self.removed_feature_indices: List[int] = []
        self.retained_feature_indices: List[int] = []

    def align_and_index_nodes(
        self,
        df_classes: pd.DataFrame,
        df_features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
        """
        Merge and strictly align features and labels by transaction ID.
        
        Guarantees that row index `i` consistently corresponds to the node index `i`
        across features, labels, and timesteps.
        """
        # Ensure string IDs
        df_classes = df_classes.copy()
        df_features = df_features.copy()
        df_classes["txId"] = df_classes["txId"].astype(str).str.strip()
        df_features["txId"] = df_features["txId"].astype(str).str.strip()

        # Inner merge ensures absolute alignment
        df_merged = pd.merge(df_features, df_classes, on="txId", how="inner")

        # Sort deterministically by timestep then txId to ensure stable ordering
        df_merged = df_merged.sort_values(by=["time_step", "txId"]).reset_index(drop=True)

        # Build bidirectional mapping
        node_tx_ids = df_merged["txId"].tolist()
        self.tx_to_idx = {tx_id: idx for idx, tx_id in enumerate(node_tx_ids)}
        self.idx_to_tx = node_tx_ids

        print(f"[Preprocessor] Aligned {len(df_merged):,} nodes across features and classes.")
        return df_merged, self.tx_to_idx, self.idx_to_tx

    def extract_labels(self, df_aligned: pd.DataFrame) -> np.ndarray:
        """
        Extract and encode target labels into integer vector:
        1 = Illicit, 0 = Licit, -1 = Unknown.
        
        Strictly prevents unknown labels from being conflated with licit transactions.
        """
        raw_labels = df_aligned["class"].astype(str).str.strip()
        label_map = self.config.label_raw_to_int
        y = np.array([label_map.get(lbl, -1) for lbl in raw_labels], dtype=np.int64)
        return y

    def extract_timesteps(self, df_aligned: pd.DataFrame) -> np.ndarray:
        """Extract separate integer timestep vector [N]."""
        return df_aligned["time_step"].to_numpy(dtype=np.int32)

    def extract_features(
        self,
        df_aligned: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract numeric feature matrices:
        - `x_all`: shape [N, 165] (93 local + 72 aggregated)
        - `x_local`: shape [N, 93] (local transaction features only)
        
        Guarantees no txId, timestep, or target class enters the feature matrix.
        """
        feat_cols = [c for c in df_aligned.columns if c.startswith("feat_")]
        if len(feat_cols) != self.config.total_numeric_features:
            raise ValueError(
                f"Expected {self.config.total_numeric_features} feature columns, "
                f"found {len(feat_cols)}"
            )

        x_all = df_aligned[feat_cols].to_numpy(dtype=np.float32)

        # Local features: first 93 columns
        num_loc = self.config.num_local_features
        x_local = x_all[:, :num_loc]

        return x_all, x_local

    def compute_reduced_features(
        self,
        x_all: np.ndarray,
        timesteps: np.ndarray
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Compute the reduced feature set ('tx+agg_red') following Paper Section IV & V.
        
        The paper identifies highly correlated aggregated features (pairwise Pearson r > 0.90)
        and removes redundant pairs (22 features removed).
        
        Strict Leakage Prevention:
        The correlation matrix is calculated EXCLUSIVELY on training set timesteps (1..34)
        if configured, and applied to all nodes.
        """
        num_local = self.config.num_local_features
        num_total = x_all.shape[1]

        # Only aggregated features are subject to pruning
        agg_start_idx = num_local
        
        if self.config.fit_correlation_on_train_only:
            train_min, train_max = self.config.train_timesteps
            train_mask = (timesteps >= train_min) & (timesteps <= train_max)
            train_x_agg = x_all[train_mask, agg_start_idx:]
        else:
            train_x_agg = x_all[:, agg_start_idx:]

        # Compute Pearson correlation matrix among aggregated features
        corr_matrix = np.corrcoef(train_x_agg, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        num_agg = corr_matrix.shape[0]
        to_drop = set()
        threshold = self.config.correlation_threshold

        for i in range(num_agg):
            if i in to_drop:
                continue
            for j in range(i + 1, num_agg):
                if j in to_drop:
                    continue
                if abs(corr_matrix[i, j]) > threshold:
                    # Drop the secondary redundant feature
                    to_drop.add(j)

        # Global indices to drop (offset by num_local)
        dropped_global_indices = sorted([agg_start_idx + idx for idx in to_drop])
        retained_global_indices = [
            idx for idx in range(num_total) if idx not in dropped_global_indices
        ]

        self.removed_feature_indices = dropped_global_indices
        self.retained_feature_indices = retained_global_indices

        x_reduced = x_all[:, retained_global_indices]

        print(
            f"[Preprocessor] Feature pruning (correlation > {threshold:.2f}): "
            f"Evaluated {num_agg} aggregated features on train split. "
            f"Removed {len(dropped_global_indices)} collinear features. "
            f"Retained {len(retained_global_indices)} features ({num_local} local + {num_agg - len(to_drop)} aggregated)."
        )

        return x_reduced, retained_global_indices, dropped_global_indices

    def run_preprocessing_pipeline(
        self,
        df_classes: pd.DataFrame,
        df_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Execute full preprocessing pipeline and return structured tensors and mappings."""
        df_aligned, tx_to_idx, idx_to_tx = self.align_and_index_nodes(df_classes, df_features)
        y = self.extract_labels(df_aligned)
        timesteps = self.extract_timesteps(df_aligned)
        x_all, x_local = self.extract_features(df_aligned)
        x_reduced, retained_idx, dropped_idx = self.compute_reduced_features(x_all, timesteps)

        report = {
            "num_nodes": int(x_all.shape[0]),
            "feature_shapes": {
                "x_all (tx+agg)": list(x_all.shape),
                "x_local (tx)": list(x_local.shape),
                "x_reduced (tx+agg_red)": list(x_reduced.shape)
            },
            "features_removed_count": len(dropped_idx),
            "retained_feature_indices": retained_idx,
            "dropped_feature_indices": dropped_idx,
            "labels_distribution": {
                "illicit (1)": int((y == 1).sum()),
                "licit (0)": int((y == 0).sum()),
                "unknown (-1)": int((y == -1).sum())
            }
        }

        return {
            "x_all": x_all,
            "x_local": x_local,
            "x_reduced": x_reduced,
            "y": y,
            "timesteps": timesteps,
            "tx_to_idx": tx_to_idx,
            "idx_to_tx": idx_to_tx,
            "df_aligned": df_aligned,
            "report": report
        }
