"""
Data validator module for the Elliptic Bitcoin dataset.
Executes systematic integrity checks on raw files, IDs, edges, features,
labels, and temporal distributions, generating human- and machine-readable reports.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from src.config import PipelineConfig, get_default_config


class DataValidator:
    """Comprehensive validation engine for raw transaction data."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_default_config()

    def validate_files_exist(self) -> Dict[str, Any]:
        """Verify presence of all three raw data files."""
        status = {
            "classes_file_exists": self.config.raw_classes_path.exists(),
            "edgelist_file_exists": self.config.raw_edgelist_path.exists(),
            "features_file_exists": self.config.raw_features_path.exists(),
        }
        status["all_files_present"] = all(status.values())
        return status

    def validate_dataset(
        self,
        df_classes: pd.DataFrame,
        df_edgelist: pd.DataFrame,
        df_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run full validation suite on in-memory raw DataFrames.
        
        Returns:
            Dictionary containing detailed validation metrics and issue tallies.
        """
        summary: Dict[str, Any] = {
            "node_checks": {},
            "edge_checks": {},
            "feature_checks": {},
            "label_checks": {},
            "timestep_checks": {},
            "overall_status": "PASSED"
        }

        # 1. Node ID & Record Alignment Checks
        classes_tx_ids = set(df_classes["txId"])
        features_tx_ids = set(df_features["txId"])

        classes_dup_count = int(df_classes["txId"].duplicated().sum())
        features_dup_count = int(df_features["txId"].duplicated().sum())

        ids_in_both = classes_tx_ids.intersection(features_tx_ids)
        classes_only = classes_tx_ids - features_tx_ids
        features_only = features_tx_ids - classes_tx_ids

        summary["node_checks"] = {
            "total_classes_records": len(df_classes),
            "unique_classes_tx_ids": len(classes_tx_ids),
            "duplicate_classes_tx_ids": classes_dup_count,
            "total_features_records": len(df_features),
            "unique_features_tx_ids": len(features_tx_ids),
            "duplicate_features_tx_ids": features_dup_count,
            "tx_ids_shared_count": len(ids_in_both),
            "tx_ids_classes_only_count": len(classes_only),
            "tx_ids_features_only_count": len(features_only),
            "perfect_node_alignment": (len(classes_only) == 0 and len(features_only) == 0)
        }

        # 2. Label Distribution Checks
        class_counts = df_classes["class"].value_counts().to_dict()
        valid_raw_classes = set(self.config.label_raw_to_int.keys())
        unexpected_classes = set(class_counts.keys()) - valid_raw_classes

        summary["label_checks"] = {
            "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
            "illicit_count": int(class_counts.get("1", 0)),
            "licit_count": int(class_counts.get("2", 0)),
            "unknown_count": int(class_counts.get("unknown", 0)),
            "missing_or_null_labels": int(df_classes["class"].isna().sum()),
            "unexpected_class_labels": list(unexpected_classes),
            "has_unexpected_labels": len(unexpected_classes) > 0
        }

        # 3. Feature Integrity Checks
        feat_cols = [c for c in df_features.columns if c.startswith("feat_")]
        null_counts_per_col = df_features[feat_cols].isna().sum()
        cols_with_nulls = int((null_counts_per_col > 0).sum())
        total_null_entries = int(null_counts_per_col.sum())

        # Check for infinities
        numeric_matrix = df_features[feat_cols].to_numpy()
        inf_count = int(np.isinf(numeric_matrix).sum())

        summary["feature_checks"] = {
            "total_feature_columns": len(feat_cols),
            "expected_numeric_features": self.config.total_numeric_features,
            "columns_with_nulls": cols_with_nulls,
            "total_null_values": total_null_entries,
            "total_infinite_values": inf_count,
            "has_missing_or_inf": (total_null_entries > 0 or inf_count > 0)
        }

        # 4. Timestep Distribution Checks
        ts_series = df_features["time_step"]
        min_ts = int(ts_series.min())
        max_ts = int(ts_series.max())
        unique_ts_count = int(ts_series.nunique())
        null_ts_count = int(ts_series.isna().sum())
        ts_distribution = {int(k): int(v) for k, v in ts_series.value_counts().sort_index().items()}

        summary["timestep_checks"] = {
            "min_timestep": min_ts,
            "max_timestep": max_ts,
            "unique_timesteps_count": unique_ts_count,
            "null_timesteps_count": null_ts_count,
            "expected_range_1_to_49": (min_ts == 1 and max_ts == 49 and unique_ts_count == 49),
            "distribution": ts_distribution
        }

        # 5. Edgelist Integrity Checks
        all_known_nodes = features_tx_ids.union(classes_tx_ids)
        total_edges = len(df_edgelist)
        
        # Self-loops
        self_loops = int((df_edgelist["txId1"] == df_edgelist["txId2"]).sum())

        # Duplicate edges
        dup_edges = int(df_edgelist.duplicated(subset=["txId1", "txId2"]).sum())

        # Invalid endpoints
        src_set = set(df_edgelist["txId1"])
        dst_set = set(df_edgelist["txId2"])
        invalid_src = src_set - all_known_nodes
        invalid_dst = dst_set - all_known_nodes

        # Temporal edge analysis (cross-timestep check)
        tx_to_ts = dict(zip(df_features["txId"], df_features["time_step"]))
        src_ts = df_edgelist["txId1"].map(tx_to_ts)
        dst_ts = df_edgelist["txId2"].map(tx_to_ts)
        
        same_ts_mask = (src_ts == dst_ts)
        same_ts_edges = int(same_ts_mask.sum())
        cross_ts_edges = total_edges - same_ts_edges

        summary["edge_checks"] = {
            "total_edges": total_edges,
            "duplicate_edges": dup_edges,
            "self_loops": self_loops,
            "invalid_source_nodes_count": len(invalid_src),
            "invalid_dest_nodes_count": len(invalid_dst),
            "same_timestep_edges": same_ts_edges,
            "cross_timestep_edges": cross_ts_edges,
            "edges_are_isolated_snapshots": (cross_ts_edges == 0)
        }

        # Overall Status Determination
        has_critical_failure = (
            summary["node_checks"]["perfect_node_alignment"] is False
            or summary["edge_checks"]["invalid_source_nodes_count"] > 0
            or summary["edge_checks"]["invalid_dest_nodes_count"] > 0
            or summary["feature_checks"]["has_missing_or_inf"] is True
            or summary["label_checks"]["has_unexpected_labels"] is True
        )
        if has_critical_failure:
            summary["overall_status"] = "WARNINGS_OR_FAILURES_DETECTED"
        else:
            summary["overall_status"] = "PASSED"

        return summary

    def generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        """Format validation metrics into a clear markdown report."""
        nc = summary["node_checks"]
        lc = summary["label_checks"]
        fc = summary["feature_checks"]
        tc = summary["timestep_checks"]
        ec = summary["edge_checks"]

        report = f"""# Elliptic Dataset Validation Report

**Validation Status**: `{summary["overall_status"]}`

---

## 1. Node & Record Alignment
- **Classes File Records**: {nc.get("total_classes_records", 0):,} (Unique IDs: {nc.get("unique_classes_tx_ids", 0):,}, Duplicates: {nc.get("duplicate_classes_tx_ids", 0)})
- **Features File Records**: {nc.get("total_features_records", 0):,} (Unique IDs: {nc.get("unique_features_tx_ids", 0):,}, Duplicates: {nc.get("duplicate_features_tx_ids", 0)})
- **Shared IDs**: {nc.get("tx_ids_shared_count", 0):,}
- **Classes-Only IDs**: {nc.get("tx_ids_classes_only_count", 0)}
- **Features-Only IDs**: {nc.get("tx_ids_features_only_count", 0)}
- **Perfect 1:1 Alignment**: `{nc.get("perfect_node_alignment", False)}`

## 2. Target Class Distribution
- **Illicit (Class 1)**: {lc.get("illicit_count", 0):,} ({lc.get("illicit_count", 0) / max(1, nc.get("total_classes_records", 1)) * 100:.2f}%)
- **Licit (Class 2)**: {lc.get("licit_count", 0):,} ({lc.get("licit_count", 0) / max(1, nc.get("total_classes_records", 1)) * 100:.2f}%)
- **Unknown (Unlabelled)**: {lc.get("unknown_count", 0):,} ({lc.get("unknown_count", 0) / max(1, nc.get("total_classes_records", 1)) * 100:.2f}%)
- **Missing / Null Labels**: {lc.get("missing_or_null_labels", 0)}
- **Unexpected Classes**: `{lc.get("unexpected_class_labels", [])}`

## 3. Node Feature Integrity
- **Numeric Feature Columns**: {fc.get("total_feature_columns", 0)} (Expected: {fc.get("expected_numeric_features", 165)})
- **Columns With Nulls**: {fc.get("columns_with_nulls", 0)}
- **Total NaN Values**: {fc.get("total_null_values", 0)}
- **Total Infinite Values**: {fc.get("total_infinite_values", 0)}
- **Features Data Quality**: `{"CLEAN (No NaNs or Infs)" if not fc.get("has_missing_or_inf") else "ISSUES DETECTED"}`

## 4. Graph Topology & Edge Integrity
- **Total Directed Edges**: {ec.get("total_edges", 0):,}
- **Duplicate Edges**: {ec.get("duplicate_edges", 0)}
- **Self-Loops (`txId1 == txId2`)**: {ec.get("self_loops", 0)}
- **Edges with Invalid Source Node**: {ec.get("invalid_source_nodes_count", 0)}
- **Edges with Invalid Destination Node**: {ec.get("invalid_dest_nodes_count", 0)}
- **Same-Timestep Edges**: {ec.get("same_timestep_edges", 0):,} ({ec.get("same_timestep_edges", 0) / max(1, ec.get("total_edges", 1)) * 100:.2f}%)
- **Cross-Timestep Edges**: {ec.get("cross_timestep_edges", 0)}
- **Temporal Subgraph Characteristic**: `{"ISOLATED TIMESTEP SNAPSHOTS (0 cross-timestep edges)" if ec.get("edges_are_isolated_snapshots") else "CROSS-TEMPORAL EDGES PRESENT"}`

## 5. Temporal Distributions
- **Min Timestep**: {tc.get("min_timestep", 0)}
- **Max Timestep**: {tc.get("max_timestep", 0)}
- **Total Timesteps**: {tc.get("unique_timesteps_count", 0)}
- **Expected Range (1..49)**: `{tc.get("expected_range_1_to_49", False)}`

---
*Report generated automatically by `crypto-aml-gnn.src.data_validator`.*
"""
        return report

    def save_reports(self, summary: Dict[str, Any]) -> Tuple[Path, Path]:
        """Save human-readable markdown report and JSON summary to disk."""
        self.config.data_summary_dir.mkdir(parents=True, exist_ok=True)
        
        md_path = self.config.data_summary_dir / "validation_report.md"
        json_path = self.config.data_summary_dir / "validation_summary.json"

        # Write markdown report
        md_content = self.generate_markdown_report(summary)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Write machine-readable JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[DataValidator] Validation report saved: {md_path}")
        print(f"[DataValidator] Validation summary saved: {json_path}")

        return md_path, json_path
