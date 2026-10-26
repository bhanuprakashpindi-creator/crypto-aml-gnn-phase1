"""
Main execution pipeline for Phase 1:
Dataset Engineering, Preprocessing, and Graph Construction.

Usage:
    python -m scripts.run_phase1
    python -m scripts.run_phase1 --project-root .
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np

# Ensure project root is in sys.path
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.config import PipelineConfig, get_default_config
from src.data_loader import load_raw_dataset
from src.data_validator import DataValidator
from src.preprocessing import DataPreprocessor
from src.graph_builder import GraphBuilder
from src.data_splitter import DataSplitter


def run_pipeline(config: Optional[PipelineConfig] = None) -> bool:
    """Execute complete Phase 1 pipeline."""
    start_time = time.time()
    config = config or get_default_config()
    config.create_directories()

    print("=" * 80)
    print(" CRYPTO AML GRAPH NEURAL NETWORK - PHASE 1 PIPELINE")
    print(" Dataset Engineering, Preprocessing, and Graph Construction")
    print("=" * 80)
    print(f"Project Root: {config.project_root}")
    print(f"Raw Data Dir: {config.raw_data_dir}")
    print(f"Processed Dir: {config.processed_data_dir}")
    print(f"Results Dir:   {config.results_dir}")
    print("-" * 80)

    # -------------------------------------------------------------
    # Step 1: Ingestion
    # -------------------------------------------------------------
    print("\n>>> STEP 1: Verifying and Loading Raw Dataset...")
    validator = DataValidator(config)
    file_status = validator.validate_files_exist()
    if not file_status["all_files_present"]:
        print("\n[ERROR] Raw dataset files are missing!")
        for k, v in file_status.items():
            if k != "all_files_present" and not v:
                print(f"  Missing: {k}")
        print("\nPlease place the Elliptic dataset files into:")
        print(f"  {config.raw_data_dir}")
        print("Required files:")
        print(f"  1. {config.classes_filename}")
        print(f"  2. {config.edgelist_filename}")
        print(f"  3. {config.features_filename}")
        return False

    raw_data = load_raw_dataset(config)
    df_classes = raw_data["classes"]
    df_edgelist = raw_data["edgelist"]
    df_features = raw_data["features"]

    # -------------------------------------------------------------
    # Step 2: Validation
    # -------------------------------------------------------------
    print("\n>>> STEP 2: Running Comprehensive Data Validation...")
    validation_summary = validator.validate_dataset(df_classes, df_edgelist, df_features)
    val_md, val_json = validator.save_reports(validation_summary)
    print(f"[Validation Status]: {validation_summary['overall_status']}")

    # -------------------------------------------------------------
    # Step 3: Preprocessing & Node Alignment
    # -------------------------------------------------------------
    print("\n>>> STEP 3: Preprocessing Features, Labels, and Node Mappings...")
    preprocessor = DataPreprocessor(config)
    preprocessed = preprocessor.run_preprocessing_pipeline(df_classes, df_features)

    x_all = preprocessed["x_all"]
    x_local = preprocessed["x_local"]
    x_reduced = preprocessed["x_reduced"]
    y = preprocessed["y"]
    timesteps = preprocessed["timesteps"]
    tx_to_idx = preprocessed["tx_to_idx"]
    idx_to_tx = preprocessed["idx_to_tx"]

    # -------------------------------------------------------------
    # Step 4: Graph Construction
    # -------------------------------------------------------------
    print("\n>>> STEP 4: Building Transaction Graph and Edge Index...")
    graph_builder = GraphBuilder(config)
    edge_index, edge_stats = graph_builder.build_edge_index(
        df_edgelist=df_edgelist,
        tx_to_idx=tx_to_idx,
        remove_self_loops=False,
        remove_duplicates=True
    )

    # -------------------------------------------------------------
    # Step 5: Data Splitting & Leakage Analysis
    # -------------------------------------------------------------
    print("\n>>> STEP 5: Generating Temporal Splits and Leakage Audit...")
    splitter = DataSplitter(config)
    masks, split_stats = splitter.generate_temporal_masks(timesteps, y)
    leakage_assessment = splitter.analyze_temporal_leakage(edge_index, timesteps)

    # -------------------------------------------------------------
    # Step 6: Persisting Processed Outputs
    # -------------------------------------------------------------
    print("\n>>> STEP 6: Saving Processed Outputs for Downstream Teams...")
    out_dir = config.processed_data_dir

    # 1. Feature matrices (NumPy)
    np.save(out_dir / "elliptic_x_all.npy", x_all)
    np.save(out_dir / "elliptic_x_local.npy", x_local)
    np.save(out_dir / "elliptic_x_reduced.npy", x_reduced)
    print(f"  Saved node features (x_all {x_all.shape}, x_local {x_local.shape}, x_reduced {x_reduced.shape})")

    # 2. Labels and Timesteps
    np.save(out_dir / "elliptic_y.npy", y)
    np.save(out_dir / "elliptic_timesteps.npy", timesteps)
    print(f"  Saved labels (y {y.shape}) and timesteps (timesteps {timesteps.shape})")

    # 3. Edge Index
    np.save(out_dir / "elliptic_edge_index.npy", edge_index)
    print(f"  Saved edge_index (shape {edge_index.shape})")

    # 4. Split Masks
    np.savez_compressed(
        out_dir / "elliptic_masks.npz",
        train_mask=masks["train_mask"],
        val_mask=masks["val_mask"],
        test_mask=masks["test_mask"],
        train_mask_labeled=masks["train_mask_labeled"],
        val_mask_labeled=masks["val_mask_labeled"],
        test_mask_labeled=masks["test_mask_labeled"]
    )
    print("  Saved train/val/test masks (elliptic_masks.npz)")

    # 5. Node ID Mappings
    with open(out_dir / "elliptic_tx_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(tx_to_idx, f)
    with open(out_dir / "elliptic_idx_to_tx.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_tx, f)
    print(f"  Saved node mapping dictionaries ({len(tx_to_idx):,} mappings)")

    # 6. PyG Data Object & PyTorch Tensors
    if HAS_TORCH:
        pyg_data = graph_builder.create_pyg_data(
            edge_index=edge_index,
            x=x_reduced,  # default with best paper feature set
            y=y,
            timesteps=timesteps,
            train_mask=masks["train_mask"],
            val_mask=masks["val_mask"],
            test_mask=masks["test_mask"]
        )
        torch.save(pyg_data, out_dir / "elliptic_pyg_data.pt")
        
        # Save PyTorch tensor bundle
        torch.save({
            "x_all": torch.from_numpy(np.array(x_all, copy=True)).float(),
            "x_local": torch.from_numpy(np.array(x_local, copy=True)).float(),
            "x_reduced": torch.from_numpy(np.array(x_reduced, copy=True)).float(),
            "y": torch.from_numpy(np.array(y, copy=True)).long(),
            "timesteps": torch.from_numpy(np.array(timesteps, copy=True)).int(),
            "edge_index": torch.from_numpy(np.array(edge_index, copy=True)).long(),
            "train_mask": torch.from_numpy(np.array(masks["train_mask"], copy=True)).bool(),
            "val_mask": torch.from_numpy(np.array(masks["val_mask"], copy=True)).bool(),
            "test_mask": torch.from_numpy(np.array(masks["test_mask"], copy=True)).bool(),
            "train_mask_labeled": torch.from_numpy(np.array(masks["train_mask_labeled"], copy=True)).bool(),
            "val_mask_labeled": torch.from_numpy(np.array(masks["val_mask_labeled"], copy=True)).bool(),
            "test_mask_labeled": torch.from_numpy(np.array(masks["test_mask_labeled"], copy=True)).bool()
        }, out_dir / "elliptic_tensors.pt")
        print("  Saved PyG Data object (elliptic_pyg_data.pt) and PyTorch tensors (elliptic_tensors.pt)")

    # 7. Metadata Manifest
    manifest = {
        "pipeline_version": "0.1.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": "Elliptic Bitcoin Dataset",
        "num_nodes": int(x_all.shape[0]),
        "num_directed_edges": int(edge_index.shape[1]),
        "feature_dimensions": {
            "x_all": list(x_all.shape),
            "x_local": list(x_local.shape),
            "x_reduced": list(x_reduced.shape)
        },
        "label_counts": {
            "illicit": int((y == 1).sum()),
            "licit": int((y == 0).sum()),
            "unknown": int((y == -1).sum())
        },
        "temporal_splits": split_stats,
        "leakage_assessment": leakage_assessment,
        "edge_statistics": edge_stats,
        "files_generated": [
            "elliptic_x_all.npy",
            "elliptic_x_local.npy",
            "elliptic_x_reduced.npy",
            "elliptic_y.npy",
            "elliptic_timesteps.npy",
            "elliptic_edge_index.npy",
            "elliptic_masks.npz",
            "elliptic_tx_to_idx.json",
            "elliptic_idx_to_tx.json",
            "elliptic_pyg_data.pt" if HAS_TORCH else None,
            "elliptic_tensors.pt" if HAS_TORCH else None
        ]
    }
    manifest["files_generated"] = [f for f in manifest["files_generated"] if f is not None]

    manifest_path = config.data_summary_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Also save markdown summary of splits and manifest
    summary_md_path = config.data_summary_dir / "dataset_statistics.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(f"""# Elliptic Dataset Statistics and Pipeline Manifest

- **Execution Date**: {manifest['timestamp']}
- **Total Transaction Nodes**: {manifest['num_nodes']:,}
- **Total Directed Edges**: {manifest['num_directed_edges']:,}
- **Raw Features**: 165 numeric features (93 local, 72 aggregated)
- **Reduced Features (Paper Best Configuration)**: 143 numeric features (22 collinear aggregated features pruned on train split)

## Class Distribution
- **Illicit (1)**: {manifest['label_counts']['illicit']:,} ({manifest['label_counts']['illicit'] / manifest['num_nodes'] * 100:.2f}%)
- **Licit (0)**: {manifest['label_counts']['licit']:,} ({manifest['label_counts']['licit'] / manifest['num_nodes'] * 100:.2f}%)
- **Unknown (-1)**: {manifest['label_counts']['unknown']:,} ({manifest['label_counts']['unknown'] / manifest['num_nodes'] * 100:.2f}%)
- **Total Labeled Subgraph**: {manifest['label_counts']['illicit'] + manifest['label_counts']['licit']:,} transactions ({manifest['label_counts']['illicit'] / (manifest['label_counts']['illicit'] + manifest['label_counts']['licit']) * 100:.2f}% illicit class ratio)

## Temporal Split Breakdown
- **Train (Timesteps 1..34)**:
  - Total Nodes: {split_stats['train_split']['total_nodes']:,}
  - Labeled Nodes: {split_stats['train_split']['labeled_nodes']:,} ({split_stats['train_split']['illicit_count']:,} illicit, {split_stats['train_split']['licit_count']:,} licit)
  - Illicit Ratio (Labeled): {split_stats['train_split']['illicit_ratio_labeled'] * 100:.2f}%
- **Validation (Timesteps 35..41)**:
  - Total Nodes: {split_stats['val_split']['total_nodes']:,}
  - Labeled Nodes: {split_stats['val_split']['labeled_nodes']:,} ({split_stats['val_split']['illicit_count']:,} illicit, {split_stats['val_split']['licit_count']:,} licit)
  - Illicit Ratio (Labeled): {split_stats['val_split']['illicit_ratio_labeled'] * 100:.2f}%
- **Test (Timesteps 42..49)**:
  - Total Nodes: {split_stats['test_split']['total_nodes']:,}
  - Labeled Nodes: {split_stats['test_split']['labeled_nodes']:,} ({split_stats['test_split']['illicit_count']:,} illicit, {split_stats['test_split']['licit_count']:,} licit)
  - Illicit Ratio (Labeled): {split_stats['test_split']['illicit_ratio_labeled'] * 100:.2f}%

## Leakage Audit
- Cross-split edges: **{leakage_assessment['total_cross_split_edges']}**
- Topological Leakage Status: **{leakage_assessment['findings']}**
""")
    print(f"  Saved manifest ({manifest_path}) and statistics summary ({summary_md_path})")

    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"[SUCCESS] Phase 1 Pipeline completed in {elapsed:.2f} seconds.")
    print(f"Processed datasets saved to: {out_dir}")
    print("=" * 80)
    return True


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 Data Pipeline.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Path to project root directory (defaults to parent of scripts/)."
    )
    args = parser.parse_args()

    config = get_default_config(Path(args.project_root) if args.project_root else None)
    success = run_pipeline(config)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
