"""
Configuration module for the Crypto AML Graph Neural Network pipeline.
Centralizes path configurations, dataset constants, split definitions,
and feature definitions to ensure strict reproducibility.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict, Any, Optional


@dataclass
class PipelineConfig:
    """Project-wide configuration settings for Phase 1."""

    # Base Paths (default relative to repository root)
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    
    # Subdirectories
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    data_summary_dir: Path = field(init=False)
    docs_dir: Path = field(init=False)

    # Raw File Names
    classes_filename: str = "elliptic_txs_classes.csv"
    edgelist_filename: str = "elliptic_txs_edgelist.csv"
    features_filename: str = "elliptic_txs_features.csv"

    # Dataset Specifications
    total_raw_feature_cols: int = 167
    num_local_features: int = 93
    num_aggregated_features: int = 72
    total_numeric_features: int = 165  # 93 + 72

    # Label Encodings:
    # Raw source mapping: '1' -> Illicit, '2' -> Licit, 'unknown' -> Unknown
    # Internal integer mapping: Illicit = 1, Licit = 0, Unknown = -1
    label_raw_to_int: Dict[str, int] = field(default_factory=lambda: {
        "1": 1,
        "2": 0,
        "unknown": -1
    })
    label_int_to_str: Dict[int, str] = field(default_factory=lambda: {
        1: "illicit",
        0: "licit",
        -1: "unknown"
    })

    # Temporal Split Boundaries (Timesteps)
    train_timesteps: Tuple[int, int] = (1, 34)
    val_timesteps: Tuple[int, int] = (35, 41)
    test_timesteps: Tuple[int, int] = (42, 49)

    # Feature Reduction Configuration (Paper Section IV & V)
    # Exclude aggregated features with pairwise correlation > threshold
    correlation_threshold: float = 0.90
    fit_correlation_on_train_only: bool = True  # Strict leakage prevention

    # Processed Output Filenames
    processed_prefix: str = "elliptic"
    save_torch_geometric: bool = True
    save_numpy: bool = True
    save_csv_metadata: bool = True

    def __post_init__(self):
        """Construct absolute resolved paths."""
        self.project_root = Path(self.project_root).resolve()
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.results_dir = self.project_root / "results"
        self.data_summary_dir = self.results_dir / "data_summary"
        self.docs_dir = self.project_root / "docs"

    @property
    def raw_classes_path(self) -> Path:
        return self.raw_data_dir / self.classes_filename

    @property
    def raw_edgelist_path(self) -> Path:
        return self.raw_data_dir / self.edgelist_filename

    @property
    def raw_features_path(self) -> Path:
        return self.raw_data_dir / self.features_filename

    def create_directories(self) -> None:
        """Create all required project directories if they do not exist."""
        for directory in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.results_dir,
            self.data_summary_dir,
            self.docs_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary representation for logging."""
        return {
            "project_root": str(self.project_root),
            "raw_data_dir": str(self.raw_data_dir),
            "processed_data_dir": str(self.processed_data_dir),
            "results_dir": str(self.results_dir),
            "train_timesteps": list(self.train_timesteps),
            "val_timesteps": list(self.val_timesteps),
            "test_timesteps": list(self.test_timesteps),
            "correlation_threshold": self.correlation_threshold,
            "fit_correlation_on_train_only": self.fit_correlation_on_train_only,
            "num_local_features": self.num_local_features,
            "num_aggregated_features": self.num_aggregated_features,
            "total_numeric_features": self.total_numeric_features
        }


def get_default_config(project_root: Optional[Path] = None) -> PipelineConfig:
    """Helper factory for retrieving default configuration instance."""
    if project_root is not None:
        return PipelineConfig(project_root=Path(project_root))
    return PipelineConfig()
