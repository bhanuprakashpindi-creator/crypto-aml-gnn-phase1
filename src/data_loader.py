"""
Data loader module for the Elliptic Bitcoin dataset.
Handles file ingestion, header normalization, and initial type enforcement.
"""

import os
from pathlib import Path
from typing import Dict, Union, Optional
import pandas as pd

from src.config import PipelineConfig, get_default_config


def load_classes(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load transaction class labels from CSV.
    
    Robustly handles potential unusual header formats (whitespace, missing headers,
    case variations) and standardizes columns to ['txId', 'class'].
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Classes file not found at: {file_path}")

    # Inspect first line to determine header presence
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    first_tokens = [tok.strip().lower() for tok in first_line.split(",")]
    
    # Check if first line contains header keywords
    has_header = any("tx" in tok or "class" in tok for tok in first_tokens)

    if has_header:
        df = pd.read_csv(file_path, dtype=str)
        # Normalize column names: strip whitespace and lowercase
        col_map = {}
        for col in df.columns:
            clean_col = col.strip()
            if "tx" in clean_col.lower() or "id" in clean_col.lower():
                col_map[col] = "txId"
            elif "class" in clean_col.lower() or "label" in clean_col.lower():
                col_map[col] = "class"
        df = df.rename(columns=col_map)
    else:
        df = pd.read_csv(file_path, header=None, dtype=str)
        df.columns = ["txId", "class"] + [f"extra_{i}" for i in range(len(df.columns) - 2)]
        df = df[["txId", "class"]]

    # Ensure required columns exist
    if "txId" not in df.columns or "class" not in df.columns:
        raise ValueError(
            f"Failed to parse required columns ['txId', 'class'] from {file_path}. "
            f"Found columns: {list(df.columns)}"
        )

    # Clean whitespace in string values
    df["txId"] = df["txId"].astype(str).str.strip()
    df["class"] = df["class"].astype(str).str.strip()

    return df[["txId", "class"]]


def load_edgelist(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load directed transaction edge list from CSV.
    
    Standardizes source and destination transaction ID columns to ['txId1', 'txId2'].
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Edgelist file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    first_tokens = [tok.strip().lower() for tok in first_line.split(",")]
    has_header = any("tx" in tok for tok in first_tokens)

    if has_header:
        df = pd.read_csv(file_path, dtype=str)
        col_map = {}
        cols = list(df.columns)
        if len(cols) >= 2:
            col_map[cols[0]] = "txId1"
            col_map[cols[1]] = "txId2"
        df = df.rename(columns=col_map)
    else:
        df = pd.read_csv(file_path, header=None, dtype=str)
        df.columns = ["txId1", "txId2"] + [f"extra_{i}" for i in range(len(df.columns) - 2)]

    df["txId1"] = df["txId1"].astype(str).str.strip()
    df["txId2"] = df["txId2"].astype(str).str.strip()

    return df[["txId1", "txId2"]]


def load_features(file_path: Union[str, Path], total_cols: int = 167) -> pd.DataFrame:
    """
    Load raw transaction features from CSV.
    
    The raw Elliptic features file has no header.
    Column 0: txId (string/id)
    Column 1: time_step (integer 1..49)
    Columns 2..166: 165 numeric float features (93 local, 72 aggregated).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Features file not found at: {file_path}")

    # Inspect first line to check if a header was mistakenly prepended
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    
    first_tok = first_line.split(",")[0].strip()
    is_header = not first_tok.isdigit()

    if is_header:
        df = pd.read_csv(file_path)
        # Rename first two columns to standardized names
        cols = list(df.columns)
        col_map = {cols[0]: "txId", cols[1]: "time_step"}
        for i, c in enumerate(cols[2:], start=1):
            col_map[c] = f"feat_{i}"
        df = df.rename(columns=col_map)
    else:
        # Generate column names: txId, time_step, feat_1 ... feat_165
        col_names = ["txId", "time_step"] + [f"feat_{i}" for i in range(1, total_cols - 1)]
        df = pd.read_csv(file_path, header=None, names=col_names, low_memory=False)

    # Standardize types
    df["txId"] = df["txId"].astype(str).str.strip()
    df["time_step"] = pd.to_numeric(df["time_step"], errors="coerce").astype("int32")

    # Enforce float32 for all feature columns
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    return df


def load_raw_dataset(config: Optional[PipelineConfig] = None) -> Dict[str, pd.DataFrame]:
    """
    Load all three raw Elliptic dataset files according to pipeline configuration.
    
    Returns:
        dict with keys: 'classes', 'edgelist', 'features'
    """
    if config is None:
        config = get_default_config()

    print(f"[DataLoader] Ingesting classes from: {config.raw_classes_path}")
    df_classes = load_classes(config.raw_classes_path)
    print(f"[DataLoader] Loaded {len(df_classes):,} class records.")

    print(f"[DataLoader] Ingesting edgelist from: {config.raw_edgelist_path}")
    df_edgelist = load_edgelist(config.raw_edgelist_path)
    print(f"[DataLoader] Loaded {len(df_edgelist):,} edge records.")

    print(f"[DataLoader] Ingesting features from: {config.raw_features_path}")
    df_features = load_features(config.raw_features_path, total_cols=config.total_raw_feature_cols)
    print(f"[DataLoader] Loaded {len(df_features):,} feature records with {df_features.shape[1]} columns.")

    return {
        "classes": df_classes,
        "edgelist": df_edgelist,
        "features": df_features
    }
