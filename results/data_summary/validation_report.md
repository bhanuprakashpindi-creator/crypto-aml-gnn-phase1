# Elliptic Dataset Validation Report

**Validation Status**: `PASSED`

---

## 1. Node & Record Alignment
- **Classes File Records**: 203,769 (Unique IDs: 203,769, Duplicates: 0)
- **Features File Records**: 203,769 (Unique IDs: 203,769, Duplicates: 0)
- **Shared IDs**: 203,769
- **Classes-Only IDs**: 0
- **Features-Only IDs**: 0
- **Perfect 1:1 Alignment**: `True`

## 2. Target Class Distribution
- **Illicit (Class 1)**: 4,545 (2.23%)
- **Licit (Class 2)**: 42,019 (20.62%)
- **Unknown (Unlabelled)**: 157,205 (77.15%)
- **Missing / Null Labels**: 0
- **Unexpected Classes**: `[]`

## 3. Node Feature Integrity
- **Numeric Feature Columns**: 165 (Expected: 165)
- **Columns With Nulls**: 0
- **Total NaN Values**: 0
- **Total Infinite Values**: 0
- **Features Data Quality**: `CLEAN (No NaNs or Infs)`

## 4. Graph Topology & Edge Integrity
- **Total Directed Edges**: 234,355
- **Duplicate Edges**: 0
- **Self-Loops (`txId1 == txId2`)**: 0
- **Edges with Invalid Source Node**: 0
- **Edges with Invalid Destination Node**: 0
- **Same-Timestep Edges**: 234,355 (100.00%)
- **Cross-Timestep Edges**: 0
- **Temporal Subgraph Characteristic**: `ISOLATED TIMESTEP SNAPSHOTS (0 cross-timestep edges)`

## 5. Temporal Distributions
- **Min Timestep**: 1
- **Max Timestep**: 49
- **Total Timesteps**: 49
- **Expected Range (1..49)**: `True`

---
*Report generated automatically by `crypto-aml-gnn.src.data_validator`.*
