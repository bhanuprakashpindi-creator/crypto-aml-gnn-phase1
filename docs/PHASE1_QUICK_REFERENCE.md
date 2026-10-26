# Phase 1 Quick Reference Cheatsheet

**Project:** AI-Powered Cryptocurrency Anti-Money Laundering Using Graph Neural Networks  
**Phase:** Phase 1 — Dataset Engineering, Preprocessing, and Graph Construction  
**GitHub Repository:** [https://github.com/bhanuprakashpindi-creator/crypto-aml-gnn-phase1](https://github.com/bhanuprakashpindi-creator/crypto-aml-gnn-phase1)  
**Last Verified:** October 26, 2026 | **Commit:** `832ae94` | **Test Pass Rate:** 24/24 (100%)  

---

## 1. Essential Commands (Windows PowerShell)

```powershell
# Run the complete Phase 1 pipeline (full dataset: ~18 seconds)
python -m scripts.run_phase1

# Run the automated unit test suite (24 tests: ~0.44 seconds)
python -m unittest discover tests -v
```

---

## 2. Core Dataset Statistics (Verified on Actual Elliptic Files)

| Metric | Measured Value | Percentage | Notes |
| :--- | :---: | :---: | :--- |
| **Total Transaction Nodes** | `203,769` | 100.00% | Distinct Bitcoin transaction hashes |
| **Total Directed Edges** | `234,355` | — | Directed payment flows ($tx_1 \rightarrow tx_2$) |
| **Illicit Transactions (Class 1)** | `4,545` | 2.23% | 9.76% of labeled transactions |
| **Licit Transactions (Class 2)** | `42,019` | 20.62% | 90.24% of labeled transactions |
| **Unknown Transactions** | `157,205` | 77.15% | Unlabeled Bitcoin transactions |
| **Total Labeled Subgraph** | `46,564` | 22.85% | Used for supervised loss & evaluation |
| **Cross-Timestep Edges** | `0` | 0.00% | 100% of edges are within identical timesteps |
| **Self-Loops / Duplicate Edges** | `0` | 0.00% | Zero topological anomalies |

---

## 3. Label Encoding Standard

| Raw Label in CSV | Encoded Value in Pipeline | Meaning | Downstream Treatment |
| :---: | :---: | :--- | :--- |
| `'1'` | **`1`** | **Illicit** (Malicious) | Supervised target (Minority positive class) |
| `'2'` | **`0`** | **Licit** (Legitimate) | Supervised target (Majority negative class) |
| `'unknown'` | **`-1`** | **Unknown** (Unlabeled) | **Masked out** of loss function; used in GNN message passing |

> ⚠️ **NEVER** convert Unknown (`-1`) to Licit (`0`).

---

## 4. Feature Matrices & Graph Dimensions

| Array Name | Shape | Dimensionality Breakdown | Description |
| :--- | :---: | :--- | :--- |
| `elliptic_x_all.npy` | `[203769, 165]` | 93 local + 72 aggregated | Complete raw numeric features |
| `elliptic_x_local.npy`| `[203769, 93]` | 93 local only | Non-graph transaction baseline |
| `elliptic_x_reduced.npy`| `[203769, 145]` | 93 local + 52 aggregated | **Paper best configuration** (20 collinear pruned) |
| `elliptic_edge_index.npy`| `[2, 234355]` | Row 0 = src, Row 1 = dst | Directed zero-indexed edge list |
| `elliptic_y.npy` | `[203769]` | Integer values `{1, 0, -1}` | Ground truth labels |
| `elliptic_timesteps.npy`| `[203769]` | Integer values `1..49` | Two-week interval timestamps |

---

## 5. Temporal Split Boundaries

Following Weber et al. (2019) benchmark protocol:

| Split Partition | Timesteps | Total Nodes | Labeled Nodes | Illicit Nodes | Licit Nodes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | **1 – 34** | 136,265 | 29,894 | 3,462 | 26,432 |
| **Validation** | **35 – 41** | 30,680 | 7,829 | 675 | 7,154 |
| **Test** | **42 – 49** | 36,824 | 8,841 | 408 | 8,433 |

---

## 6. Research Benchmark Targets vs. Phase 1 Status

| Model (Ferretti et al., 2025) | Feature Set | Paper Illicit F1 | Paper Recall | Paper Precision | Our Project Phase 1 Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Chebyshev+Lin** *(Best Overall)* | `tx+agg_red` | **0.906** (±0.007) | **0.863** (±0.012) | **0.955** (±0.008) | **Not yet trained** *(Reserved for Phase 3)* |
| GATv2+Lin | `tx+agg_red` | 0.887 | 0.842 | 0.938 | **Not yet trained** *(Reserved for Phase 3)* |
| GraphSAGE (Base) | `tx+agg` | 0.835 | 0.818 | 0.852 | **Not yet trained** *(Reserved for Phase 3)* |
| Random Forest Baseline | All | 0.782 | 0.651 | 0.960 | **Not yet trained** *(Reserved for Phase 2)* |
| GCN+Lin | `tx+agg_red` | 0.667 | 0.621 | 0.720 | **Not yet trained** *(Reserved for Phase 3)* |

> 🚨 **VIVA WARNING:** Do **not** claim Phase 1 achieved an F1 of 0.906. That is the **research paper's target score**. Phase 1 created the graph that enables Phase 3 to train and reach that target score.

---

## 7. Key Processed Files Under `data/processed/`

- `elliptic_pyg_data.pt`: PyTorch Geometric container ready for `DataLoader` and GNN models.
- `elliptic_tensors.pt`: Standalone PyTorch tensor dictionary for standard PyTorch workflows.
- `elliptic_masks.npz`: Boolean masks (`train_mask`, `val_mask`, `test_mask`, `train_mask_labeled`, etc.).
- `elliptic_tx_to_idx.json`: Hash map mapping original Bitcoin `txId` $\rightarrow$ continuous integer node index.
