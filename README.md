# AI-Powered Cryptocurrency Anti-Money Laundering Using Graph Neural Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.5%2B-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end research engineering pipeline for cryptocurrency Anti-Money Laundering (AML) classification using Graph Neural Networks on the Elliptic Bitcoin dataset.

---

## Project Overview & Research Context

Cryptocurrency transaction networks present unique compliance and regulatory enforcement challenges due to pseudonymity, complex peel chains, and automated transaction obfuscation. Traditional transaction-level machine learning models (e.g., Random Forests, Logistic Regression) evaluate individual transactions in isolation or rely on rudimentary statistical aggregations. 

This project implements the foundational graph engineering and preprocessing architecture to classify illicit Bitcoin transactions according to the state-of-the-art methodology published by Ferretti et al. (University of Bologna, IEEE Access, 2025).

### Primary Research Reference
- **Title**: *Enhancing Anti-Money Laundering Frameworks: An Application of Graph Neural Networks in Cryptocurrency Transaction Classification*
- **Authors**: Stefano Ferretti, Gabriele D'Angelo, Vittorio Ghini (University of Bologna)
- **Publication**: IEEE Access, Vol. 13, 2025, pp. 50204–50218
- **DOI**: [10.1109/ACCESS.2025.3552240](https://doi.org/10.1109/ACCESS.2025.3552240)
- **Comprehensive Analysis**: See [`docs/RESEARCH_REFERENCE.md`](docs/RESEARCH_REFERENCE.md).

### The Best-Performing Model Identified in the Paper
The authors evaluated Graph Convolutional Networks (GCN), Graph Attention Networks (GAT, GATv2), GraphSAGE, and Chebyshev Spectral Graph Convolutions across several feature representations and architectural variations.

The **Chebyshev+Lin** architecture utilizing the reduced aggregated feature set (**`tx+agg_red`**) achieved the highest overall detection capability:
- **Illicit F1-Score**: **0.906** (±0.007) — Highest across all tested architectures
- **Illicit Precision**: **0.955** (±0.008)
- **Illicit Recall**: **0.863** (±0.012)
- **Macro-Average F1**: **0.983** (±0.002)
- **Comparison to Prior Baselines**: Outperforms the previous standard Random Forest baseline (F1 = 0.782, Recall = 0.651) and standard spatial GCNs (F1 ≤ 0.667).

---

## Four-Phase Project Division

1. **Phase 1 (This Repository)**: Dataset Engineering, Preprocessing, and Graph Construction.
2. **Phase 2**: Random Forest Baseline and Tabular Evaluation.
3. **Phase 3**: Graph Neural Networks (Chebyshev, GATv2, GraphSAGE) and Comparative Evaluation.
4. **Phase 4**: Streamlit Dashboard and Interactive Forensic System Integration.

---

## Phase 1 Deliverables & Architecture

Phase 1 establishes a modular, reproducible, leakage-free data engineering pipeline:

```
crypto-aml-gnn/
│
├── data/
│   ├── raw/                 # Raw Elliptic CSV files (ignored from git)
│   └── processed/           # Processed tensors, edge_index, masks, and mappings
│
├── docs/
│   └── RESEARCH_REFERENCE.md # In-depth paper review, benchmark metrics & methodology
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Centralized settings, constants, and path resolvers
│   ├── data_loader.py       # Robust CSV ingestion with header normalization
│   ├── data_validator.py    # Multi-point structural & topological validation suite
│   ├── preprocessing.py     # Alignment, label encoding, and leakage-free feature pruning
│   ├── graph_builder.py     # Zero-indexed edge_index & PyTorch Geometric Data builder
│   └── data_splitter.py     # Temporal split partitioning & topological leakage audit
│
├── scripts/
│   ├── __init__.py
│   └── run_phase1.py        # Automated one-command pipeline execution entrypoint
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_data_validator.py
│   ├── test_preprocessing.py
│   ├── test_graph_builder.py
│   └── test_data_splitter.py
│
├── results/
│   └── data_summary/        # Validation reports, split breakdowns, and manifests
│
├── requirements.txt         # Pinned python dependencies
├── .gitignore               # Strict exclusion of raw data, caches, and large tensors
├── README.md                # Project documentation and reproduction guide
└── PHASE1_HANDOVER.md       # Teammate integration guide for Phases 2 & 3
```

---

## Verified Dataset Statistics (Measured from Actual Elliptic Data)

- **Total Transaction Nodes**: `203,769`
- **Total Directed Edges**: `234,355`
- **Raw Feature Columns**: `167` (1 txId + 1 timestep + 93 local features + 72 aggregated features)
- **Target Class Distribution**:
  - `Illicit` (Raw `1` $\rightarrow$ Encoded `1`): **4,545** (2.23% of total; 9.76% of labeled transactions)
  - `Licit` (Raw `2` $\rightarrow$ Encoded `0`): **42,019** (20.62% of total; 90.24% of labeled transactions)
  - `Unknown` (Raw `unknown` $\rightarrow$ Encoded `-1`): **157,205** (77.15% of total)
- **Graph Snapshot Nature**: All 234,355 edges connect transactions within the **exact same timestep** (0 cross-timestep edges).

---

## Setup & Execution Guide (Windows PowerShell)

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Git installed and configured on PATH

### 2. Environment Setup
Open Windows PowerShell:
```powershell
# Clone repository
git clone https://github.com/Bhanu-prakash02/crypto-aml-gnn-phase1.git
cd crypto-aml-gnn-phase1

# Create virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Placing the Raw Dataset
Due to redistribution licensing, raw dataset files are not tracked in Git.
Download the **Elliptic Data Set** from Kaggle:
👉 [https://www.kaggle.com/datasets/ellipticco/elliptic-data-set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

Place the three uncompressed CSV files into `data/raw/`:
```
crypto-aml-gnn/
└── data/
    └── raw/
        ├── elliptic_txs_classes.csv
        ├── elliptic_txs_edgelist.csv
        └── elliptic_txs_features.csv
```

### 4. Running the Complete Phase 1 Pipeline
From the project root:
```powershell
python -m scripts.run_phase1
```
The command automatically:
1. Validates all input files, ID alignments, and feature column types.
2. Formats labels (`1` = illicit, `0` = licit, `-1` = unknown).
3. Fits the Pearson correlation filter ($r > 0.90$) strictly on training timesteps (1..34) to prune 20 collinear aggregated features without data leakage.
4. Builds zero-indexed `edge_index` and PyG `Data` structures.
5. Evaluates temporal boundary leakage.
6. Serializes outputs to `data/processed/` and logs markdown reports in `results/data_summary/`.

### 5. Running the Automated Test Suite
Run the 24 unit tests covering synthetic fixtures:
```powershell
python -m unittest discover tests -v
```

---

## Processed Output Specifications (`data/processed/`)

| Filename | Format | Shape / Type | Description |
| :--- | :--- | :--- | :--- |
| `elliptic_x_all.npy` | NumPy float32 | `[203769, 165]` | Full numeric feature matrix (93 local + 72 aggregated). |
| `elliptic_x_local.npy` | NumPy float32 | `[203769, 93]` | Local transaction-only features (`tx` configuration). |
| `elliptic_x_reduced.npy` | NumPy float32 | `[203769, 145]` | Pruned feature set (`tx+agg_red`) with collinear features removed. |
| `elliptic_y.npy` | NumPy int64 | `[203769]` | Target labels (`1` = illicit, `0` = licit, `-1` = unknown). |
| `elliptic_timesteps.npy` | NumPy int32 | `[203769]` | Discrete timesteps `1..49` for temporal splitting. |
| `elliptic_edge_index.npy` | NumPy int64 | `[2, 234355]` | Directed edge indices mapped to node indices `0..203768`. |
| `elliptic_masks.npz` | NumPy archive | Compressed booleans | `train_mask`, `val_mask`, `test_mask`, and labeled counterparts. |
| `elliptic_tx_to_idx.json` | JSON dict | `203,769` keys | Mapping: Original Bitcoin `txId` $\rightarrow$ continuous node index. |
| `elliptic_idx_to_tx.json` | JSON list | `203,769` elements | Mapping: Continuous node index $\rightarrow$ original Bitcoin `txId`. |
| `elliptic_pyg_data.pt` | PyTorch Geometric | `Data` object | Ready-to-train PyG container (`x`, `edge_index`, `y`, masks). |
| `elliptic_tensors.pt` | PyTorch dict | Dictionary of Tensors | All tensors bundled for pure PyTorch workflows without PyG. |

---

## Methodological Decisions & Leakage Prevention

1. **Unknown Label Isolation**: Unknown transactions ($77.15\%$) are explicitly encoded as `-1`. They are never converted to licit. In downstream loss functions and evaluation, calculate metrics using `train_mask_labeled` / `test_mask_labeled` to evaluate only labeled transactions while allowing the GNN to message-pass across unknown nodes.
2. **Temporal Splitting Protocol**:
   - **Train**: Timesteps `1..34` (136,265 nodes, 29,894 labeled: 3,462 illicit, 26,432 licit).
   - **Validation**: Timesteps `35..41` (30,680 nodes, 7,829 labeled: 675 illicit, 7,154 licit).
   - **Test**: Timesteps `42..49` (36,824 nodes, 8,841 labeled: 408 illicit, 8,433 licit).
3. **Leakage Prevention During Preprocessing**:
   - Pearson correlation thresholding ($r > 0.90$) for `tx+agg_red` is computed **exclusively on the training split** (timesteps 1..34).
   - Validation and test splits never influence feature selection or scaling.
4. **Topological Leakage Audit**:
   - Our topological analysis verified that **0 edges** cross timestep boundaries. The Elliptic dataset is composed of 49 independent temporal subgraphs.

---

## Troubleshooting

- **Missing Files**: Ensure the uncompressed files in `data/raw/` match the exact filenames: `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`, and `elliptic_txs_features.csv`.
- **Memory Consumption**: Ingesting the 689 MB features CSV requires ~1.5 GB of RAM during loading. The generated NumPy `.npy` files load in under 200 ms.
- **PowerShell Execution Policy**: If `Activate.ps1` gives a script execution policy error, run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
