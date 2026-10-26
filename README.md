# AI-Powered Cryptocurrency Anti-Money Laundering Using Graph Neural Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyG 2.5+](https://img.shields.io/badge/PyG-2.5%2B-3C2179.svg)](https://pyg.org/)
[![Tests: 24 Passed](https://img.shields.io/badge/tests-24%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end research engineering pipeline for cryptocurrency Anti-Money Laundering (AML) classification using Graph Neural Networks on the Elliptic Bitcoin dataset.

---

## Project Overview

Cryptocurrency transaction networks present unique regulatory and compliance challenges due to pseudonymity and automated obfuscation strategies (such as peel chains and multi-hop mixing). Standard tabular machine learning approaches evaluate transactions in isolation, ignoring the rich relational topology of payment flows.

This project implements the foundational graph engineering and preprocessing architecture to classify illicit Bitcoin transactions according to the state-of-the-art methodology published by Ferretti et al. (*IEEE Access*, 2025).

### Four-Phase Project Structure
1. **Phase 1 (This Repository)**: Dataset Engineering, Validation, Leakage-Free Preprocessing, and Graph Construction.
2. **Phase 2**: Random Forest Baseline and Tabular Evaluation.
3. **Phase 3**: Graph Neural Networks (Chebyshev, GATv2, GraphSAGE) and Comparative Evaluation.
4. **Phase 4**: Streamlit Interactive Forensic Dashboard and System Integration.

---

## Comprehensive Project Documentation Suite

For exhaustive technical and academic details, consult our dedicated documentation modules:

| Document | Purpose / Key Contents |
| :--- | :--- |
| 📖 [**Phase 1 Technical Report**](docs/PHASE1_TECHNICAL_REPORT.md) | **Primary academic & engineering report** (20 sections covering problem statement, methodology, leakage audit, validation, and architecture). |
| 📊 [**Research Metrics & Scorecard**](docs/RESEARCH_METRICS_AND_SCORECARD.md) | **Official scorecard** isolating Paper Benchmarks (Category A), Phase 1 Engineering Results (Category B), and Future Model Scores (Category C). |
| 🎓 [**Viva Defense Notes**](docs/PHASE1_VIVA_NOTES.md) | **Presentation defense guide** with 2-minute pitch, deep-dive technical explanations, and **25 top viva questions and answers**. |
| ⚡ [**Quick Reference Cheatsheet**](docs/PHASE1_QUICK_REFERENCE.md) | **Compact one-page review** covering key statistics, label encodings, split boundaries, and essential commands. |
| 🔬 [**Research Paper Reference**](docs/RESEARCH_REFERENCE.md) | In-depth review of Ferretti et al. (IEEE Access, 2025) and mathematical basis of Chebyshev spectral convolutions. |
| 🤝 [**Phase 1 Handover Guide**](PHASE1_HANDOVER.md) | Concrete Python integration code snippets for Phase 2 (Random Forest) and Phase 3 (PyG GNN) teammates. |

---

## Research Paper Reference & Benchmark Target

- **Paper Title**: *Enhancing Anti-Money Laundering Frameworks: An Application of Graph Neural Networks in Cryptocurrency Transaction Classification*
- **Authors**: Stefano Ferretti, Gabriele D’Angelo, and Vittorio Ghini (University of Bologna, Italy)
- **Publication**: *IEEE Access*, Volume 13, 2025, pp. 50204–50218 (DOI: [10.1109/ACCESS.2025.3552240](https://doi.org/10.1109/ACCESS.2025.3552240))
- **Best-Performing Configuration**: **`Chebyshev+Lin` with reduced aggregated features (`tx+agg_red`)**:
  - **Illicit F1-Score**: **0.906** (±0.007)
  - **Illicit Precision**: **0.955** (±0.008)
  - **Illicit Recall**: **0.863** (±0.012)
  - **Macro-F1**: **0.983** (±0.002)

> ⚠️ **IMPORTANT NOTE:** These scores represent the **published research benchmark** that will be targeted during Phase 3. Phase 1 provides the data and graph structures; it does **not** train classification models.

---

## Verified Dataset Statistics (Measured from Actual Elliptic Data)

- **Total Transaction Nodes**: `203,769`
- **Total Directed Edges**: `234,355`
- **Raw Feature Columns**: `167` (1 txId + 1 timestep + 93 local + 72 aggregated neighbor features)
- **Class Breakdown**:
  - `Illicit` (Class 1): **4,545** (2.23% of total; 9.76% of labeled transactions)
  - `Licit` (Class 2): **42,019** (20.62% of total; 90.24% of labeled transactions)
  - `Unknown`: **157,205** (77.15% of total; encoded as `-1`, masked out of loss)
- **Topological Finding**: All 234,355 directed edges connect transactions within the **exact same timestep** (0 cross-timestep edges).

---

## Setup & Quickstart (Windows PowerShell)

### 1. Environment Setup
```powershell
# Clone repository
git clone https://github.com/bhanuprakashpindi-creator/crypto-aml-gnn-phase1.git
cd crypto-aml-gnn-phase1

# Virtual environment setup
python -m venv venv
.\venv\Scripts\Activate.ps1

# Dependency installation
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Placing the Raw Dataset
Download the uncompressed **Elliptic Data Set** from [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place the three CSV files in `data/raw/`:
```
crypto-aml-gnn/
└── data/
    └── raw/
        ├── elliptic_txs_classes.csv
        ├── elliptic_txs_edgelist.csv
        └── elliptic_txs_features.csv
```

### 3. Running the Pipeline & Test Suite
```powershell
# Execute complete Phase 1 pipeline (full dataset: ~18 seconds)
python -m scripts.run_phase1

# Run automated unit test suite (24 tests: ~0.44 seconds)
python -m unittest discover tests -v
```

---

## Processed Output Specifications (`data/processed/`)

| Filename | Format | Shape / Type | Description |
| :--- | :--- | :--- | :--- |
| `elliptic_x_all.npy` | NumPy float32 | `[203769, 165]` | Full numeric features (93 local + 72 aggregated). |
| `elliptic_x_local.npy` | NumPy float32 | `[203769, 93]` | Local-only transaction features. |
| `elliptic_x_reduced.npy` | NumPy float32 | `[203769, 145]` | **Paper best feature set** (20 collinear pruned on train). |
| `elliptic_y.npy` | NumPy int64 | `[203769]` | Labels: `1` = illicit, `0` = licit, `-1` = unknown. |
| `elliptic_timesteps.npy` | NumPy int32 | `[203769]` | Discrete timestep indicators `1..49`. |
| `elliptic_edge_index.npy` | NumPy int64 | `[2, 234355]` | Directed edge indices mapped to continuous node IDs. |
| `elliptic_masks.npz` | NumPy archive | Compressed bools | `train_mask`, `val_mask`, `test_mask` + labeled subsets. |
| `elliptic_tx_to_idx.json` | JSON dict | 203,769 keys | Maps original Bitcoin `txId` $\rightarrow$ node index. |
| `elliptic_idx_to_tx.json` | JSON list | 203,769 items | Maps node index $\rightarrow$ original Bitcoin `txId`. |
| `elliptic_pyg_data.pt` | PyG Data | Container | Ready-to-train PyTorch Geometric graph object. |
| `elliptic_tensors.pt` | PyTorch Dict | 12 Tensors | Pure PyTorch tensor dictionary bundle. |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
