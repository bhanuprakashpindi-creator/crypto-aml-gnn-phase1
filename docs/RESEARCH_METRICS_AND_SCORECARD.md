# Research Metrics and Project Scorecard

**Project Title:** AI-Powered Cryptocurrency Anti-Money Laundering Using Graph Neural Networks  
**Repository:** [https://github.com/bhanuprakashpindi-creator/crypto-aml-gnn-phase1](https://github.com/bhanuprakashpindi-creator/crypto-aml-gnn-phase1)  
**Phase:** Phase 1 — Dataset Engineering, Preprocessing, and Graph Construction  
**Document Purpose:** Clear, unambiguous distinction between published research benchmarks, Phase 1 engineering results, and future model evaluation scorecards.

---

## Document Taxonomy: The Three Metrics Categories

To prevent methodological confusion and ensure academic integrity, this scorecard separates project measurements into three strictly isolated tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PROJECT METRICS TIERS                           │
├────────────────────────────┬─────────────────────────────┬─────────────┤
│   CATEGORY A               │   CATEGORY B                │ CATEGORY C  │
│   Published Research Paper │   Phase 1 Engineering &     │ Our Future  │
│   Benchmark Results        │   Data Quality Results      │ Model Scores│
│   (Ferretti et al., 2025)  │   (Actually Measured Today) │ (Phase 2/3) │
└────────────────────────────┴─────────────────────────────┴─────────────┘
```

---

## Category A: Research Paper Benchmark Results (Ferretti et al., 2025)

The metrics below represent the official published empirical findings from the research paper:  
*“Enhancing Anti-Money Laundering Frameworks: An Application of Graph Neural Networks in Cryptocurrency Transaction Classification”* (IEEE Access, Vol. 13, 2025).

### A.1 Published GNN Performance Benchmark (Paper Table 4 & Section V)

All values represent mean performance across 15 experimental repetitions (with standard deviation in parentheses):

| Model Configuration | Feature Set | Illicit Precision | Illicit Recall | Illicit F1-Score | Macro-Average F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Chebyshev+Lin** *(Best Overall)* | `tx+agg_red` | **0.955** (±0.008) | **0.863** (±0.012) | **0.906** (±0.007) | **0.983** (±0.002) |
| Chebyshev+Lin | `tx+agg` | 0.942 | 0.851 | 0.894 | 0.978 |
| GATv2+Lin | `tx+agg_red` | 0.938 | 0.842 | 0.887 | 0.976 |
| GATv2+Lin+Skip | `tx+agg_red` | 0.941 | 0.838 | 0.886 | 0.975 |
| GraphSAGE (Base) | `tx+agg` | 0.852 | **0.818** | 0.835 | 0.958 |
| Random Forest Baseline ([Weber et al., 2019]) | All features | 0.960 | 0.651 | 0.782 | — |
| GCN+Lin | `tx+agg_red` | 0.720 | 0.621 | 0.667 | 0.912 |

### A.2 The Meaning of Evaluation Metrics in Cryptocurrency AML
In financial compliance and forensic investigations, metrics carry asymmetric consequences:

1. **Illicit Precision**:
   $$\text{Precision} = \frac{TP}{TP + FP}$$
   Measures the fraction of transactions flagged as illicit that are truly illicit. High precision prevents compliance analyst fatigue and avoids freezing legitimate user accounts.
2. **Illicit Recall (Sensitivity)**:
   $$\text{Recall} = \frac{TP}{TP + FN}$$
   Measures the fraction of all actual illicit transactions correctly detected by the model. **In AML, recall is of paramount importance.** A false negative represents an unflagged money laundering transaction that successfully escapes the banking or exchange perimeter.
3. **Illicit F1-Score**:
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
   Harmonic mean providing a balanced measurement under extreme class imbalance.
4. **Macro-Average F1**:
   Unweighted mean of F1-scores across both licit and illicit classes. Reflects balanced accuracy across both minority and majority groups.

### A.3 The Precision-Recall Trade-Off in AML
The published Random Forest baseline achieved high precision (0.960) but poor recall (0.651), missing approximately **35% of all illicit transactions**. The Chebyshev+Lin model boosts recall to **0.863** while maintaining **0.955** precision, reducing missed laundering volume by over 60%.

---

## Category B: Phase 1 Engineering & Data Quality Results

These measurements reflect the verified data engineering, graph construction, and structural integrity audits produced by our Phase 1 codebase on the actual Elliptic dataset.

> **NOTE:** These are software engineering and data quality metrics, **not** machine learning classification scores.

### B.1 Verified Dataset & Topological Statistics
* **Total Transaction Nodes**: `203,769`
* **Total Directed Edges**: `234,355`
* **Raw Ingested Columns**: `167`
* **Local Transaction Features**: `93`
* **Aggregated Neighbor Features**: `72`
* **Collinear Aggregated Features Pruned (Train split, $r > 0.90$)**: `20`
* **Retained Reduced Features (`tx+agg_red`)**: `145` (93 local + 52 aggregated)
* **Discrete Timesteps**: `49` (two-week intervals)
* **Ground Truth Classes**:
  * Illicit (1): `4,545` (2.23% of total; 9.76% of labeled transactions)
  * Licit (0): `42,019` (20.62% of total; 90.24% of labeled transactions)
  * Unknown (-1): `157,205` (77.15% of total)

### B.2 Temporal Split Partitioning
* **Train Split (Timesteps 1–34)**: 136,265 nodes (29,894 labeled: 3,462 illicit, 26,432 licit)
* **Validation Split (Timesteps 35–41)**: 30,680 nodes (7,829 labeled: 675 illicit, 7,154 licit)
* **Test Split (Timesteps 42–49)**: 36,824 nodes (8,841 labeled: 408 illicit, 8,433 licit)

### B.3 Validation & Execution Integrity
* **Automated Unit Tests**: `24 / 24 PASSED` (100% passing in 0.443s)
* **Structural Anomalies Detected**: `0` duplicate IDs, `0` missing IDs, `0` invalid endpoints, `0` self-loops, `0` NaNs/Infs.
* **Topological Leakage**: `0` cross-split edges (100% temporally isolated snapshots).
* **Pipeline Execution Runtime**: `18.22 seconds` on local host (Intel/AMD x86_64 CPU).

---

## Category C: Our Model Results (Scorecard Template for Phases 2 & 3)

The table below is the **active project scorecard**. In Phase 1, no machine learning models have been trained or evaluated; therefore, all cells are marked as `Not yet evaluated`.

> **STRICT PROTOCOL FOR FUTURE PHASES:**
> 1. All models must be evaluated strictly on the **Test Split (Timesteps 42–49)**.
> 2. Evaluation must be computed **exclusively on labeled nodes** (`y != -1`) using `test_mask_labeled`.
> 3. Training scores must **never** be substituted for test scores.
> 4. Research paper scores must **never** be copied into this table as our project's results.

| Model | Feature Set | Precision (Illicit) | Recall (Illicit) | F1-Score (Illicit) | Macro-Average F1 | Overall Accuracy | Evaluation Status | Assigned Phase |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | `x_all` (165 feats) | — | — | — | — | — | *Not yet evaluated* | Phase 2 |
| **Random Forest** | `x_reduced` (145 feats) | — | — | — | — | — | *Not yet evaluated* | Phase 2 |
| **GCN+Lin** | `x_reduced` | — | — | — | — | — | *Not yet evaluated* | Phase 3 |
| **GraphSAGE (Base)** | `x_all` | — | — | — | — | — | *Not yet evaluated* | Phase 3 |
| **GATv2+Lin** | `x_reduced` | — | — | — | — | — | *Not yet evaluated* | Phase 3 |
| **Chebyshev+Lin** *(Research Target)* | `x_reduced` | — | — | — | — | — | *Not yet evaluated* | Phase 3 |

---

## Summary of Next Steps to Populate Category C

1. **Phase 2 Engineer**: Run Random Forest training script on `elliptic_x_all.npy` and `elliptic_x_reduced.npy` using `elliptic_masks.npz`. Record precision, recall, and F1 on Class 1 in Category C.
2. **Phase 3 Engineer**: Load `elliptic_pyg_data.pt`, instantiate GNN architectures in PyTorch Geometric, train on `train_mask`, select checkpoints via `val_mask`, and log final test performance on `test_mask` in Category C.
