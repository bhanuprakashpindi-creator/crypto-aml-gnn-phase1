# Research Reference: Enhancing Anti-Money Laundering Frameworks via Graph Neural Networks

## 1. Paper Overview and Metadata

- **Title**: Enhancing Anti-Money Laundering Frameworks: An Application of Graph Neural Networks in Cryptocurrency Transaction Classification
- **Authors**: Stefano Ferretti, Gabriele D'Angelo, Vittorio Ghini
- **Affiliation**: Department of Computer Science and Engineering (DISI), University of Bologna, 40126 Bologna, Italy
- **Publication**: IEEE Access, Volume 13, 2025, pp. 50204–50218
- **Digital Object Identifier (DOI)**: [10.1109/ACCESS.2025.3552240](https://doi.org/10.1109/ACCESS.2025.3552240)
- **Grant/Funding Support**: European Union – NextGenerationEU (PNRR Mission 4 – Component 2, PRIN 2022, Grant 2022N2NH42)

---

## 2. Dataset Analysis & Topological Findings

The paper conducts an extensive empirical investigation using the **Elliptic Bitcoin Transaction Dataset** ([Weber et al., 2019]):

- **Total Transaction Nodes**: 203,769
- **Total Directed Edges**: 234,355
- **Feature Dimensionality**: 167 raw columns in feature matrix:
  - Column 0: `txId` (unique transaction identifier)
  - Column 1: `time_step` (discrete time step from 1 to 49, representing two-week intervals)
  - Columns 2 to 94: **93 local transaction features** (extracted directly from transaction payloads, e.g., input/output counts, transaction fees, BTC volumes)
  - Columns 95 to 166: **72 aggregated neighbor features** (derived from 1-hop neighbor transaction characteristics such as minimum, maximum, standard deviation, and sum of neighbor values)
- **Class Labels**:
  - `illicit` (encoded as `1` in raw dataset): 4,545 transactions (~2.23% of total; 9.76% of labeled subset)
  - `licit` (encoded as `2` in raw dataset): 42,019 transactions (~20.62% of total; 90.24% of labeled subset)
  - `unknown` (encoded as `unknown` in raw dataset): 157,205 transactions (~77.15% of total)

### Key Topological Observations from the Paper
1. **Degree Distribution**:
   - Licit transactions have a noticeably higher mean degree (3.10) compared to illicit transactions (2.01) and all transactions (2.30).
   - Maximum degree recorded is 473 (a licit transaction), whereas illicit transactions have a restricted range (1–177, median = 1).
   - This indicates illicit actors often execute isolated, pass-through, or peel-chain transactions to evade detection.
2. **Centrality & Clustering**:
   - Licit nodes demonstrate significantly higher degree centrality and closeness centrality than illicit transactions.
   - Illicit nodes exhibit a lower clustering coefficient, confirming that laundering entities minimize inter-neighbor connectivity to prevent cluster-wide blacklisting.
3. **Temporal Subgraph Snapshot Nature**:
   - All 234,355 edges in the dataset connect nodes occurring strictly within the *same* timestep. No cross-timestep edges exist in the raw edgelist. Each timestep represents a distinct, time-stamped Bitcoin graph snapshot.

---

## 3. Evaluated Models and Architectures

The authors benchmarked multiple GNN families across three architectural configurations:

### GNN Convolutions Tested
1. **GCN (Graph Convolutional Network)**: Standard first-order localized spectral approximation ([Kipf & Welling, 2017]).
2. **GAT (Graph Attention Network)**: Multi-head masked self-attention over node neighborhoods ([Veličković et al., 2018]).
3. **GATv2**: Dynamic attention mechanism fixing GAT's static attention limitation ([Brody et al., 2022]).
4. **GraphSAGE (SAGE)**: Inductive neighborhood aggregation with mean pooling ([Hamilton et al., 2017]).
5. **Chebyshev Spectral Convolution (ChebNet)**: Fast spectral graph convolution utilizing truncated Chebyshev polynomial expansion of order $K$ ([Defferrard et al., 2016]).

### Architectural Variants
- **Base**: Two stacked graph convolutional layers with intermediate ReLU and dropout.
- **+Lin**: Two stacked graph convolutional layers followed by a final linear projection layer and log-softmax output probabilities.
- **+Lin+Skip**: Two stacked graph convolutional layers with residual/skip connection ($\mathbf{h}^{(1)} + \mathbf{h}^{(2)}$) fed into a final linear projection layer.

### Feature Variants Tested
- `tx`: Only local transaction features (93 features).
- `tx+agg`: All transaction and aggregated neighbor features (165 features).
- `tx+agg_red`: Transaction features + reduced aggregated features. The paper identified that 22 aggregated features exhibited pairwise Pearson correlation $> 0.90$. Removing these redundant features yielded 143 clean features, reducing collinearity.

---

## 4. Best-Performing Model & Quantitative Evidence

### Summary of Performance (from Paper Section V & Table 4)

| Model Configuration | Feature Set | Illicit Precision | Illicit Recall | Illicit F1-Score | Macro-Average F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Chebyshev+Lin** (Best Overall) | `tx+agg_red` | **0.955** (±0.008) | **0.863** (±0.012) | **0.906** (±0.007) | **0.983** (±0.002) |
| Chebyshev+Lin | `tx+agg` | 0.942 | 0.851 | 0.894 | 0.978 |
| GATv2+Lin | `tx+agg_red` | 0.938 | 0.842 | 0.887 | 0.976 |
| GATv2+Lin+Skip | `tx+agg_red` | 0.941 | 0.838 | 0.886 | 0.975 |
| GraphSAGE (Base) | `tx+agg` | 0.852 | **0.818** | 0.835 | 0.958 |
| Random Forest Baseline ([Weber et al., 2019]) | All features | 0.960 | 0.651 | 0.782 | — |
| GCN+Lin | `tx+agg_red` | 0.720 | 0.621 | 0.667 | 0.912 |

### Why Chebyshev+Lin Achieved Superiority
1. **$K$-hop Spectral Representation**: First-order spatial convolutions (GCN) suffer from severe receptive field limitations on sparse cryptocurrency directed acyclic transaction chains. Chebyshev polynomials of order $K$ allow spectral filtering over multi-hop paths without expensive full Laplacian decomposition.
2. **Linear Decoupling (`+Lin`)**: Projecting the spectral graph representations through a decoupled linear layer mitigates over-smoothing and enhances separation between illicit and licit decision boundaries.
3. **Collinearity Elimination (`tx+agg_red`)**: Redundant aggregated features with high correlation ($r > 0.90$) introduce collinear noise in spectral filters. Pruning these 22 features provided the highest generalization stability (F1 = 0.906).
4. **Significant Improvement Over Non-Graph Baselines**: The Chebyshev+Lin model boosts illicit recall from 0.651 (Random Forest) to 0.863 and F1 from 0.782 to 0.906, demonstrating the decisive value of topological modeling for cryptocurrency AML.

---

## 5. Experimental Setup & Reproducibility Constraints

- **Optimizer**: Adam with initial learning rate $\eta = 9 \times 10^{-3}$, dynamically reduced by a factor of 0.9 when validation loss failed to improve for 100 epochs (`ReduceLROnPlateau`).
- **Loss Function**: Binary Cross Entropy (BCE) with class weighting for severe imbalance.
- **Hidden Dimensions**: 64 units for `tx` configuration, 110 units for `tx+agg` and `tx+agg_red` configurations.
- **Training Epochs**: Up to 13,000 epochs with Early Stopping (patience = 1,000 epochs).
- **Repetitions**: 15 independent runs reported with mean and standard deviation.
- **Hardware Used**: University of Bologna HPC cluster node: 8-core CPU, 64 GB RAM, NVIDIA L40 GPU (48 GB GDDR6 VRAM, CUDA 11.8).

---

## 6. Phase 1 Implementation Directives for Downstream Compatibility

To guarantee seamless integration for Phase 2 (Random Forest), Phase 3 (GNNs & Chebyshev), and Phase 4 (Streamlit Dashboard):

1. **Strict Data Separation**:
   - `x` must contain only numeric features (never `txId`, `time_step`, or class labels).
   - Original `txId` mapping to continuous node indices `[0, N-1]` must be stored explicitly in JSON/NumPy for downstream node-level queries.
2. **Feature Set Flexibility**:
   - Support `tx` (93 local features), `tx+agg` (165 features), and compute the correlation filter for `tx+agg_red` (143 features) fitted strictly on the training set to prevent data leakage.
3. **Temporal Split Boundaries**:
   - Standard Weber et al. temporal protocol: Timesteps 1–34 (Train), 35–41 (Validation), 42–49 (Test).
   - Leakage prevention: Zero test/validation labels or features used during preprocessing transformations.
4. **PyTorch Geometric Compatibility**:
   - Directed edge index tensor `[2, E]` saved with zero-indexed node identifiers.
   - Class labels encoded as: `0` = Licit, `1` = Illicit, `-1` = Unknown.
