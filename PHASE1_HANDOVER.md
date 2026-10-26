# Phase 1 Handover Guide: Downstream Integration

Welcome to the team! Phase 1 has completed the dataset engineering, structural validation, feature alignment, graph construction, and leakage-free temporal splitting for the **Elliptic Bitcoin Transaction Dataset**.

This document outlines the exact files, data representations, and code patterns needed for **Phase 2 (Random Forest Baseline)** and **Phase 3 (Graph Neural Networks & Chebyshev Models)**.

---

## 1. Quick File Map for Downstream Phases

All processed files are generated under `data/processed/`.

| File | Shape / Type | Recommended For | Purpose |
| :--- | :--- | :---: | :--- |
| `elliptic_x_all.npy` | `[203769, 165]` float32 | Phase 2 & 3 | Full feature matrix (93 local + 72 aggregated). |
| `elliptic_x_local.npy` | `[203769, 93]` float32 | Phase 2 & 3 | Local-only feature matrix (`tx` baseline). |
| `elliptic_x_reduced.npy` | `[203769, 145]` float32 | **Phase 2 & 3** | **Paper best feature set (`tx+agg_red`)** with collinear features removed. |
| `elliptic_y.npy` | `[203769]` int64 | Phase 2 & 3 | Labels: `1` = Illicit, `0` = Licit, `-1` = Unknown. |
| `elliptic_masks.npz` | Compressed NumPy | Phase 2 & 3 | Contains `train_mask`, `val_mask`, `test_mask`, and labeled masks. |
| `elliptic_edge_index.npy` | `[2, 234355]` int64 | **Phase 3** | Directed edge list mapped to node indices `0..203768`. |
| `elliptic_pyg_data.pt` | PyTorch Geometric `Data` | **Phase 3** | Pre-packaged PyG `Data` object ready for PyG loaders and GNN layers. |
| `elliptic_tensors.pt` | PyTorch Tensor Dict | Phase 2 & 3 | All tensors bundled for pure PyTorch without PyG. |
| `elliptic_tx_to_idx.json` | JSON Dictionary | Phase 2, 3, 4 | Lookup original Bitcoin `txId` $\rightarrow$ continuous integer node index. |
| `elliptic_idx_to_tx.json` | JSON List | Phase 2, 3, 4 | Lookup continuous integer node index $\rightarrow$ original Bitcoin `txId`. |

---

## 2. Instructions for Phase 2: Random Forest Baseline

The Phase 2 engineer is responsible for establishing standard machine learning baselines (Random Forest, Logistic Regression, XGBoost/LightGBM) on tabular node features.

### Crucial Directives for Phase 2:
1. **Filter Out Unknown Labels**: Models must only be trained and evaluated on labeled nodes where `y != -1`. Use `train_mask_labeled` and `test_mask_labeled`.
2. **Class Imbalance Handling**: Illicit transactions represent ~9.76% of labeled data. Use class weighting (e.g. `class_weight='balanced'` in scikit-learn).
3. **Primary Evaluation Metrics**: Evaluate on the **Illicit class (Class 1)**:
   - Illicit Precision
   - Illicit Recall
   - Illicit F1-Score
   - Macro-Average F1

### Phase 2 Python Integration Example:
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# 1. Load preprocessed features, labels, and masks
x = np.load("data/processed/elliptic_x_all.npy")      # Or elliptic_x_reduced.npy
y = np.load("data/processed/elliptic_y.npy")
masks = np.load("data/processed/elliptic_masks.npz")

train_mask = masks["train_mask_labeled"]  # boolean array: train timesteps 1..34 & y != -1
test_mask = masks["test_mask_labeled"]    # boolean array: test timesteps 42..49 & y != -1

# 2. Extract training and testing sets
X_train, y_train = x[train_mask], y[train_mask]
X_test, y_test = x[test_mask], y[test_mask]

print(f"Training on {len(X_train):,} labeled nodes (Illicit: {(y_train == 1).sum():,})")
print(f"Testing on {len(X_test):,} labeled nodes (Illicit: {(y_test == 1).sum():,})")

# 3. Train Random Forest baseline
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 4. Evaluate against paper baseline (Weber et al. / Ferretti et al. benchmark: F1 ~ 0.782, Recall ~ 0.651)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Licit (0)", "Illicit (1)"], digits=4))
```

---

## 3. Instructions for Phase 3: Graph Neural Networks & Chebyshev Models

The Phase 3 engineer is responsible for implementing GNN architectures (GCN, GraphSAGE, GATv2, and Chebyshev spectral networks).

### Loading the Pre-Built PyTorch Geometric Graph
Phase 1 has already bundled the graph into a PyTorch Geometric `Data` object:

```python
import torch

# Load complete graph object
data = torch.load("data/processed/elliptic_pyg_data.pt", weights_only=False)

print(data)
# Expected output:
# Data(x=[203769, 145], edge_index=[2, 234355], y=[203769], timesteps=[203769], train_mask=[203769], val_mask=[203769], test_mask=[203769])

# To train on labeled nodes only:
labeled_mask = (data.y != -1)
train_idx = (data.train_mask & labeled_mask).nonzero(as_tuple=True)[0]
val_idx = (data.val_mask & labeled_mask).nonzero(as_tuple=True)[0]
test_idx = (data.test_mask & labeled_mask).nonzero(as_tuple=True)[0]
```

### Implementing the Research Paper's Best-Performing Model (`Chebyshev+Lin`)

According to Ferretti et al. (IEEE Access, 2025), the **Chebyshev+Lin** architecture with `tx+agg_red` achieved:
- Illicit F1: **0.906**
- Illicit Precision: **0.955**
- Illicit Recall: **0.863**
- Macro-Average F1: **0.983**

Here is the exact reference architecture ready for Phase 3:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class ChebyshevPlusLin(nn.Module):
    """
    Chebyshev Spectral GNN with Final Linear Layer (Chebyshev+Lin).
    Architecture specified by Ferretti et al. (Section IV-B & Table 3).
    """
    def __init__(self, in_channels: int = 145, hidden_channels: int = 110, out_channels: int = 2, K: int = 3, dropout: float = 0.5):
        super().__init__()
        # First Chebyshev Spectral Convolution
        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)
        # Second Chebyshev Spectral Convolution (preserves embedding dimension)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=K)
        # Final Linear classification projection
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1: ChebConv -> ReLU -> Dropout
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2: ChebConv -> ReLU -> Dropout
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Final Linear Layer + LogSoftmax
        out = self.lin(h)
        return F.log_softmax(out, dim=-1)

# Usage in Phase 3:
# Note: Spectral convolutions require undirected graphs or explicit normalization
from torch_geometric.utils import to_undirected

data = torch.load("data/processed/elliptic_pyg_data.pt", weights_only=False)
undirected_edge_index = to_undirected(data.edge_index)

model = ChebyshevPlusLin(in_channels=data.x.shape[1], hidden_channels=110, K=3)
out = model(data.x, undirected_edge_index)
print("Logits shape:", out.shape)  # [203769, 2]
```

### Paper Training Hyperparameters to Replicate:
- **Optimizer**: Adam with learning rate $\eta = 9 \times 10^{-3}$.
- **Learning Rate Scheduler**: Reduce learning rate by factor $0.90$ if validation loss does not improve for 100 epochs (`ReduceLROnPlateau(factor=0.9, patience=100)`).
- **Loss Function**: `nn.CrossEntropyLoss(weight=class_weights)` or `nn.NLLLoss(weight=class_weights)` with inverse class frequency weights to counteract the 10:1 licit:illicit imbalance.
- **Early Stopping**: Early stop after 1,000 epochs of non-decreasing validation loss (max 13,000 epochs).

---

## 4. Invariant Rules Across All Phases

To prevent methodology errors:
1. **Never treat Unknown (`-1`) as Licit (`0`)**.
2. **Never evaluate models on nodes from the training timesteps**. Testing must be strictly on timesteps 42..49.
3. **Preserve original transaction mappings** using `elliptic_tx_to_idx.json` so Phase 4 dashboard users can search real Bitcoin transaction hashes.
