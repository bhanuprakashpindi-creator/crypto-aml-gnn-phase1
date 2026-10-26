# Phase 1 Viva & Presentation Defense Notes

**Project Title:** AI-Powered Cryptocurrency Anti-Money Laundering Using Graph Neural Networks  
**Repository:** [https://github.com/bhanuprakashpindi-creator/crypto-aml-gnn-phase1](https://github.com/bhanuprakashpindi-creator/crypto-aml-gnn-phase1)  
**Target Audience:** Academic Examiners, Project Review Panels, Technical Teammates, and Research Advisors  
**Prepared By:** Pindi Bhanu Prakash (Phase 1 Lead Engineer)  

---

## 1. The Two-Minute Project Pitch (Elevator Pitch)

> "Cryptocurrency networks such as Bitcoin process billions of dollars daily with cryptographic pseudonymity. While transaction records are completely public, illicit actors—such as ransomware cartels, darknet markets, and sanctions evaders—exploit this pseudonymity by splitting and routing funds through complex 'peel chains' and multi-hop mixing transactions.
> 
> Traditional Anti-Money Laundering (AML) software relies on tabular rules and basic machine learning, which evaluate transactions as isolated data points. They miss the broader structural context. Our research project addresses this fundamental limitation by utilizing **Graph Neural Networks (GNNs)**. GNNs treat transactions as nodes and payment flows as directed edges, allowing neural models to propagate neighborhood context across multiple hops.
> 
> Our team has divided this college research project into four clean phases:
> - **Phase 1 (My Responsibility)**: Dataset Engineering, Validation, Leakage-Free Preprocessing, and Graph Construction.
> - **Phase 2**: Random Forest Tabular Baseline and Feature Evaluation.
> - **Phase 3**: Graph Neural Networks (Chebyshev, GATv2, GraphSAGE) and Comparative Model Evaluation.
> - **Phase 4**: Streamlit Interactive Forensic Dashboard and System Integration.
> 
> In Phase 1, I have built a completely clean, reproducible data pipeline from scratch, validated over 203,000 transactions and 234,000 edges, generated PyG-compatible graph representations, and published our work to GitHub with 100% automated test coverage."

---

## 2. The Two-Minute Phase 1 Pitch (My Specific Contribution)

> "As the Phase 1 engineer, my responsibility was to build the foundational data and graph architecture upon which the other three team members depend. If Phase 1 has subtle bugs—such as index misalignment, data leakage, or corrupted edge endpoints—every downstream model in Phase 2 and Phase 3 will fail or produce scientifically invalid results.
> 
> Specifically, I implemented:
> 1. **Robust Ingestion & Validation**: Ingested the Elliptic Bitcoin dataset and built an 11-point validation engine that proved the dataset has zero duplicate IDs, zero missing values, and zero invalid edge endpoints.
> 2. **Leakage-Free Preprocessing**: Mapped original transaction hashes into continuous zero-indexed node integers `[0, 203768]`. I isolated the numeric feature matrix from labels and timesteps, and computed a Pearson correlation filter ($r > 0.90$) **strictly on the training timesteps** to prune 20 redundant aggregated features without lookahead bias.
> 3. **Graph Construction**: Built PyTorch Geometric directed `edge_index` tensors `[2, 234355]` and complete `Data` containers.
> 4. **Temporal Partitioning**: Partitioned the graph chronologically into Train (timesteps 1–34), Validation (35–41), and Test (42–49), auditing all edges to confirm 100% topological isolation.
> 
> Phase 1 is fully tested with 24 passing unit tests, executes in under 19 seconds, and serializes 11 standardized artifacts to `data/processed/`."

---

## 3. Core Technical Concepts Deep Dive

### 3.1 Why the Elliptic Dataset is the Gold Standard for AML Research
The Elliptic dataset, introduced by Weber et al. (MIT & IBM, 2019), is the largest real-world benchmark of labeled Bitcoin transactions. Unlike synthetic datasets, it reflects actual Bitcoin blockchain transactions associated with confirmed illicit entities (ransomware syndicates, scams, terrorist financing) and licit entities (regulated exchanges, miners, merchants).

### 3.2 What the Three Raw CSV Files Contain
1. `elliptic_txs_classes.csv` (3.3 MB): Contains two columns: `txId` (unique transaction hash identifier) and `class` (`'1'` for illicit, `'2'` for licit, and `'unknown'` for unlabeled).
2. `elliptic_txs_edgelist.csv` (4.5 MB): Contains directed edge pairs: `txId1` (source transaction) and `txId2` (destination transaction where funds flow).
3. `elliptic_txs_features.csv` (689.7 MB): A headerless matrix of 167 columns:
   - Column 0: `txId`
   - Column 1: `time_step` (discrete interval 1..49)
   - Columns 2–94: 93 local transaction features
   - Columns 95–166: 72 aggregated neighbor features

### 3.3 What Timesteps and Transaction IDs Represent
- **`txId`**: An integer identifier representing a distinct Bitcoin transaction.
- **`time_step`**: An anonymized temporal marker from 1 to 49. Each timestep represents a discrete time window of approximately two weeks of Bitcoin blockchain activity.

### 3.4 Difference Between Local and Aggregated Features
- **Local Features (93 features)**: Extracted strictly from the single transaction payload: input count, output count, transaction fees, output Bitcoin volume, and script characteristics.
- **Aggregated Features (72 features)**: Extracted from 1-hop neighboring transactions in the backward and forward flow: min, max, standard deviation, and sum of neighboring fees and volumes.

### 3.5 Why Unknown Labels are Kept Separate
77.15% (157,205 nodes) are unlabeled (`unknown`). In our pipeline, they are encoded as `-1`. They must **never** be treated as licit transactions because doing so would falsely train the model that millions of unclassified transactions are legitimate. Instead, unknown transactions participate in graph message passing (providing structural pathways) but are masked out of the classification loss function.

### 3.6 What `edge_index` Means in PyTorch Geometric
In PyTorch Geometric, graphs are represented using coordinate list format (COO) as a 2D integer tensor of shape `[2, E]`:
- Row 0 contains source node indices ($u$).
- Row 1 contains target node indices ($v$).
Each column represents a directed edge $u \rightarrow v$.

### 3.7 Why the Feature Set is Reduced
The paper observed that 20–22 aggregated neighbor features had pairwise Pearson correlation $r > 0.90$. High collinearity causes parameter instability in GNNs and introduces redundant noise. Pruning these features leaves a clean feature set of **145 features** (`tx+agg_red`).

### 3.8 Why Correlation Pruning Must be Fitted on Training Data Only
Fitting correlation analysis on the full dataset leaks information about the feature distributions of the validation and test sets into the feature selector. To guarantee scientific integrity, our pipeline fits the correlation matrix **exclusively on timesteps 1–34** and applies the resulting column mask across all nodes.

### 3.9 Why Temporal Splits are Used Instead of Random Splits
Randomly shuffling transaction nodes across train and test sets would cause **temporal data leakage**. A model would be trained on transactions from week 40 to predict transactions from week 10. In real-world AML, models can only learn from past history to detect future crimes. Therefore, the dataset is split chronologically: Timesteps 1–34 (Train), 35–41 (Validation), and 42–49 (Test).

### 3.10 The Research Paper's Best-Performing Model
Ferretti et al. (IEEE Access, 2025) identified **`Chebyshev+Lin` with `tx+agg_red`** as the best-performing model:
- **Illicit F1-Score**: **0.906** (±0.007)
- **Illicit Precision**: **0.955** (±0.008)
- **Illicit Recall**: **0.863** (±0.012)
- **Macro-F1**: **0.983** (±0.002)

### 3.11 Why Those Scores are NOT Our Phase 1 Scores
Phase 1 is strictly data engineering and graph preparation. No neural networks or Random Forests were trained in Phase 1. Claiming the paper's model scores as our own would constitute scientific fabrication. Our team will measure our actual model scores when Phase 2 and Phase 3 train their models on our processed graph.

---

## 4. Top 25 Likely Viva Questions and Model Answers

### General AML & Problem Context

#### Q1: What is cryptocurrency money laundering, and why is it difficult to detect?
**Answer:** Cryptocurrency money laundering involves obfuscating the origin of illegally obtained funds (from ransomware, darknet markets, fraud) by moving them through hundreds of pseudonymous blockchain transactions. It is difficult to detect because actors use automated peel chains, unhosted wallets, and decentralized mixers to dissipate transactional trails.

#### Q2: Why are standard tabular machine learning models insufficient for AML?
**Answer:** Tabular models evaluate transactions independently based solely on immediate attributes (e.g., fee, BTC amount). They cannot see relational structure—such as whether a counterparty received funds from a darknet market three hops upstream. Without topological context, standard models miss complex layering patterns.

#### Q3: Why are Graph Neural Networks uniquely suited for this task?
**Answer:** GNNs operate directly on graph structures. Through neighborhood message passing, each node iteratively aggregates feature representations from its neighbors. This allows a transaction's embedding to capture both its individual characteristics and its multi-hop topological context.

---

### Dataset & Topology

#### Q4: What dataset did you use, and what are its key statistics?
**Answer:** We used the Elliptic Bitcoin Transaction Dataset, comprising **203,769 transaction nodes**, **234,355 directed edges**, and **167 raw feature columns** spanning **49 discrete timesteps**.

#### Q5: What is the class distribution in the Elliptic dataset?
**Answer:** The dataset is severely imbalanced:
- **Illicit (Class 1)**: 4,545 transactions (2.23% of total; 9.76% of labeled transactions)
- **Licit (Class 2)**: 42,019 transactions (20.62% of total; 90.24% of labeled transactions)
- **Unknown**: 157,205 transactions (77.15% of total)

#### Q6: Why did you not delete or label the unknown transactions as licit?
**Answer:** Deleting unknown transactions would destroy the graph's structural connectivity, disconnecting payment paths. Labeling them as licit would introduce severe false-negative bias. We encode them as `-1` and mask them out of the supervised loss function while allowing GNNs to pass messages through them.

#### Q7: What did your audit discover about the edges across timesteps?
**Answer:** We discovered that **100% of the 234,355 edges connect transactions within the exact same timestep**. There are zero cross-timestep edges. The dataset is essentially an ensemble of 49 independent temporal graph snapshots.

---

### Data Engineering & Preprocessing

#### Q8: How did you handle the unusual header formatting in the classes file?
**Answer:** In `src/data_loader.py`, we implemented dynamic header inspection. The function inspects the first line, detects whether header keywords exist, strips leading and trailing whitespace, and normalizes column names to `['txId', 'class']`.

#### Q9: How are original Bitcoin transaction IDs mapped to graph node indices?
**Answer:** We sort all transactions deterministically by `[time_step, txId]` and create a bijective mapping: `tx_to_idx` (dictionary mapping string txId to an integer $0 \dots 203,768$) and `idx_to_tx` (array mapping integer index back to original txId). Both are saved as JSON files in `data/processed/`.

#### Q10: How do you prevent data leakage into the node feature matrix?
**Answer:** In `src/preprocessing.py`, we explicitly separate transaction IDs, timesteps, and target labels from the numeric feature matrix. The feature matrices (`x_all`, `x_local`, `x_reduced`) contain strictly floating-point numeric features.

#### Q11: What is the difference between `x_all`, `x_local`, and `x_reduced`?
**Answer:**
- `x_all`: All 165 numeric features (93 local + 72 aggregated neighbor features).
- `x_local`: 93 local-only transaction features.
- `x_reduced`: 145 features resulting from pruning 20 highly correlated aggregated features ($r > 0.90$).

#### Q12: Why does your implementation retain 145 features while the research paper reported 143 features?
**Answer:** The paper pruned 22 features across the entire dataset. In our pipeline, we strictly enforce **leakage prevention by computing the correlation matrix exclusively on the training split (timesteps 1–34)**. On the training split, 20 features exceeded the 0.90 correlation threshold, leaving 145 features. This minor difference protects our evaluation from lookahead bias.

---

### Graph Construction & PyG

#### Q13: What format does your graph builder output?
**Answer:** In `src/graph_builder.py`, we construct a PyTorch Geometric `Data` object containing `x`, directed `edge_index` (tensor of shape `[2, 234355]`), `y`, `timesteps`, and boolean split masks. We also serialize a standalone PyTorch dictionary (`elliptic_tensors.pt`) for pure PyTorch workflows.

#### Q14: Did you find any self-loops or duplicate edges in the raw dataset?
**Answer:** Our validation engine proved that the raw edgelist contains **zero self-loops** (`txId1 == txId2`), **zero duplicate directed edges**, and **zero invalid endpoints** referencing non-existent transactions.

#### Q15: Is your transaction graph directed or undirected?
**Answer:** The raw graph and our primary `edge_index` are strictly **directed**, preserving the flow of funds from inputs to outputs. However, for Phase 3 models that require symmetric adjacency (such as Chebyshev spectral convolutions or standard GCNs), `PHASE1_HANDOVER.md` provides utility instructions to apply `to_undirected(edge_index)`.

---

### Splitting & Leakage

#### Q16: What temporal split boundaries did you implement?
**Answer:** Following Weber et al. (2019):
- **Training Split**: Timesteps 1–34 (136,265 nodes; 29,894 labeled)
- **Validation Split**: Timesteps 35–41 (30,680 nodes; 7,829 labeled)
- **Test Split**: Timesteps 42–49 (36,824 nodes; 8,841 labeled)

#### Q17: What is topological data leakage, and how did you verify it was absent?
**Answer:** Topological leakage occurs when edges cross from test or validation sets into the training set, allowing message passing to leak future information. In `src/data_splitter.py`, we audited every edge against node timestep boundaries and verified that **0 edges cross split boundaries**.

#### Q18: What are labeled masks, and why are they necessary?
**Answer:** Labeled masks (`train_mask_labeled`, `test_mask_labeled`) are boolean masks that select only nodes belonging to a specific split whose label is known (`y != -1`). They ensure that downstream models compute loss and metrics only on confirmed licit and illicit transactions.

---

### Research Reference & Future Phases

#### Q19: Which model performed best in the research paper?
**Answer:** The **Chebyshev+Lin** architecture using `tx+agg_red` achieved the highest performance: Illicit F1 = **0.906**, Precision = **0.955**, Recall = **0.863**, and Macro-F1 = **0.983**.

#### Q20: Why did Chebyshev convolutions outperform standard GCNs?
**Answer:** Standard GCNs approximate only first-order (1-hop) neighborhoods. Chebyshev spectral convolutions use truncated polynomials of order $K$, capturing multi-hop neighborhood dynamics without expensive full matrix eigendecomposition. This is critical for capturing extended Bitcoin peel chains.

#### Q21: What is the primary evaluation metric for AML, and why?
**Answer:** While F1-score provides a balanced metric, **Illicit Recall is the most critical compliance metric**. A low recall means money launderers successfully bypass detection. High precision ensures investigators do not waste resources on false alarms.

#### Q22: What will Phase 2 implement?
**Answer:** Phase 2 will implement tabular baselines (Random Forest, Logistic Regression, XGBoost) using our `elliptic_x_all.npy` and `elliptic_masks.npz` files to establish the benchmark without graph structure.

#### Q23: What will Phase 3 implement?
**Answer:** Phase 3 will implement and benchmark Graph Neural Networks (GCN, GraphSAGE, GATv2, and Chebyshev+Lin) using our `elliptic_pyg_data.pt` container to replicate and extend the paper's findings.

#### Q24: What will Phase 4 implement?
**Answer:** Phase 4 will build an interactive Streamlit forensic dashboard allowing analysts to input real transaction IDs, inspect subgraph neighborhoods, and view real-time AML risk classifications.

#### Q25: How do you prove your Phase 1 code works correctly?
**Answer:**
1. Our 24-test unit suite runs and passes in **0.443 seconds** (`Ran 24 tests ... OK`).
2. The end-to-end pipeline executes on the full 203,769-node dataset in **18.22 seconds**, generating 11 verified artifacts in `data/processed/`.
3. All code is version-controlled and pushed to GitHub with a clean commit history.
