# Experimentation Log & Roadmap: Business Entity Resolution

> **Purpose**:  
> This document governs our experimental methodology, hypothesis backlog, and systematic tracking of all models, blocking strategies, and postprocessing techniques. Every idea, trial, parameter change, and metric evaluation will be recorded here to ensure scientific rigor and steady progress.

---

## 1. Evaluation Protocol & Local Validation Strategy

To guarantee that our local metrics reflect leaderboard reality, we follow a strict validation protocol:

### 1.1 Metric: Macro $F_{0.5}$
$$\text{Macro } F_{0.5} = \frac{1}{|S_1|} \sum_{i \in S_1} F_{0.5}(i)$$
where for each Source 1 record $i$:
$$F_{0.5}(i) = \begin{cases}
1.0 & \text{if } |T_i| = 0 \text{ and } |P_i| = 0 \quad (\text{Correct Singleton}) \\
0.0 & \text{if } |T_i| = 0 \text{ and } |P_i| > 0 \quad (\text{False Positive on Singleton}) \\
0.0 & \text{if } |T_i| > 0 \text{ and } |P_i| = 0 \quad (\text{Missed Match}) \\
\frac{1.25 \times P_i \times R_i}{0.25 \times P_i + R_i} & \text{otherwise, where } P_i = \frac{|P_i \cap T_i|}{|P_i|}, R_i = \frac{|P_i \cap T_i|}{|T_i|}
\end{cases}$$

### 1.2 Splitting Strategy
- **Reference Group Split**: Train/Validation splits **must be grouped by `source1_entity_id`**. Never leak records from the same S1 entity across folds.
- **Simulating Unseen Countries (France)**:
  - The test set includes France, while the training set only includes US and India.
  - To prevent overfitting to US/India address quirks, we create a cross-country validation split (e.g., train on US, validate on India, or train on India, validate on US) to test how robust our features and normalizations are on unseen country data.

### 1.3 Key Metrics to Track on Every Experiment
1. **Blocking Recall Ceiling**: Fraction of ground truth matches captured in the candidate set:
   $$\text{Recall}_{\text{blocking}} = \frac{|\text{Candidates} \cap \text{Ground Truth}|}{|\text{Ground Truth}|}$$
2. **Reduction Ratio / Candidate Pair Count**: Total candidate pairs evaluated per S1 entity. (Goal: High recall ceiling $\ge 98\%$ while keeping candidates per S1 entity $\le 50$).
3. **Macro $F_{0.5}$**: Target metric.
4. **Macro Precision & Macro Recall**: To understand error modes.
5. **Singleton Accuracy**: Accuracy on true singletons (must be maintained very high).

---

## 2. Experiment Tracks & Hypotheses Backlog

### Track A: Data Cleaning & Normalization
- [ ] **EXP-A1: Legal Entity Suffix Normalization**
  - *Hypothesis*: Business names often vary solely by corporate suffixes (`Pvt Ltd`, `Private Limited`, `Corp`, `Corporation`, `Inc`, `LLC`, `GmbH`, `SA`, `SAS`). Normalizing or stripping these while retaining a cleaned base name will boost token and character similarity.
- [ ] **EXP-A2: Address Standardizer & Landmark Parsing**
  - *Hypothesis*: Standardizing common street tokens (`st` -> `street`, `rd` -> `road`, `ave` -> `avenue`, `flr` -> `floor`, `opp` -> `opposite`, `nr` -> `near`) and isolating postal codes/numbers will reduce noise across different data sources.
- [ ] **EXP-A3: Transliteration & Typo Robustness**
  - *Hypothesis*: Unicode NFKD normalization, lowercase conversion, and punctuation stripping (`&` -> `and`, removing hyphens/periods) will resolve casing and encoding discrepancies across sources.

### Track B: Blocking / Candidate Generation
- [ ] **EXP-B1: Country-Partitioned Inverted Index**
  - *Hypothesis*: Since businesses operate in specific countries, restricting candidates to records with matching `country` labels drastically cuts pairwise search complexity without hurting recall.
- [ ] **EXP-B2: Multi-Pass Inverted Index (TF-IDF & Character N-Grams)**
  - *Hypothesis*: An inverted index combining:
    1. 3-to-5 character n-grams of business name
    2. Distinct tokens in business name
    3. Digits / postal codes / house numbers in address
    will achieve $>98\%$ recall ceiling while limiting candidate pairs to $<30$ per S1 entity.
- [ ] **EXP-B3: MinHash / Locality Sensitive Hashing (LSH)**
  - *Hypothesis*: MinHash LSH on sub-word token sets allows fast, sub-linear time candidate retrieval for large datasets.

### Track C: Feature Engineering
- [ ] **EXP-C1: Multi-Granularity String Distances**
  - *Features*:
    - Levenshtein distance & ratio (name & address)
    - Jaro & Jaro-Winkler similarity (rewards common prefixes in names)
    - Token Sort Ratio & Token Set Ratio (invariant to word order changes)
    - Longest Common Substring & Subsequence ratio
- [ ] **EXP-C2: Address-Specific Matching Features**
  - *Features*:
    - Numerical overlap ratio (do building numbers, PIN codes, or suite numbers match?)
    - Jaccard similarity of address tokens
    - Landmark compatibility score
- [ ] **EXP-C3: Open-World Generalization Features**
  - *Hypothesis*: Features should measure relative similarity and token consistency rather than raw country identity, ensuring high performance on France test records.

### Track D: Matching Models & Scoring
- [ ] **EXP-D1: Baseline Rule/Similarity Matcher**
  - High-precision string similarity threshold baseline to establish lower-bound benchmark.
- [ ] **EXP-D2: Gradient Boosted Decision Tree (LightGBM)**
  - Train LightGBM classifier with pairwise features to predict match probability $P(\text{match} \mid S_1, S_{2/3})$.
- [ ] **EXP-D3: Lightweight Cross-Encoder / Bi-Encoder Representations**
  - Evaluate sentence transformers (e.g. `all-MiniLM-L6-v2` or `paraphrase-multilingual-MiniLM-L12-v2`, strictly Apache 2.0 / MIT and $<8\text{B}$ params) for semantic embedding similarity.

### Track E: Threshold Optimization & Postprocessing
- [ ] **EXP-E1: Threshold Search for $F_{0.5}$**
  - *Hypothesis*: Because $F_{0.5}$ penalizes precision twice as much, the optimal decision threshold will be significantly higher than standard $0.5$ (likely $0.70 - 0.85$).
- [ ] **EXP-E2: Singleton Guard Filter**
  - *Hypothesis*: If the top candidate for an S1 entity has score $< \tau_{\text{singleton}}$, predict empty list to capture the $1.0$ singleton score and avoid disastrous $0.0$ penalty.

---

## 3. Experiment Tracker Log

| Run ID | Date | Description / Hypothesis | Blocking Recall | Candidate Count / S1 | Val Precision | Val Recall | Val Macro $F_{0.5}$ | Status | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-000** | 2026-09-25 | Environment setup, evaluation suite, and full exploratory data analysis (EDA) notebook | N/A | N/A | N/A | N/A | N/A | Completed | Baseline Ready |
| **EXP-001** | 2026-09-25 | Country-Partitioned Multi-Key Inverted Index + Heuristic RapidFuzz Scorer ($\tau = 82.0$) | 91.60% | 24.54 | 0.9212 | 0.8603 | **0.8938** | Completed | Baseline Established |
| **EXP-002** | 2026-09-25 | Multi-Granularity Tokenization (Spaceless + Shingles + 5-grams + Devanagari + DBA + France) + Strip-Mall & IDF Guards ($\tau = 83.0$) | **97.94%** | 34.82 | **0.9501** | **0.9186** | **0.9340** | Completed | **Adopted (+4.02 F0.5, +6.34% Blocking Recall)** |
| **EXP-003** | *Planned* | Rich Pairwise Feature Engineering (20+ features) + LightGBM GBDT Classifier + GroupKFold Out-of-Fold Thresholding | 97.94%+ | ~35 | TBD | TBD | TBD | Ready | Pending Run |

---

## 4. Immediate Next Steps

1. [x] Read challenge specification PDF thoroughly.
2. [x] Scaffold project directory structure (`dataset/`, `output/`, `utils/`, `code/business_entity_resolution/src/`).
3. [x] Implement submission format validator (`utils/validate_submission.py`).
4. [x] Implement local evaluation metric calculator (`utils/metrics.py`).
5. [x] Set up Python virtual environment (`venv/`) with JupyterLab and ML libraries.
6. [x] Build interactive EDA notebook ([`notebooks/01_exploratory_data_analysis.ipynb`](notebooks/01_exploratory_data_analysis.ipynb)) and launcher script ([`start_jupyter.sh`](start_jupyter.sh)).
7. [x] Perform Exploratory Data Analysis (EDA) on training data: singletons (~5.6%), country invariance (100%), noise types, missing address handling.
8. [x] Build and run **EXP-001** baseline on 20,000 validation benchmark ($F_{0.5} = 0.8938$, Blocking Recall $= 91.60\%$).
9. [x] Implement and run **EXP-002** with multi-granularity tokenization, Devanagari transliteration, URL/DBA handling, and precision guards ($F_{0.5} = 0.9340$, Blocking Recall $= 97.94\%$).
10. [ ] Build **EXP-003**: Engineer 20+ pairwise features (primary street number exact match, token IDF overlap, character/word distances, rank features) and train a **LightGBM Classifier** to push Macro $F_{0.5}$ above **0.95+**.
11. [ ] Add GPU-accelerated multilingual subword embeddings (RTX 2050) for hard cross-script and semantic candidate pairs.
