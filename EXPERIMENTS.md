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
| **EXP-003** | 2026-09-26 | 26 Pairwise & Group-Relative Features + LightGBM trained on **5 Lakh (500,000)** $S_1$ entities + 5-Fold GroupKFold OOF ($P \ge 0.80$) | **97.97%** | 34.82 | **0.9823** | **0.9482** | **0.9701** | Completed | **Adopted (+3.61 F0.5, +8.59% Singleton Acc)** |
| **EXP-004a** | 2026-09-26 | **GPU XGBoost (`cuda:0` on RTX 2050)** trained on **15 Lakh (1,500,000)** $S_1$ entities + 30 Features + 9-Script Indic Transliteration ($P \ge 0.80$) | **98.06%** | 34.82 | **0.9851** | **0.9519** | **0.9734** | Completed | **Adopted (+0.33 F0.5 over EXP-003)** |
| **EXP-004b** | 2026-09-26 | **LightGBM** trained on **15 Lakh (1,500,000)** $S_1$ entities + 30 Features + 9-Script Indic Transliteration ($P \ge 0.80$) | **98.06%** | 34.82 | **0.9854** | **0.9529** | **0.9740** | Completed | **New Best Single Model (+0.39 F0.5, 98.54% Precision)** |
| **EXP-004c** | 2026-09-26 | **50/50 Ensemble (GPU XGBoost + LightGBM)** on 15 Lakh $S_1$ entities ($P \ge 0.75$ / $P \ge 0.80$) | **98.06%** | 34.82 | **0.9841** | **0.9563** | **0.9738** | Completed | **Highest Stage-1 Recall (`95.63%`)** |
| **EXP-005** | 2026-09-26 | **Stage-2 Group-Context Meta-Reranker (48 Features: 30 Pairwise + 18 Group Prob Context)** + Hierarchical LGB/GPU-XGB Ensemble ($\tau = 0.74$) on `val_pool` (`234k`) | **98.06%** | 34.82 | **0.9865** | **0.9553** | **0.9760** | Completed | **Adopted (+0.20 F0.5, Singleton Acc jumped 94.64% $\to$ 97.02%!)** |
| **EXP-005 (Full-Pool Baseline)** | 2026-09-26 | **Reproduced `82%` Official Leaderboard Score Locally** on Full `4,133,346` (`41.3 Lakh`) India Pool (`train_source2 + train_source3`) at $\tau = 0.74$ | 90.75% | 35.00 | 0.8403 | 0.8765 | **0.8299** | Completed | **Root Cause Identified (`4%` Val Distractor Pool Trap + Scale Shift)** |
| **EXP-006 (Step 1)** | 2026-09-26 | **Full-Pool Scale-Invariant Blocker (`token_df` scaling + Phonetic Skeleton `ps:`/`ph:` + Name+Location `nc:` + Multi-Channel Synergy Quotas) + Pairwise Cross-Script Alignment** evaluated on Full `4,133,346` India Pool (Zero Retraining) | **95.73%** | 45.00 | **0.9179** | **0.9083** (`0.9438` @0.74) | **0.9059** | Completed | **Adopted (+7.60 Full-Pool F0.5, +4.98% Full-Pool Blocking Recall, +27.05% Singleton Acc!)** |
| **EXP-006 (Step 2 — India 4.13M Pool)** | 2026-09-26 | **40-Feature Full-Pool Disambiguation + Stage-1/Stage-2 GBDT Retrained on `1.03 Crore` Pool** evaluated on all `7,812` India Val $S_1$ vs `4,133,346` Pool | **95.10%** | 35.00 | **0.9713** | **0.9004** | **0.9479** | Completed | **Adopted (+11.80 points over 0.8299 Full-Pool Baseline; Singleton Acc 53.28% $\to$ 94.71%!)** |
| **EXP-006 (Step 2 — Full 1.03 Crore Pool)** | 2026-09-26 | **Complete `10,320,219` (`1.03 Crore`) Full-Pool Benchmark** (`20,000` Val $S_1$ + `24,000` Ext $S_1$, 40 Pairwise + 18 Group Context = 58 Features, $\tau_{\text{top1}}=0.74, \tau_{\text{sec}}=0.72$) | **96.43%** | 35.00 | **0.9778** (`0.9789` @0.74) | **0.9132** | **0.9571** (`US`: **0.9631** \| **Live LB: `94.00%`**) | Completed | **Verified +12% Live Leaderboard Jump (`82%` $\to$ `94%`)** |
| **EXP-007 (Full 1.03 Crore Pool)** | 2026-09-27 | **50 Pairwise & Sibling-Cluster Features + 18 Group Context = 68 Stage-2 Features** (Pure-Street Similarity, 3+ Digit Building Rescue, Sibling Address Cluster Consensus, Missing-Address Group Separator, French Legal/Region Tokens, Alphanumeric Number Splitting, GPU 4-Model Stage-1 + Stage-2 Ensemble + Drop-Off Ratio $\ge 0.75$ + Bipartite Mutual Exclusion) | **96.43%** | 35.00 | **0.9820** | **0.9208** (`0.9260` @0.68) | **0.9626** (`India`: **0.9518**, `US`: **0.9688**) | Completed | **Adopted (+0.55 F0.5, +640 TPs, -184 FPs on 1.03 Crore Pool)** |
| **EXP-007b (Full 1.03 Crore Pool + Upgraded Blocker)** | 2026-09-27 | **Deep Error-Driven Preprocessor, Blocker & Feature Upgrades**: Fixed 4 Indic script bugs (Bengali/Odia `য়`/`ୟ`, Gurmukhi Tippi/Addak/`ਸ਼`, zero-width joiners), standalone `private`/`pvt` & `publiclimited`, consecutive duplicate word/bigram collapse, OCR digit-in-word fix, exact 48-stem synthetic vocabulary, Exact Initials Acronym Rescue (`100%` precision), Synthetic Generic-Word Substitution (`97.8%` precision), Shifted Primary Street Number (`prim_num_match=0.75`), 1-Digit Suffix Truncation & $\pm 2$ Perturbation, `loc_conflict` fix (`pradesh`/`nadu`, Telangana/Andhra), 3-level `legal_type_conflict`, and clean `s1_same_name_diff_addr_cnt` | **99.92%** (`+2,410` TPs rescued) | 50.00 | **0.9832** (`0.9890` Bipartite) | **0.9494** (`0.9513` Bipartite) | **0.9719** (`US`: **0.9742**, `India`: **0.9683** \| **Full-Pop Bipartite: `0.9775`**) | Completed | **Target Achieved (`97.19%` Raw 20k-Slice / `97.75%` Full-Population Bipartite Competition Regime!)** |

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
10. [x] Build and run **EXP-003**: Engineer 26 pairwise & group-relative features and train **LightGBM** on **5 Lakh (500,000)** $S_1$ entities ($F_{0.5} = 0.9701$, Precision $= 0.9823$, Recall $= 0.9482$, Singleton Acc $= 95.06\%$).
11. [x] Build and run **EXP-004**: Train GPU-accelerated **XGBoost (`device="cuda:0"`, `tree_method="hist"`)** + **LightGBM** on **15 Lakh (1,500,000)** $S_1$ entities with 30 features and 9-script Indic transliteration (**Blocking Recall $= 98.06\%$**, **Macro $F_{0.5} = 0.9740$**).
12. [x] Build and run **EXP-005**: Train **Stage-2 Group-Context Meta-Reranker (48 Features)** over Stage-1 OOF probabilities (**Macro $F_{0.5} = 0.9760$**, **Precision $= 98.65\%$**, **Recall $= 95.53\%$**, **Singleton Acc $= 97.02\%$**).
13. [x] **Generate & Validate Official Test Submission (`output/matching_results.tsv` & `output/candidate_pairs.tsv`)** across all `1,732,544` test $S_1$ entities and `9,969,589` candidates (**`PASS — no blocking issues found. Safe to submit.`** with `--check-ids`).
14. [x] **Step 1 (`EXP-006` Full-Pool Blocker & Cross-Script Phonetic Alignment)**: Raised Full `4.13M` India Pool Blocking Recall from `90.75%` $\to$ **`95.73%`** and Full-Pool Macro $F_{0.5}$ from `0.8299` $\to$ **`0.9059` (+7.60 points)** before any model retraining.
15. [x] **Step 2 (`EXP-006` Full-Pool 40-Feature Disambiguation & 1.03-Crore Model Retraining)**: Retrained Stage-1 (GPU XGBoost + LightGBM) and Stage-2 (Group-Context Meta-Reranker) on **`1,542,984` Full-Pool Candidate Pairs** mined from the **complete `10,320,219` (`1.03 Crore`) pool**, pushing Full-Pool Macro $F_{0.5}$ from `0.8299` $\to$ **`0.9571` (`95.71%` overall | `94.79%` India | `96.31%` US)**!
16. [x] **Regenerate & Validate Official Test Submission (`output/matching_results.tsv`)** using the trained `EXP-006` Full-Pool pipeline across all `1,732,544` test $S_1$ entities and `9,969,589` candidates (`107,138` Singletons = `6.18%`, `5,563,476` matched links, **Live Leaderboard Score jumped `82%` $\to$ `94%`**).
17. [x] **Step 1 & Step 2 (`EXP-007` 50-Feature Full-Pool Disambiguation + GPU Multi-Model Ensemble)**: Engineered 10 new domain/sibling-cluster features (`50` base + `18` group context = `68` Stage-2 features), raising Full-Pool Macro $F_{0.5}$ from `0.9571` $\to$ **`0.9626` (`98.20%` Precision, `+640` TPs, `-184` FPs on `1.03 Crore` pool)**!
18. [x] **Step 3 (`EXP-007b` Deep Error-Driven Preprocessor, Blocker & Feature Upgrades)**: Rescued `2,410 / 2,457` (`98.1%`) missed ground-truth links in the blocker (**Blocking Recall `96.43%` $\to$ `99.92%`**) and raised Full-Pool Validation Macro $F_{0.5}$ to **`0.9719` (`97.19%` Raw 20k-Slice | `US: 97.42%` | `India: 96.83%`)** and **`0.9775` (`97.75%`)** under the Full-Population Bipartite Competition Regime!
19. [x] **Step 4 (`EXP-007b` Test Submission Regeneration + Global Bipartite Mutual Exclusion)**: Regenerated and validated both [`output/matching_results.tsv`](output/matching_results.tsv) (`92 MB`, `103,170` empty = `5.95%` singletons, `5,662,989` matched links, `49,655` cross-$S_1$ duplicate claims pruned) and [`output/candidate_pairs.tsv`](output/candidate_pairs.tsv) (`661 MB`, `1,732,543 / 1,732,544` non-empty) with **`PASS — no blocking issues found. Safe to submit.`** (including `--check-ids`).


