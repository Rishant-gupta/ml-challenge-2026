# Project Memory: Business Entity Resolution Challenge (Time Machine Capsule)

> **Role of this Document**:  
> This file acts as our permanent, living project memory capsule. Every major milestone, technical decision, learning, experiment summary, and challenge pivot is documented chronologically here. Anyone opening this repository should be able to step into our "time machine" and understand exactly where we started, what we discovered, why specific choices were made, and how the system evolved.

---

## 1. Challenge & Mission Overview

### 1.1 The Core Problem
In large-scale commercial platforms, business identity data arrives asynchronously from multiple independent external and internal sources ($S_1, S_2, S_3$). Each source contributes partial, noisy, unstandardized fragments of information regarding real-world businesses. These records share **no common primary keys or identifiers**.

- **Goal**: Build a high-precision Machine Learning pipeline that, given records from 3 independent sources, links matching records from **Source 2 ($S_2$)** and **Source 3 ($S_3$)** to each deduplicated reference record in **Source 1 ($S_1$)**.
- **Cardinality**: A Source 1 entity may match zero (singleton), one, or multiple records across Source 2 and Source 3.

### 1.2 Data Characteristics & Noise Patterns
- **Input Data Format**: Tab-separated (`.tsv`) with columns:
  - `entity_id`: Identifier starting with prefix `S1-`, `S2-`, or `S3-`.
  - `business_name`: Business name containing abbreviations (`Corp` vs `Corporation`, `Pvt` vs `Private`, `Ltd` vs `Limited`), legal suffixes, transliteration variants, typos, punctuation differences (`&` vs `and`), word order transpositions.
  - `business_address`: Address string with missing postal codes/states, landmark references (e.g., *Near SBI ATM*), abbreviations (`Rd` vs `Road`, `St` vs `Street`), component reordering, municipal numbering quirks.
  - `country`: String label. Training covers **US** and **India**. The test set introduces **France** (an unseen country requiring zero-shot / open-world generalization; no hardcoded country filters or one-hot encodings!).

### 1.3 Key Constraints & Rules
- **Strict Prohibition on External Lookups**: No external APIs, commercial entity resolution APIs, government registration lookups, geocoding engines, or web scraping allowed. Disqualification applies to any external lookups.
- **Model Size & License**: Maximum **8 Billion parameters**, permissible under **MIT** or **Apache 2.0** licenses.
- **Evaluation Metric ($F_{0.5}$ Macro)**:
  $$\text{Macro } F_{0.5} = \frac{1}{|S_1|} \sum_{i \in S_1} F_{0.5}(i)$$
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  - **Precision is weighted 2× over recall** ($\beta = 0.5$). False merges (merging distinct companies) are penalized twice as heavily as missed matches.
  - **Singletons**: Source 1 entities with no true matches score **1.0** if predicted empty, and **0.0** if *any* false positive is predicted. Singletons make up a significant portion of real-world ER datasets, so aggressive over-matching is suicidal for the leaderboard score.

---

## 2. Deliverables & Submission Contract

The challenge requires a strict final package structure:
```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv       # Scored on leaderboard
│   └── candidate_pairs.tsv        # Scored for blocking audit
├── code/
│   └── business_entity_resolution/
│       ├── src/                   # Pipeline source code
│       ├── README.md              # Reproduction instructions
│       └── requirements.txt       # Pinned dependencies
└── Documentation_template.md      # Methodology write-up
```

### Output File Specifications:
1. **`output/matching_results.tsv`**:
   - Header: `source1_entity_id\tmatched_entity_ids`
   - Every single $S_1$ entity from test set must have exactly one line.
   - `matched_entity_ids` is comma-separated list of $S_2$ and $S_3$ IDs (or empty if singleton).
2. **`output/candidate_pairs.tsv`**:
   - Header: `source1_entity_id\tcandidate_entity_ids`
   - Must contain all candidates evaluated by the matching model.
   - All predicted matches in `matching_results.tsv` must be a subset of `candidate_pairs.tsv`.

---

## 3. High-Level Architecture Blueprint

```
Raw Sources (S1, S2, S3)
        │
        ▼
[ 1. Ingestion & Preprocessing ]
  - Country-aware text normalization
  - Legal suffix stripping & expansion
  - Address token standardization (road/street/suite/floor)
  - Landmark & noise character cleaning
        │
        ▼
[ 2. Blocking / Candidate Generation ]
  - Filter candidate space from O(|S1| * (|S2| + |S3|)) down to manageable top-K
  - Country-conditioned blocking (entities match within same country)
  - Multi-index inverted index (TF-IDF character n-grams, business name token overlap, Soundex/Metaphone)
  - High Recall Ceiling (target: >= 98% recall of true matches in candidate set)
        │
        ▼
[ 3. Candidate Pairs ] ───► output/candidate_pairs.tsv
        │
        ▼
[ 4. Pairwise Feature Engineering ]
  - String similarities: Levenshtein, Jaro-Winkler, Token Sort/Set Ratio, Jaccard
  - TF-IDF Cosine & Embedding Cosine similarity
  - Address component matches: Token overlap, number matching, city/pincode consistency
  - Name abbreviation / acronym matching
        │
        ▼
[ 5. Matching Model & Scoring ]
  - Gradient Boosted Decision Tree (LightGBM/CatBoost) / Bi-Encoder / Cross-Encoder
  - Predict match probability P(match | pair)
        │
        ▼
[ 6. Precision-Tuned Postprocessing & Singleton Filtering ]
  - Threshold optimization tuned specifically for Macro F_0.5
  - Singleton confidence guard: suppress low-confidence single links
  - Mutual exclusivity & transitivity enforcement (if applicable)
        │
        ▼
[ 7. Submission Output & Validation ] ───► output/matching_results.tsv
  - Run utils/validate_submission.py to guarantee 0 formatting defects
```

---

## 4. Time Machine Capsule: Project Journey Log

### [Entry 001] — 2026-09-25: Project Kickoff & Workspace Inception
- **Author**: Antigravity & Engineering Team
- **Milestones Completed**:
  1. Ingested and comprehensively analyzed `Untitled.pdf` describing the Business Entity Resolution Challenge.
  2. Located full dataset (previously in `6ab10eb3b23ba_student_resource/`) and established top-level links under `dataset/train` and `dataset/test`.
  3. Identified crucial challenge scale & hardware constraints:
     - **Scale**:
       - `train_source1.tsv`: 2,206,821 records (~201MB)
       - `train_source2.tsv`: 5,034,616 records (~467MB)
       - `train_source3.tsv`: 5,285,603 records (~481MB)
       - `train_ground_truth.tsv`: 2,206,821 rows (~122MB)
       - `test_source1.tsv`: 1,732,544 records (~167MB)
       - `test_source2.tsv`: 4,887,273 records (~486MB)
       - `test_source3.tsv`: 5,082,316 records (~483MB)
       - Total: **~26.4 million records** across all sources! Exhaustive pairwise comparison would require $1.73 \times 10^{13}$ operations (computationally impossible without aggressive blocking).
     - **Hardware**: 12 CPU cores, 15GB RAM (available ~7.4GB), NVIDIA RTX 2050 (4GB VRAM).
     - **Memory Strategy**: Processing must be memory-efficient, chunked or partitioned by country, avoiding massive naive Python object dictionaries.
     - **Noise Insights**: Multilingual/multiscript (e.g. Hindi Devanagari alongside English in India records), legal suffixes (`Pvt Ltd`, `LLP`, `Inc`), domain-style business names (`foo.com`), addresses with varying granularities and missing components.
     - **Open-set country**: Test set introduces **France**, which is absent from training (US & India only). Pipelines must not hardcode country assumptions.
     - **Evaluation metric**: Macro $F_{0.5}$ penalizes false positives heavily ($2\times$ weight on precision). Singletons are scored strictly (1.0 vs 0.0).
     - **Strict no-external-lookup policy**: Fully local ML models and algorithms.
     - **Mandatory dual output files**: `matching_results.tsv` and `candidate_pairs.tsv`.
  4. Established clean repository structure:
     - `dataset/train/` and `dataset/test/`
     - `output/`
     - `utils/validate_submission.py` (official script copied and verified)
     - `utils/metrics.py` (official Macro $F_{0.5}$ suite tested and verified with PDF test case)
     - `code/business_entity_resolution/` (modular src: `data_loader.py`, `preprocessor.py`, `README.md`, `requirements.txt`)
     - `Documentation_template.md`
     - `PROJECT_MEMORY.md` (this file)
     - `EXPERIMENTS.md` (experiment roadmap and logs)
- **Key Learnings & Strategy Insights**:
  - Entity Resolution has two distinct bottlenecks:
    1. *Blocking*: If true match is pruned during blocking, it can never be recovered. Recall must be maximized here with low computational footprint.
    2. *Matching & Thresholding*: Because of the $F_{0.5}$ metric, when in doubt, *do not match*. A false match on a singleton reduces its score from 1.0 to 0.0!
- **Current Status**: Project environment fully scaffolded, dataset linked and inspected, documentation and memory systems active. Ready for dataset exploration and baseline development.

---

### [Entry 002] — 2026-09-25: In-Depth Exploratory Data Analysis & Jupyter Environment
- **Author**: Antigravity & Engineering Team
- **Milestones Completed**:
  1. Built and activated isolated Python virtual environment (`venv/`) equipped with `jupyterlab`, `pandas 2.2+`, `scikit-learn`, `lightgbm`, `rapidfuzz`, `seaborn`, `matplotlib`, and `tqdm`.
  2. Created configured launcher [`start_jupyter.sh`](start_jupyter.sh) with sandbox-safe workspace paths for Jupyter configuration, runtime, and data directories.
  3. Created and executed [`notebooks/01_exploratory_data_analysis.ipynb`](notebooks/01_exploratory_data_analysis.ipynb), pre-rendering all statistics, histograms, KDE plots, and noise inspections.
- **Key EDA Discoveries & Empirical Insights**:
  1. **Strict Country Partitioning**:
     - Evaluated ground truth pairs: **100% of matches stay strictly within the same country**.
     - An Indian entity in $S_1$ will never match a US or French entity in $S_2/S_3$.
     - *Implication*: We can partition the blocking and candidate search into independent country buckets (`US`, `India`, `France`), drastically cutting pairwise search space by $60\% - 80\%$ and keeping memory footprint manageable on our 15GB RAM machine.
  2. **Singleton Profile & Match Cardinality**:
     - **$\sim 5.6\%$ of Source 1 entities are pure singletons (0 matches)**.
     - Non-singletons match an average of **3.46 records** (peaking at 3-4 matches, with max 11).
     - Breakdown: $\sim 48\%$ of matches originate from Source 2 and $\sim 52\%$ from Source 3.
  3. **Data Completeness**:
     - Reference Source 1 is pristine ($0\%$ missing names, addresses, or countries).
     - Sources 2 and 3 contain **$\sim 3.3\%$ missing addresses** and rare missing names. Any matching feature must handle null addresses gracefully (falling back to strong name matching).
  4. **Unseen Country Breakdown (France)**:
     - Training set: US ($60\%$), India ($40\%$).
     - Test set: India ($47\%$), US ($38\%$), France ($15\%$).
     - France represents nearly 260,000 Source 1 test entities! Features must be completely language/country agnostic (relative token sort ratios, character n-grams, Levenshtein distances).
  5. **Observed Noise Taxonomy**:
     - *Legal Suffix Inconsistencies*: `BS Projects Ltd Ltd` vs `BS-Projects` vs `BS Projects`.
     - *DBA (Doing Business As)*: `Beloavi d/b/a Novent Owl PLLC` matches `Novent Owl PLLC`.
     - *Address Transpositions*: City/State preceding street name (`Des Plaines, IL, 9308 Home Court` vs `9308 Home Ct, Des Plaines, IL`).
     - *Numeric Variations*: Leading zeros (`004303 ELKINS AVE` vs `4303 Elkins Avenue`).
     - *Non-Latin Script*: Hindi Devanagari names in India records (`राम मार्केटिंग प्राइवेट लिमिटेड`).
  6. **Directory Consolidation & Cleanup**:
     - Removed redundant archive folders (`6ab10eb3b23ba_student_resource/` and `__MACOSX/`).
     - Consolidated all real TSV dataset files directly into a single unified [`dataset/train/`](dataset/train) and [`dataset/test/`](dataset/test) structure.
     - Cleaned up temporary generator scripts and caches; workspace is lean and canonical.
- **Current Status**: Workspace cleaned and consolidated. EDA notebook generated, verified, and saved with complete output. Next milestone: Baseline country-partitioned blocking and string similarity matcher (EXP-001).

---

### [Entry 003] — 2026-09-25: EXP-001 Results & Systematic Error Analysis
- **Author**: Antigravity & Engineering Team
- **Why We Chose EXP-001**:
  - We needed an end-to-end, deterministic, fast baseline to validate our two-stage architecture: **Stage 1 (Country-Partitioned Inverted Index Blocking)** and **Stage 2 (Heuristic String/Address Matcher + $F_{0.5}$ Threshold Tuning)**.
  - Established a dedicated, reproducible 20,000 $S_1$ entity local validation benchmark (`dataset/val/`) with 468,785 candidate records preserving exact singleton and US/India distributions.
- **Implementation Highlights**:
  - `blocking.py`: Partitioned inverted index by `country`. Multi-key hashing on cleaned name tokens, 4-char prefixes, address number-token pairs, and postal codes with inverse document frequency (IDF) scoring. Top-$25$ candidate retrieval.
  - `matcher.py`: Multi-field composite scoring using `RapidFuzz` Token Sort Ratio, Token Set Ratio, Jaro-Winkler, number overlap bonuses, and missing-address fallback handling.
- **Empirical Results on 20,000 Validation Entities**:
  - **Runtime**: Blocking took **21.30s**; Scoring 490,848 pairs took **24.71s** (Total: $< 47$ seconds).
  - **Search Space Reduction**: Pruned 9.38 Billion pairwise comparisons down to 490,848 pairs (**99.995% reduction**, avg 24.54 candidates per $S_1$).
  - **Blocking Recall Ceiling**: **91.60%** (captured 63,009 / 68,785 true links in top-25).
  - **Threshold Calibration for Macro $F_{0.5}$**:
    - $\tau = 68.0 \implies F_{0.5} = 0.6572$ (Precision: 64.22%, Recall: 88.68%, Singleton Acc: 31.83%)
    - $\tau = 75.0 \implies F_{0.5} = 0.8352$ (Precision: 84.47%, Recall: 87.93%, Singleton Acc: 65.19%)
    - $\tau = 80.0 \implies F_{0.5} = 0.8927$ (Precision: 91.72%, Recall: 86.81%, Singleton Acc: 80.85%)
    - **$\tau = 82.0 \implies \mathbf{F_{0.5} = 0.8938}$ (Precision: 92.12%, Recall: 86.03%, Singleton Acc: 82.47%)**
    - $\tau = 86.0 \implies F_{0.5} = 0.8821$ (Precision: 93.11%, Recall: 79.52%, Singleton Acc: 91.74%)
- **Diagnostic Error Analysis — Where EXP-001 Lacks**:
  1. *Commercial Building / Strip Mall False Positives*:
     - Example: `Congregation Church` at `929 Ridge Road` falsely matched `Service Certified LLC` and `CERTIFIED NEUBERGER [LLC]` at `929 RIDGE RD` (Score 84.0).
     - *Cause*: Our address rescue rule boosted pairs with identical addresses even when name similarity was near zero.
     - *Takeaway*: Never match distinct businesses purely on address — physical buildings host multiple separate companies!
  2. *Domain Name & Website URL Formatting*:
     - Example: S1 `PUS Horizon Paper` missed GT `pushorizonpaper.com` and `pushorizonpapercom`.
     - Example: S1 `D/U Brc LLC` missed GT `dubrc.com`.
     - *Cause*: Tokenizing `pushorizonpaper.com` yields 1 monolithic token, missing token-based inverted index keys.
     - *Takeaway*: Preprocessing must strip `.com`/`.net`/`.org` and test concatenated name string matching.
  3. *Cross-Script / Hindi Devanagari Mismatches*:
     - Example: S1 `Alpha Marketing Private Limited` missed GT `अल्फा मार्केटिंग प्राइवेट लिमिटेड` (Score 55.6).
     - *Cause*: String edit distances between English and Devanagari characters are 0 without transliteration.
     - *Takeaway*: Implement phonetic / transliteration normalization for Indian records.
  4. *Blocking Recall Ceiling Bottleneck (91.60%)*:
     - ~8.4% of true links are lost in blocking and can never be recovered by any downstream model.
     - *Takeaway*: EXP-002 must expand blocking keys to include character n-grams and domain-stripped tokens to raise the recall ceiling to $\ge 97\%$.
- **Current Status**: Solid baseline established ($F_{0.5} = 0.8938$). Clear blueprint for EXP-002 (Enhanced Blocking & URL/Script Normalization).

---
*(Additional entries will be logged continuously as experiments are executed and progress is made)*
