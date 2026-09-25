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
*(Additional entries will be logged continuously as experiments are executed and progress is made)*
