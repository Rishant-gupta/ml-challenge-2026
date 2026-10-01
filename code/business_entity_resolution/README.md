# Business Entity Resolution Pipeline — ML Challenge 2026

An end-to-end, high-performance machine learning system for resolving noisy, heterogeneously formatted business records across multi-million entity pools into a deduplicated reference entity catalog.

---

## 1. System Overview & Architecture

Resolving business entities at scale (1.73M Source 1 entities evaluated against ~10M Source 2 & Source 3 records) presents both an algorithmic disambiguation challenge and a severe computational bottleneck ($>17 \text{ trillion}$ candidate comparisons). 

Our solution implements a multi-stage architecture engineered specifically for the **Macro $F_{0.5}$** metric (which weights precision $2\times$ over recall) and bounded memory execution (< 8 GB RAM):

```
                   [Source 1, Source 2, Source 3 Records]
                                      │
                                      ▼
                   1. Multilingual Preprocessor & Cleaner
          (Indic 9-Script Transliteration, French Legal Suffixes,
             DBA/Concatenation Splitting, Synthetic Deduplication)
                                      │
                                      ▼
                   2. Scale-Invariant Multi-Channel Blocker
           (Zero-COW array('I') Inverted Indexing, Channel Quotas,
              Logarithmic IDF Attenuation, Top-30 Retrieval)
                                      │
                                      ▼
                    3. Pairwise Feature Engineering (50-D)
          (String Edit Metrics, Alphanumeric Perturbation, Shared
            House Numbers, Sibling Cluster Consensus, Strip-Mall Guard)
                                      │
                                      ▼
                4. Stage-1 Hierarchical GBDT Ensemble (50-D)
             (55% Pure In-Distribution + 45% Extended Mining
                      GPU-Hist XGBoost + LightGBM)
                                      │
                                      ▼
             5. Stage-2 Group-Context Meta-Reranker (68-D)
         (Group Relative Probabilities, Top-1 Margin, Candidate Rank)
                                      │
                                      ▼
             6. Global Bipartite Mutual Exclusion & Precision Guard
       (Enforces 100% Single-Parent Disjoint Ownership Across Country,
             Tuned Decision Rule with Drop-Off Ratio & Singleton Guard)
                                      │
                                      ▼
          [output/matching_results.tsv] & [output/candidate_pairs.tsv]
```

---

## 2. Hardware & Environment Specifications

- **Tested OS**: Linux (Ubuntu 22.04 LTS / Debian-based systems), x86_64
- **Python Version**: Python 3.10+
- **System Memory (RAM)**: 16 GB recommended (streaming chunked execution keeps active memory $< 6.5 \text{ GB}$)
- **GPU (Optional)**: NVIDIA CUDA GPU supported for training (`tree_method="hist"` / `device="cuda"`); inference runs fully optimized on standard multi-core CPU.
- **Disk Space**: ~5 GB for code, pretrained model checkpoints, and generated submission files.

---

## 3. Setup & Installation

### Step 1: Create Virtual Environment
```bash
# From the project root or code directory
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Pinned Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. End-to-End Execution Instructions

### A. Fast Reproduction: Generate Official Submission Files (< 90 mins)
Using the pre-trained Stage-1 and Stage-2 model checkpoints shipped in `output/models/`, run the end-to-end prediction pipeline:

```bash
# Run from code/business_entity_resolution directory:
python3 src/run_pipeline.py \
    --test-dir ../../dataset/test \
    --output-dir ../../output \
    --model-dir ../../output/models \
    --num-workers 4
```

This single command:
1. Partitions the ~1.73M test entities by country (`France`, `US`, `India`).
2. Streams candidate records from `test_source2.tsv` and `test_source3.tsv` through memory-bounded posting lists.
3. Computes the 50 pairwise and 18 group-context features.
4. Generates match probabilities with the hierarchical GBDT ensemble.
5. Applies Global Bipartite Deduplication to ensure zero cross-entity duplicate assignments.
6. Writes:
   - `output/matching_results.tsv` (1,732,544 rows: `source1_entity_id \t matched_entity_ids`)
   - `output/candidate_pairs.tsv` (1,732,544 rows: `source1_entity_id \t candidate_entity_ids`)
7. Automatically executes the official format validator (`utils/validate_submission.py`).

### B. Retrain Models from Scratch (Optional)
To retrain the 50-feature Stage-1 models and 68-feature Stage-2 meta-rerankers from raw training data:

```bash
python3 src/run_pipeline.py \
    --mode train \
    --train-dir ../../dataset/train
```

---

## 5. Validating Submission Outputs

To manually run the official submission validation suite:

```bash
# Standard structural & format check
python3 ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test

# Exhaustive check including target ID existence in test set
python3 ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test \
    --check-ids
```

---

## 6. Key Engineering & Methodological Decisions

1. **Country Partitioning**:
   - Analysis of training ground truth revealed that 100.0% of matches occur strictly within the same country. Partitioning blocking into independent country subsets (`France`, `US`, `India`) reduces search complexity by $>80\%$ with zero recall loss.

2. **Zero-COW Memory Optimization**:
   - Standard Python dictionary posting lists with string IDs cause severe Copy-on-Write (COW) memory paging during multi-process execution. We store integer indices in contiguous `array('I')` unsigned 32-bit buffers (4 bytes/candidate), keeping active memory bounded under 6.5 GB.

3. **Multi-Channel Quota Allocation**:
   - To prevent high-frequency token keys from monopolizing candidate slots, retrieval enforces channel quotas (58% cross-field synergy, 16% core name, 12% phonetic skeleton, 14% street address anchor).

4. **Addressing Domain Specifics & Indic Scripts**:
   - Indian entities frequently mix Latin transliterations with Devanagari, Gurmukhi, Bengali, and Odia scripts. Our preprocessor incorporates phonetic consonant transliteration to bridge script divergence.
   - For France (unseen zero-shot test distribution), corporate structures (`SARL`, `SAS`, `EURL`, `SCI`) and regional street abbreviations are explicitly normalized.

5. **Metric-Aligned Decision Thresholds ($F_{0.5}$)**:
   - Because $F_{0.5}$ penalizes precision twice as heavily as recall ($\beta^2 = 0.25$), standard $0.50$ probability thresholds produce catastrophic false positive penalties. Calibrated thresholds ($\tau_{\text{top1}} \ge 0.72 - 0.78$) with secondary drop-off ratio filtering ($\ge 0.76$) protect singleton precision.

6. **Global Bipartite Mutual Exclusion**:
   - Ground truth establishes that each Source 2/3 entity belongs to at most one Source 1 entity. If multiple Source 1 entities claim the same candidate ID, only the highest-confidence entity retains the link, pruning tens of thousands of duplicate false positives.
