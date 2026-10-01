# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Rishant Gupta  
**Team Members:** Rishant Gupta  
**Submission Date:** September 28, 2026  

---

## 1. Executive Summary

We present an end-to-end, high-precision Business Entity Resolution system designed to resolve noisy, unstandardized business entities across heterogeneous data sources into a canonical reference entity catalog. Our architecture combines a scale-invariant, multi-channel inverted index with zero-copy integer posting lists for candidate blocking, a 50-dimensional pairwise and sibling-cluster feature extraction pipeline, a hierarchical Gradient Boosted Decision Tree (GBDT) ensemble (GPU XGBoost + LightGBM), and a Stage-2 Group-Context Meta-Reranker. To optimize directly for the Macro $F_{0.5}$ metric (which penalizes precision loss twice as heavily as recall), we introduce Global Bipartite Mutual Exclusion to enforce disjoint entity ownership across test partitions alongside metric-calibrated dual-threshold decision rules, achieving a 94.0% – 95.2% score on the official competition leaderboard.

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory data analysis across the 1.80M training records and 1.73M test records revealed several foundational domain characteristics and noise patterns:
1. **Strict Country Partitioning**: Evaluating 7.64M ground-truth links confirmed that 100.00% of true matches occur strictly within the same country partition (`US`, `India`, `France`). No cross-border matches exist.
2. **Cardinality & Singleton Distribution**: Approximately 5.8% of Source 1 reference entities are pure singletons (zero corresponding records in Source 2 or Source 3). Non-singletons link to an average of 3.46 records (peaking at 3–4 matches). In the Macro $F_{0.5}$ metric, predicting an empty list for a true singleton yields a perfect 1.0 score, whereas any spurious false-positive assignment drops that entity's score to 0.0.
3. **Severe Multi-Lingual & Cross-Script Noise**: Indian records frequently exhibit script divergence where the reference entity is in Latin English but Source 2/3 entities are written in Indic scripts (Devanagari, Gurmukhi, Bengali, Odia) or phonetic colloquial spellings.
4. **Structural Commercial Address Noise**: High-density physical locations (e.g., commercial complexes, strip malls, industrial parks) host dozens of distinct corporations sharing identical postal addresses and PIN codes. Naive address-matching models produce disastrous false-positive clusters on such entities.
5. **Zero-Shot Country Generalization (France)**: The test dataset introduced France (~15% of all test entities, 259,452 Source 1 records) with zero training examples. Features and normalizations had to generalize across French legal entity structures (`SARL`, `SAS`, `EURL`, `SCI`) and address tokens (`rue`, `avenue`, `bd`, `impasse`).

### 2.2 Solution Strategy

To balance recall, precision, and bounded memory consumption (< 8 GB RAM) across 17+ trillion possible pairwise comparisons, we designed a two-stage hierarchical pipeline:

**Approach Type:** Multi-Channel Blocking + Two-Stage GBDT Reranking + Global Bipartite Mutual Exclusion  
**Core Innovations:**
- **Zero-COW (Copy-on-Write) Inverted Indexing**: Storing candidate postings as `array('I')` unsigned 32-bit integer arrays (4 bytes/candidate) rather than Python object sets, preventing memory paging thrashing across parallel worker processes.
- **Phonetic & Consonant Skeleton Alignment**: Transliteration of 9 Indic scripts and phonetic skeleton hashing to bridge English-Devanagari and colloquial phonetic variations.
- **Strip-Mall False Positive Guards**: Specialized pairwise feature logic enforcing core name similarity floors whenever physical addresses overlap identically.
- **Stage-2 Group-Context Meta-Reranker**: A second-stage model that consumes relative probability ranks, score margins, and candidate drop-off ratios to make holistic group decisions rather than independent pairwise classifications.
- **Global Bipartite Mutual Exclusion**: Post-processing optimization enforcing the single-parent property (each Source 2/3 record belongs to at most one Source 1 entity).

---

## 3. Candidate Generation (Blocking)

### 3.1 Blocking Keys Used
To achieve high recall without memory explosion, we implemented a multi-granularity inverted index generating complementary key channels:
- `ns:` **Spaceless Business Name**: Exact concatenated cleaned business names (rescuing spacing/punctuation splits).
- `nc:` **Name + Location Compound**: Concatenation of the first significant name token and city/state.
- `ps:` / `ph:` **Phonetic Consonant Skeleton**: Sound-based consonant skeleton representations of significant tokens.
- `sh:` **Sorted 2-Word Shingles**: Ordered pairs of distinctive name tokens.
- `af:` **Address Fingerprints**: 3-token normalized address anchor n-grams.
- `an:` **Address Number-Word Anchors**: Street building numbers paired with distinctive street words.
- `nt:` **Distinctive Word Tokens**: Filtered against common legal stopwords with logarithmic inverse document frequency (IDF) attenuation.

### 3.2 Candidate Pairs Generated
- **Retrieval Depth**: Top-30 candidates per Source 1 entity ($k = 30$).
- **Total Test Candidates Generated**: Across 1,732,544 test entities, exactly 51,976,320 candidate evaluations were performed.
- **Search Space Reduction**: Pruned the theoretical comparison space of $1.73\text{M} \times 9.97\text{M} \approx 1.72 \times 10^{13}$ pairs down to $5.2 \times 10^7$ pairs — a **99.9997% reduction** in search space.

### 3.3 Ensuring True Matches Were Not Lost
- **Multi-Channel Quota Allocation**: To prevent high-frequency token matches from crowding out high-precision address or phonetic matches, top-$k$ retrieval enforces channel quotas: 58% cross-field synergy, 16% core name tokens, 12% phonetic skeletons, and 14% street address anchors.
- **Scale-Invariant Token Scaling**: Document frequencies in the inverted index are scaled relative to country pool size, ensuring identical key selectivity across datasets of varying magnitude.
- **Validation Blocker Recall**: Verified on the full-pool 20,000 validation benchmark against 10.3M candidate records, achieving **99.92% recall** (68,732 / 68,785 ground truth links retained).

---

## 4. Matching Model

### 4.1 Features Used (68 Features Total: 50 Pairwise + 18 Group Context)

#### Name Features (22 Features):
- RapidFuzz token sort ratio, token set ratio, partial token ratio, and Levenshtein similarity.
- Spaceless string similarity and character 4-gram overlap.
- Phonetic skeleton edit distance and consonant sequence match.
- Exact initials acronym match (e.g., "BMH" $\leftrightarrow$ "Beloit Memorial Hospital").
- Synthetic generic-word substitution indicator (e.g., "Service" / "Solutions" substituting distinctive words).
- Legal entity suffix compatibility (exact match, soft variation, hard class conflict).
- Synthetic coined-alias vocabulary filter (handling 48 synthetic prefix stems).

#### Address Features (18 Features):
- Pure street token sort and token set ratios (excluding city, state, postal code).
- Street number matching: exact primary number match, unit/suite shifted number match, $\pm 2$ address perturbation match.
- Numeric multiset Jaccard similarity and shared 3+ digit number count.
- Postal code / PIN code exact and prefix compatibility.
- City and state conflict penalties.

#### Disambiguation & Context Features (10 Features):
- Strip-mall false positive guard (flags identical address with zero core name token overlap).
- Sibling address cluster consensus and candidate cluster size.
- S1 same-name different-address count (separating multi-branch franchises from single-location entities).

#### Stage-2 Group-Context Meta Features (18 Features):
- Candidate Stage-1 probability rank within the Source 1 candidate group.
- Top-1 probability, score margin to top candidate ($\Delta = p_1 - p_i$), and ratio to top candidate ($p_i / p_1$).
- Group size, group score mean, group score variance, and top-2 score margin.

### 4.2 Model Type & Training Configuration
- **Stage 1 (Pairwise Classifier)**: Ensemble combining GPU-accelerated XGBoost (`tree_method="hist"`, `device="cuda"`, max_depth=8) and LightGBM (num_leaves=63, learning_rate=0.04). Trained on 50 pairwise features over both pure validation distributions and full-pool hard-negative candidate pairs.
- **Stage 2 (Group-Context Meta-Reranker)**: LightGBM and XGBoost rerankers trained on 68 combined features using 5-Fold GroupKFold (grouping strictly by `source1_entity_id`) to prevent data leakage.
- **Ensemble Blend**: Final link probability is computed as $P = 0.80 \times P_{\text{Stage2}} + 0.20 \times P_{\text{Stage1}}$.

### 4.3 Threshold Selection & Post-Processing
- **Metric Calibration for $F_{0.5}$**:
  Standard $0.50$ probability thresholds produce catastrophic precision degradation on Macro $F_{0.5}$. We calibrated country-specific dual-threshold parameters:
  - US: $\tau_{\text{top1}} = 0.72$, $\tau_{\text{secondary}} = 0.72$, $\text{ratio} \ge 0.76$, $\tau_{\text{consensus}} = 0.68$
  - India: $\tau_{\text{top1}} = 0.74$, $\tau_{\text{secondary}} = 0.74$, $\text{ratio} \ge 0.76$, $\tau_{\text{consensus}} = 0.70$
  - France: $\tau_{\text{top1}} = 0.78$, $\tau_{\text{secondary}} = 0.76$, $\text{ratio} \ge 0.80$, $\tau_{\text{consensus}} = 0.74$
- **Global Bipartite Mutual Exclusion**:
  Because each ground-truth Source 2/3 entity belongs to at most one Source 1 parent, whenever multiple Source 1 entities claim the same candidate ID, only the entity with the highest probability retains the claim. This eliminated 49,655 cross-entity duplicate false positives across the test set.

---

## 5. Results & Error Analysis

### 5.1 Progression & Metric Scores

| Pipeline Stage | Blocker Recall | Precision | Recall | Singleton Accuracy | Macro $F_{0.5}$ | Official Live LB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (EXP-001)** | 91.60% | 0.9212 | 0.8603 | 82.47% | 0.8938 | — |
| **Multi-Granularity Tokenizer (EXP-002)** | 97.94% | 0.9501 | 0.9186 | 86.47% | 0.9340 | — |
| **GBDT Pairwise Scoring (EXP-003)** | 97.97% | 0.9823 | 0.9482 | 95.06% | 0.9701 | — |
| **Initial Full-Pool Baseline (EXP-005 Full)** | 90.75% | 0.8403 | 0.8765 | 53.28% | 0.8299 | **82.0%** |
| **Full-Pool Two-Stage GBDT (EXP-006)** | 96.43% | 0.9778 | 0.9132 | 96.17% | 0.9571 | **94.0%** |
| **50-Feature + Bipartite Ensemble (EXP-007b)** | **99.92%** | **0.9832** | **0.9494** | **95.66%** | **0.9719** | **95.2%** |

### 5.2 Common False Positives (Wrong Merges) & Mitigations
- **Physical Co-location / Strip-Malls**: Separate businesses operating at the same street address (e.g., medical clinics, retail plazas) sharing postal codes.  
  *Mitigation*: The `strip_mall_guard` and `pure_street_sort_sim` features enforce hard name token similarity thresholds when addresses are identical.
- **Franchise / Branch Name Collisions**: Corporations with identical brand names operating across different cities.  
  *Mitigation*: The `s1_same_name_diff_addr_cnt` feature penalizes ambiguous candidate groups that share name tokens but exhibit geographic divergence.
- **Cross-Entity Multi-Claiming**: A candidate record being linked to multiple plausible Source 1 parents.  
  *Mitigation*: Global Bipartite Mutual Exclusion pruned over 49,000 duplicate links, ensuring strictly 1-to-many disjoint entity ownership.

### 5.3 Common False Negatives (Missed Matches) & Mitigations
- **Cross-Script Transliteration**: Indian business names written in Devanagari or other Indic scripts diverging from Latin English transliterations.  
  *Mitigation*: Preprocessing transliterates 9 Indic scripts and extracts phonetic consonant skeletons, boosting blocking recall from 90.75% to >99%.
- **Synthetically Duplicated Words & Legal Suffix Reordering**: Synthetic noise duplicating words (e.g., "Kanchenjunga Kanchenjunga Buildcon") or reordering legal entity types.  
  *Mitigation*: Automatic deduplication of consecutive identical word tokens and comprehensive normalization of legal corporate classes across US, India, and France.

---

## 6. Conclusion

By systematically diagnosing error modes across full candidate pools rather than artificial sub-samples, we transformed business entity resolution from an unscalable $O(N^2)$ problem into a high-throughput, memory-bounded pipeline. The combination of scale-invariant inverted indexing with `array('I')` posting buffers, 50-dimensional domain feature engineering, two-stage GBDT reranking, and Global Bipartite Mutual Exclusion delivered high accuracy (94.0% – 95.2% leaderboard Macro $F_{0.5}$) while running efficiently within 16 GB RAM and completing full 1.73M entity test inference in under 90 minutes.

---

## Appendix

### A. Code Artefacts & Structure

The complete runnable codebase is self-contained in `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── README.md                 # Step-by-step reproduction instructions
├── requirements.txt          # Pinned dependency environment
└── src/
    ├── run_pipeline.py       # Main end-to-end execution CLI entrypoint
    ├── preprocessor.py       # Text cleaning, Indic transliteration, legal suffix normalization
    ├── blocking.py           # Inverted index blocking, key extraction, channel quotas
    ├── features.py           # 50-dimensional pairwise & sibling feature extraction
    ├── matcher.py            # Heuristic similarity scoring and string matching logic
    ├── data_loader.py        # Streaming TSV loaders and EntityRecord data structures
    ├── exp005_postproc_reranker.py # Stage-2 group context feature generation
    ├── exp007_gpu_pipeline.py# 5-Fold GroupKFold model training & validation pipeline
    └── generate_submission_exp007.py # Official test submission generation engine
```

**Single Command Reproduction**:
```bash
python3 src/run_pipeline.py \
    --test-dir ../../dataset/test \
    --output-dir ../../output \
    --model-dir ../../output/models \
    --num-workers 4
```

### B. Output Submission Verification

Both required submission files (`matching_results.tsv` and `candidate_pairs.tsv`) have been verified using the official submission validator:

```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (110004 empty, 1622540 non-empty).
  candidate_pairs.tsv: 1732544 rows (1 empty, 1732543 non-empty).

PASS — no blocking issues found. Safe to submit.
```
*(100% ID compliance, zero missing entities, zero invalid IDs, and matching results are a strict subset of candidate pairs).*
