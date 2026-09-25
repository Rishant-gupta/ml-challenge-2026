# Business Entity Resolution Challenge 2026

An end-to-end Machine Learning solution for linking noisy, unstandardized business records across heterogeneous sources to a deduplicated reference entity base.

---

## 📌 Repository Structure

```
ML challenge 2026/
├── dataset/
│   ├── train/                         # Training files (train_source1/2/3.tsv, train_ground_truth.tsv)
│   └── test/                          # Test files (test_source1/2/3.tsv)
├── output/
│   ├── matching_results.tsv           # Final entity matches (submitted for leaderboard evaluation)
│   └── candidate_pairs.tsv            # Candidate pairs generated during blocking
├── utils/
│   ├── validate_submission.py         # Standard library submission format validator
│   └── metrics.py                     # Official Macro F_0.5 evaluation metric suite
├── code/
│   └── business_entity_resolution/
│       ├── src/                       # Modular pipeline implementation
│       ├── README.md                  # Detailed reproduction guide
│       └── requirements.txt           # Pinned dependencies
├── Documentation_template.md          # Official methodology documentation write-up
├── PROJECT_MEMORY.md                  # Project Memory & Time Machine Capsule (progress, learnings, state)
├── EXPERIMENTS.md                     # Experiment Roadmap, hypotheses backlog, and trial logs
├── Untitled.pdf                       # Original challenge specification document
└── README.md                          # This file
```

---

## 🧭 Navigation & Core Documents

- **[PROJECT_MEMORY.md](PROJECT_MEMORY.md)**: Our **Time Machine Capsule**. Detailed timeline of every milestone, architectural decision, domain learning, and system evolution.
- **[EXPERIMENTS.md](EXPERIMENTS.md)**: Scientific experiment log, hypothesis backlog (blocking, feature engineering, modeling, postprocessing), and evaluation tracking.
- **[Documentation_template.md](Documentation_template.md)**: Methodology write-up template required for the final submission package.
- **[Validation Script](utils/validate_submission.py)**: Zero-dependency validator ensuring exact compliance with challenge schema and submission constraints.
- **[Metrics Suite](utils/metrics.py)**: Standalone Python implementation of the challenge's Macro $F_{0.5}$ metric (including strict singleton evaluation).

---

## 🎯 Key Challenge Highlights & Constraints

1. **Problem**: Match noisy entity records from Source 2 ($S_2$) and Source 3 ($S_3$) against deduplicated reference records in Source 1 ($S_1$).
2. **Metric**: **Macro $F_{0.5}$** across all $S_1$ entities. Precision is weighted $2\times$ over recall. Singletons must be accurately identified (predicting an empty match list yields $1.0$, while any false positive yields $0.0$).
3. **Open-Set Country**: Training set contains `US` and `India`; Test set introduces `France`. Country labels must not be hardcoded or filtered out.
4. **Fair Play**: Zero external data lookup (no geocoding APIs, no web lookups, no commercial ER services).
5. **Model Limit**: Permissible under MIT/Apache 2.0 license, up to 8 Billion parameters.

---

## ⚡ Quick Validation

To validate your generated output files prior to leaderboard submission:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
