# Business Entity Resolution Pipeline

This repository contains the self-contained, end-to-end Machine Learning pipeline for the Business Entity Resolution Challenge.

## Pipeline Architecture
1. **Data Ingestion**: Reads tab-separated source records from Source 1, Source 2, and Source 3.
2. **Preprocessing & Normalization**: Strips legal suffixes, normalizes diacritics/casing, standardizes address tokens.
3. **Blocking / Candidate Generation**: Country-partitioned multi-index candidate generator producing `output/candidate_pairs.tsv`.
4. **Feature Extraction**: Pairwise string distances (Levenshtein, Jaro-Winkler, token sort/set ratio), address token overlap, numeric matches.
5. **Matching & Scoring**: GBDT pairwise scoring model predicting match probability.
6. **Precision-Tuned Thresholding**: Postprocessing calibrated for Macro $F_{0.5}$ with singleton protection.
7. **Output Export & Validation**: Writes `output/matching_results.tsv` and verifies against `utils/validate_submission.py`.

## Setup Instructions

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Running End-to-End Inference

To run the complete pipeline on the test dataset and generate submission outputs:

```bash
python3 src/run_pipeline.py \
    --train-dir ../../dataset/train \
    --test-dir ../../dataset/test \
    --output-dir ../../output
```

## Validating Outputs

```bash
python3 ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```
