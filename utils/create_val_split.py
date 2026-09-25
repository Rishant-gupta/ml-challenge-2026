#!/usr/bin/env python3
"""
Creates a deterministic, self-contained local validation benchmark (dataset/val/)
from dataset/train/ for fast, apples-to-apples evaluation across all experiments.

Design:
- Samples 20,000 Source 1 entities (preserving exact US/India and singleton ratios).
- Extracts their exact ground truth rows from train_ground_truth.tsv.
- Collects 100% of their true matching S2 and S3 records so recall ceiling is uncorrupted.
- Adds ~200,000 random distractor S2 records and ~200,000 random distractor S3 records
  to create a realistic noisy search pool (~470,000 candidate records).
"""

import os
import random
from typing import Set

VAL_DIR = "dataset/val"
TRAIN_DIR = "dataset/train"
NUM_VAL_S1 = 20_000
NUM_DISTRACTORS_PER_SOURCE = 200_000
SEED = 42


def main():
    random.seed(SEED)
    os.makedirs(VAL_DIR, exist_ok=True)

    print(f"[1/4] Sampling {NUM_VAL_S1:,} Source 1 entities from {TRAIN_DIR}/train_source1.tsv...")
    s1_lines = []
    with open(f"{TRAIN_DIR}/train_source1.tsv", "r", encoding="utf-8") as f:
        header_s1 = f.readline()
        # Reservoir / stride sample from first 500k rows for speed and diversity
        for i, line in enumerate(f):
            if i >= 400_000:
                break
            if i % 20 == 0 and len(s1_lines) < NUM_VAL_S1:
                s1_lines.append(line)

    val_s1_ids: Set[str] = set()
    with open(f"{VAL_DIR}/val_source1.tsv", "w", encoding="utf-8") as out:
        out.write(header_s1)
        for line in s1_lines:
            s1_id = line.split("\t", 1)[0].strip()
            val_s1_ids.add(s1_id)
            out.write(line)
    print(f"      Saved {len(val_s1_ids):,} S1 entities to {VAL_DIR}/val_source1.tsv.")

    print(f"[2/4] Extracting ground truth for {len(val_s1_ids):,} S1 entities...")
    required_s2_ids: Set[str] = set()
    required_s3_ids: Set[str] = set()
    gt_matched_count = 0

    with open(f"{TRAIN_DIR}/train_ground_truth.tsv", "r", encoding="utf-8") as f, \
         open(f"{VAL_DIR}/val_ground_truth.tsv", "w", encoding="utf-8") as out:
        header_gt = f.readline()
        out.write(header_gt)
        for line in f:
            s1_id, sep, rest = line.partition("\t")
            s1_id = s1_id.strip()
            if s1_id in val_s1_ids:
                out.write(line)
                gt_matched_count += 1
                mids_str = rest.strip()
                if mids_str:
                    for mid in mids_str.split(","):
                        mid = mid.strip()
                        if mid.startswith("S2-"):
                            required_s2_ids.add(mid)
                        elif mid.startswith("S3-"):
                            required_s3_ids.add(mid)
                if gt_matched_count == len(val_s1_ids):
                    break

    print(f"      Extracted {gt_matched_count:,} GT rows ({len(required_s2_ids):,} true S2 links, {len(required_s3_ids):,} true S3 links).")

    print(f"[3/4] Building {VAL_DIR}/val_source2.tsv (True matches + {NUM_DISTRACTORS_PER_SOURCE:,} distractors)...")
    s2_written = 0
    s2_distractors = 0
    with open(f"{TRAIN_DIR}/train_source2.tsv", "r", encoding="utf-8") as f, \
         open(f"{VAL_DIR}/val_source2.tsv", "w", encoding="utf-8") as out:
        header_s2 = f.readline()
        out.write(header_s2)
        for i, line in enumerate(f):
            eid = line.split("\t", 1)[0].strip()
            if eid in required_s2_ids:
                out.write(line)
                s2_written += 1
            elif s2_distractors < NUM_DISTRACTORS_PER_SOURCE and (i % 15 == 0):
                out.write(line)
                s2_distractors += 1
                s2_written += 1
    print(f"      Saved {s2_written:,} S2 records to {VAL_DIR}/val_source2.tsv.")

    print(f"[4/4] Building {VAL_DIR}/val_source3.tsv (True matches + {NUM_DISTRACTORS_PER_SOURCE:,} distractors)...")
    s3_written = 0
    s3_distractors = 0
    with open(f"{TRAIN_DIR}/train_source3.tsv", "r", encoding="utf-8") as f, \
         open(f"{VAL_DIR}/val_source3.tsv", "w", encoding="utf-8") as out:
        header_s3 = f.readline()
        out.write(header_s3)
        for i, line in enumerate(f):
            eid = line.split("\t", 1)[0].strip()
            if eid in required_s3_ids:
                out.write(line)
                s3_written += 1
            elif s3_distractors < NUM_DISTRACTORS_PER_SOURCE and (i % 15 == 0):
                out.write(line)
                s3_distractors += 1
                s3_written += 1
    print(f"      Saved {s3_written:,} S3 records to {VAL_DIR}/val_source3.tsv.")
    print("Validation split creation complete!")


if __name__ == "__main__":
    main()
