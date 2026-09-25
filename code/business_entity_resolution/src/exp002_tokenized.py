#!/usr/bin/env python3
"""
EXP-002: Multi-Granularity Tokenization & Normalization + Strip-Mall & IDF Precision Guards.
Evaluates blocking recall ceiling, candidate count, Macro F_0.5 threshold calibration,
side-by-side comparison vs EXP-001, and residual error analysis on dataset/val/.
"""

import os
import sys
import time
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from blocking import CountryPartitionedBlocker
from data_loader import EntityRecord, load_ground_truth_tsv, load_source_tsv
from matcher import compute_heuristic_pair_score
from utils.metrics import evaluate_blocking_recall, evaluate_predictions


def main():
    val_dir = os.path.join(PROJECT_ROOT, "dataset/val")
    t0 = time.time()

    print("=" * 80)
    print("🚀 RUNNING EXP-002: MULTI-GRANULARITY TOKENIZATION + PRECISION-GUARDED MATCHER")
    print("=" * 80)

    # 1. Load Validation Data
    print("\n[1/5] Loading validation dataset...")
    s1_records = load_source_tsv(os.path.join(val_dir, "val_source1.tsv"))
    s2_records = load_source_tsv(os.path.join(val_dir, "val_source2.tsv"))
    s3_records = load_source_tsv(os.path.join(val_dir, "val_source3.tsv"))
    ground_truth = load_ground_truth_tsv(os.path.join(val_dir, "val_ground_truth.tsv"))

    cand_pool: Dict[str, EntityRecord] = {}
    cand_pool.update(s2_records)
    cand_pool.update(s3_records)
    print(f"  Loaded S1: {len(s1_records):,} | S2+S3 Pool: {len(cand_pool):,} ({time.time() - t0:.2f}s)")

    # 2. Fit Upgraded Multi-Granularity Blocker & Generate Candidates
    t_block = time.time()
    print("\n[2/5] Building Multi-Granularity Inverted Index & Generating Candidates (top_k=35)...")
    blocker = CountryPartitionedBlocker(max_postings_per_key=2500, top_k=35)
    blocker.fit(s2_records, s3_records)
    candidates_list_map = blocker.generate_candidates(s1_records)
    candidates_set_map = {k: set(v) for k, v in candidates_list_map.items()}

    block_stats = evaluate_blocking_recall(ground_truth, candidates_set_map)
    print(f"  Blocking completed in {time.time() - t_block:.2f}s")
    print(f"  --> EXP-002 Blocking Recall Ceiling: {block_stats['recall_ceiling'] * 100:.2f}% "
          f"({block_stats['captured_links']:,} / {block_stats['total_true_links']:,} true links) "
          f"[EXP-001 was 91.60%]")
    print(f"  --> Avg Candidates per S1:           {block_stats['avg_candidates_per_s1']:.2f} "
          f"(Total pairs: {block_stats['total_candidates']:,})")

    # 3. Score Candidate Pairs
    t_score = time.time()
    print("\n[3/5] Scoring candidate pairs with EXP-002 Matcher...")
    scored_pairs: Dict[str, List[Tuple[str, float]]] = {}
    for s1_id, cand_ids in candidates_list_map.items():
        s1_rec = s1_records[s1_id]
        c_df = blocker.token_df.get(s1_rec.country)
        pair_scores = []
        for cid in cand_ids:
            cand_rec = cand_pool[cid]
            score = compute_heuristic_pair_score(s1_rec, cand_rec, country_token_df=c_df)
            pair_scores.append((cid, score))
        scored_pairs[s1_id] = pair_scores
    print(f"  Scoring completed in {time.time() - t_score:.2f}s")

    # 4. Threshold Calibration Sweep for Macro F_0.5
    print("\n[4/5] Sweeping Decision Thresholds for Macro F_0.5 Optimization:")
    print("-" * 85)
    print(f"{'Threshold':>10} | {'Macro F_0.5':>12} | {'Precision':>10} | {'Recall':>10} | {'Singleton Acc':>14} | {'Non-Sing F0.5':>14}")
    print("-" * 85)

    best_thresh = 82.0
    best_metrics = None
    thresholds = [74.0, 76.0, 78.0, 80.0, 81.0, 82.0, 83.0, 84.0, 86.0, 88.0]

    for tau in thresholds:
        preds: Dict[str, Set[str]] = {}
        for s1_id, pairs in scored_pairs.items():
            preds[s1_id] = {cid for cid, sc in pairs if sc >= tau}

        m = evaluate_predictions(ground_truth, preds)
        print(
            f"{tau:>10.1f} | {m['macro_f05']:>12.4f} | {m['macro_precision']:>10.4f} | "
            f"{m['macro_recall']:>10.4f} | {m['singleton_accuracy']*100:>13.2f}% | {m['non_singleton_f05']:>14.4f}"
        )
        if best_metrics is None or m["macro_f05"] > best_metrics["macro_f05"]:
            best_metrics = m
            best_thresh = tau

    print("-" * 85)
    print(f"🏆 EXP-002 Best Threshold: tau = {best_thresh:.1f} --> Macro F_0.5 = {best_metrics['macro_f05']:.4f} "
          f"(Precision: {best_metrics['macro_precision']:.4f}, Recall: {best_metrics['macro_recall']:.4f}, "
          f"Singleton Acc: {best_metrics['singleton_accuracy']*100:.2f}%)")

    # 5. Residual Error Analysis
    print("\n[5/5] Residual Error Analysis (Where EXP-002 Still Lacks):")
    blocking_misses = []
    false_positives = []
    false_negatives = []

    for s1_id, true_set in ground_truth.items():
        cand_scores = dict(scored_pairs.get(s1_id, []))
        cand_set = set(cand_scores.keys())
        pred_set = {cid for cid, sc in cand_scores.items() if sc >= best_thresh}

        for tid in (true_set - cand_set):
            if len(blocking_misses) < 5:
                blocking_misses.append((s1_id, tid))

        for fpid in (pred_set - true_set):
            if len(false_positives) < 5:
                false_positives.append((s1_id, fpid, cand_scores[fpid]))

        for fnid in ((true_set & cand_set) - pred_set):
            if len(false_negatives) < 5:
                false_negatives.append((s1_id, fnid, cand_scores[fnid]))

    print("\n  A. Sample Remaining BLOCKING MISSES:")
    for s1_id, tid in blocking_misses:
        r1, r2 = s1_records[s1_id], cand_pool[tid]
        print(f"     S1 [{s1_id}] ({r1.country}): Name={r1.business_name!r} | Addr={r1.business_address!r}")
        print(f"     GT [{tid}] ({r2.country}): Name={r2.business_name!r} | Addr={r2.business_address!r}\n")

    print(f"  B. Sample Remaining FALSE POSITIVES (Scored >= {best_thresh:.1f}):")
    for s1_id, fpid, sc in false_positives:
        r1, r2 = s1_records[s1_id], cand_pool[fpid]
        print(f"     Score={sc:.1f} | S1 [{s1_id}]: Name={r1.business_name!r} | Addr={r1.business_address!r}")
        print(f"               | FP [{fpid}]: Name={r2.business_name!r} | Addr={r2.business_address!r}\n")

    print(f"  C. Sample Remaining FALSE NEGATIVES (Scored < {best_thresh:.1f}):")
    for s1_id, fnid, sc in false_negatives:
        r1, r2 = s1_records[s1_id], cand_pool[fnid]
        print(f"     Score={sc:.1f} | S1 [{s1_id}]: Name={r1.business_name!r} | Addr={r1.business_address!r}")
        print(f"               | FN [{fnid}]: Name={r2.business_name!r} | Addr={r2.business_address!r}\n")


if __name__ == "__main__":
    main()
