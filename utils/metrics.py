#!/usr/bin/env python3
"""
Evaluation metrics for Business Entity Resolution Challenge.
Computes Macro F_0.5, Macro Precision, Macro Recall, and Singleton statistics.
"""

from typing import Dict, List, Set, Tuple


def compute_entity_f05(
    true_matches: Set[str],
    pred_matches: Set[str],
) -> Tuple[float, float, float]:
    """
    Computes (f05, precision, recall) for a single Source 1 entity.
    
    Rules according to challenge spec:
    - If true_matches is empty (singleton):
        - if pred_matches is empty -> F_0.5 = 1.0, precision = 1.0, recall = 1.0
        - if pred_matches is non-empty -> F_0.5 = 0.0, precision = 0.0, recall = 1.0
    - If true_matches is non-empty:
        - if pred_matches is empty -> F_0.5 = 0.0, precision = 0.0, recall = 0.0
        - otherwise:
            precision = |pred & true| / |pred|
            recall = |pred & true| / |true|
            if precision + recall == 0:
                F_0.5 = 0.0
            else:
                F_0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)
    """
    if len(true_matches) == 0:
        if len(pred_matches) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 1.0
    else:
        if len(pred_matches) == 0:
            return 0.0, 0.0, 0.0
        tp = len(true_matches & pred_matches)
        precision = tp / len(pred_matches)
        recall = tp / len(true_matches)
        denominator = 0.25 * precision + recall
        if denominator == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * precision * recall) / denominator
        return f05, precision, recall


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> Dict[str, float]:
    """
    Evaluates predictions dictionary vs ground_truth dictionary.
    Both dictionaries map source1_entity_id -> set of matched_entity_ids.
    """
    s1_entities = sorted(list(ground_truth.keys()))
    if not s1_entities:
        return {"macro_f05": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "count": 0}

    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0

    singleton_count = 0
    singleton_correct = 0

    non_singleton_count = 0
    non_singleton_f05 = 0.0

    for s1_id in s1_entities:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        f05, prec, rec = compute_entity_f05(true_set, pred_set)
        total_f05 += f05
        total_prec += prec
        total_rec += rec

        if len(true_set) == 0:
            singleton_count += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            non_singleton_count += 1
            non_singleton_f05 += f05

    n = len(s1_entities)
    return {
        "macro_f05": total_f05 / n,
        "macro_precision": total_prec / n,
        "macro_recall": total_rec / n,
        "total_entities": n,
        "singleton_count": singleton_count,
        "singleton_accuracy": singleton_correct / max(1, singleton_count),
        "non_singleton_count": non_singleton_count,
        "non_singleton_f05": non_singleton_f05 / max(1, non_singleton_count),
    }


def evaluate_blocking_recall(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
) -> Dict[str, float]:
    """
    Evaluates blocking quality:
    - Recall ceiling: fraction of all true links captured in candidate sets
    - Average candidates per entity
    """
    total_true_links = 0
    captured_links = 0
    total_candidates = 0

    for s1_id, true_set in ground_truth.items():
        total_true_links += len(true_set)
        cand_set = candidates.get(s1_id, set())
        captured_links += len(true_set & cand_set)
        total_candidates += len(cand_set)

    n = len(ground_truth)
    recall_ceiling = captured_links / max(1, total_true_links)
    avg_candidates = total_candidates / max(1, n)

    return {
        "recall_ceiling": recall_ceiling,
        "total_true_links": total_true_links,
        "captured_links": captured_links,
        "total_candidates": total_candidates,
        "avg_candidates_per_s1": avg_candidates,
    }


if __name__ == "__main__":
    # Test example from challenge PDF (page 6)
    # Ground truth: S1-00001 -> [S2-00047, S3-00812]
    # Prediction: S1-00001 -> [S2-00047, S2-00193, S3-00812]
    # Precision = 2/3 = 0.6667, Recall = 2/2 = 1.0, F0.5 = 0.7143
    gt = {"S1-00001": {"S2-00047", "S3-00812"}}
    pr = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
    res = evaluate_predictions(gt, pr)
    print("Self-test PDF example result:", res)
    assert abs(res["macro_f05"] - 0.7142857) < 1e-4, "Self-test failed!"
    print("Self-test verified successfully.")
