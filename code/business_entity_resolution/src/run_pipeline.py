#!/usr/bin/env python3
"""
Business Entity Resolution — End-to-End Execution Pipeline
ML Challenge 2026

Main reproduction entry point for the business entity resolution system.
Handles candidate generation (blocking), pairwise & group feature extraction,
hierarchical GBDT model scoring, and global bipartite mutual exclusion.

Usage:
    # 1. Run full test set inference and generate official outputs:
    python3 src/run_pipeline.py --test-dir ../../dataset/test --output-dir ../../output

    # 2. Retrain models from scratch using training data:
    python3 src/run_pipeline.py --mode train --train-dir ../../dataset/train
"""

import argparse
import gc
import heapq
import math
import os
import subprocess
import sys
import time
from array import array
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure project root and src/ are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from blocking import extract_blocking_keys
from data_loader import EntityRecord
from exp005_postproc_reranker import STAGE2_CONTEXT_FEATURE_NAMES, build_stage2_context_features
from features import FEATURE_NAMES, extract_features_for_s1_group
from preprocessor import (
    clean_address,
    clean_business_name,
    extract_name_variants,
    phonetic_skeleton_word,
)

ALL_STAGE2_FEATURE_NAMES = list(FEATURE_NAMES) + STAGE2_CONTEXT_FEATURE_NAMES

# Shared read-only worker globals (populated prior to worker fork, frozen via gc.freeze)
_G_C_IDX: Dict[str, array] = {}
_G_TOKEN_DF: Dict[str, int] = {}
_G_CAND_EIDS: List[str] = []
_G_CAND_NAMES: List[str] = []
_G_CAND_ADDRS: List[str] = []
_G_COUNTRY: str = ""
_G_TOP_K: int = 30

# Worker-local models loaded post-fork
_W_S1_LGB_PURE = None
_W_S1_LGB_EXT = None
_W_S1_XGB_PURE = None
_W_S1_XGB_PURE_END: int = 0
_W_S1_XGB_EXT = None
_W_S1_XGB_EXT_END: int = 0
_W_S2_LGB = None
_W_S2_XGB = None
_W_S2_XGB_END: int = 0
_W_XGB_MOD = None

# Base weights for multi-channel blocking query resolution
_PREFIX_BASE_W = {
    "ns:": 3.5,  # Spaceless business name (high precision)
    "nc:": 3.5,  # Name + City compound
    "ps:": 2.8,  # Phonetic consonant skeleton
    "af:": 2.8,  # Address fingerprint
    "sh:": 2.2,  # Word shingles
    "an:": 2.2,  # Address number-word anchors
    "nt:": 1.5,  # Raw business name tokens
    "aw:": 1.5,  # Address word-pair shingles
    "pc:": 1.5,  # Postal codes
    "ph:": 1.25, # Secondary phonetic keys
    "ng:": 0.35, # 5-gram subwords
    "np:": 0.45,
}

# Decision thresholds tuned on validation split for Macro F_0.5:
# (tau_top1, tau_sec, min_ratio, rescue_consensus_tau, dedup_margin)
# Note: F_0.5 weights precision 2x over recall (beta=0.5 -> beta^2=0.25).
# A strict top1 threshold avoids costly false positive singletons.
_COUNTRY_THRESHOLDS = {
    "US": (0.72, 0.72, 0.76, 0.68, 0.0),
    "India": (0.74, 0.74, 0.76, 0.70, 0.0),
    "France": (0.78, 0.76, 0.80, 0.74, 0.0),  # Extra precision guard for zero-shot country
}


def _clear_caches() -> None:
    clean_business_name.cache_clear()
    extract_name_variants.cache_clear()
    clean_address.cache_clear()
    phonetic_skeleton_word.cache_clear()


def _get_xgb_end_round(bst) -> int:
    best_it = bst.attr("best_iteration")
    if best_it is not None:
        return int(best_it) + 1
    return bst.num_boosted_rounds()


def _extract_s1_keys_worker(batch: List[Tuple[str, str, str]]) -> Set[str]:
    country = _G_COUNTRY
    keys: Set[str] = set()
    for s1_id, b_name, b_addr in batch:
        rec = EntityRecord(s1_id, b_name, b_addr, country)
        keys.update(extract_blocking_keys(rec))
    _clear_caches()
    return keys


def _query_candidates(s1_record: EntityRecord) -> List[int]:
    """
    Candidate retrieval using inverted index over array('I') 32-bit integer postings.
    Uses multi-channel scoring + guaranteed quotas across name, phonetic, and address signals.
    """
    c_idx = _G_C_IDX
    if not c_idx:
        return []

    name_scores: Dict[int, float] = defaultdict(float)
    skel_scores: Dict[int, float] = defaultdict(float)
    addr_scores: Dict[int, float] = defaultdict(float)

    for key in extract_blocking_keys(s1_record):
        postings = c_idx.get(key)
        if not postings:
            continue
        df = len(postings)
        prefix = key[:3]
        base_w = _PREFIX_BASE_W.get(prefix, 0.45)
        # IDF attenuation: penalize common keys logarithmically
        w = base_w / (math.log(2.0 + df) ** 2.0)

        if prefix in ("af:", "an:", "aw:", "pc:"):
            for cid in postings:
                addr_scores[cid] += w
        elif prefix == "nc:":
            for cid in postings:
                name_scores[cid] += w
                skel_scores[cid] += w
                addr_scores[cid] += w
        else:
            for cid in postings:
                name_scores[cid] += w
                if prefix in ("ps:", "ph:"):
                    skel_scores[cid] += w

    if not name_scores and not addr_scores:
        return []

    k = _G_TOP_K
    all_cands = set(name_scores.keys()) | set(addr_scores.keys())
    if len(all_cands) <= k:
        return [
            cid
            for cid, _ in sorted(
                ((c, name_scores.get(c, 0.0) + addr_scores.get(c, 0.0)) for c in all_cands),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

    # Cross-field synergy bonus: rewards candidates matching both name and address
    synergy_scores: Dict[int, float] = {}
    for cid in all_cands:
        ns = name_scores.get(cid, 0.0)
        ads = addr_scores.get(cid, 0.0)
        bonus = 0.50 if (ns > 0.0 and ads > 0.0) else 0.0
        synergy_scores[cid] = ns + 0.6 * ads + bonus

    top_syn = [cid for cid, _ in heapq.nlargest(k, synergy_scores.items(), key=lambda x: x[1])]
    top_name = (
        [cid for cid, _ in heapq.nlargest(12, name_scores.items(), key=lambda x: x[1])]
        if name_scores
        else []
    )
    top_skel = (
        [cid for cid, _ in heapq.nlargest(10, skel_scores.items(), key=lambda x: x[1])]
        if skel_scores
        else []
    )
    top_addr = (
        [cid for cid, _ in heapq.nlargest(10, addr_scores.items(), key=lambda x: x[1])]
        if addr_scores
        else []
    )

    selected: List[int] = []
    seen: Set[int] = set()

    def add_from(lst: List[int], limit: int) -> None:
        c = 0
        for cid in lst:
            if cid not in seen:
                seen.add(cid)
                selected.append(cid)
                c += 1
                if c >= limit or len(selected) >= k:
                    break

    # Channel quota distribution
    q_syn = max(1, int(k * 0.58))
    q_name = max(1, int(k * 0.16))
    q_skel = max(1, int(k * 0.12))
    q_addr = max(1, k - q_syn - q_name - q_skel)

    add_from(top_syn, q_syn)
    add_from(top_name, q_name)
    add_from(top_skel, q_skel)
    add_from(top_addr, q_addr)
    if len(selected) < k:
        add_from(top_syn, k - len(selected))

    return selected


def _init_worker(model_dir: str) -> None:
    """
    Initializes models inside each child worker process.
    Forces single-threaded OpenMP/BLAS per worker to prevent CPU thread contention.
    """
    global _W_S1_LGB_PURE, _W_S1_LGB_EXT
    global _W_S1_XGB_PURE, _W_S1_XGB_PURE_END, _W_S1_XGB_EXT, _W_S1_XGB_EXT_END
    global _W_S2_LGB, _W_S2_XGB, _W_S2_XGB_END, _W_XGB_MOD

    os.environ["OMP_NUM_THREADS"] = "1"
    import lightgbm as lgb
    import xgboost as xgb

    _W_XGB_MOD = xgb

    # Stage-1 50-Feature Models
    _W_S1_LGB_PURE = lgb.Booster(model_file=os.path.join(model_dir, "lgbm_exp007_pure_fold0.txt"))
    _W_S1_LGB_PURE.params["num_threads"] = 1

    _W_S1_LGB_EXT = lgb.Booster(model_file=os.path.join(model_dir, "lgbm_exp007_ext_fold1.txt"))
    _W_S1_LGB_EXT.params["num_threads"] = 1

    _W_S1_XGB_PURE = xgb.Booster()
    _W_S1_XGB_PURE.load_model(os.path.join(model_dir, "xgb_exp007_pure_fold2.json"))
    _W_S1_XGB_PURE.set_param({"nthread": 1, "device": "cpu"})
    _W_S1_XGB_PURE_END = _get_xgb_end_round(_W_S1_XGB_PURE)

    _W_S1_XGB_EXT = xgb.Booster()
    _W_S1_XGB_EXT.load_model(os.path.join(model_dir, "xgb_exp007_ext_fold3.json"))
    _W_S1_XGB_EXT.set_param({"nthread": 1, "device": "cpu"})
    _W_S1_XGB_EXT_END = _get_xgb_end_round(_W_S1_XGB_EXT)

    # Stage-2 68-Feature Group Meta-Reranker
    _W_S2_LGB = lgb.Booster(model_file=os.path.join(model_dir, "stage2_lgb_exp007_fold0.txt"))
    _W_S2_LGB.params["num_threads"] = 1

    _W_S2_XGB = xgb.Booster()
    _W_S2_XGB.load_model(os.path.join(model_dir, "stage2_xgb_exp007_fold2.json"))
    _W_S2_XGB.set_param({"nthread": 1, "device": "cpu"})
    _W_S2_XGB_END = _get_xgb_end_round(_W_S2_XGB)


def _predict_batch_worker(
    s1_batch: List[Tuple[str, str, str]]
) -> List[Tuple[str, str, str]]:
    country = _G_COUNTRY
    c_df = _G_TOKEN_DF
    eids = _G_CAND_EIDS
    names = _G_CAND_NAMES
    addrs = _G_CAND_ADDRS

    active_s1_indices: List[int] = []
    batch_cids: List[List[str]] = []
    group_lengths: List[int] = []
    X_parts: List[np.ndarray] = []

    results: List[Tuple[str, str, str]] = [None] * len(s1_batch)  # type: ignore

    for idx, (s1_id, b_name, b_addr) in enumerate(s1_batch):
        s1_rec = EntityRecord(s1_id, b_name, b_addr, country)
        cand_ints = _query_candidates(s1_rec)
        if not cand_ints:
            results[idx] = (s1_id, "", "")
            continue

        c_recs = [EntityRecord(eids[ci], names[ci], addrs[ci], country) for ci in cand_ints]
        cids_str = [eids[ci] for ci in cand_ints]
        feats = extract_features_for_s1_group(s1_rec, c_recs, country_token_df=c_df)

        active_s1_indices.append(idx)
        batch_cids.append(cids_str)
        group_lengths.append(len(cand_ints))
        X_parts.append(feats)

    if not X_parts:
        _clear_caches()
        return results

    X_base = np.vstack(X_parts)

    # Stage-1 ensemble predictions (50 features)
    p_lgb_p = _W_S1_LGB_PURE.predict(X_base, num_threads=1).astype(np.float32)
    p_lgb_e = _W_S1_LGB_EXT.predict(X_base, num_threads=1).astype(np.float32)
    p_lgb = 0.55 * p_lgb_p + 0.45 * p_lgb_e

    dmat_s1 = _W_XGB_MOD.DMatrix(X_base, feature_names=FEATURE_NAMES)
    p_xgb_p = _W_S1_XGB_PURE.predict(dmat_s1, iteration_range=(0, _W_S1_XGB_PURE_END)).astype(np.float32)
    p_xgb_e = _W_S1_XGB_EXT.predict(dmat_s1, iteration_range=(0, _W_S1_XGB_EXT_END)).astype(np.float32)
    p_xgb = 0.55 * p_xgb_p + 0.45 * p_xgb_e

    # Stage-2 Group Context features (68 features)
    X_stage2 = build_stage2_context_features(X_base, p_lgb, p_xgb, group_lengths, blend_w_lgb=0.55)
    p_s2_lgb = _W_S2_LGB.predict(X_stage2, num_threads=1).astype(np.float32)
    dmat_s2 = _W_XGB_MOD.DMatrix(X_stage2, feature_names=ALL_STAGE2_FEATURE_NAMES)
    p_s2_xgb = _W_S2_XGB.predict(dmat_s2, iteration_range=(0, _W_S2_XGB_END)).astype(np.float32)

    p_s1_blend = 0.55 * p_lgb + 0.45 * p_xgb
    p_s2_ens = 0.50 * p_s2_lgb + 0.50 * p_s2_xgb
    p_final = 0.80 * p_s2_ens + 0.20 * p_s1_blend

    offset = 0
    for active_pos, s1_idx in enumerate(active_s1_indices):
        s1_id = s1_batch[s1_idx][0]
        cids = batch_cids[active_pos]
        length = group_lengths[active_pos]
        end = offset + length
        g_probs = p_final[offset:end]
        offset = end

        pairs = sorted(
            ((cids[i], float(g_probs[i])) for i in range(length)),
            key=lambda x: x[1],
            reverse=True,
        )
        cand_csv = ",".join(cid for cid, _ in pairs)
        # Retain candidate scores >= 0.44 for global bipartite deduplication
        scored_str = ",".join(f"{cid}:{prob:.5f}" for cid, prob in pairs if prob >= 0.44)
        results[s1_idx] = (s1_id, cand_csv, scored_str)

    _clear_caches()
    return results


def load_country_s1_records(path: str, country: str) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            if not line.endswith(country + "\n") and not line.endswith(country + "\r\n"):
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[3].strip() == country:
                rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return rows


def stream_index_candidates(
    test_dir: str,
    country: str,
    s1_chunk: List[Tuple[str, str, str]],
    cached_tdf: Optional[Dict[str, int]] = None,
    num_workers: int = 4,
) -> Tuple[Dict[str, array], Dict[str, int], List[str], List[str], List[str]]:
    """
    Streams test_source2 and test_source3, indexing candidate keys with pruning limits.
    Stores posting lists as array('I') unsigned 32-bit integers to prevent COW memory bloat.
    """
    global _G_COUNTRY
    _G_COUNTRY = country
    t0 = time.time()

    # Step 1: Extract all keys needed by this S1 chunk
    key_batches = [s1_chunk[i : i + 4000] for i in range(0, len(s1_chunk), 4000)]
    needed_keys: Set[str] = set()
    with Pool(processes=num_workers) as key_pool:
        for kset in key_pool.imap_unordered(_extract_s1_keys_worker, key_batches):
            needed_keys.update(kset)
    del key_batches
    gc.collect()

    print(
        f"    [1/3] Extracted {len(needed_keys):,} blocking keys from {len(s1_chunk):,} "
        f"{country} S1 records ({time.time() - t0:.1f}s)",
        flush=True,
    )

    # Step 2: Stream candidate files
    t_scan = time.time()
    c_idx: Dict[str, array] = {}
    build_tdf = cached_tdf is None
    c_tdf: Dict[str, int] = defaultdict(int) if build_tdf else cached_tdf  # type: ignore

    high_prec_limit = 2500
    mid_prec_limit = 1000
    sub_limit = 250

    cand_eids: List[str] = []
    cand_names: List[str] = []
    cand_addrs: List[str] = []
    doc_cnt = 0

    for fname in ("test_source2.tsv", "test_source3.tsv"):
        fpath = os.path.join(test_dir, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                if not line.endswith(country + "\n") and not line.endswith(country + "\r\n"):
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4 or parts[3].strip() != country:
                    continue

                eid = parts[0].strip()
                b_name = parts[1].strip()
                b_addr = parts[2].strip()
                doc_cnt += 1

                rec = EntityRecord(eid, b_name, b_addr, country)
                keys = extract_blocking_keys(rec)

                if build_tdf:
                    for key in keys:
                        if key.startswith("nt:"):
                            c_tdf[key[3:]] += 1

                matched_keys = [k for k in keys if k in needed_keys]
                if matched_keys:
                    cid_int = len(cand_eids)
                    added = False
                    for key in matched_keys:
                        arr = c_idx.get(key)
                        if arr is None:
                            arr = array("I")
                            c_idx[key] = arr

                        if key.startswith(("ng:", "np:")):
                            lim = sub_limit
                        elif key.startswith(("ns:", "ps:", "nc:", "sh:", "af:", "an:")):
                            lim = high_prec_limit
                        else:
                            lim = mid_prec_limit

                        if len(arr) >= lim:
                            del c_idx[key]
                            needed_keys.discard(key)
                        else:
                            arr.append(cid_int)
                            added = True

                    if added:
                        cand_eids.append(eid)
                        cand_names.append(b_name)
                        cand_addrs.append(b_addr)

    needed_keys.clear()
    _clear_caches()

    if build_tdf:
        ref_scale = 234_392.0 / max(1.0, float(doc_cnt))
        if ref_scale < 0.95:
            for tok, raw_df in list(c_tdf.items()):
                c_tdf[tok] = max(1, int(round(raw_df * ref_scale)))
        c_tdf = dict(c_tdf)

    gc.collect()
    print(
        f"    [2/3] Indexed {doc_cnt:,} {country} records -> {len(c_idx):,} active posting keys, "
        f"{len(cand_eids):,} reachable candidates ({time.time() - t_scan:.1f}s)",
        flush=True,
    )
    return c_idx, c_tdf, cand_eids, cand_names, cand_addrs


def apply_bipartite_mutual_exclusion(
    country: str,
    raw_scores_path: str,
    match_output_path: str,
) -> Tuple[int, int]:
    """
    Enforces the single-parent property across all S1 entities in a country.
    In the ground truth, every S2/S3 record belongs to at most ONE S1 parent entity.
    This resolves cross-entity duplicate assignments, boosting precision.
    """
    t0 = time.time()
    tau_top1, tau_sec, min_ratio, rescue_tau, dedup_margin = _COUNTRY_THRESHOLDS[country]

    # Pass 1: Global best ownership resolution
    cand_best_prob: Dict[str, float] = {}
    cand_best_s1: Dict[str, str] = {}

    with open(raw_scores_path, "r", encoding="utf-8") as f:
        for line in f:
            s1_id, tab, rest = line.partition("\t")
            if not tab:
                continue
            s_str = rest.strip()
            if not s_str:
                continue
            for item in s_str.split(","):
                cid, _, p_str = item.partition(":")
                prob = float(p_str)
                if prob > cand_best_prob.get(cid, -1.0):
                    cand_best_prob[cid] = prob
                    cand_best_s1[cid] = s1_id

    # Pass 2: Filter by ownership & apply precision thresholds
    non_sing = 0
    tot_links = 0
    tmp_out = match_output_path + ".tmp"

    with open(raw_scores_path, "r", encoding="utf-8") as fin, open(tmp_out, "w", encoding="utf-8") as fout:
        for line in fin:
            s1_id, tab, rest = line.partition("\t")
            if not tab:
                continue
            s_str = rest.strip()
            if not s_str:
                fout.write(f"{s1_id}\t\n")
                continue

            valid_pairs: List[Tuple[str, float]] = []
            for item in s_str.split(","):
                cid, _, p_str = item.partition(":")
                prob = float(p_str)
                if cand_best_s1.get(cid) == s1_id:
                    valid_pairs.append((cid, prob))

            if not valid_pairs:
                fout.write(f"{s1_id}\t\n")
                continue

            p1 = valid_pairs[0][1]
            p2 = valid_pairs[1][1] if len(valid_pairs) > 1 else 0.0
            is_non_singleton = (p1 >= tau_top1) or (p1 >= rescue_tau and p2 >= rescue_tau)
            if not is_non_singleton:
                fout.write(f"{s1_id}\t\n")
                continue

            matched = [valid_pairs[0][0]]
            for cid, prob in valid_pairs[1:]:
                if prob >= tau_sec and (prob / (p1 + 1e-6)) >= min_ratio:
                    matched.append(cid)
                else:
                    break

            non_sing += 1
            tot_links += len(matched)
            fout.write(f"{s1_id}\t{','.join(matched)}\n")

    os.replace(tmp_out, match_output_path)
    print(
        f"    [Bipartite Dedup - {country}] Processed in {time.time() - t0:.1f}s | "
        f"Non-Singletons: {non_sing:,}, Total Links: {tot_links:,}",
        flush=True,
    )
    return non_sing, tot_links


def run_country_inference(
    country: str,
    test_dir: str,
    model_dir: str,
    ckpt_dir: str,
    num_workers: int = 4,
    chunk_size: int = 260_000,
) -> Tuple[str, str]:
    global _G_C_IDX, _G_TOKEN_DF, _G_CAND_EIDS, _G_CAND_NAMES, _G_CAND_ADDRS, _G_COUNTRY

    match_ckpt = os.path.join(ckpt_dir, f"matching_{country}.tsv")
    cand_ckpt = os.path.join(ckpt_dir, f"candidates_{country}.tsv")
    raw_scores_ckpt = os.path.join(ckpt_dir, f"raw_scores_{country}.tsv")

    s1_rows = load_country_s1_records(os.path.join(test_dir, "test_source1.tsv"), country)
    expected_count = len(s1_rows)

    if os.path.isfile(match_ckpt) and os.path.isfile(cand_ckpt):
        with open(match_ckpt, "r", encoding="utf-8") as fm, open(cand_ckpt, "r", encoding="utf-8") as fc:
            m_len = sum(1 for _ in fm)
            c_len = sum(1 for _ in fc)
        if m_len == expected_count and c_len == expected_count:
            print(f"[{country}] Checkpoints verified ({expected_count:,} records). Skipping inference.")
            return match_ckpt, cand_ckpt

    t_start = time.time()
    print(f"\nProcessing {country} ({expected_count:,} Source 1 entities)...")

    # Resume capability from existing partial raw scores
    done_ids: Set[str] = set()
    if os.path.isfile(raw_scores_ckpt + ".tmp") and os.path.isfile(cand_ckpt + ".tmp"):
        with open(cand_ckpt + ".tmp", "r", encoding="utf-8") as f:
            for line in f:
                sid, tab, _ = line.partition("\t")
                if tab and sid:
                    done_ids.add(sid)
        if done_ids:
            print(f"  Resuming from {len(done_ids):,} existing records...")

    remaining_rows = [r for r in s1_rows if r[0] not in done_ids] if done_ids else s1_rows
    del s1_rows, done_ids
    gc.collect()

    chunks = [remaining_rows[i : i + chunk_size] for i in range(0, len(remaining_rows), chunk_size)]
    del remaining_rows
    gc.collect()

    cached_tdf: Optional[Dict[str, int]] = None
    processed = 0

    for chunk_idx, s1_chunk in enumerate(chunks, 1):
        print(f"  Chunk {chunk_idx}/{len(chunks)} ({len(s1_chunk):,} entities)")
        c_idx, cached_tdf, cand_eids, cand_names, cand_addrs = stream_index_candidates(
            test_dir=test_dir,
            country=country,
            s1_chunk=s1_chunk,
            cached_tdf=cached_tdf,
            num_workers=num_workers,
        )

        _G_C_IDX = c_idx
        _G_TOKEN_DF = cached_tdf
        _G_CAND_EIDS = cand_eids
        _G_CAND_NAMES = cand_names
        _G_CAND_ADDRS = cand_addrs
        _G_COUNTRY = country

        batch_size = 500
        batches = [s1_chunk[i : i + batch_size] for i in range(0, len(s1_chunk), batch_size)]

        gc.freeze()
        open_mode = "a" if (chunk_idx > 1 or os.path.isfile(raw_scores_ckpt + ".tmp")) else "w"
        with open(raw_scores_ckpt + ".tmp", open_mode, encoding="utf-8") as fs, open(
            cand_ckpt + ".tmp", open_mode, encoding="utf-8"
        ) as fc:
            with Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(model_dir,),
                maxtasksperchild=16,
            ) as pool:
                for b_idx, batch_out in enumerate(pool.imap(_predict_batch_worker, batches), 1):
                    s_lines = [f"{sid}\t{sc}\n" for sid, _, sc in batch_out]
                    c_lines = [f"{sid}\t{cc}\n" for sid, cc, _ in batch_out]
                    fs.writelines(s_lines)
                    fc.writelines(c_lines)
                    fs.flush()
                    fc.flush()
                    processed += len(batch_out)

                    if b_idx % 40 == 0 or b_idx == len(batches):
                        print(f"    Scored {processed:,}/{expected_count:,} records...", flush=True)
        gc.unfreeze()

        _G_C_IDX.clear()
        _G_CAND_EIDS.clear()
        _G_CAND_NAMES.clear()
        _G_CAND_ADDRS.clear()
        del c_idx, cand_eids, cand_names, cand_addrs, batches
        _clear_caches()
        gc.collect()

    os.replace(raw_scores_ckpt + ".tmp", raw_scores_ckpt)
    os.replace(cand_ckpt + ".tmp", cand_ckpt)

    apply_bipartite_mutual_exclusion(
        country=country,
        raw_scores_path=raw_scores_ckpt,
        match_output_path=match_ckpt,
    )

    print(f"Finished {country} in {(time.time() - t_start)/60:.1f} minutes.")
    return match_ckpt, cand_ckpt


def run_prediction_pipeline(
    test_dir: str,
    output_dir: str,
    model_dir: str,
    num_workers: int = 4,
) -> None:
    t_total = time.time()
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Business Entity Resolution Pipeline — Inference Execution")
    print("=" * 80)

    # Process countries: France (smallest) -> US -> India
    countries = ["France", "US", "India"]
    for country in countries:
        run_country_inference(
            country=country,
            test_dir=test_dir,
            model_dir=model_dir,
            ckpt_dir=ckpt_dir,
            num_workers=num_workers,
        )

    print("\nMerging country predictions into final submission files...")
    s1_source_path = os.path.join(test_dir, "test_source1.tsv")
    ordered_s1_ids: List[str] = []
    with open(s1_source_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            if line.strip():
                ordered_s1_ids.append(line.split("\t", 1)[0].strip())

    # 1. matching_results.tsv
    final_match_path = os.path.join(output_dir, "matching_results.tsv")
    match_map: Dict[str, str] = {}
    for country in countries:
        m_path = os.path.join(ckpt_dir, f"matching_{country}.tsv")
        if os.path.isfile(m_path):
            with open(m_path, "r", encoding="utf-8") as f:
                for line in f:
                    sid, _, rest = line.partition("\t")
                    match_map[sid] = rest

    with open(final_match_path, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in ordered_s1_ids:
            fm.write(f"{sid}\t{match_map.get(sid, chr(10))}")
    del match_map
    gc.collect()
    print(f"  [OK] Generated {final_match_path} ({len(ordered_s1_ids):,} rows)")

    # 2. candidate_pairs.tsv
    final_cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    cand_map: Dict[str, str] = {}
    for country in countries:
        c_path = os.path.join(ckpt_dir, f"candidates_{country}.tsv")
        if os.path.isfile(c_path):
            with open(c_path, "r", encoding="utf-8") as f:
                for line in f:
                    sid, _, rest = line.partition("\t")
                    cand_map[sid] = rest

    with open(final_cand_path, "w", encoding="utf-8") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in ordered_s1_ids:
            fc.write(f"{sid}\t{cand_map.get(sid, chr(10))}")
    del cand_map
    gc.collect()
    print(f"  [OK] Generated {final_cand_path} ({len(ordered_s1_ids):,} rows)")

    # Run official validator check
    val_script = os.path.join(PROJECT_ROOT, "utils/validate_submission.py")
    if os.path.isfile(val_script):
        print("\nExecuting validation check (utils/validate_submission.py)...")
        res = subprocess.run(
            [sys.executable, val_script, "--matching", final_match_path, "--candidate", final_cand_path, "--test-dir", test_dir],
            capture_output=True,
            text=True,
        )
        print(res.stdout)
        if res.stderr:
            print(res.stderr)

    print(f"\nPipeline finished in {(time.time() - t_total) / 60:.1f} minutes.")


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution End-to-End Execution Pipeline")
    parser.add_argument("--mode", choices=["predict", "train"], default="predict", help="Operation mode (default: predict)")
    parser.add_argument("--test-dir", default=os.path.join(PROJECT_ROOT, "dataset/test"), help="Test data directory")
    parser.add_argument("--train-dir", default=os.path.join(PROJECT_ROOT, "dataset/train"), help="Training data directory")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "output"), help="Output directory for generated TSVs")
    parser.add_argument("--model-dir", default=os.path.join(PROJECT_ROOT, "output/models"), help="Pretrained model directory")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel worker processes (default: 4)")
    args = parser.parse_args()

    if args.mode == "train":
        print("Launching full retraining workflow (exp007_gpu_pipeline.py)...")
        train_script = os.path.join(SCRIPT_DIR, "exp007_gpu_pipeline.py")
        subprocess.run([sys.executable, train_script], check=True)
    else:
        run_prediction_pipeline(
            test_dir=os.path.abspath(args.test_dir),
            output_dir=os.path.abspath(args.output_dir),
            model_dir=os.path.abspath(args.model_dir),
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
