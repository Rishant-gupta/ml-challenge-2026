"""
Matching and Scoring module for Business Entity Resolution (EXP-002).
Upgrades over EXP-001:
1. Multi-variant DBA name matching ('d/b/a', 'a/k/a')
2. Spaceless concatenated name ratio (matches 'PUS Horizon Paper' <-> 'pushorizonpaper.com' at 100%)
3. Country-aware address normalization (US/India state codes, French street types)
4. Strip-Mall False Positive Guard + Coined-Alias Exact-Address Rescue
5. IDF Rarity-Guarded Missing-Address Fallback
"""

import re
from typing import Dict, Optional, Set
from rapidfuzz import distance, fuzz

from blocking import GENERIC_CORP_WORDS, PURE_STOP_WORDS
from data_loader import EntityRecord
from preprocessor import clean_address, clean_business_name, extract_name_variants

DIGIT_RE = re.compile(r"\b\d+\b")


def extract_numbers(text: str) -> Set[str]:
    """Extracts normalized digit sequences (without leading zeros)."""
    nums = set()
    for m in DIGIT_RE.findall(text):
        norm = m.lstrip("0")
        if norm:
            nums.add(norm)
    return nums


def is_single_coined_alias(raw_name: str, stripped_name: str) -> bool:
    """
    Checks if a business name is a single-word coined trade alias (e.g. 'DREXTAVO', 'Gildcalo')
    without standard corporate suffixes or generic business words.
    """
    tokens = stripped_name.split()
    if len(tokens) != 1:
        return False
    word = tokens[0]
    if len(word) < 5 or word in GENERIC_CORP_WORDS or word in PURE_STOP_WORDS:
        return False
    # Ensure raw name was also just 1 token (no 'LLC', 'Inc', etc.)
    full_clean, _ = clean_business_name(raw_name)
    if len(full_clean.split()) != 1:
        return False
    return True


def compute_heuristic_pair_score(
    s1_rec: EntityRecord,
    cand_rec: EntityRecord,
    country_token_df: Optional[Dict[str, int]] = None,
) -> float:
    """
    Computes a composite similarity score in [0.0, 100.0] between an S1 entity
    and a candidate S2/S3 entity.
    """
    if s1_rec.country != cand_rec.country:
        return 0.0

    # 1. Multi-variant & Spaceless Name Similarities
    vars1 = extract_name_variants(s1_rec.business_name)
    vars2 = extract_name_variants(cand_rec.business_name)

    best_name_score = 0.0
    best_strict_name = 0.0
    best_name_sort = 0.0

    for v1 in vars1:
        s_v1 = v1.replace(" ", "")
        for v2 in vars2:
            s_v2 = v2.replace(" ", "")
            n_sort = fuzz.token_sort_ratio(v1, v2)
            n_set = fuzz.token_set_ratio(v1, v2)
            n_jw = distance.JaroWinkler.similarity(v1, v2) * 100.0
            n_spaceless = fuzz.ratio(s_v1, s_v2) if (s_v1 and s_v2) else 0.0

            strict_sim = max(n_sort, n_spaceless, 0.92 * n_jw)
            # Only give full credit to token_set_ratio if length ratio isn't too lopsided
            len_ratio = min(len(s_v1), len(s_v2)) / max(1, max(len(s_v1), len(s_v2)))
            set_weight = 0.95 if len_ratio >= 0.6 else 0.82
            var_score = max(strict_sim, set_weight * n_set)

            if var_score > best_name_score:
                best_name_score = var_score
            if strict_sim > best_strict_name:
                best_strict_name = strict_sim
            if n_sort > best_name_sort:
                best_name_sort = n_sort

    # 2. Address Similarities
    addr1 = clean_address(s1_rec.business_address, s1_rec.country)
    addr2 = clean_address(cand_rec.business_address, cand_rec.country)

    if not addr2:
        # Candidate has missing address (~3.3% of S2/S3 records).
        # Must rely on strict name match (not subset match) and token rarity (IDF guard)
        # to prevent false positives on generic names like 'Vision Care' or 'Family Medicine'.
        base_missing = 0.90 * best_strict_name
        if country_token_df is not None:
            sig_tokens = [
                t for t in vars1[0].split()
                if len(t) >= 2 and t not in GENERIC_CORP_WORDS
            ]
            if sig_tokens:
                min_df = min(country_token_df.get(t, 0) for t in sig_tokens)
                # If all words in the name are fairly common (appear >120 times) and name has <=2 words, penalize
                if min_df > 120 and len(sig_tokens) <= 2:
                    base_missing -= 8.0
                elif min_df <= 25 and best_strict_name >= 95.0:
                    base_missing += 3.0
            else:
                base_missing -= 12.0
        return min(100.0, max(0.0, base_missing))

    addr_sort = fuzz.token_sort_ratio(addr1, addr2)
    addr_set = fuzz.token_set_ratio(addr1, addr2)

    nums1 = extract_numbers(addr1)
    nums2 = extract_numbers(addr2)
    shared_nums = nums1 & nums2
    if nums1 and nums2:
        overlap = len(shared_nums) / max(1, min(len(nums1), len(nums2)))
        num_bonus = 6.0 * overlap - 5.0 * (1.0 - overlap)
    else:
        num_bonus = 0.0

    addr_score = min(100.0, max(0.0, max(addr_sort, 0.93 * addr_set) + num_bonus))

    # 3. Combine Name and Address Scores
    # Strip-Mall Guard: If name similarity is very weak (<42%), heavily dampen composite score
    # UNLESS it qualifies for the Coined-Alias Exact-Address Rescue below.
    composite = 0.60 * best_name_score + 0.40 * addr_score

    # Rescue Rule A: Strong Name + Strong Address Synergy
    if best_strict_name >= 95.0 and (len(shared_nums) >= 1 or addr_set >= 72.0):
        composite = max(composite, 88.0)

    # Rescue Rule B: Partial Name Overlap (e.g. 'Mccullough Adamas' vs 'Service Mccullough') + Very Strong Address
    if best_name_score >= 58.0 and addr_score >= 84.0 and (len(shared_nums) >= 1 or not nums1 or not nums2):
        composite = max(composite, 83.5)

    # Rescue Rule C: Coined Single-Word Alias at Exact Same Address (e.g. 'BS Projects' <-> 'Gildcalo', 'DREXTAVO')
    if best_name_score < 45.0:
        if (
            (addr_sort >= 88.0 or addr_set >= 92.0)
            and len(shared_nums) >= 1
            and (
                is_single_coined_alias(cand_rec.business_name, vars2[0])
                or is_single_coined_alias(s1_rec.business_name, vars1[0])
            )
        ):
            composite = max(composite, 83.5)
        else:
            # Strip-Mall Penalty: Two multi-word companies with completely different names at the same address
            composite = min(composite, 72.0)

    return min(100.0, max(0.0, composite))
