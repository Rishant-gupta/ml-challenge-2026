"""
Matching and Scoring module for Business Entity Resolution.
EXP-001: Heuristic Multi-Field String Similarity Scorer.
"""

import re
from typing import Dict, List, Set, Tuple
from rapidfuzz import distance, fuzz

from data_loader import EntityRecord
from preprocessor import clean_address, clean_business_name

DIGIT_RE = re.compile(r"\b\d+\b")


def extract_numbers(text: str) -> Set[str]:
    """Extracts normalized digit sequences (without leading zeros)."""
    nums = set()
    for m in DIGIT_RE.findall(text):
        norm = m.lstrip("0")
        if norm:
            nums.add(norm)
    return nums


def compute_heuristic_pair_score(
    s1_rec: EntityRecord,
    cand_rec: EntityRecord,
) -> float:
    """
    Computes a composite similarity score in [0.0, 100.0] between an S1 entity
    and a candidate S2/S3 entity.

    Components:
    1. Cleaned Name Similarity (Token Sort Ratio, Token Set Ratio, Jaro-Winkler)
    2. Cleaned Address Similarity (Token Sort Ratio, Token Set Ratio, Number Overlap)
    3. Missing-address fallback and exact-address DBA rescue rule
    """
    if s1_rec.country != cand_rec.country:
        return 0.0

    # 1. Name similarities
    full_n1, strip_n1 = clean_business_name(s1_rec.business_name)
    full_n2, strip_n2 = clean_business_name(cand_rec.business_name)

    # Handle DBA ("d/b/a" or "dba") trade name patterns by also checking token_set_ratio
    name_sort = fuzz.token_sort_ratio(strip_n1, strip_n2)
    name_set = fuzz.token_set_ratio(strip_n1, strip_n2)
    name_jw = distance.JaroWinkler.similarity(strip_n1, strip_n2) * 100.0

    # Effective name score
    name_score = max(name_sort, 0.94 * name_set, 0.92 * name_jw)

    # 2. Address similarities
    addr1 = clean_address(s1_rec.business_address)
    addr2 = clean_address(cand_rec.business_address)

    if not addr2:
        # Candidate has missing address (~3.3% of S2/S3 records)
        # Require high name similarity to avoid false positives
        return 0.90 * name_score

    addr_sort = fuzz.token_sort_ratio(addr1, addr2)
    addr_set = fuzz.token_set_ratio(addr1, addr2)

    # Number overlap bonus/penalty (Street number, suite, PIN/ZIP code)
    nums1 = extract_numbers(addr1)
    nums2 = extract_numbers(addr2)
    if nums1 and nums2:
        overlap = len(nums1 & nums2) / max(1, min(len(nums1), len(nums2)))
        num_bonus = 6.0 * overlap - 4.0 * (1.0 - overlap)
    else:
        num_bonus = 0.0

    addr_score = min(100.0, max(0.0, max(addr_sort, 0.92 * addr_set) + num_bonus))

    # 3. Combine Name and Address scores
    composite = 0.62 * name_score + 0.38 * addr_score

    # Special Rule A: Exact/Near-Exact Address Rescue (for DBA / Trade Name changes like "BS Projects" -> "Gildcalo")
    if addr_sort >= 95.0 and len(nums1 & nums2) >= 1:
        composite = max(composite, 84.0)

    # Special Rule B: Exact Stripped Name + Shared Address Number
    if name_sort >= 98.0 and (len(nums1 & nums2) >= 1 or addr_set >= 75.0):
        composite = max(composite, 88.0)

    return min(100.0, max(0.0, composite))
