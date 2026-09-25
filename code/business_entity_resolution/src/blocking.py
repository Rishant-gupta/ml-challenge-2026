"""
Blocking / Candidate Generation module for Business Entity Resolution.
Implements Country-Partitioned Multi-Key Inverted Index with IDF weighting.
"""

import math
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from data_loader import EntityRecord
from preprocessor import clean_address, clean_business_name

STOP_TOKENS = {
    "the", "and", "of", "in", "at", "for", "to", "on", "by", "with",
    "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
    "co", "company", "llp", "pllc", "plc", "sa", "sas", "sarl", "gmbh",
    "group", "services", "solutions", "enterprises", "enterprise", "holdings",
    "international", "global", "india", "usa", "us", "france",
}

ADDR_STOP_TOKENS = {
    "road", "street", "avenue", "boulevard", "lane", "drive", "court", "ct",
    "floor", "ground", "first", "second", "third", "unit", "suite", "apartment",
    "shop", "no", "plot", "flat", "sector", "phase", "block", "near", "opposite",
    "behind", "beside", "main", "cross", "stage", "layout", "colony", "nagar",
    "vihar", "marg", "puram", "khand", "peth", "bazar", "market", "plaza",
    "building", "complex", "tower", "towers", "center", "centre", "park",
    "north", "south", "east", "west", "new", "old", "city", "state", "district",
    "po", "box", "post", "office", "pin", "zip",
}

DIGIT_SEQ_RE = re.compile(r"\b\d+\b")


def normalize_number_token(num_str: str) -> str:
    """Strips leading zeros from numbers (e.g., '004303' -> '4303')."""
    stripped = num_str.lstrip("0")
    return stripped if stripped else "0"


def extract_blocking_keys(record: EntityRecord) -> List[str]:
    """
    Extracts blocking keys for a record:
    1. Name word tokens: 'nt:<token>'
    2. Name 4-char prefixes (for typo tolerance): 'np:<prefix4>'
    3. Address number + rare address token combinations: 'an:<num>_<token>'
    4. Rare address tokens: 'at:<token>'
    """
    keys: Set[str] = set()
    _, stripped_name = clean_business_name(record.business_name)
    name_tokens = [
        t for t in stripped_name.split()
        if len(t) >= 2 and t not in STOP_TOKENS
    ]

    for tok in name_tokens:
        keys.add(f"nt:{tok}")
        if len(tok) >= 4 and not tok.isdigit():
            keys.add(f"np:{tok[:4]}")

    # Bigram of first two significant name tokens
    if len(name_tokens) >= 2:
        keys.add(f"nb:{name_tokens[0]}_{name_tokens[1]}")

    # Address keys
    if record.business_address:
        cleaned_addr = clean_address(record.business_address)
        addr_tokens = cleaned_addr.split()
        nums = []
        words = []
        for tok in addr_tokens:
            if tok.isdigit():
                norm_num = normalize_number_token(tok)
                if len(norm_num) >= 2:
                    nums.append(norm_num)
            elif len(tok) >= 3 and tok not in ADDR_STOP_TOKENS:
                words.append(tok)

        # Combine first couple of numbers with distinctive address words
        # (Captures exact location matches even when business name is a DBA/trade name)
        for num in nums[:3]:
            for w in words[:4]:
                keys.add(f"an:{num}_{w}")
            if len(num) >= 5:  # Postal code / PIN code
                keys.add(f"pc:{num}")

    return list(keys)


class CountryPartitionedBlocker:
    """
    Builds a country-partitioned inverted index over Source 2 and Source 3 records,
    and retrieves top-K candidate IDs per Source 1 entity using IDF scoring.
    """

    def __init__(
        self,
        max_postings_per_key: int = 1500,
        top_k: int = 25,
    ):
        self.max_postings_per_key = max_postings_per_key
        self.top_k = top_k
        # country -> {key -> list of entity_ids}
        self.index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.key_weights: Dict[str, Dict[str, float]] = defaultdict(dict)

    def fit(
        self,
        s2_records: Dict[str, EntityRecord],
        s3_records: Dict[str, EntityRecord],
    ) -> None:
        """Indexes all S2 and S3 records partitioned by country."""
        for records_dict in (s2_records, s3_records):
            for eid, rec in records_dict.items():
                country = rec.country
                for key in extract_blocking_keys(rec):
                    self.index[country][key].append(eid)

        # Prune overly common keys and compute IDF weights
        for country, c_idx in self.index.items():
            keys_to_delete = []
            for key, postings in c_idx.items():
                df = len(postings)
                if df > self.max_postings_per_key:
                    keys_to_delete.append(key)
                else:
                    # Prefix keys have slightly lower base weight than exact token or address-number keys
                    if key.startswith("np:"):
                        base_w = 0.6
                    elif key.startswith("nb:") or key.startswith("an:"):
                        base_w = 1.8
                    else:
                        base_w = 1.2
                    self.key_weights[country][key] = base_w / math.log(2.0 + df)

            for k in keys_to_delete:
                del c_idx[k]

    def query_entity(self, s1_record: EntityRecord) -> List[str]:
        """Retrieves top-K candidate entity IDs from S2/S3 for a single S1 record."""
        country = s1_record.country
        c_idx = self.index.get(country)
        if not c_idx:
            return []

        c_weights = self.key_weights[country]
        scores: Dict[str, float] = defaultdict(float)

        for key in extract_blocking_keys(s1_record):
            postings = c_idx.get(key)
            if postings:
                w = c_weights[key]
                for cand_id in postings:
                    scores[cand_id] += w

        if not scores:
            return []

        if len(scores) <= self.top_k:
            sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        else:
            sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]

        return [cid for cid, _ in sorted_cands]

    def generate_candidates(
        self,
        s1_records: Dict[str, EntityRecord],
    ) -> Dict[str, List[str]]:
        """Generates candidate lists for all S1 records."""
        results: Dict[str, List[str]] = {}
        for s1_id, rec in s1_records.items():
            results[s1_id] = self.query_entity(rec)
        return results
