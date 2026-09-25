"""
Blocking / Candidate Generation module for Business Entity Resolution (EXP-002).
Implements Country-Partitioned Multi-Granularity Tokenizer & Inverted Index:
- Spaceless concatenated name keys ('ns:')
- DBA-aware word token keys ('nt:')
- Sorted 2-word shingle keys ('sh:') to rescue common industry words
- 5-char subword n-gram keys ('ng:') for compound words and typos
- Address number+word ('an:'), address word-pair ('aw:'), and postal code ('pc:') keys
"""

import math
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set

from data_loader import EntityRecord
from preprocessor import clean_address, extract_name_variants

PURE_STOP_WORDS = {
    "the", "and", "of", "in", "at", "for", "to", "on", "by", "with", "de", "du", "des", "la", "le", "les",
    "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
    "co", "company", "llp", "pllc", "plc", "sa", "sas", "sasu", "sarl", "eurl", "sci", "gmbh",
    "dba", "aka",
}

GENERIC_CORP_WORDS = PURE_STOP_WORDS | {
    "group", "services", "service", "solutions", "enterprises", "enterprise", "holdings",
    "international", "global", "india", "usa", "us", "france", "center", "centre",
}

ADDR_STOP_TOKENS = {
    "road", "street", "avenue", "boulevard", "lane", "drive", "court", "place", "allee", "rue",
    "floor", "ground", "first", "second", "third", "unit", "suite", "apartment",
    "shop", "no", "plot", "flat", "sector", "phase", "block", "near", "opposite",
    "behind", "beside", "main", "cross", "stage", "layout", "colony", "nagar",
    "vihar", "marg", "puram", "khand", "peth", "bazar", "market", "plaza",
    "building", "complex", "tower", "towers", "center", "centre", "park",
    "north", "south", "east", "west", "new", "old", "city", "state", "district",
    "po", "box", "post", "office", "pin", "zip", "de", "du", "des", "la", "le", "les", "bis",
    # Common state names (too broad for address word pairs)
    "uttar", "pradesh", "maharashtra", "delhi", "karnataka", "tamil", "nadu", "gujarat",
    "madhya", "haryana", "rajasthan", "telangana", "andhra", "kerala", "punjab", "bengal",
    "california", "texas", "florida", "illinois", "pennsylvania", "ohio", "georgia",
    "michigan", "carolina", "jersey", "virginia", "washington", "arizona", "tennessee",
    "indiana", "missouri", "maryland", "wisconsin", "colorado", "minnesota", "alabama",
    "kentucky", "oregon", "oklahoma", "connecticut", "utah",
}


def extract_blocking_keys(record: EntityRecord) -> List[str]:
    """
    Extracts multi-granularity blocking keys for a record.
    """
    keys: Set[str] = set()
    variants = extract_name_variants(record.business_name)

    for stripped_name in variants:
        raw_tokens = [t for t in stripped_name.split() if t not in PURE_STOP_WORDS]
        sig_tokens = [t for t in raw_tokens if len(t) >= 2 and t not in GENERIC_CORP_WORDS]

        # 1. Spaceless concatenated name (catches URLs & joined words like 'pushorizonpaper')
        spaceless = "".join(sig_tokens)
        if len(spaceless) >= 4:
            keys.add(f"ns:{spaceless}")
            # 2. Subword 5-char n-grams on spaceless string (catches compound words & typos)
            if len(spaceless) >= 5:
                step = 1 if len(spaceless) <= 12 else 2
                for i in range(0, min(len(spaceless) - 4, 14), step):
                    keys.add(f"ng:{spaceless[i:i+5]}")

        # 3. Individual significant word tokens & 4-char prefixes
        for tok in sig_tokens:
            keys.add(f"nt:{tok}")
            if len(tok) >= 4 and not tok.isdigit():
                keys.add(f"np:{tok[:4]}")

        # 4. Sorted 2-word Shingles (uses raw_tokens so pairs like 'cardiology_center' or 'clinic_highland' are indexed!)
        shingle_tokens = sorted(set(t for t in raw_tokens if len(t) >= 3))[:6]
        if len(shingle_tokens) >= 2:
            for w1, w2 in combinations(shingle_tokens, 2):
                keys.add(f"sh:{w1}_{w2}")

    # 5. Address Keys
    if record.business_address:
        cleaned_addr = clean_address(record.business_address, record.country)
        addr_tokens = cleaned_addr.split()
        nums = []
        words = []
        for tok in addr_tokens:
            if tok.isdigit():
                if len(tok) >= 2:
                    nums.append(tok)
            elif len(tok) >= 3 and tok not in ADDR_STOP_TOKENS:
                words.append(tok)

        for num in nums[:3]:
            for w in words[:4]:
                keys.add(f"an:{num}_{w}")
            if len(num) >= 5:
                keys.add(f"pc:{num}")

        # Address rare word pairs (when street number is missing in one source)
        distinct_addr_words = sorted(set(words[:5]))
        if len(distinct_addr_words) >= 2:
            for w1, w2 in combinations(distinct_addr_words, 2):
                keys.add(f"aw:{w1}_{w2}")

    return list(keys)


class CountryPartitionedBlocker:
    """
    Builds a country-partitioned inverted index over Source 2 and Source 3 records,
    and retrieves top-K candidate IDs per Source 1 entity using IDF scoring.
     Also exposes token document frequencies for downstream matcher precision guards.
    """

    def __init__(
        self,
        max_postings_per_key: int = 2500,
        top_k: int = 35,
    ):
        self.max_postings_per_key = max_postings_per_key
        self.top_k = top_k
        self.index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.key_weights: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.token_df: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

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
                    if key.startswith("nt:"):
                        self.token_df[country][key[3:]] += 1

        # Prune overly common keys and compute IDF weights
        for country, c_idx in self.index.items():
            keys_to_delete = []
            for key, postings in c_idx.items():
                df = len(postings)
                if df > self.max_postings_per_key:
                    keys_to_delete.append(key)
                else:
                    prefix = key[:3]
                    if prefix == "ns:":
                        base_w = 3.0  # Exact spaceless name match is ultra-informative
                    elif prefix in ("sh:", "an:"):
                        base_w = 2.2  # Word-pair shingle & address number+word anchor
                    elif prefix in ("nt:", "aw:", "pc:"):
                        base_w = 1.4  # Single name word, address word-pair, postal code
                    elif prefix == "ng:":
                        base_w = 0.55 # Subword 5-gram
                    else:
                        base_w = 0.65 # 4-char prefix
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
