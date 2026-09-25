"""
Preprocessor and text normalization module for Business Entity Resolution.
Cleans business names and addresses without external API calls.
"""

import re
import unicodedata
from typing import Tuple


# Common legal suffixes across US, India, and general international business
LEGAL_SUFFIXES_REGEX = re.compile(
    r"\b("
    r"private\s+limited|pvt\s+ltd|pvt\s+limited|p\s+ltd|"
    r"corporation|corp|incorporated|inc|"
    r"limited\s+liability\s+company|llc|l\.l\.c\.|"
    r"limited|ltd|"
    r"public\s+limited\s+company|plc|"
    r"societe\s+anonyme|s\.a\.|sa|"
    r"societe\s+par\s+actions\s+simplifiee|s\.a\.s\.|sas|"
    r"sarl|s\.a\.r\.l\.|"
    r"company|co|"
    r"enterprise|enterprises|services|solutions|group|holdings"
    r")\b",
    re.IGNORECASE,
)

# Address abbreviations and expansions
ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bdr\b": "drive",
    r"\bflr\b": "floor",
    r"\bfl\b": "floor",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bplz\b": "plaza",
    r"\bctr\b": "center",
    r"\bdept\b": "department",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"\bsec\b": "sector",
    r"\bdist\b": "district",
    r"\bhwy\b": "highway",
}

ADDR_COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), repl)
    for pattern, repl in ADDRESS_ABBREVIATIONS.items()
]


def strip_accents(text: str) -> str:
    """Normalizes accented/diacritic characters to ASCII equivalents."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_general_text(text: str) -> str:
    """General lowercase, accent removal, punctuation normalization."""
    if not text:
        return ""
    text = strip_accents(text).lower()
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_business_name(name: str) -> Tuple[str, str]:
    """
    Returns (cleaned_full_name, name_stripped_of_legal_suffixes).
    """
    cleaned_full = normalize_general_text(name)
    stripped = LEGAL_SUFFIXES_REGEX.sub(" ", cleaned_full)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return cleaned_full, stripped if stripped else cleaned_full


def clean_address(address: str) -> str:
    """
    Standardizes address tokens, expands common road/street abbreviations,
    removes extra noise punctuation.
    """
    cleaned = normalize_general_text(address)
    for pattern, repl in ADDR_COMPILED_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
