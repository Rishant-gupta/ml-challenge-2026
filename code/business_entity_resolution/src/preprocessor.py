"""
Preprocessor and text normalization module for Business Entity Resolution (EXP-002).
Includes:
- Hindi Devanagari -> Latin phonetic & lexical transliteration
- URL / Domain stripping (.com, .net, .org, www.)
- Dotted/slashed acronym collapsing (D/U -> du, S.A.S. -> sas)
- DBA ('d/b/a', 'a/k/a') multi-variant name extraction
- US, India, and France legal suffix stripping
- Country-aware state and street abbreviation standardization
- Ordinal and leading-zero number normalization
"""

import re
import unicodedata
from functools import lru_cache
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 1. Hindi Devanagari -> Latin Transliteration (Lexical + Phonetic)
# ---------------------------------------------------------------------------
DEVANAGARI_RANGE_RE = re.compile(r"[\u0900-\u097F]")

HINDI_CORP_LEXICON = {
    "प्राइवेट": "private",
    "प्राईवेट": "private",
    "लिमिटेड": "limited",
    "एलएलपी": "llp",
    "मार्केटिंग": "marketing",
    "एंटरप्राइजेज": "enterprises",
    "इंटरप्राइजेज": "enterprises",
    "एंटरप्राइज": "enterprise",
    "सर्विसेज": "services",
    "सर्विस": "service",
    "सॉल्यूशंस": "solutions",
    "सोल्यूशंस": "solutions",
    "प्रॉपर्टीज": "properties",
    "प्रोपर्टीज": "properties",
    "इंडिया": "india",
    "ट्रेडिंग": "trading",
    "ट्रेडर्स": "traders",
    "कंपनी": "company",
    "कम्पनी": "company",
    "ग्रुप": "group",
    "इंफ्रास्ट्रक्चर": "infrastructure",
    "इंफ्रा": "infra",
    "टेक्नोलॉजीज": "technologies",
    "टेक्नोलॉजी": "technology",
    "कंसल्टेंट्स": "consultants",
    "कंसल्टेंसी": "consultancy",
    "एसोसिएट्स": "associates",
    "लॉजिस्टिक्स": "logistics",
    "एक्सपोर्ट्स": "exports",
    "एक्सपोर्ट": "export",
    "इंपोर्ट्स": "imports",
    "फाइनेंस": "finance",
    "मोटर्स": "motors",
    "फार्मा": "pharma",
    "फार्मास्युटिकल्स": "pharmaceuticals",
    "केमिकल्स": "chemicals",
    "टेक्सटाइल्स": "textiles",
    "इंडस्ट्रीज": "industries",
    "प्रोजेक्ट्स": "projects",
    "प्रोजेक्ट": "project",
    "बिल्डर्स": "builders",
    "डेवलपर्स": "developers",
    "होटल": "hotel",
    "हॉस्पिटल": "hospital",
    "अल्फा": "alpha",
    "बीटा": "beta",
    "डेल्टा": "delta",
    "ग्लोबल": "global",
    "इंटरनेशनल": "international",
    "नेशनल": "national",
}

DEV_VOWELS = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
}

DEV_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
}

DEV_MATRAS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ॉ": "o", "ॅ": "e",
}

VIRAMA = "्"
ANUSVARA = ("ं", "ँ")


def transliterate_devanagari_word(word: str) -> str:
    """Converts a single Devanagari token to Latin script."""
    if word in HINDI_CORP_LEXICON:
        return HINDI_CORP_LEXICON[word]

    out = []
    n = len(word)
    i = 0
    while i < n:
        ch = word[i]
        # Handle consonant + nukta
        if i + 1 < n and word[i + 1] == "़":
            ch = ch + "़"
            i += 1

        if ch in DEV_VOWELS:
            out.append(DEV_VOWELS[ch])
        elif ch in DEV_CONSONANTS:
            base = DEV_CONSONANTS[ch]
            # Look ahead for matra or virama
            if i + 1 < n:
                nxt = word[i + 1]
                if nxt == VIRAMA:
                    out.append(base)
                    i += 2
                    continue
                elif nxt in DEV_MATRAS:
                    out.append(base + DEV_MATRAS[nxt])
                    i += 2
                    continue
            # Word-final consonant in Hindi drops inherent schwa 'a'
            if i == n - 1:
                out.append(base)
            else:
                out.append(base + "a")
        elif ch in DEV_MATRAS:
            out.append(DEV_MATRAS[ch])
        elif ch in ANUSVARA:
            out.append("n")
        elif ch != VIRAMA:
            out.append(ch)
        i += 1
    return "".join(out)


def transliterate_if_needed(text: str) -> str:
    """Transliterates any Devanagari words in text into Latin equivalents."""
    if not DEVANAGARI_RANGE_RE.search(text):
        return text
    tokens = text.split()
    converted = [
        transliterate_devanagari_word(tok) if DEVANAGARI_RANGE_RE.search(tok) else tok
        for tok in tokens
    ]
    return " ".join(converted)


# ---------------------------------------------------------------------------
# 2. Domain / URL, Initials, DBA & Legal Suffix Patterns
# ---------------------------------------------------------------------------
DBA_SPLIT_RE = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.|dba|a\s*/\s*k\s*/\s*a|aka)\b", re.IGNORECASE)
URL_PREFIX_RE = re.compile(r"\b(?:https?://)?(?:www\.)?", re.IGNORECASE)
DOMAIN_SUFFIX_RE = re.compile(r"\.(?:com|net|org|co\.in|in|fr|us|io|biz|info|edu|gov)\b", re.IGNORECASE)
CONCAT_COM_RE = re.compile(r"^([a-z0-9]{6,})com$")

# Matches dotted or slashed initials like "D/U", "B.S.", "S.A.S."
INITIALS_3_RE = re.compile(r"\b([a-zA-Z])\s*[./]\s*([a-zA-Z])\s*[./]\s*([a-zA-Z])\b")
INITIALS_2_RE = re.compile(r"\b([a-zA-Z])\s*[./]\s*([a-zA-Z])\b")

LEGAL_SUFFIXES_REGEX = re.compile(
    r"\b("
    r"private\s+limited|pvt\s+ltd|pvt\s+limited|p\s+ltd|"
    r"corporation|corp|incorporated|inc|"
    r"limited\s+liability\s+partnership|llp|"
    r"professional\s+limited\s+liability\s+company|pllc|"
    r"limited\s+liability\s+company|llc|"
    r"limited\s+partnership|lp|"
    r"public\s+limited\s+company|plc|"
    r"limited|ltd|"
    r"societe\s+a\s+responsabilite\s+limitee|sarl|"
    r"societe\s+par\s+actions\s+simplifiee\s+unipersonnelle|sasu|"
    r"societe\s+par\s+actions\s+simplifiee|sas|"
    r"entreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee|eurl|"
    r"societe\s+civile\s+immobiliere|sci|"
    r"societe\s+anonyme|sa|"
    r"et\s+fils|and\s+fils|fils|et\s+freres|and\s+freres|freres|cie|groupe|"
    r"company|co|group|holdings"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# 3. Address & State Standardization (US, India, France)
# ---------------------------------------------------------------------------
ORDINAL_NUM_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
LEADING_ZERO_NUM_RE = re.compile(r"\b0+(\d+)\b")

ADDRESS_ABBREVIATIONS = [
    (re.compile(r"\brd\b"), "road"),
    (re.compile(r"\bst\b"), "street"),
    (re.compile(r"\bave\b"), "avenue"),
    (re.compile(r"\bav\b"), "avenue"),
    (re.compile(r"\bblvd\b"), "boulevard"),
    (re.compile(r"\bbd\b"), "boulevard"),
    (re.compile(r"\bln\b"), "lane"),
    (re.compile(r"\bdr\b"), "drive"),
    (re.compile(r"\bct\b"), "court"),
    (re.compile(r"\bpkwy\b"), "parkway"),
    (re.compile(r"\bhwy\b"), "highway"),
    (re.compile(r"\bcir\b"), "circle"),
    (re.compile(r"\bplz\b"), "plaza"),
    (re.compile(r"\bflr\b"), "floor"),
    (re.compile(r"\bfl\b"), "floor"),
    (re.compile(r"\bste\b"), "suite"),
    (re.compile(r"\bapt\b"), "apartment"),
    (re.compile(r"\bctr\b"), "center"),
    (re.compile(r"\bcentre\b"), "center"),
    (re.compile(r"\bopp\b"), "opposite"),
    (re.compile(r"\bnr\b"), "near"),
    (re.compile(r"\bsec\b"), "sector"),
    (re.compile(r"\bdist\b"), "district"),
]

US_STATE_MAP = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

INDIA_STATE_MAP = {
    "up": "uttar pradesh", "dl": "delhi", "mh": "maharashtra", "ka": "karnataka",
    "tn": "tamil nadu", "gj": "gujarat", "wb": "west bengal", "mp": "madhya pradesh",
    "hr": "haryana", "rj": "rajasthan", "ap": "andhra pradesh", "tg": "telangana",
    "ts": "telangana", "kl": "kerala", "pb": "punjab", "br": "bihar",
    "od": "odisha", "jh": "jharkhand", "cg": "chhattisgarh", "uk": "uttarakhand",
    "ua": "uttarakhand", "hp": "himachal pradesh", "jk": "jammu and kashmir",
    "as": "assam", "ga": "goa", "ch": "chandigarh",
}

FRANCE_STREET_MAP = {
    "r": "rue", "av": "avenue", "bd": "boulevard", "all": "allee",
    "pl": "place", "imp": "impasse", "rte": "route", "ch": "chemin",
}


def strip_accents(text: str) -> str:
    """Normalizes accented/diacritic characters to ASCII equivalents."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_general_text(text: str, is_name: bool = False) -> str:
    """General lowercase, Devanagari transliteration, accent & punctuation normalization."""
    if not text:
        return ""
    text = transliterate_if_needed(text)
    text = strip_accents(text).lower()

    if is_name:
        text = DOMAIN_SUFFIX_RE.sub(" ", text)
        text = URL_PREFIX_RE.sub(" ", text)

    # Collapse dotted/slashed initials before removing punctuation
    text = INITIALS_3_RE.sub(r"\1\2\3", text)
    text = INITIALS_2_RE.sub(r"\1\2", text)

    text = text.replace("&", " and ")
    text = text.replace("+", " plus ")
    text = text.replace("@", " at ")
    text = re.sub(r"['’`]", "", text)  # Collapse apostrophes ("orelee's" -> "orelees")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if is_name and " " not in text:
        m = CONCAT_COM_RE.match(text)
        if m:
            text = m.group(1)

    return text


@lru_cache(maxsize=500_000)
def clean_business_name(name: str) -> Tuple[str, str]:
    """
    Returns (cleaned_full_name, name_stripped_of_legal_suffixes).
    Cached for fast repeated lookup across blocking and scoring.
    """
    cleaned_full = normalize_general_text(name, is_name=True)
    stripped = LEGAL_SUFFIXES_REGEX.sub(" ", cleaned_full)
    # Run twice in case of repeated suffixes like "Ltd Ltd"
    stripped = LEGAL_SUFFIXES_REGEX.sub(" ", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return cleaned_full, (stripped if stripped else cleaned_full)


@lru_cache(maxsize=500_000)
def extract_name_variants(name: str) -> Tuple[str, ...]:
    """
    Returns a tuple of stripped name variants.
    If the name contains 'd/b/a' or 'a/k/a', includes both the full stripped name
    and each individual part stripped of legal suffixes.
    """
    _, full_stripped = clean_business_name(name)
    if not name or not DBA_SPLIT_RE.search(name):
        return (full_stripped,)

    parts = DBA_SPLIT_RE.split(name)
    variants = [full_stripped]
    for p in parts:
        p = p.strip()
        if p:
            _, s_p = clean_business_name(p)
            if s_p and s_p not in variants:
                variants.append(s_p)
    return tuple(variants)


@lru_cache(maxsize=500_000)
def clean_address(address: str, country: str = "") -> str:
    """
    Standardizes address tokens, expands road/street/state abbreviations,
    normalizes leading zeroes and ordinals.
    """
    if not address:
        return ""
    cleaned = normalize_general_text(address, is_name=False)
    cleaned = ORDINAL_NUM_RE.sub(r"\1", cleaned)
    cleaned = LEADING_ZERO_NUM_RE.sub(r"\1", cleaned)

    for pattern, repl in ADDRESS_ABBREVIATIONS:
        cleaned = pattern.sub(repl, cleaned)

    tokens = cleaned.split()
    if country == "US":
        tokens = [US_STATE_MAP.get(t, t) for t in tokens]
    elif country == "India":
        tokens = [INDIA_STATE_MAP.get(t, t) for t in tokens]
    elif country == "France":
        tokens = [FRANCE_STREET_MAP.get(t, t) for t in tokens]

    return " ".join(tokens).strip()
