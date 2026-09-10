"""Normalization helpers for enriched attributes and map coordinates.

Pure functions (no Spark/SDK) so they are unit-testable and can be reused by
the enrichment notebook and the app:

* :func:`normalize_office_policy` / :func:`normalize_benefits_rating` collapse
  the free-text the LLM returns into the fixed lowercase buckets the filters
  expect, so the filter dropdowns show a small, clean set of values.
* :func:`jitter_coord` deterministically nudges co-located points apart so many
  listings sharing one city's coordinates don't stack into a single map marker.

Validates: standardized filter values + map density for the redesigned app.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Tuple

# Fixed buckets the UI filters on.
OFFICE_POLICY_VALUES = ("onsite", "hybrid", "remote", "unknown")
BENEFITS_RATING_VALUES = ("excellent", "good", "basic", "unknown")

# Synonyms → canonical bucket. Keys are matched case-insensitively as
# substrings so e.g. "fully remote" → "remote", "on-site" → "onsite".
_OFFICE_POLICY_SYNONYMS = {
    "remote": "remote",
    "work from home": "remote",
    "wfh": "remote",
    "home office": "remote",
    "hybrid": "hybrid",
    "flexible": "hybrid",
    "onsite": "onsite",
    "on-site": "onsite",
    "on site": "onsite",
    "in office": "onsite",
    "in-office": "onsite",
    "office-based": "onsite",
    "vor ort": "onsite",
}

_BENEFITS_SYNONYMS = {
    "excellent": "excellent",
    "great": "excellent",
    "outstanding": "excellent",
    "generous": "excellent",
    "good": "good",
    "solid": "good",
    "decent": "good",
    "basic": "basic",
    "minimal": "basic",
    "standard": "basic",
    "none": "basic",
    "limited": "basic",
}


def _normalize(value: Optional[str], synonyms: dict, valid: tuple) -> str:
    """Collapse a free-text value into one of ``valid`` via ``synonyms``.

    Returns ``"unknown"`` when the value is empty or unrecognized.
    """
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    # Exact bucket match first.
    if text in valid:
        return text
    # Longest synonym key that appears in the text wins (so "on-site" beats a
    # stray "site" and "fully remote" maps to remote).
    for key in sorted(synonyms, key=len, reverse=True):
        if key in text:
            return synonyms[key]
    return "unknown"


def normalize_office_policy(value: Optional[str]) -> str:
    """Map any LLM office-policy string to {onsite,hybrid,remote,unknown}."""
    return _normalize(value, _OFFICE_POLICY_SYNONYMS, OFFICE_POLICY_VALUES)


def normalize_benefits_rating(value: Optional[str]) -> str:
    """Map any LLM benefits string to {excellent,good,basic,unknown}."""
    return _normalize(value, _BENEFITS_SYNONYMS, BENEFITS_RATING_VALUES)


def normalize_company_vibe(value: Optional[str]) -> str:
    """Tidy the company-vibe descriptor: trimmed, lowercased, bounded length.

    Vibe is intentionally free-text (not a fixed bucket) but we still normalize
    casing/whitespace so the filter's distinct values aren't fragmented by
    trivial differences.
    """
    if value is None:
        return "unknown"
    text = " ".join(str(value).strip().lower().split())
    if not text:
        return "unknown"
    return text[:40]


# ---------------------------------------------------------------------------
# Emoji bucketing — collapse each filter into 3-5 fixed, emoji-labeled groups
# ---------------------------------------------------------------------------
# Each filter's messy free-text values are summarized into a small set of
# stable, human-friendly buckets. The app's dropdowns offer exactly these
# labels (plus "All"), so every filter has 4-5 choices max. Matching is a
# case-insensitive substring/keyword scan; the FIRST rule that matches wins,
# so rules are ordered from most-specific to least-specific. Anything
# unmatched falls into the group's "unknown" bucket.

UNKNOWN_BUCKET = "❓ Unknown"

# (bucket_label, [keywords]) — order matters (first match wins).
_SENIORITY_RULES = [
    ("👑 Lead / Executive", ["executive", "director", "head", "lead", "principal", "chief", "vp", "manager", "expert", "facharzt"]),
    ("🎯 Senior", ["senior", "sr.", "sr ", "mid-senior"]),
    ("🌱 Entry / Junior", ["entry", "junior", "intern", "trainee", "apprentice", "graduate", "student", "assistant", "einsteiger", "ausbildung"]),
    ("🧑\u200d💼 Mid-level", ["mid", "experienced", "professional", "regular", "intermediate"]),
]

_EMPLOYMENT_RULES = [
    ("🎓 Trainee / Intern", ["intern", "trainee", "apprentice", "ausbildung", "werkstudent", "praktik", "dual"]),
    ("📃 Contract / Temp", ["contract", "temporary", "temp", "freelance", "self-employed", "befristet", "hourly", "franchise", "seasonal"]),
    ("🕐 Full-time", ["full-time", "full time", "fulltime", "vollzeit", "permanent", "festanstellung", "unbefristet"]),
    ("🕜 Part-time", ["part-time", "part time", "parttime", "teilzeit", "minijob"]),
]

_COMPANY_SIZE_RULES = [
    ("🚀 Startup", ["startup", "start-up"]),
    ("🏛️ Large / Enterprise", ["enterprise", "large", "corporate", "multinational", "konzern", "established"]),
    ("🏬 Medium", ["medium", "mid-size", "mid size", "mid-sized", "midsize", "mittel", "sme", "small to medium", "small-to-medium", "small-medium", "small/medium", "small to med"]),
    ("🏢 Small", ["small", "klein", "boutique"]),
]

_INDUSTRY_RULES = [
    ("💻 Tech / IT", ["tech", "it", "software", "digital", "data", "telecom", "internet", "computer", "saas", "ai", "cyber", "electronic"]),
    ("🏥 Healthcare / Social", ["health", "medical", "care", "pharma", "hospital", "clinic", "social", "nursing", "bio", "life science", "wellness", "education"]),
    ("🏭 Industry / Engineering", ["manufactur", "automotive", "aerospace", "engineering", "energy", "chemical", "industrial", "production", "food", "machinery", "electr", "construction", "logistic", "transport", "supply", "building", "utilit", "agriculture"]),
    ("🏦 Business / Public", ["finance", "financ", "insurance", "bank", "consult", "legal", "account", "real estate", "professional service", "marketing", "media", "hr", "recruit", "public", "government", "retail", "hospitality", "tourism", "administration", "non-profit", "nonprofit", "ngo", "arts", "sport"]),
]

_VIBE_RULES = [
    ("🚀 Startup / Innovative", ["startup", "start-up", "fast-paced", "dynamic", "agile", "scale-up", "young", "innovat", "modern", "creative", "cutting-edge", "forward", "tech-driven", "progressive"]),
    ("👨\u200d👩\u200d👧 Family / Team", ["family", "famili", "team", "friendly", "collaborat", "close-knit", "people-first"]),
    ("🌍 Mission-driven / Social", ["mission", "social", "purpose", "sustainab", "impact", "responsible", "value", "public service", "non-profit"]),
    ("👔 Corporate / Traditional", ["corporate", "traditional", "professional", "established", "formal", "structured", "conservative", "large"]),
]

_OFFICE_POLICY_BUCKETS = {
    "onsite": "🏢 Onsite",
    "hybrid": "🔀 Hybrid",
    "remote": "🏠 Remote",
    "unknown": UNKNOWN_BUCKET,
}

_BENEFITS_BUCKETS = {
    "excellent": "⭐ Excellent",
    "good": "👍 Good",
    "basic": "🟡 Basic",
    "unknown": UNKNOWN_BUCKET,
}


def _bucketize(value: Optional[str], rules) -> str:
    """Return the first bucket whose keyword appears in ``value``.

    Case-insensitive substring match; falls back to :data:`UNKNOWN_BUCKET`.
    """
    if value is None:
        return UNKNOWN_BUCKET
    text = str(value).strip().lower()
    if not text or text == "unknown":
        return UNKNOWN_BUCKET
    for label, keywords in rules:
        for kw in keywords:
            if kw in text:
                return label
    return UNKNOWN_BUCKET


def bucket_seniority(value: Optional[str]) -> str:
    """Bucket seniority into 5 emoji groups."""
    return _bucketize(value, _SENIORITY_RULES)


def bucket_employment_type(value: Optional[str]) -> str:
    """Bucket employment type into 5 emoji groups."""
    return _bucketize(value, _EMPLOYMENT_RULES)


def bucket_company_size(value: Optional[str]) -> str:
    """Bucket company size into 5 emoji groups."""
    return _bucketize(value, _COMPANY_SIZE_RULES)


def bucket_industry(value: Optional[str]) -> str:
    """Bucket industry into 5 emoji groups."""
    return _bucketize(value, _INDUSTRY_RULES)


def bucket_company_vibe(value: Optional[str]) -> str:
    """Bucket company vibe into 5 emoji groups."""
    return _bucketize(value, _VIBE_RULES)


def bucket_office_policy(value: Optional[str]) -> str:
    """Map the already-standardized office policy to its emoji bucket."""
    return _OFFICE_POLICY_BUCKETS.get(normalize_office_policy(value), UNKNOWN_BUCKET)


def bucket_benefits_rating(value: Optional[str]) -> str:
    """Map the already-standardized benefits rating to its emoji bucket."""
    return _BENEFITS_BUCKETS.get(normalize_benefits_rating(value), UNKNOWN_BUCKET)


# The exact, fixed choice lists the app's filter dropdowns present (each
# prefixed with "All" by the UI). Keeping these here keeps the buckets and the
# dropdown options in one place.
BUCKET_CHOICES = {
    "seniority_level": [label for label, _ in _SENIORITY_RULES] + [UNKNOWN_BUCKET],
    "employment_type": [label for label, _ in _EMPLOYMENT_RULES] + [UNKNOWN_BUCKET],
    "company_size_band": [label for label, _ in _COMPANY_SIZE_RULES] + [UNKNOWN_BUCKET],
    "industry": [label for label, _ in _INDUSTRY_RULES] + [UNKNOWN_BUCKET],
    "company_vibe": [label for label, _ in _VIBE_RULES] + [UNKNOWN_BUCKET],
    "office_policy": ["🏢 Onsite", "🔀 Hybrid", "🏠 Remote", UNKNOWN_BUCKET],
    "benefits_rating": ["⭐ Excellent", "👍 Good", "🟡 Basic", UNKNOWN_BUCKET],
}

# Maps a filter column to the function that buckets a raw value for it.
BUCKETIZERS = {
    "seniority_level": bucket_seniority,
    "employment_type": bucket_employment_type,
    "company_size_band": bucket_company_size,
    "industry": bucket_industry,
    "company_vibe": bucket_company_vibe,
    "office_policy": bucket_office_policy,
    "benefits_rating": bucket_benefits_rating,
}


# Deterministic jitter so overlapping city-level coordinates spread into a
# small cloud instead of a single stacked marker. ~0.05° ≈ 5 km, enough to
# separate markers visually without moving a listing to the wrong region.
_JITTER_DEGREES = 0.05


def _stable_unit(seed: str) -> float:
    """Map a seed string deterministically to a float in [-1, 1)."""
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    # Use 8 hex chars → int, scale to [0,1), shift to [-1,1).
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    return frac * 2.0 - 1.0


def jitter_coord(
    latitude: float,
    longitude: float,
    seed: str,
    amount_degrees: float = _JITTER_DEGREES,
) -> Tuple[float, float]:
    """Return coordinates nudged deterministically by up to ``amount_degrees``.

    Args:
        latitude/longitude: The base (city-level) coordinates.
        seed: A per-listing stable string (e.g. listing_id) so the same
            listing always lands in the same spot across refreshes.
        amount_degrees: Maximum offset applied to each axis.

    Returns:
        ``(lat, lon)`` offset by a deterministic amount derived from ``seed``.
    """
    lat_off = _stable_unit(seed + ":lat") * amount_degrees
    lon_off = _stable_unit(seed + ":lon") * amount_degrees
    return latitude + lat_off, longitude + lon_off
