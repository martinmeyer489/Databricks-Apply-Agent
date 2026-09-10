"""Unit tests for enriched-attribute + coordinate normalization."""

from src.pipelines.attribute_normalization import (
    BENEFITS_RATING_VALUES,
    OFFICE_POLICY_VALUES,
    jitter_coord,
    normalize_benefits_rating,
    normalize_company_vibe,
    normalize_office_policy,
)


def test_office_policy_maps_to_fixed_buckets():
    assert normalize_office_policy("Hybrid") == "hybrid"
    assert normalize_office_policy("HYBRID") == "hybrid"
    assert normalize_office_policy("fully remote") == "remote"
    assert normalize_office_policy("work from home") == "remote"
    assert normalize_office_policy("on-site") == "onsite"
    assert normalize_office_policy("in office") == "onsite"
    assert normalize_office_policy("vor ort") == "onsite"
    assert normalize_office_policy(None) == "unknown"
    assert normalize_office_policy("") == "unknown"
    assert normalize_office_policy("something weird") == "unknown"


def test_office_policy_only_returns_valid_values():
    for v in ["hybrid", "Remote", "onsite", "", None, "gibberish", "flexible"]:
        assert normalize_office_policy(v) in OFFICE_POLICY_VALUES


def test_benefits_rating_maps_to_fixed_buckets():
    assert normalize_benefits_rating("Excellent") == "excellent"
    assert normalize_benefits_rating("great") == "excellent"
    assert normalize_benefits_rating("solid") == "good"
    assert normalize_benefits_rating("minimal") == "basic"
    assert normalize_benefits_rating(None) == "unknown"


def test_benefits_rating_only_returns_valid_values():
    for v in ["excellent", "Good", "basic", "", None, "???", "generous"]:
        assert normalize_benefits_rating(v) in BENEFITS_RATING_VALUES


def test_company_vibe_normalized_lowercase_trimmed():
    assert normalize_company_vibe("  Fast-Paced  Startup ") == "fast-paced startup"
    assert normalize_company_vibe(None) == "unknown"
    assert normalize_company_vibe("") == "unknown"
    assert len(normalize_company_vibe("x" * 100)) <= 40


def test_jitter_is_deterministic_and_bounded():
    a = jitter_coord(52.5, 13.4, "listing-1")
    b = jitter_coord(52.5, 13.4, "listing-1")
    c = jitter_coord(52.5, 13.4, "listing-2")
    assert a == b  # deterministic for the same seed
    assert a != c  # different seed → different offset
    # Bounded within the default amount (0.05°) on each axis.
    assert abs(a[0] - 52.5) <= 0.05
    assert abs(a[1] - 13.4) <= 0.05


# --- Emoji bucketing --------------------------------------------------------

from src.pipelines.attribute_normalization import (  # noqa: E402
    BUCKET_CHOICES,
    BUCKETIZERS,
    UNKNOWN_BUCKET,
    bucket_benefits_rating,
    bucket_company_size,
    bucket_company_vibe,
    bucket_employment_type,
    bucket_industry,
    bucket_office_policy,
    bucket_seniority,
)


def test_every_filter_has_3_to_5_buckets():
    for name, choices in BUCKET_CHOICES.items():
        assert 3 <= len(choices) <= 5, f"{name} has {len(choices)} buckets"


def test_all_bucket_labels_have_emoji_and_text():
    # Each label must contain a non-ASCII emoji char AND alphabetic text.
    for choices in BUCKET_CHOICES.values():
        for label in choices:
            has_emoji = any(ord(ch) > 0x2000 for ch in label)
            has_text = any(ch.isalpha() for ch in label)
            assert has_emoji and has_text, f"label {label!r} missing emoji or text"


def test_seniority_bucketing_summarizes_messy_values():
    assert bucket_seniority("entry-level") == "🌱 Entry / Junior"
    assert bucket_seniority("Junior") == "🌱 Entry / Junior"
    assert bucket_seniority("Senior") == "🎯 Senior"
    assert bucket_seniority("manager") == "👑 Lead / Executive"
    assert bucket_seniority("mid-level") == "🧑\u200d💼 Mid-level"
    assert bucket_seniority(None) == UNKNOWN_BUCKET
    assert bucket_seniority("") == UNKNOWN_BUCKET


def test_employment_bucketing():
    assert bucket_employment_type("full-time") == "🕐 Full-time"
    assert bucket_employment_type("Vollzeit") == "🕐 Full-time"
    assert bucket_employment_type("part-time") == "🕜 Part-time"
    assert bucket_employment_type("apprenticeship") == "🎓 Trainee / Intern"
    assert bucket_employment_type("contract") == "📃 Contract / Temp"


def test_company_size_bucketing():
    assert bucket_company_size("startup") == "🚀 Startup"
    assert bucket_company_size("enterprise") == "🏛️ Large / Enterprise"
    assert bucket_company_size("medium") == "🏬 Medium"
    assert bucket_company_size("small") == "🏢 Small"


def test_industry_bucketing():
    assert bucket_industry("software development") == "💻 Tech / IT"
    assert bucket_industry("healthcare") == "🏥 Healthcare / Social"
    assert bucket_industry("automotive") == "🏭 Industry / Engineering"
    assert bucket_industry("finance") == "🏦 Business / Public"


def test_vibe_bucketing():
    assert bucket_company_vibe("fast-paced startup") == "🚀 Startup / Innovative"
    assert bucket_company_vibe("family-owned") == "👨\u200d👩\u200d👧 Family / Team"
    assert bucket_company_vibe("mission-driven") == "🌍 Mission-driven / Social"
    assert bucket_company_vibe("corporate") == "👔 Corporate / Traditional"


def test_office_and_benefits_bucketing():
    assert bucket_office_policy("hybrid") == "🔀 Hybrid"
    assert bucket_office_policy("Onsite") == "🏢 Onsite"
    assert bucket_benefits_rating("excellent") == "⭐ Excellent"
    assert bucket_benefits_rating("basic") == "🟡 Basic"


def test_bucketizers_always_return_a_declared_choice():
    # Any input maps to one of the fixed choices for that filter.
    samples = ["", None, "weird value", "Senior", "startup", "healthcare"]
    for col, fn in BUCKETIZERS.items():
        valid = set(BUCKET_CHOICES[col])
        for s in samples:
            assert fn(s) in valid
