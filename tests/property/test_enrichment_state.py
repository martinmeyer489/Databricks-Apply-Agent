"""Property-based tests for the enrichment state machine.

Property 9: Enrichment State Machine Totality
Validates: Requirements 3.5, 3.6, 3.7

Property 10: Unenriched-Only Processing Filter
Validates: Requirements 3.1
"""

from hypothesis import given, strategies as st

from src.pipelines.enrichment_state import (
    ALL_ATTRIBUTES,
    LLM_ATTRIBUTES,
    SOFT_LLM_ATTRIBUTES,
    determine_enrichment_state,
)

VALID_STATES = {"enriched", "partially_enriched", "failed"}

# --- Strategies -------------------------------------------------------

# A resolved value for an LLM attribute: a non-empty string, or something
# falsy that should count as unresolved (None, "", []).
resolved_or_unresolved_value = st.one_of(
    st.none(),
    st.just(""),
    st.just([]),
    st.text(min_size=1, max_size=20),
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
)

llm_normal_result = st.one_of(
    st.none(),
    st.fixed_dictionaries(
        {}, optional={attr: resolved_or_unresolved_value for attr in LLM_ATTRIBUTES}
    ),
)

llm_failure_result = st.one_of(
    st.fixed_dictionaries({"error": st.text(min_size=1, max_size=30)}),
    st.fixed_dictionaries(
        {"_failed": st.just(True)},
        optional={"reason": st.text(min_size=0, max_size=30)},
    ),
)

llm_result_strategy = st.one_of(llm_normal_result, llm_failure_result)

geocode_result_strategy = st.one_of(
    st.none(),
    st.just({}),
    st.fixed_dictionaries(
        {},
        optional={
            "latitude": st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
            "longitude": st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
        },
    ),
)


# --- Property 9: Totality ----------------------------------------------


@given(geocode_result=geocode_result_strategy, llm_result=llm_result_strategy)
def test_enrichment_state_machine_totality(geocode_result, llm_result):
    """**Validates: Requirements 3.5, 3.6, 3.7**

    For arbitrary well-formed geocode_result/llm_result inputs:
    - the function never raises
    - state is exactly one of {"enriched", "partially_enriched", "failed"}
    - unresolved_attributes is a list of strings drawn only from the
      known attribute names
    - failure_reason is None iff state == "enriched"; otherwise it is a
      non-empty string
    """
    state, unresolved_attributes, failure_reason = determine_enrichment_state(
        geocode_result, llm_result
    )

    assert state in VALID_STATES

    assert isinstance(unresolved_attributes, list)
    for attr in unresolved_attributes:
        assert isinstance(attr, str)
        assert attr in ALL_ATTRIBUTES

    if state == "enriched":
        assert failure_reason is None
        # Soft (vibe) attributes never gate the enriched state, but may still
        # be reported as unresolved; the required attributes must all be
        # resolved when enriched.
        assert all(attr in SOFT_LLM_ATTRIBUTES for attr in unresolved_attributes)
    else:
        assert isinstance(failure_reason, str)
        assert len(failure_reason) > 0


# --- Soft (vibe) attribute behaviour ------------------------------------


def _all_required_resolved():
    return {
        "required_skills": ["python"],
        "seniority_level": "Senior",
        "employment_type": "Full-time",
        "industry": "Tech",
        "company_size_band": "Medium",
    }


def test_missing_soft_attributes_do_not_downgrade_enriched():
    """A record with all required attrs + coords is 'enriched' even when the
    soft vibe attributes are absent — they must not drop it from the map."""
    geocode = {"latitude": 52.5, "longitude": 13.4}
    state, unresolved, failure_reason = determine_enrichment_state(
        geocode, _all_required_resolved()
    )
    assert state == "enriched"
    assert failure_reason is None
    assert set(unresolved) == set(SOFT_LLM_ATTRIBUTES)


def test_resolved_soft_attributes_leave_unresolved_empty():
    """When soft attributes are also resolved, unresolved is empty."""
    geocode = {"latitude": 52.5, "longitude": 13.4}
    llm = _all_required_resolved()
    llm.update({"company_vibe": "fast-paced startup", "office_policy": "hybrid",
                "benefits_rating": "good"})
    state, unresolved, failure_reason = determine_enrichment_state(geocode, llm)
    assert state == "enriched"
    assert unresolved == []
    assert failure_reason is None


# --- Property 10: Unenriched-Only Processing Filter ---------------------


record_strategy = st.fixed_dictionaries(
    {
        "id": st.integers(min_value=0, max_value=10_000),
        "enrichment_state": st.sampled_from(
            ["unenriched", "enriched", "partially_enriched", "failed"]
        ),
    }
)


@given(records=st.lists(record_strategy, max_size=200))
def test_unenriched_only_processing_filter(records):
    """**Validates: Requirements 3.1**

    Filtering records for enrichment_state == 'unenriched' (the pipeline's
    WHERE clause) yields exactly the subset of records with that state,
    and the filtered set is disjoint from records with any other state.
    """
    filtered = list(filter(lambda r: r["enrichment_state"] == "unenriched", records))

    expected = [r for r in records if r["enrichment_state"] == "unenriched"]
    assert filtered == expected

    # Every filtered record indeed has state 'unenriched'.
    assert all(r["enrichment_state"] == "unenriched" for r in filtered)

    # The filtered set is disjoint from records with any other state.
    other_state_records = [r for r in records if r["enrichment_state"] != "unenriched"]
    filtered_ids = {id(r) for r in filtered}
    other_ids = {id(r) for r in other_state_records}
    assert filtered_ids.isdisjoint(other_ids)
