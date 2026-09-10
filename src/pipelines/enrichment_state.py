"""Enrichment state machine for the Silver-tier enrichment pipeline.

Determines the enrichment state of a Job_Listing from the outcome of the
two independent enrichment steps: geocode resolution and LLM attribute
extraction (Requirements 3.5, 3.6, 3.7, 3.8).

Interface contract
-------------------
``geocode_result``:
    A ``dict`` with resolved ``latitude`` and ``longitude`` keys (both
    non-``None``) when the location text was matched in the
    Geocode_Lookup reference dataset. Pass ``None`` or ``{}`` (or a dict
    missing/``None``-valued ``latitude``/``longitude``) when the lookup
    produced no match (Requirement 3.7).

``llm_result``:
    Either:

    - A ``dict`` keyed by the five LLM-derived attributes
      (``required_skills``, ``seniority_level``, ``employment_type``,
      ``industry``, ``company_size_band``), where each value is the
      resolved attribute or ``None``/empty if that attribute could not
      be extracted (Requirement 3.7). Any attribute key omitted from the
      dict is treated the same as ``None``.
    - A **failure marker** signalling that LLM extraction itself timed
      out or raised an error for the whole record (Requirement 3.8),
      rather than merely returning a partial result. This is expressed
      as a dict containing either an ``"error"`` key (truthy value, e.g.
      an exception message or ``"timeout"``) or a ``"_failed"`` key set
      to ``True``, optionally with a ``"reason"`` key describing why.
      Example: ``{"error": "timeout after 60s"}`` or
      ``{"_failed": True, "reason": "RateLimitError: ..."}``.

Returns
-------
A tuple ``(state, unresolved_attributes, failure_reason)``:

- ``state`` is exactly one of ``"enriched"``, ``"partially_enriched"``,
  or ``"failed"``.
- ``unresolved_attributes`` lists the names of attributes that could not
  be resolved, drawn from ``{"location", "required_skills",
  "seniority_level", "employment_type", "industry", "company_size_band"}``.
  Empty for ``"enriched"``. For ``"failed"``, this lists every attribute
  (the whole record could not be processed).
- ``failure_reason`` is ``None`` for ``"enriched"``, and a human-readable
  string describing why for ``"partially_enriched"`` and ``"failed"``.
"""

from typing import List, Optional, Tuple

LLM_ATTRIBUTES = (
    "required_skills",
    "seniority_level",
    "employment_type",
    "industry",
    "company_size_band",
)

# Soft, LLM-inferred "vibe" attributes used only as optional filters
# (company_vibe, office_policy, benefits_rating). Unlike LLM_ATTRIBUTES these
# are best-effort: a job description often does not state them, so their
# absence must NOT downgrade a listing to partially_enriched or drop it from
# the map (which filters on enrichment_state = 'enriched'). They are recorded
# in unresolved_attributes for observability but never gate the state.
SOFT_LLM_ATTRIBUTES = (
    "company_vibe",
    "office_policy",
    "benefits_rating",
)

LOCATION_ATTRIBUTE = "location"

ALL_ATTRIBUTES = (LOCATION_ATTRIBUTE,) + LLM_ATTRIBUTES + SOFT_LLM_ATTRIBUTES


def _is_geocode_resolved(geocode_result: Optional[dict]) -> bool:
    """Return True iff the geocode lookup produced usable coordinates."""
    if not geocode_result:
        return False
    latitude = geocode_result.get("latitude")
    longitude = geocode_result.get("longitude")
    return latitude is not None and longitude is not None


def _llm_failure_reason(llm_result: Optional[dict]) -> Optional[str]:
    """Return a failure reason string if llm_result signals a hard failure.

    A hard failure is signalled by an ``"error"`` key with a truthy value,
    or a ``"_failed"`` key set to True. Returns None if llm_result does
    not represent a hard failure (i.e. it is a normal, possibly partial,
    extraction result).
    """
    if not isinstance(llm_result, dict):
        return None

    error = llm_result.get("error")
    if error:
        return str(error)

    if llm_result.get("_failed"):
        reason = llm_result.get("reason")
        return str(reason) if reason else "LLM extraction failed"

    return None


def _unresolved_llm_fields(llm_result: Optional[dict]) -> List[str]:
    """Return the names of LLM attributes not resolved in llm_result."""
    unresolved: List[str] = []
    result = llm_result or {}
    for attribute in LLM_ATTRIBUTES:
        value = result.get(attribute)
        if value is None:
            unresolved.append(attribute)
        elif isinstance(value, (list, str)) and len(value) == 0:
            unresolved.append(attribute)
    return unresolved


def _unresolved_soft_fields(llm_result: Optional[dict]) -> List[str]:
    """Return the names of soft (vibe) attributes not resolved in llm_result."""
    unresolved: List[str] = []
    result = llm_result or {}
    for attribute in SOFT_LLM_ATTRIBUTES:
        value = result.get(attribute)
        if value is None:
            unresolved.append(attribute)
        elif isinstance(value, (list, str)) and len(value) == 0:
            unresolved.append(attribute)
    return unresolved


def determine_enrichment_state(
    geocode_result: Optional[dict],
    llm_result: Optional[dict],
) -> Tuple[str, List[str], Optional[str]]:
    """Determine the enrichment state for a single Job_Listing.

    Args:
        geocode_result: The geocode lookup result, or None/{} if
            unresolved. See module docstring for the contract.
        llm_result: The LLM extraction result, or a failure marker dict.
            See module docstring for the contract.

    Returns:
        A tuple ``(state, unresolved_attributes, failure_reason)``.
    """
    # A hard failure (timeout or exception) during LLM extraction takes
    # precedence: the whole record could not be processed.
    hard_failure_reason = _llm_failure_reason(llm_result)
    if hard_failure_reason is not None:
        return "failed", list(ALL_ATTRIBUTES), hard_failure_reason

    geocode_resolved = _is_geocode_resolved(geocode_result)
    unresolved_llm = _unresolved_llm_fields(llm_result)
    # Soft attributes are reported as unresolved when missing, but do not
    # affect the enriched/partially_enriched decision below.
    unresolved_soft = _unresolved_soft_fields(llm_result)

    unresolved: List[str] = []
    if not geocode_resolved:
        unresolved.append(LOCATION_ATTRIBUTE)
    unresolved.extend(unresolved_llm)

    # Enriched state is decided solely by location + the required LLM
    # attributes; soft vibe attributes never downgrade it.
    if not unresolved:
        return "enriched", list(unresolved_soft), None

    unresolved.extend(unresolved_soft)

    reasons = []
    if not geocode_resolved:
        reasons.append("no matching Geocode_Lookup entry for location text")
    if unresolved_llm:
        reasons.append(
            "Foundation_Model_APIs returned a subset of required attributes: "
            + ", ".join(unresolved_llm)
        )
    failure_reason = "; ".join(reasons)

    return "partially_enriched", unresolved, failure_reason
