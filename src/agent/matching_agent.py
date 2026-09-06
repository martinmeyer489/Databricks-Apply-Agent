"""Matching Agent.

Authored with the Agent Framework `ResponsesAgent` interface. Matches a
User_Profile against the job listing corpus using UC Function tools for
all retrieval, distance computation, and profile lookup, then ranks and
returns the top candidates within a 60-second budget.

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10,
7.11, 9.1
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional

LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

UC_FUNCTION_NAMES = [
    "job_agent.gold.search_listings",
    "job_agent.gold.compute_commute_distance",
    "job_agent.gold.get_user_profile",
    "job_agent.gold.draft_application",
]

MAX_SEARCH_CANDIDATES = 200
MAX_RESULTS = 50
MAX_EXPLANATION_CHARS = 500
MATCHING_TIMEOUT_SECONDS = 60

NO_MATCHES_SUGGESTION = (
    "No matches found within {radius} km. Try increasing your commute "
    "radius or updating your CV."
)


# --------------------------------------------------------------------------
# Agent construction (Req 7.1, 7.2, 9.1)
# --------------------------------------------------------------------------


def build_agent(warehouse_id: Optional[str] = None):
    """Build the tool callables backing the Matching Agent.

    The matching flow in :class:`MatchingAgent` drives retrieval by calling
    three UC Function tools directly (``get_user_profile``,
    ``search_listings``, ``compute_commute_distance``) rather than delegating
    to a free-form LLM agent loop, so this function returns a plain dict of
    callables wired to those Unity Catalog functions. Each callable executes
    the corresponding ``job_agent.gold.*`` UC function on the serverless SQL
    warehouse via the Databricks SDK statement execution API.

    (Earlier revisions returned a LangChain ``AgentExecutor`` here, but it was
    never invoked by the matching flow and its constructor API was removed in
    LangChain 1.x; returning the tools dict directly both fixes that import
    break and ensures the deployed model actually has working tools.)

    Args:
        warehouse_id: SQL warehouse used to execute the UC functions. If not
            provided, falls back to the ``SQL_WAREHOUSE_ID`` / ``DATABRICKS_WAREHOUSE_ID``
            environment variable.

    Returns:
        A dict with keys ``get_user_profile``, ``search_listings``, and
        ``compute_commute_distance`` mapping to callables.
    """
    import os

    import mlflow
    from databricks.sdk import WorkspaceClient

    mlflow.langchain.autolog()

    resolved_warehouse_id = (
        warehouse_id
        or os.environ.get("SQL_WAREHOUSE_ID")
        or os.environ.get("DATABRICKS_WAREHOUSE_ID")
    )

    w = WorkspaceClient()

    def _run(statement: str) -> list[list[Any]]:
        result = w.statement_execution.execute_statement(
            warehouse_id=resolved_warehouse_id,
            statement=statement,
            wait_timeout="50s",
        )
        if result.result and result.result.data_array:
            return result.result.data_array
        return []

    def _sql_str(value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    def get_user_profile(profile_id: str) -> Dict[str, Any]:
        rows = _run(
            "SELECT * FROM "
            f"job_agent.gold.get_user_profile({_sql_str(profile_id)})"
        )
        if not rows:
            return {}
        (
            skills,
            years_of_experience,
            job_title_history,
            qualifications_summary,
            home_latitude,
            home_longitude,
            home_location_name,
            commute_radius_km,
        ) = rows[0]
        return {
            "skills": skills,
            "years_of_experience": years_of_experience,
            "job_title_history": job_title_history,
            "qualifications_summary": qualifications_summary,
            "home_latitude": float(home_latitude) if home_latitude is not None else None,
            "home_longitude": float(home_longitude) if home_longitude is not None else None,
            "home_location_name": home_location_name,
            "commute_radius_km": int(commute_radius_km) if commute_radius_km is not None else None,
        }

    def search_listings(query_text: str, max_results: int = MAX_SEARCH_CANDIDATES) -> List[Dict[str, Any]]:
        rows = _run(
            "SELECT listing_id, job_title, company_name, latitude, longitude, "
            "enrichment_state, similarity_score FROM "
            f"job_agent.gold.search_listings({_sql_str(query_text)}, {int(max_results)})"
        )
        results = []
        for r in rows:
            results.append(
                {
                    "listing_id": r[0],
                    "job_title": r[1],
                    "company_name": r[2],
                    "latitude": float(r[3]) if r[3] is not None else None,
                    "longitude": float(r[4]) if r[4] is not None else None,
                    "enrichment_state": r[5],
                    "similarity_score": float(r[6]) if r[6] is not None else 0.0,
                }
            )
        return results

    def compute_commute_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        rows = _run(
            "SELECT job_agent.gold.compute_commute_distance("
            f"{float(lat1)}, {float(lon1)}, {float(lat2)}, {float(lon2)})"
        )
        return float(rows[0][0]) if rows and rows[0][0] is not None else float("inf")

    return {
        "get_user_profile": get_user_profile,
        "search_listings": search_listings,
        "compute_commute_distance": compute_commute_distance,
    }


# --------------------------------------------------------------------------
# Pure matching logic (Req 7.4 - 7.10)
# --------------------------------------------------------------------------


REQUIRED_PROFILE_FIELDS = ("skills", "home_coordinates", "commute_radius_km")


def validate_profile_completeness(profile: Dict[str, Any]) -> List[str]:
    """Return the list of missing required User_Profile fields.

    Mirrors Requirement 7.10: if the profile is missing the skills list,
    the home coordinates, or the commute radius, the caller must be told
    exactly which fields are missing so no retrieval is attempted.

    Args:
        profile: dict that may contain `skills`, `home_latitude`,
            `home_longitude`, and `commute_radius_km`.

    Returns:
        A list containing zero or more of: "skills", "home_coordinates",
        "commute_radius_km". Empty list means the profile is complete.
    """
    missing: List[str] = []

    if not profile.get("skills"):
        missing.append("skills")

    if profile.get("home_latitude") is None or profile.get("home_longitude") is None:
        missing.append("home_coordinates")

    if profile.get("commute_radius_km") is None:
        missing.append("commute_radius_km")

    return missing


def build_search_query_text(profile: Dict[str, Any]) -> str:
    """Build the Vector Search query text from a User_Profile.

    Concatenates skills, job title history, and qualifications summary, as
    required for the `search_listings` semantic query (Req 7.4).
    """
    skills = profile.get("skills") or []
    job_titles = profile.get("job_title_history") or []
    qualifications_summary = profile.get("qualifications_summary") or ""

    parts = [", ".join(skills), ", ".join(job_titles), qualifications_summary]
    return " ".join(part for part in parts if part).strip()


def filter_by_commute_radius(
    candidates: List[Dict[str, Any]],
    home_lat: float,
    home_lon: float,
    radius_km: float,
    distance_fn,
) -> List[Dict[str, Any]]:
    """Exclude candidates whose commute distance exceeds `radius_km` (Req 7.6).

    Each surviving candidate has its computed `distance_km` (rounded to 1
    decimal place by `distance_fn`) attached.

    Args:
        candidates: list of dicts, each with `latitude`/`longitude`.
        home_lat: User_Profile home latitude.
        home_lon: User_Profile home longitude.
        radius_km: Commute_Radius in kilometres.
        distance_fn: callable `(lat1, lon1, lat2, lon2) -> float` used to
            compute the commute distance (the `compute_commute_distance`
            UC Function tool, or the equivalent haversine helper).

    Returns:
        The subset of `candidates` within `radius_km`, each augmented with
        a `distance_km` key.
    """
    kept = []
    for candidate in candidates:
        distance_km = distance_fn(
            home_lat, home_lon, candidate["latitude"], candidate["longitude"]
        )
        if distance_km <= radius_km:
            enriched = dict(candidate)
            enriched["distance_km"] = round(distance_km, 1)
            kept.append(enriched)
    return kept


def score_relevance(candidate: Dict[str, Any]) -> int:
    """Assign a Relevance_Score from 0 to 100 inclusive (Req 7.5).

    The candidate's `similarity_score` (as returned by `search_listings`,
    typically in [0, 1]) is scaled to the 0-100 range and clamped so
    upstream floating point noise can never escape the documented bounds.
    """
    similarity_score = candidate.get("similarity_score") or 0.0
    scaled = round(similarity_score * 100)
    return max(0, min(100, scaled))


def build_match_explanation(candidate: Dict[str, Any], profile: Dict[str, Any]) -> str:
    """Build a match explanation for a candidate, truncated to 500 chars (Req 7.8)."""
    job_title = candidate.get("job_title", "this role")
    company_name = candidate.get("company_name", "the company")
    distance_km = candidate.get("distance_km")

    explanation = (
        f"Matched to {job_title} at {company_name} based on profile skill "
        f"and experience overlap"
    )
    if distance_km is not None:
        explanation += f", {distance_km} km from your home location"
    explanation += "."

    return truncate_explanation(explanation, MAX_EXPLANATION_CHARS)


def truncate_explanation(text: str, max_len: int = MAX_EXPLANATION_CHARS) -> str:
    """Truncate `text` to at most `max_len` characters (Req 7.8)."""
    return text[:max_len]


def rank_and_limit_matches(
    scored_candidates: List[Dict[str, Any]], max_results: int = MAX_RESULTS
) -> List[Dict[str, Any]]:
    """Sort candidates by Relevance_Score descending and cap at `max_results` (Req 7.7)."""
    ranked = sorted(scored_candidates, key=lambda c: c["relevance_score"], reverse=True)
    return ranked[:max_results]


def build_empty_result_message(radius_km: float) -> str:
    """Build the suggestion message returned when no candidates remain (Req 7.9)."""
    return NO_MATCHES_SUGGESTION.format(radius=radius_km)


# --------------------------------------------------------------------------
# ResponsesAgent implementation
# --------------------------------------------------------------------------


class MatchingAgent:
    """Matching Agent authored with the Agent Framework `ResponsesAgent` interface.

    On Databricks, this class subclasses `mlflow.pyfunc.ResponsesAgent` so it
    can be logged to MLflow, registered in Unity Catalog, and deployed to a
    Model Serving endpoint (Req 7.1). The base class is resolved lazily in
    `__init__` (rather than at import time) so this module can be imported
    and unit-tested outside a Databricks runtime, where `mlflow` may not be
    installed.

    Tool access is injected via `tools` so the class can be tested with
    fakes; in production, `tools` wraps the UC Function calls exposed by
    `build_agent()`'s `UCFunctionToolkit`.
    """

    def __init__(self, tools: Optional[Dict[str, Any]] = None, agent: Optional[Any] = None):
        self.tools = tools or {}
        self._agent = agent

    # -- tool call helpers (delegate to injected callables, or the real
    #    UC Function tools / agent executor in production) --------------

    def _get_user_profile(self, profile_id: str) -> Dict[str, Any]:
        return self.tools["get_user_profile"](profile_id)

    def _search_listings(self, query_text: str, max_results: int = MAX_SEARCH_CANDIDATES) -> List[Dict[str, Any]]:
        return self.tools["search_listings"](query_text, max_results)

    def _compute_commute_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        return self.tools["compute_commute_distance"](lat1, lon1, lat2, lon2)

    # -- core matching flow (Req 7) ---------------------------------------

    def match(self, profile_id: str) -> Dict[str, Any]:
        """Run the full matching flow for `profile_id` (Req 7.1-7.10).

        Returns a dict shaped like:
            {"error": "...", "missing_fields": [...]}   -- on incomplete profile
            {"results": [...], "message": "..."}         -- otherwise (message
                                                             present only when
                                                             results is empty)
        """
        profile = self._get_user_profile(profile_id)

        missing_fields = validate_profile_completeness(profile)
        if missing_fields:
            return {
                "error": f"Cannot match: please complete {', '.join(missing_fields)} first.",
                "missing_fields": missing_fields,
            }

        query_text = build_search_query_text(profile)
        candidates = self._search_listings(query_text, MAX_SEARCH_CANDIDATES)

        within_radius = filter_by_commute_radius(
            candidates,
            profile["home_latitude"],
            profile["home_longitude"],
            profile["commute_radius_km"],
            self._compute_commute_distance,
        )

        if not within_radius:
            return {
                "results": [],
                "message": build_empty_result_message(profile["commute_radius_km"]),
            }

        scored = []
        for candidate in within_radius:
            relevance_score = score_relevance(candidate)
            explanation = build_match_explanation(candidate, profile)
            scored.append(
                {
                    **candidate,
                    "relevance_score": relevance_score,
                    "explanation": explanation,
                }
            )

        results = rank_and_limit_matches(scored, MAX_RESULTS)
        return {"results": results}

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """`ResponsesAgent` entry point (Req 7.1).

        Extracts `profile_id` from the incoming request, runs `match`
        under a best-effort 60-second budget (Req 7.11), and returns a
        response dict. Tool call latencies are bounded by the underlying
        UC Functions / Vector Search SLAs; the `ThreadPoolExecutor`
        timeout here is a defensive backstop so a single invocation never
        exceeds the documented 60-second contract from the caller's
        perspective.
        """
        profile_id = request.get("profile_id") if isinstance(request, dict) else None
        if not profile_id:
            return {"error": "Cannot match: request is missing profile_id."}

        with ThreadPoolExecutor(max_workers=1) as executor:
            future: Future = executor.submit(self.match, profile_id)
            try:
                return future.result(timeout=MATCHING_TIMEOUT_SECONDS)
            except FuturesTimeoutError as exc:
                raise TimeoutError(
                    f"Matching timed out after {MATCHING_TIMEOUT_SECONDS}s"
                ) from exc


# --------------------------------------------------------------------------
# MLflow pyfunc adapter (Req 7.1, 9.1)
# --------------------------------------------------------------------------


def _extract_profile_id(model_input: Any) -> Optional[str]:
    """Pull `profile_id` out of the various shapes MLflow may hand `predict`.

    Model Serving / ``mlflow.pyfunc`` can deliver the input as a dict, a list
    of dicts (batch), or a pandas DataFrame with a ``profile_id`` column.
    """
    # pandas DataFrame (has a to_dict method)
    if hasattr(model_input, "to_dict") and not isinstance(model_input, dict):
        try:
            records = model_input.to_dict("records")
            if records:
                return records[0].get("profile_id")
        except Exception:  # noqa: BLE001
            return None
    if isinstance(model_input, dict):
        return model_input.get("profile_id")
    if isinstance(model_input, list) and model_input:
        first = model_input[0]
        if isinstance(first, dict):
            return first.get("profile_id")
    return None


class MatchingAgentModel:
    """`mlflow.pyfunc.PythonModel` adapter around :class:`MatchingAgent`.

    Subclassing is done lazily via ``__init_subclass__``-free indirection so
    this module still imports cleanly outside a Databricks/mlflow runtime
    (the unit tests import :class:`MatchingAgent` only). In production,
    ``make_pyfunc_model()`` returns an instance whose class actually derives
    from ``mlflow.pyfunc.PythonModel``.
    """

    def load_context(self, context):  # noqa: D401 - mlflow hook
        # Tools are constructed at serving time so the served model calls the
        # gold.* UC functions via the SQL warehouse (resolved from the
        # SQL_WAREHOUSE_ID env var configured on the serving endpoint).
        self._agent = MatchingAgent(tools=build_agent())

    def predict(self, context, model_input, params=None):  # noqa: D401 - mlflow hook
        if not hasattr(self, "_agent") or self._agent is None:
            self._agent = MatchingAgent(tools=build_agent())
        profile_id = _extract_profile_id(model_input)
        return self._agent.predict({"profile_id": profile_id})


def make_pyfunc_model():
    """Return an ``mlflow.pyfunc.PythonModel`` instance wrapping the agent.

    Resolved lazily so importing this module does not require mlflow.
    """
    import mlflow

    class _MatchingAgentPyfunc(mlflow.pyfunc.PythonModel, MatchingAgentModel):
        pass

    return _MatchingAgentPyfunc()
