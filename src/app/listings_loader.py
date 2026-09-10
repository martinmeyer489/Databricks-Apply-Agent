"""Listings loader for the map + chat frontend.

Reads enriched job listings from ``{catalog}.silver.enriched_listings`` via the
serverless SQL warehouse (Databricks SDK statement execution), applying the
map filters as SQL predicates so only the rows the user asked for come back.

Kept independent of ``gradio``/``plotly`` so it can be unit-tested without a UI
or a plotting backend. The only heavy dependency is the Databricks SDK, and
even that is imported lazily inside :func:`load_listings` so importing this
module (e.g. to build filter choices) never requires a workspace connection.

Validates: map data source for the redesigned app (silver.enriched_listings).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Columns selected for the map, hover cards, filters, and the chat context.
LISTING_COLUMNS = (
    "listing_id",
    "job_title",
    "company_name",
    "location_text",
    "source_url",
    "latitude",
    "longitude",
    "seniority_level",
    "employment_type",
    "industry",
    "company_size_band",
    "company_vibe",
    "office_policy",
    "benefits_rating",
    "required_skills_text",
)

# Filterable categorical attributes → the enriched-listings column backing them.
FILTER_COLUMNS = {
    "industry": "industry",
    "seniority_level": "seniority_level",
    "employment_type": "employment_type",
    "company_size_band": "company_size_band",
    "office_policy": "office_policy",
    "benefits_rating": "benefits_rating",
    "company_vibe": "company_vibe",
}

# Hard cap so a filter that matches everything can't drag the whole corpus into
# the browser. Clustering on the map keeps large result sets legible, so this
# is generous; the table shows the same rows.
MAX_LISTINGS = 2000


def _catalog() -> str:
    """Unity Catalog root, from ``CATALOG_NAME`` (mirrors app.yml default)."""
    return os.environ.get("CATALOG_NAME", "job_agent")


def enriched_listings_table() -> str:
    """Three-level name of the enriched listings table."""
    return f"{_catalog()}.silver.enriched_listings"


def _resolve_warehouse_id(warehouse_id: Optional[str]) -> str:
    """Resolve the SQL warehouse id from arg or environment.

    Mirrors ``location_resolver.resolve_location_via_warehouse`` so the whole
    app authenticates to compute the same way.
    """
    resolved = (
        warehouse_id
        or os.environ.get("SQL_WAREHOUSE_ID")
        or os.environ.get("DATABRICKS_WAREHOUSE_ID")
    )
    if not resolved:
        raise RuntimeError(
            "No SQL warehouse id available for loading listings (set SQL_WAREHOUSE_ID)."
        )
    return resolved


def _sql_literal(value: str) -> str:
    """Escape a string for safe inline use in a SQL single-quoted literal."""
    return value.replace("'", "''")


def build_listings_query(
    filters: Optional[Dict[str, Any]] = None,
    skill_query: Optional[str] = None,
    limit: int = MAX_LISTINGS,
) -> str:
    """Build the parameter-free SELECT for the enriched listings map.

    Args:
        filters: Mapping of filter name (a key of :data:`FILTER_COLUMNS`) to a
            selected value. Empty / ``None`` / ``"All"`` values are ignored.
        skill_query: Optional free-text term matched (case-insensitively)
            against the job title and the flattened required-skills text.
        limit: Maximum rows to return (capped at :data:`MAX_LISTINGS`).

    Returns:
        A SQL string. Only rows with a resolved latitude/longitude are
        returned, since a listing without coordinates cannot be placed on the
        map.
    """
    filters = filters or {}
    columns = ", ".join(LISTING_COLUMNS)
    where = [
        "enrichment_state = 'enriched'",
        "latitude IS NOT NULL",
        "longitude IS NOT NULL",
    ]

    for filter_name, column in FILTER_COLUMNS.items():
        selected = filters.get(filter_name)
        if selected and selected != "All":
            where.append(f"{column} = '{_sql_literal(str(selected))}'")

    if skill_query and skill_query.strip():
        term = _sql_literal(skill_query.strip().lower())
        where.append(
            "(LOWER(job_title) LIKE '%" + term + "%' "
            "OR LOWER(COALESCE(required_skills_text, '')) LIKE '%" + term + "%')"
        )

    capped = max(1, min(int(limit), MAX_LISTINGS))
    return (
        f"SELECT {columns} FROM {enriched_listings_table()} "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY company_name, job_title "
        f"LIMIT {capped}"
    )


def _rows_to_dicts(columns: List[str], data_array: List[List[Any]]) -> List[Dict[str, Any]]:
    """Zip a statement-execution ``data_array`` into a list of row dicts.

    Latitude/longitude come back as strings from the statement API; coerce them
    to floats (dropping rows whose coordinates can't be parsed) so the map layer
    receives numeric positions.
    """
    listings: List[Dict[str, Any]] = []
    for raw in data_array or []:
        row = dict(zip(columns, raw))
        try:
            row["latitude"] = float(row["latitude"])
            row["longitude"] = float(row["longitude"])
        except (TypeError, ValueError):
            continue
        listings.append(row)
    return listings


def load_listings(
    filters: Optional[Dict[str, Any]] = None,
    skill_query: Optional[str] = None,
    warehouse_id: Optional[str] = None,
    limit: int = MAX_LISTINGS,
) -> List[Dict[str, Any]]:
    """Load filtered enriched listings from the SQL warehouse.

    Args:
        filters: See :func:`build_listings_query`.
        skill_query: See :func:`build_listings_query`.
        warehouse_id: SQL warehouse id; falls back to ``SQL_WAREHOUSE_ID`` /
            ``DATABRICKS_WAREHOUSE_ID``.
        limit: Maximum rows to return.

    Returns:
        A list of listing dicts (keys = :data:`LISTING_COLUMNS`, with
        ``latitude``/``longitude`` as floats).

    Raises:
        RuntimeError: If no warehouse id is available.
    """
    resolved_warehouse_id = _resolve_warehouse_id(warehouse_id)
    statement = build_listings_query(filters=filters, skill_query=skill_query, limit=limit)

    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient()
    result = client.statement_execution.execute_statement(
        warehouse_id=resolved_warehouse_id,
        statement=statement,
        wait_timeout="30s",
    )
    data_array = (result.result.data_array if result.result else None) or []
    return _rows_to_dicts(list(LISTING_COLUMNS), data_array)


def load_filter_options(
    warehouse_id: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Fetch the distinct values for each categorical filter.

    Returns a mapping of filter name → sorted distinct values (each list
    prefixed with ``"All"`` so the UI can offer a no-filter choice). On any
    failure to reach the warehouse, returns just ``{"All"}`` per filter so the
    UI still renders (the map load will surface the real error).
    """
    options: Dict[str, List[str]] = {name: ["All"] for name in FILTER_COLUMNS}
    try:
        resolved_warehouse_id = _resolve_warehouse_id(warehouse_id)
        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient()
        for filter_name, column in FILTER_COLUMNS.items():
            statement = (
                f"SELECT DISTINCT {column} FROM {enriched_listings_table()} "
                f"WHERE {column} IS NOT NULL AND enrichment_state = 'enriched' "
                f"ORDER BY {column} LIMIT 100"
            )
            result = client.statement_execution.execute_statement(
                warehouse_id=resolved_warehouse_id,
                statement=statement,
                wait_timeout="30s",
            )
            data_array = (result.result.data_array if result.result else None) or []
            values = [row[0] for row in data_array if row and row[0]]
            options[filter_name] = ["All", *values]
    except Exception:  # noqa: BLE001 - degrade gracefully to "All" only
        return {name: ["All"] for name in FILTER_COLUMNS}
    return options
