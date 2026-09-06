"""User location resolution against the offline Geocode_Lookup dataset.

Resolves a user-submitted home location string (city name or postal code)
to latitude/longitude coordinates by querying `job_agent.ops.geocode_lookup`
via the SQL warehouse. No outbound network requests are issued.

Validates: Requirements 6.1, 6.5, 6.6, 6.7, 6.8
"""

from __future__ import annotations

from typing import Optional

from src.utils.input_validation import validate_location_input

GEOCODE_LOOKUP_TABLE = "job_agent.ops.geocode_lookup"

RESOLVE_LOCATION_QUERY = f"""
SELECT city_name, latitude, longitude
FROM {GEOCODE_LOOKUP_TABLE}
WHERE LOWER(city_name) = LOWER(:input) OR postal_code = :input
LIMIT 1
"""


def resolve_location(spark, input_text: str) -> Optional[dict]:
    """Resolve a home location input to a city name and coordinates.

    Validates `input_text` (1-200 characters, Requirement 6.6), then queries
    `job_agent.ops.geocode_lookup` via the SQL warehouse, matching on a
    case-insensitive city name or an exact postal code (Requirement 6.5).

    Args:
        spark: The active SparkSession (connected to the SQL warehouse).
        input_text: The user-submitted home location string (city name or
            postal code).

    Returns:
        A dict with keys `city_name`, `latitude`, `longitude` for the first
        matching row, or `None` if no entry matches (Requirement 6.7).

    Raises:
        ValueError: If `input_text` fails validation (empty or >200 chars),
            with the validation error message (Requirement 6.6).
    """
    is_valid, message = validate_location_input(input_text)
    if not is_valid:
        raise ValueError(message)

    result_df = spark.sql(RESOLVE_LOCATION_QUERY, args={"input": input_text})
    rows = result_df.collect()

    if not rows:
        return None

    row = rows[0]
    return {
        "city_name": row.city_name,
        "latitude": row.latitude,
        "longitude": row.longitude,
    }


def resolve_location_via_warehouse(input_text: str, warehouse_id: Optional[str] = None) -> Optional[dict]:
    """Resolve a home location using the SQL warehouse via the Databricks SDK.

    Unlike :func:`resolve_location` (which needs a SparkSession /
    Databricks Connect), this path uses ``WorkspaceClient().statement_execution``
    so it works inside a Databricks App container, where no ambient Spark
    session exists. Matches case-insensitive city name or exact postal code
    against ``job_agent.ops.geocode_lookup``.

    Args:
        input_text: The user-submitted home location string.
        warehouse_id: SQL warehouse id. Falls back to the ``SQL_WAREHOUSE_ID``
            / ``DATABRICKS_WAREHOUSE_ID`` env var.

    Returns:
        A dict with ``city_name``/``latitude``/``longitude`` for the first
        match, or ``None`` if nothing matches.

    Raises:
        ValueError: If ``input_text`` fails validation.
        RuntimeError: If no warehouse id is available.
    """
    import os

    from databricks.sdk import WorkspaceClient

    is_valid, message = validate_location_input(input_text)
    if not is_valid:
        raise ValueError(message)

    resolved_warehouse_id = (
        warehouse_id
        or os.environ.get("SQL_WAREHOUSE_ID")
        or os.environ.get("DATABRICKS_WAREHOUSE_ID")
    )
    if not resolved_warehouse_id:
        raise RuntimeError(
            "No SQL warehouse id available for location resolution "
            "(set SQL_WAREHOUSE_ID)."
        )

    safe_input = input_text.replace("'", "''")
    statement = (
        "SELECT city_name, latitude, longitude "
        f"FROM {GEOCODE_LOOKUP_TABLE} "
        f"WHERE LOWER(city_name) = LOWER('{safe_input}') OR postal_code = '{safe_input}' "
        "LIMIT 1"
    )

    w = WorkspaceClient()
    result = w.statement_execution.execute_statement(
        warehouse_id=resolved_warehouse_id,
        statement=statement,
        wait_timeout="30s",
    )
    rows = (result.result.data_array if result.result else None) or []
    if not rows:
        return None

    city_name, latitude, longitude = rows[0]
    return {
        "city_name": city_name,
        "latitude": float(latitude) if latitude is not None else None,
        "longitude": float(longitude) if longitude is not None else None,
    }
