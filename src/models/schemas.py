"""PySpark schema definitions for all Unity Catalog Delta tables.

Each schema mirrors the table definitions in the design document's
"Data Models" section. Schemas are expressed with `pyspark.sql.types` so
they can be used directly with `spark.createDataFrame(data, schema=...)`
or DDL generation in the catalog/table setup notebook.

Requirements: 2.2, 3.1, 4.4, 11.2
"""

from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Bronze tier
# ---------------------------------------------------------------------------

BRONZE_JOB_LISTINGS_SCHEMA = StructType(
    [
        StructField("listing_id", StringType(), nullable=False),
        StructField("job_title", StringType(), nullable=False),
        StructField("company_name", StringType(), nullable=False),
        StructField("job_description", StringType(), nullable=True),
        StructField("location_text", StringType(), nullable=True),
        StructField("source_url", StringType(), nullable=False),
        StructField("ingestion_timestamp", TimestampType(), nullable=False),
        StructField("ingestion_mode", StringType(), nullable=False),
        StructField("enrichment_state", StringType(), nullable=False),
        StructField("source_domain", StringType(), nullable=False),
    ]
)

BRONZE_INGESTION_ERRORS_SCHEMA = StructType(
    [
        StructField("error_id", StringType(), nullable=False),
        StructField("source_domain", StringType(), nullable=True),
        StructField("source_url", StringType(), nullable=True),
        StructField("error_type", StringType(), nullable=False),
        StructField("error_message", StringType(), nullable=False),
        StructField("missing_fields", ArrayType(StringType()), nullable=True),
        StructField("error_timestamp", TimestampType(), nullable=False),
    ]
)

# ---------------------------------------------------------------------------
# Silver tier
# ---------------------------------------------------------------------------

SILVER_ENRICHED_LISTINGS_SCHEMA = StructType(
    [
        StructField("listing_id", StringType(), nullable=False),
        StructField("job_title", StringType(), nullable=False),
        StructField("company_name", StringType(), nullable=False),
        StructField("job_description", StringType(), nullable=True),
        StructField("location_text", StringType(), nullable=True),
        StructField("source_url", StringType(), nullable=False),
        StructField("latitude", DoubleType(), nullable=True),
        StructField("longitude", DoubleType(), nullable=True),
        StructField("required_skills", ArrayType(StringType()), nullable=True),
        StructField("required_skills_text", StringType(), nullable=True),
        StructField("seniority_level", StringType(), nullable=True),
        StructField("employment_type", StringType(), nullable=True),
        StructField("industry", StringType(), nullable=True),
        StructField("company_size_band", StringType(), nullable=True),
        StructField("company_vibe", StringType(), nullable=True),
        StructField("office_policy", StringType(), nullable=True),
        StructField("benefits_rating", StringType(), nullable=True),
        StructField("enrichment_state", StringType(), nullable=False),
        StructField("unresolved_attributes", ArrayType(StringType()), nullable=True),
        StructField("failure_reason", StringType(), nullable=True),
        StructField("enrichment_timestamp", TimestampType(), nullable=False),
        StructField("embedding_text", StringType(), nullable=True),
    ]
)

SILVER_ENRICHED_LISTINGS_CHUNKS_SCHEMA = StructType(
    [
        StructField("chunk_id", StringType(), nullable=False),
        StructField("listing_id", StringType(), nullable=False),
        StructField("chunk_index", IntegerType(), nullable=False),
        StructField("embedding_text", StringType(), nullable=False),
        StructField("job_title", StringType(), nullable=False),
        StructField("company_name", StringType(), nullable=False),
        StructField("latitude", DoubleType(), nullable=True),
        StructField("longitude", DoubleType(), nullable=True),
        StructField("enrichment_state", StringType(), nullable=False),
    ]
)

# ---------------------------------------------------------------------------
# Gold tier
# ---------------------------------------------------------------------------

GOLD_USER_PROFILES_SCHEMA = StructType(
    [
        StructField("profile_id", StringType(), nullable=False),
        StructField("skills", ArrayType(StringType()), nullable=True),
        StructField("years_of_experience", IntegerType(), nullable=True),
        StructField("education_history", StringType(), nullable=True),
        StructField("job_title_history", ArrayType(StringType()), nullable=True),
        StructField("qualifications_summary", StringType(), nullable=True),
        StructField("home_latitude", DoubleType(), nullable=True),
        StructField("home_longitude", DoubleType(), nullable=True),
        StructField("home_location_name", StringType(), nullable=True),
        StructField("commute_radius_km", IntegerType(), nullable=False),
        StructField("cv_file_path", StringType(), nullable=True),
        StructField("created_at", TimestampType(), nullable=False),
        StructField("updated_at", TimestampType(), nullable=False),
    ]
)

GOLD_PIPELINE_SUMMARY_SCHEMA = StructType(
    [
        StructField("run_id", StringType(), nullable=False),
        StructField("pipeline_name", StringType(), nullable=False),
        StructField("ingestion_mode", StringType(), nullable=True),
        StructField("records_added", IntegerType(), nullable=False),
        StructField("records_updated", IntegerType(), nullable=False),
        StructField("records_skipped", IntegerType(), nullable=False),
        StructField("source_errors", IntegerType(), nullable=False),
        StructField("enriched_count", IntegerType(), nullable=True),
        StructField("partially_enriched_count", IntegerType(), nullable=True),
        StructField("failed_count", IntegerType(), nullable=True),
        StructField("unenriched_count", IntegerType(), nullable=True),
        StructField("run_timestamp", TimestampType(), nullable=False),
    ]
)

# ---------------------------------------------------------------------------
# Ops tier
# ---------------------------------------------------------------------------

OPS_REACHABILITY_REPORT_SCHEMA = StructType(
    [
        StructField("domain", StringType(), nullable=False),
        StructField("probe_timestamp", TimestampType(), nullable=False),
        StructField("outcome", StringType(), nullable=False),
        StructField("http_status_code", IntegerType(), nullable=True),
        StructField("error_message", StringType(), nullable=True),
        StructField("reachable_count", IntegerType(), nullable=True),
        StructField("blocked_count", IntegerType(), nullable=True),
    ]
)

OPS_GEOCODE_LOOKUP_SCHEMA = StructType(
    [
        StructField("city_name", StringType(), nullable=False),
        StructField("postal_code", StringType(), nullable=True),
        StructField("country", StringType(), nullable=False),
        StructField("latitude", DoubleType(), nullable=False),
        StructField("longitude", DoubleType(), nullable=False),
    ]
)

OPS_BATCH_CHECKPOINTS_SCHEMA = StructType(
    [
        StructField("pipeline_name", StringType(), nullable=False),
        StructField("run_id", StringType(), nullable=False),
        StructField("last_successful_batch", IntegerType(), nullable=False),
        StructField("checkpoint_timestamp", TimestampType(), nullable=False),
    ]
)

OPS_INDEXING_ERRORS_SCHEMA = StructType(
    [
        StructField("error_id", StringType(), nullable=False),
        StructField("error_message", StringType(), nullable=False),
        StructField("error_timestamp", TimestampType(), nullable=False),
    ]
)

OPS_EVALUATION_DATASET_SCHEMA = StructType(
    [
        StructField("case_id", StringType(), nullable=False),
        StructField("user_profile", StringType(), nullable=False),
        StructField("expected_listing_ids", ArrayType(StringType()), nullable=False),
        StructField("description", StringType(), nullable=True),
    ]
)

OPS_EVALUATION_RESULTS_SCHEMA = StructType(
    [
        StructField("eval_run_id", StringType(), nullable=False),
        StructField("model_version", IntegerType(), nullable=False),
        StructField("match_relevance_mean", DoubleType(), nullable=False),
        StructField("groundedness_mean", DoubleType(), nullable=False),
        StructField("eval_timestamp", TimestampType(), nullable=False),
    ]
)

# ---------------------------------------------------------------------------
# Registry — maps fully-qualified (schema.table) names to their StructType.
# Used by the catalog/table setup notebook (task 3.2) to create all tables.
# ---------------------------------------------------------------------------

TABLE_SCHEMAS = {
    "bronze.job_listings": BRONZE_JOB_LISTINGS_SCHEMA,
    "bronze.ingestion_errors": BRONZE_INGESTION_ERRORS_SCHEMA,
    "silver.enriched_listings": SILVER_ENRICHED_LISTINGS_SCHEMA,
    "silver.enriched_listings_chunks": SILVER_ENRICHED_LISTINGS_CHUNKS_SCHEMA,
    "gold.user_profiles": GOLD_USER_PROFILES_SCHEMA,
    "gold.pipeline_summary": GOLD_PIPELINE_SUMMARY_SCHEMA,
    "ops.reachability_report": OPS_REACHABILITY_REPORT_SCHEMA,
    "ops.geocode_lookup": OPS_GEOCODE_LOOKUP_SCHEMA,
    "ops.batch_checkpoints": OPS_BATCH_CHECKPOINTS_SCHEMA,
    "ops.indexing_errors": OPS_INDEXING_ERRORS_SCHEMA,
    "ops.evaluation_dataset": OPS_EVALUATION_DATASET_SCHEMA,
    "ops.evaluation_results": OPS_EVALUATION_RESULTS_SCHEMA,
}
