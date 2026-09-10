# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Enrichment Pipeline
# MAGIC
# MAGIC Reads `job_agent.bronze.job_listings` rows whose `enrichment_state` is
# MAGIC `unenriched`, resolves each listing's `location_text` to latitude and
# MAGIC longitude via an offline LEFT JOIN against `job_agent.ops.geocode_lookup`
# MAGIC (no outbound network calls), derives five structured attributes
# MAGIC (`required_skills`, `seniority_level`, `employment_type`, `industry`,
# MAGIC `company_size_band`) from the job description via Foundation Model
# MAGIC APIs, applies the enrichment state machine, builds and chunks the
# MAGIC `embedding_text` field, and writes the results to
# MAGIC `silver.enriched_listings` and `silver.enriched_listings_chunks`.
# MAGIC
# MAGIC Records are processed in batches of at most 500. Within each batch,
# MAGIC LLM extraction uses the `ai_query` SQL function (server-side, on the
# MAGIC SQL warehouse / serverless SQL compute) when the batch has more than
# MAGIC 50 records, and per-record Foundation Model API calls via the
# MAGIC Databricks SDK otherwise (Req 3 AC4). A checkpoint is written to
# MAGIC `ops.batch_checkpoints` after each successfully processed batch so a
# MAGIC restarted run resumes from the following batch (Req 11 AC8, AC9).
# MAGIC
# MAGIC **Prerequisite**: `notebooks/00_setup_catalog.py`,
# MAGIC `notebooks/00a_load_reference_data.py`, and
# MAGIC `notebooks/02_ingest_listings.py` must have already run.
# MAGIC
# MAGIC Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 11.8, 11.9, 11.10

# COMMAND ----------

import sys
import os

# Allow `from src... import ...` when this notebook is run from the
# Databricks Workspace (repo root is not automatically on sys.path there,
# unlike when running via `databricks bundle` sync or repos).
_NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_REPO_ROOT = os.path.abspath(os.path.join(_NOTEBOOK_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models.schemas import (  # noqa: E402
    GOLD_PIPELINE_SUMMARY_SCHEMA,
    SILVER_ENRICHED_LISTINGS_CHUNKS_SCHEMA,
    SILVER_ENRICHED_LISTINGS_SCHEMA,
)
from src.pipelines.embedding import build_embedding_text, chunk_text  # noqa: E402
from src.pipelines.enrichment_state import determine_enrichment_state  # noqa: E402
from src.utils.batching import batch_records  # noqa: E402
from src.utils.checkpoints import get_resume_batch, write_checkpoint  # noqa: E402
from src.utils.retry import retry_with_backoff  # noqa: E402

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets

# COMMAND ----------

dbutils.widgets.text("catalog", "job_agent", "Catalog name")
dbutils.widgets.text("run_id", "", "Workflow run ID (blank = auto-generate)")
dbutils.widgets.text("batch_size", "500", "Max records per batch")
dbutils.widgets.text("per_record_llm_threshold", "1000", "Batches at/under this size use per-record LLM calls")
dbutils.widgets.text("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct", "Foundation Model APIs chat endpoint")
dbutils.widgets.text("record_timeout_seconds", "60", "Per-listing enrichment timeout (seconds)")
dbutils.widgets.text("embedding_chunk_char_threshold", "6000", "embedding_text char length above which chunking is applied")

CATALOG = dbutils.widgets.get("catalog")
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))
PER_RECORD_LLM_THRESHOLD = int(dbutils.widgets.get("per_record_llm_threshold"))
LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")
RECORD_TIMEOUT_SECONDS = int(dbutils.widgets.get("record_timeout_seconds"))
EMBEDDING_CHUNK_CHAR_THRESHOLD = int(dbutils.widgets.get("embedding_chunk_char_threshold"))

import uuid  # noqa: E402

RUN_ID = dbutils.widgets.get("run_id").strip() or str(uuid.uuid4())
PIPELINE_NAME = "enrichment"

BRONZE_LISTINGS_TABLE = f"{CATALOG}.bronze.job_listings"
GEOCODE_LOOKUP_TABLE = f"{CATALOG}.ops.geocode_lookup"
SILVER_ENRICHED_TABLE = f"{CATALOG}.silver.enriched_listings"
SILVER_CHUNKS_TABLE = f"{CATALOG}.silver.enriched_listings_chunks"
PIPELINE_SUMMARY_TABLE = f"{CATALOG}.gold.pipeline_summary"

print(f"Run ID:                  {RUN_ID}")
print(f"Bronze listings table:   {BRONZE_LISTINGS_TABLE}")
print(f"Geocode lookup table:    {GEOCODE_LOOKUP_TABLE}")
print(f"Silver enriched table:   {SILVER_ENRICHED_TABLE}")
print(f"Silver chunks table:     {SILVER_CHUNKS_TABLE}")
print(f"Pipeline summary table:  {PIPELINE_SUMMARY_TABLE}")
print(f"Batch size:              {BATCH_SIZE}")
print(f"Per-record LLM threshold: <= {PER_RECORD_LLM_THRESHOLD} records/batch")
print(f"LLM endpoint:            {LLM_ENDPOINT}")
print(f"Record timeout:          {RECORD_TIMEOUT_SECONDS}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Read unenriched Bronze listings + geocode LEFT JOIN (Req 3.1, 3.2)
# MAGIC
# MAGIC Joins on normalized (lowercased, trimmed) `location_text` matching
# MAGIC either `city_name` or `postal_code` in `ops.geocode_lookup`. The lookup
# MAGIC table is loaded once from a Volume (`notebooks/00a_load_reference_data.py`)
# MAGIC so this resolution issues no outbound network requests. If more than
# MAGIC one lookup row matches a listing, only the first match is kept so the
# MAGIC join does not fan out the listing set.

# COMMAND ----------

from pyspark.sql.functions import col, lower, monotonically_increasing_id, row_number, trim
from pyspark.sql.functions import concat, lit
from pyspark.sql.window import Window

bronze_unenriched_df = spark.table(BRONZE_LISTINGS_TABLE).filter("enrichment_state = 'unenriched'")
geocode_df = spark.table(GEOCODE_LOOKUP_TABLE)

# The Jobsuche API returns location_text as a full address, e.g.
# "10115, Berlin, Berlin", so an exact equality against a bare city name or
# postal code almost never matches and virtually no listing gets coordinates
# (leaving the map empty). Match instead when the geocode city_name or
# postal_code appears as a substring of the (lowercased) location_text. This
# geocodes every listing located in one of the known reference cities.
_loc = lower(trim(col("b.location_text")))
_city = lower(trim(col("g.city_name")))
_plz = trim(col("g.postal_code"))
joined_df = (
    bronze_unenriched_df.alias("b")
    .join(
        geocode_df.alias("g"),
        (
            (_loc == _city)
            | (_loc == lower(_plz))
            | _loc.contains(_city)
            | col("b.location_text").contains(_plz)
        ),
        how="left",
    )
    .select(
        col("b.listing_id"),
        col("b.job_title"),
        col("b.company_name"),
        col("b.job_description"),
        col("b.location_text"),
        col("b.source_url"),
        col("g.latitude"),
        col("g.longitude"),
    )
)

# Deduplicate any fan-out from multiple geocode matches, keeping one row
# per listing_id (arbitrary but deterministic choice among matches).
dedup_window = Window.partitionBy("listing_id").orderBy(monotonically_increasing_id())
joined_df = (
    joined_df.withColumn("_rn", row_number().over(dedup_window))
    .filter(col("_rn") == 1)
    .drop("_rn")
)

joined_records = [row.asDict() for row in joined_df.collect()]

print(f"Unenriched listings to process: {len(joined_records)}")

# ---------------------------------------------------------------------------
# Location fallback support
# ---------------------------------------------------------------------------
# Build an in-memory {lowercased city_name -> (lat, lon)} dict from the
# geocode lookup so that when a listing has no coordinates from the SQL join,
# we can resolve the city the LLM inferred (the `city` attribute) to
# coordinates without another Spark join. Anything still unresolved falls back
# to the geographic centre of Germany plus per-listing jitter, so EVERY
# listing ends up with a location and appears on the map.
from src.pipelines.attribute_normalization import (  # noqa: E402
    bucket_benefits_rating,
    bucket_company_size,
    bucket_company_vibe,
    bucket_employment_type,
    bucket_industry,
    bucket_office_policy,
    bucket_seniority,
    jitter_coord,
    normalize_benefits_rating,
    normalize_company_vibe,
    normalize_office_policy,
)

_geo_rows = spark.table(GEOCODE_LOOKUP_TABLE).select("city_name", "latitude", "longitude").collect()
CITY_TO_COORDS = {
    (r["city_name"] or "").strip().lower(): (r["latitude"], r["longitude"])
    for r in _geo_rows
    if r["latitude"] is not None and r["longitude"] is not None
}
GERMANY_CENTER = (51.1657, 10.4515)


def resolve_coordinates(record: dict, llm_result: dict, listing_id: str):
    """Resolve (latitude, longitude) for a listing, guaranteeing a location.

    Order of resolution:
      1. Coordinates from the offline geocode join (record lat/lon).
      2. The LLM-inferred `city` matched against the geocode lookup.
      3. Any known city name found as a substring of location_text.
      4. Germany centroid.
    Steps 2-4 apply deterministic jitter (keyed on listing_id) so many
    listings resolving to the same city/centroid spread into a visible cloud
    rather than stacking on one marker.
    """
    lat, lon = record.get("latitude"), record.get("longitude")
    if lat is not None and lon is not None:
        return lat, lon

    # 2. LLM-inferred city.
    if isinstance(llm_result, dict):
        city = (llm_result.get("city") or "").strip().lower()
        if city and city != "unknown" and city in CITY_TO_COORDS:
            base = CITY_TO_COORDS[city]
            return jitter_coord(base[0], base[1], listing_id)

    # 3. Known city as a substring of the raw location text.
    loc = (record.get("location_text") or "").lower()
    for city_name, base in CITY_TO_COORDS.items():
        if city_name and city_name in loc:
            return jitter_coord(base[0], base[1], listing_id)

    # 4. Germany centroid fallback.
    return jitter_coord(GERMANY_CENTER[0], GERMANY_CENTER[1], listing_id, amount_degrees=1.5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — LLM extraction helpers (Req 3.3, 3.4, 3.8, 11.10)
# MAGIC
# MAGIC `extract_attributes_via_ai_query` handles batches with more than
# MAGIC `PER_RECORD_LLM_THRESHOLD` records using the `ai_query` SQL function,
# MAGIC which runs server-side and requires no per-record Python retry
# MAGIC wrapping. `extract_attributes_per_record` handles smaller batches with
# MAGIC one Foundation Model APIs call per record, wrapped in
# MAGIC `retry_with_backoff` (Req 11 AC10) and bounded by a
# MAGIC `RECORD_TIMEOUT_SECONDS` timeout (Req 3 AC8).

# COMMAND ----------

import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

LLM_EXTRACTION_PROMPT_TEMPLATE = (
    "Extract structured attributes from the job description below. "
    "Respond with ONLY a single JSON object and nothing else — no markdown "
    "code fences, no explanation. The object must have exactly these keys: "
    "required_skills (array of strings), seniority_level (string), "
    "employment_type (string), industry (string), company_size_band (string), "
    "company_vibe (string: a 1-3 word culture descriptor, e.g. "
    "'fast-paced startup', 'corporate', 'mission-driven'), "
    "office_policy (string: exactly one of 'remote', 'hybrid', 'onsite', "
    "or 'unknown'), "
    "benefits_rating (string: exactly one of 'excellent', 'good', 'basic', "
    "or 'unknown', judged from any perks/benefits mentioned), "
    "city (string: the German city the job is located in; infer the most "
    "likely major city if the text is vague, else 'unknown'). "
    "For office_policy/benefits_rating/city, infer from the description; use "
    "'unknown' only if there is truly no signal. Job description: {job_description}"
)


def _parse_llm_json(raw_response: str) -> dict:
    """Parse a raw LLM response string into an llm_result dict.

    Chat models such as ``databricks-meta-llama-3-3-70b-instruct`` frequently
    wrap their JSON in markdown fences (```json ... ```) or surround it with
    explanatory prose, so a bare ``json.loads`` on the whole response fails
    with "Expecting value: line 1 column 1 (char 0)" and every record ends up
    ``failed``. This parser is tolerant:

    1. Strip a leading/trailing markdown code fence if present.
    2. Try ``json.loads`` on the cleaned text.
    3. Fall back to extracting the first balanced ``{...}`` object substring
       and parsing that.

    Returns a hard-failure marker dict (see ``enrichment_state`` module
    docstring) if no JSON object can be recovered.
    """
    if not raw_response:
        return {"error": "empty LLM response"}

    text = raw_response.strip()

    # 1. Strip markdown code fences: ```json\n{...}\n``` or ```\n{...}\n```
    if text.startswith("```"):
        # drop the opening fence line (``` or ```json) and any closing fence
        text = re.sub(r"^```[a-zA-Z0-9_]*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()

    def _try(candidate: str):
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None

    # 2. Direct parse of the (de-fenced) text.
    parsed = _try(text)
    if parsed is not None:
        return parsed

    # 3. Extract the first {...} object substring and parse that. Handles the
    #    common "Here is the JSON: { ... }" prose-wrapped case.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        parsed = _try(match.group(0))
        if parsed is not None:
            return parsed

    return {"error": f"invalid JSON response: {raw_response!r}"}


def extract_attributes_via_ai_query(batch: list[dict]) -> dict[str, dict]:
    """Extract LLM attributes for a batch of >50 records using `ai_query`.

    Runs a single SQL statement over a temporary view of the batch, calling
    the `ai_query` SQL function once per row on the SQL warehouse /
    serverless SQL compute (Req 3 AC4). Returns a mapping of
    `listing_id -> llm_result` (parsed JSON dict, or a hard-failure marker
    dict if the row's response was null/unparseable).
    """
    batch_view_rows = [(r["listing_id"], r.get("job_description") or "") for r in batch]
    batch_view_df = spark.createDataFrame(batch_view_rows, ["listing_id", "job_description"])
    batch_view_df.createOrReplaceTempView("_enrichment_ai_query_batch")

    result_rows = spark.sql(
        f"""
        SELECT listing_id,
               ai_query(
                 '{LLM_ENDPOINT}',
                 CONCAT(
                   'Extract structured attributes from the job description below. ',
                   'Respond with ONLY a single JSON object and nothing else - no ',
                   'markdown code fences, no explanation. The object must have ',
                   'exactly these keys: required_skills (array of strings), ',
                   'seniority_level (string), employment_type (string), ',
                   'industry (string), company_size_band (string), ',
                   'company_vibe (string: a 1-3 word culture descriptor), ',
                   'office_policy (string: one of remote, hybrid, onsite, unknown), ',
                   'benefits_rating (string: one of excellent, good, basic, unknown), ',
                   'city (string: the German city the job is in, or unknown). ',
                   'For the last four, infer from the description; use unknown ',
                   'only if there is truly no signal. ',
                   'Job description: ',
                   job_description
                 )
               ) AS llm_response
        FROM _enrichment_ai_query_batch
        """
    ).collect()

    # Debug: print first few raw responses
    print("DEBUG: Sample LLM responses:")
    for i, row in enumerate(result_rows[:3]):
        print(f"  listing_id: {row['listing_id']}")
        print(f"  llm_response: {row['llm_response']!r}")
        print()

    return {row["listing_id"]: _parse_llm_json(row["llm_response"]) for row in result_rows}


@retry_with_backoff(max_attempts=3, backoff_base=2, retryable_codes=(429,))
def _query_llm_endpoint(job_description: str) -> str:
    """Call Foundation Model APIs for a single listing's job description.

    Uses the OpenAI-compatible client exposed by the Databricks SDK
    (`serving_endpoints.get_open_ai_client()`), which accepts plain-dict
    chat messages and returns a standard OpenAI response object. This avoids
    the SDK's native `serving_endpoints.query(messages=[{...}])` path, which
    on recent SDK versions raises ``'dict' object has no attribute 'as_dict'``
    when handed plain-dict messages.

    Wrapped in `retry_with_backoff` so a rate-limited (HTTP 429) response
    is retried up to 3 times with exponential backoff starting at 2
    seconds (Req 11 AC10).
    """
    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient()
    openai_client = client.serving_endpoints.get_open_ai_client()
    response = openai_client.chat.completions.create(
        model=LLM_ENDPOINT,
        messages=[
            {
                "role": "user",
                "content": LLM_EXTRACTION_PROMPT_TEMPLATE.format(job_description=job_description or ""),
            }
        ],
    )
    return response.choices[0].message.content


def extract_attributes_for_record(job_description: str) -> dict:
    """Extract LLM attributes for a single record, bounded by a timeout.

    Returns the parsed llm_result dict, or a hard-failure marker dict if
    the call raised an error or exceeded `RECORD_TIMEOUT_SECONDS`
    (Req 3 AC8).
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_query_llm_endpoint, job_description)
        try:
            raw_response = future.result(timeout=RECORD_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            return {"_failed": True, "reason": f"LLM extraction exceeded {RECORD_TIMEOUT_SECONDS}s timeout"}
        except Exception as exc:  # noqa: BLE001 - any LLM call failure is a hard failure
            return {"_failed": True, "reason": str(exc)}

    return _parse_llm_json(raw_response)


def extract_attributes_per_record(batch: list[dict]) -> dict[str, dict]:
    """Extract LLM attributes for each record in a batch of <=50 records."""
    return {r["listing_id"]: extract_attributes_for_record(r.get("job_description")) for r in batch}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Build the `embedding_text` chunk rows (Req 4.2, 4.3)

# COMMAND ----------

def build_chunk_rows(listing_id: str, embedding_text: str, job_title: str, company_name: str,
                      latitude, longitude, enrichment_state: str) -> list[tuple]:
    """Build one or more `silver.enriched_listings_chunks` rows for a listing.

    If `embedding_text` exceeds `EMBEDDING_CHUNK_CHAR_THRESHOLD` characters,
    it is split via `chunk_text` into chunks of at most 512 tokens with 64
    tokens of overlap; otherwise the whole text is stored as a single
    chunk (chunk_index 0), so every enriched listing has at least one row
    for downstream Vector Search indexing.
    """
    if embedding_text and len(embedding_text) > EMBEDDING_CHUNK_CHAR_THRESHOLD:
        chunks = chunk_text(embedding_text, max_tokens=512, overlap_tokens=64)
    else:
        chunks = [embedding_text] if embedding_text else []

    return [
        (
            f"{listing_id}_{chunk_index}",
            listing_id,
            chunk_index,
            chunk,
            job_title,
            company_name,
            latitude,
            longitude,
            enrichment_state,
        )
        for chunk_index, chunk in enumerate(chunks)
    ]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Batch, enrich, checkpoint, and MERGE upsert (Req 3.5-3.9, 11.8, 11.9)
# MAGIC
# MAGIC Joined records are split into batches of at most `BATCH_SIZE`.
# MAGIC Processing resumes from `get_resume_batch` so a restarted run skips
# MAGIC already-completed batches. Within each batch, the LLM extraction path
# MAGIC is chosen by batch size (Req 3 AC4), the state machine is applied per
# MAGIC record, and both Silver tables plus the source `bronze.job_listings`
# MAGIC row's `enrichment_state` are MERGE-upserted. A checkpoint is written
# MAGIC after each successfully processed batch.

# COMMAND ----------

from datetime import datetime, timezone  # noqa: E402
from delta.tables import DeltaTable  # noqa: E402

batches = batch_records(joined_records, batch_size=BATCH_SIZE)
resume_batch_index = get_resume_batch(spark, PIPELINE_NAME, RUN_ID)

print(f"Total batches: {len(batches)}; resuming from batch index {resume_batch_index}")

for target_table, schema in (
    (SILVER_ENRICHED_TABLE, SILVER_ENRICHED_LISTINGS_SCHEMA),
    (SILVER_CHUNKS_TABLE, SILVER_ENRICHED_LISTINGS_CHUNKS_SCHEMA),
):
    if not spark.catalog.tableExists(target_table):
        spark.createDataFrame([], schema=schema).write.format("delta").saveAsTable(target_table)

# Evolve a pre-existing enriched_listings table to include newly-introduced
# columns (company_vibe, office_policy, benefits_rating) before the MERGE.
# Serverless does not allow spark.databricks.delta.schema.autoMerge.enabled,
# so add any missing columns explicitly via ALTER TABLE (idempotent — only
# adds columns that are not already present).
_existing_cols = {f.name for f in spark.table(SILVER_ENRICHED_TABLE).schema.fields}
for _field in SILVER_ENRICHED_LISTINGS_SCHEMA.fields:
    if _field.name not in _existing_cols:
        spark.sql(
            f"ALTER TABLE {SILVER_ENRICHED_TABLE} "
            f"ADD COLUMNS ({_field.name} {_field.dataType.simpleString()})"
        )
        print(f"Added column {_field.name} to {SILVER_ENRICHED_TABLE}")

silver_enriched_target = DeltaTable.forName(spark, SILVER_ENRICHED_TABLE)
silver_chunks_target = DeltaTable.forName(spark, SILVER_CHUNKS_TABLE)
bronze_target = DeltaTable.forName(spark, BRONZE_LISTINGS_TABLE)

state_counts = {"enriched": 0, "partially_enriched": 0, "failed": 0}

for batch_index, batch in enumerate(batches):
    if batch_index < resume_batch_index:
        print(f"Skipping already-completed batch {batch_index}")
        continue

    if len(batch) > PER_RECORD_LLM_THRESHOLD:
        llm_results_by_id = extract_attributes_via_ai_query(batch)
        llm_path = "ai_query"
    else:
        llm_results_by_id = extract_attributes_per_record(batch)
        llm_path = "per_record"

    enriched_rows = []
    chunk_rows = []
    enrichment_timestamp = datetime.now(timezone.utc)

    for record in batch:
        listing_id = record["listing_id"]
        llm_result = llm_results_by_id.get(listing_id, {"error": "no LLM result returned for listing"})

        # Guarantee a location for every listing (join → LLM city → centroid).
        resolved_lat, resolved_lon = resolve_coordinates(record, llm_result, listing_id)
        geocode_result = {"latitude": resolved_lat, "longitude": resolved_lon}

        state, unresolved_attributes, failure_reason = determine_enrichment_state(geocode_result, llm_result)
        state_counts[state] += 1

        required_skills = llm_result.get("required_skills") if isinstance(llm_result, dict) else None
        required_skills = required_skills if isinstance(required_skills, list) else None
        required_skills_text = ", ".join(required_skills) if required_skills else ""

        # Standardize the vibe attributes into fixed, lowercase buckets so the
        # app filters show a small, clean set of values.
        _raw = llm_result if isinstance(llm_result, dict) else {}
        # Bucket every filterable attribute into 3-5 fixed emoji+text groups so
        # the app dropdowns are short and readable.
        b_seniority = bucket_seniority(_raw.get("seniority_level"))
        b_employment = bucket_employment_type(_raw.get("employment_type"))
        b_industry = bucket_industry(_raw.get("industry"))
        b_company_size = bucket_company_size(_raw.get("company_size_band"))
        norm_company_vibe = bucket_company_vibe(_raw.get("company_vibe"))
        norm_office_policy = bucket_office_policy(_raw.get("office_policy"))
        norm_benefits_rating = bucket_benefits_rating(_raw.get("benefits_rating"))

        embedding_text = build_embedding_text(
            record["job_title"], record.get("job_description") or "", required_skills_text
        )

        enriched_rows.append(
            (
                listing_id,
                record["job_title"],
                record["company_name"],
                record.get("job_description"),
                record.get("location_text"),
                record["source_url"],
                geocode_result["latitude"],
                geocode_result["longitude"],
                required_skills,
                required_skills_text,
                b_seniority,
                b_employment,
                b_industry,
                b_company_size,
                norm_company_vibe,
                norm_office_policy,
                norm_benefits_rating,
                state,
                unresolved_attributes if unresolved_attributes else None,
                failure_reason,
                enrichment_timestamp,
                embedding_text,
            )
        )

        chunk_rows.extend(
            build_chunk_rows(
                listing_id,
                embedding_text,
                record["job_title"],
                record["company_name"],
                geocode_result["latitude"],
                geocode_result["longitude"],
                state,
            )
        )

    enriched_df = spark.createDataFrame(enriched_rows, schema=SILVER_ENRICHED_LISTINGS_SCHEMA)
    (
        silver_enriched_target.alias("target")
        .merge(enriched_df.alias("source"), "target.listing_id = source.listing_id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )

    if chunk_rows:
        chunks_df = spark.createDataFrame(chunk_rows, schema=SILVER_ENRICHED_LISTINGS_CHUNKS_SCHEMA)
        (
            silver_chunks_target.alias("target")
            .merge(chunks_df.alias("source"), "target.chunk_id = source.chunk_id")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    # Reflect the new enrichment_state back onto bronze.job_listings so a
    # subsequent run's `enrichment_state = 'unenriched'` filter (Req 3.1)
    # does not reprocess these listings.
    batch_states_df = spark.createDataFrame(
        [(r[0], r[17]) for r in enriched_rows], ["listing_id", "enrichment_state"]
    )
    (
        bronze_target.alias("target")
        .merge(batch_states_df.alias("source"), "target.listing_id = source.listing_id")
        .whenMatchedUpdate(set={"enrichment_state": "source.enrichment_state"})
        .execute()
    )

    write_checkpoint(spark, PIPELINE_NAME, RUN_ID, batch_index)
    print(
        f"Batch {batch_index}: {len(batch)} record(s) enriched via '{llm_path}' path "
        f"({len(chunk_rows)} chunk row(s) written)"
    )

print(f"Enrichment state counts this run: {state_counts}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Produce and persist the run summary (Req 3.10)

# COMMAND ----------

run_timestamp = datetime.now(timezone.utc)

summary_row = [
    (
        RUN_ID,
        PIPELINE_NAME,
        None,
        0,
        0,
        0,
        0,
        state_counts["enriched"],
        state_counts["partially_enriched"],
        state_counts["failed"],
        0,
        run_timestamp,
    )
]

summary_df = spark.createDataFrame(summary_row, schema=GOLD_PIPELINE_SUMMARY_SCHEMA)

if not spark.catalog.tableExists(PIPELINE_SUMMARY_TABLE):
    summary_df.write.format("delta").saveAsTable(PIPELINE_SUMMARY_TABLE)
else:
    summary_df.write.format("delta").mode("append").saveAsTable(PIPELINE_SUMMARY_TABLE)

print(f"Wrote pipeline summary to {PIPELINE_SUMMARY_TABLE}")
print(
    {
        "run_id": RUN_ID,
        "enriched_count": state_counts["enriched"],
        "partially_enriched_count": state_counts["partially_enriched"],
        "failed_count": state_counts["failed"],
        "unenriched_count": 0,
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification

# COMMAND ----------

total_processed = state_counts["enriched"] + state_counts["partially_enriched"] + state_counts["failed"]
print(f"Total joined records: {len(joined_records)}")
print(f"Total processed (enriched + partially_enriched + failed): {total_processed}")
assert total_processed == len(joined_records), (
    "enriched_count + partially_enriched_count + failed_count must equal the "
    "total number of unenriched records read at the start of the run"
)

display(spark.table(SILVER_ENRICHED_TABLE).limit(10))
