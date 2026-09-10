# Databricks notebook source
# MAGIC %md
# MAGIC # 00a — Load Reference Data: Geocode Lookup
# MAGIC
# MAGIC Uploads the bundled `geocode_lookup.csv` reference dataset (city/postal-code →
# MAGIC lat/lng mappings for major European and US cities) to the
# MAGIC `job_agent.volumes.geocode_lookup` Unity Catalog Volume, then loads it into
# MAGIC the `job_agent.ops.geocode_lookup` Delta table.
# MAGIC
# MAGIC This keeps geocoding offline during the enrichment pipeline (Requirement 3.2)
# MAGIC and powers location resolution for the app's Location & Commute tab
# MAGIC (Requirement 6.5).
# MAGIC
# MAGIC **Prerequisite**: `notebooks/00_setup_catalog.py` must have already created
# MAGIC the `job_agent` catalog, the `ops` and `volumes` schemas, the
# MAGIC `volumes.geocode_lookup` Volume, and the empty `ops.geocode_lookup` Delta
# MAGIC table.
# MAGIC
# MAGIC Requirements: 3.2, 6.5

# COMMAND ----------

# MAGIC %pip install --quiet databricks-sdk

# COMMAND ----------

dbutils.widgets.text("catalog", "job_agent", "Catalog name")
dbutils.widgets.text("schema", "ops", "Schema name")
dbutils.widgets.text("volume_schema", "volumes", "Volume schema name")
dbutils.widgets.text("volume", "geocode_lookup", "Volume name")
dbutils.widgets.text(
    "local_csv_path",
    "../data/geocode_lookup/geocode_lookup.csv",
    "Path to the local geocode_lookup.csv (relative to this notebook)",
)

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME_SCHEMA = dbutils.widgets.get("volume_schema")
VOLUME = dbutils.widgets.get("volume")
LOCAL_CSV_PATH = dbutils.widgets.get("local_csv_path")

VOLUME_PATH = f"/Volumes/{CATALOG}/{VOLUME_SCHEMA}/{VOLUME}"
VOLUME_CSV_PATH = f"{VOLUME_PATH}/geocode_lookup.csv"
TABLE_NAME = f"{CATALOG}.{SCHEMA}.geocode_lookup"

print(f"Local CSV path:  {LOCAL_CSV_PATH}")
print(f"Volume CSV path: {VOLUME_CSV_PATH}")
print(f"Target table:    {TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Upload the CSV to the `volumes.geocode_lookup` Volume
# MAGIC
# MAGIC Reads the CSV bundled in the repo (`data/geocode_lookup/geocode_lookup.csv`)
# MAGIC and copies it into the Unity Catalog Volume
# MAGIC (`job_agent.volumes.geocode_lookup`) so it is durable and shareable across
# MAGIC workspace users, without requiring outbound network access.

# COMMAND ----------

import os
import shutil

# Idempotency guard (important for the daily scheduled pipeline): the geocode
# reference dataset is static, so if the target table already holds a healthy
# number of rows there is nothing to reload. This also makes the task robust
# when the bundled CSV is not reachable from the serverless working directory
# (Workspace file sync), which would otherwise fail the whole pipeline on an
# essentially no-op step.
_already_loaded = False
if spark.catalog.tableExists(TABLE_NAME):
    try:
        _existing_rows = spark.table(TABLE_NAME).count()
    except Exception:  # noqa: BLE001
        _existing_rows = 0
    if _existing_rows >= 100:
        print(
            f"{TABLE_NAME} already has {_existing_rows} rows — skipping reload "
            "of the static geocode reference dataset."
        )
        _already_loaded = True

if _already_loaded:
    dbutils.notebook.exit(
        f"skipped: {TABLE_NAME} already populated ({_existing_rows} rows)"
    )

os.makedirs(VOLUME_PATH, exist_ok=True)

# Resolve the local CSV path against the notebook's working directory when run
# from the Databricks Repos/Workspace file tree.
resolved_local_path = LOCAL_CSV_PATH
if not os.path.isabs(resolved_local_path):
    notebook_dir = os.path.dirname(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    )
    # Fall back to the current working directory if the notebook path can't be
    # resolved (e.g. when running this file as a plain Python script locally).
    resolved_local_path = os.path.abspath(LOCAL_CSV_PATH)

if not os.path.exists(resolved_local_path):
    raise FileNotFoundError(
        f"Could not find geocode_lookup.csv at '{resolved_local_path}'. "
        "Set the 'local_csv_path' widget to the correct location."
    )

shutil.copyfile(resolved_local_path, VOLUME_CSV_PATH)
print(f"Uploaded {resolved_local_path} -> {VOLUME_CSV_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Load the CSV into the `ops.geocode_lookup` Delta table
# MAGIC
# MAGIC Reads the CSV from the Volume using the schema defined in
# MAGIC `src/models/schemas.py` (`OPS_GEOCODE_LOOKUP_SCHEMA`), normalizes
# MAGIC `city_name` to lowercase for case-insensitive matching (per the design's
# MAGIC `LOWER(city_name) = LOWER(:input)` lookup pattern), and overwrites the
# MAGIC target Delta table.

# COMMAND ----------

import sys

sys.path.append("../src")

from models.schemas import OPS_GEOCODE_LOOKUP_SCHEMA  # noqa: E402
from pyspark.sql.functions import lower, trim  # noqa: E402

raw_df = (
    spark.read.format("csv")
    .option("header", "true")
    .schema(OPS_GEOCODE_LOOKUP_SCHEMA)
    .load(VOLUME_CSV_PATH)
)

geocode_df = raw_df.withColumn("city_name", lower(trim(raw_df.city_name)))

row_count = geocode_df.count()
print(f"Read {row_count} rows from {VOLUME_CSV_PATH}")

if row_count < 100:
    raise ValueError(
        f"Expected at least 100 geocode_lookup entries, found {row_count}."
    )

# COMMAND ----------

(
    geocode_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

print(f"Loaded {row_count} rows into {TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Validate the loaded table
# MAGIC
# MAGIC Sanity-checks row count and latitude/longitude ranges after the load.

# COMMAND ----------

from pyspark.sql.functions import col

loaded_df = spark.table(TABLE_NAME)
loaded_count = loaded_df.count()

invalid_coords = loaded_df.filter(
    (col("latitude") < -90)
    | (col("latitude") > 90)
    | (col("longitude") < -180)
    | (col("longitude") > 180)
).count()

print(f"{TABLE_NAME}: {loaded_count} rows, {invalid_coords} with out-of-range coordinates")

assert loaded_count >= 100, f"Expected >= 100 rows in {TABLE_NAME}, found {loaded_count}"
assert invalid_coords == 0, f"Found {invalid_coords} rows with invalid lat/lng ranges"

display(loaded_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

summary = {
    "table": TABLE_NAME,
    "volume_csv_path": VOLUME_CSV_PATH,
    "rows_loaded": loaded_count,
}
print(summary)
