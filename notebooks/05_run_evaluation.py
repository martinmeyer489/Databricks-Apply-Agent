# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Evaluation Runner
# MAGIC
# MAGIC Scores a registered Matching Agent version
# MAGIC (`job_agent.gold.matching_agent`) against the labelled evaluation
# MAGIC cases in `job_agent.ops.evaluation_dataset` (seeded by
# MAGIC `notebooks/05a_seed_evaluation_dataset.py`, task 17.1) using two
# MAGIC MLflow LLM-as-judge scorers, and records the per-scorer aggregate
# MAGIC results to `job_agent.ops.evaluation_results` (Req 9 AC3-AC4):
# MAGIC
# MAGIC - **`match_relevance`** (1-5): given a User_Profile and the Job_Listings
# MAGIC   the agent returned for it, an LLM judge rates how relevant the
# MAGIC   matches are.
# MAGIC - **`cover_letter_groundedness`** (1-5): given a User_Profile, a
# MAGIC   selected Job_Listing, and a generated cover letter, an LLM judge
# MAGIC   rates how grounded the letter is in the profile and listing.
# MAGIC
# MAGIC The row written to `ops.evaluation_results` is the input the
# MAGIC deployment gate (`src/utils/deployment_gate.py`, task 17.3) checks
# MAGIC before a version is promoted to the Model Serving endpoint (Req 9
# MAGIC AC5-6) — see `notebooks/07_register_deploy_agent.py`.
# MAGIC
# MAGIC **Bridging note**: `ops.evaluation_dataset` stores each case's
# MAGIC `user_profile` as raw fields (skills, location text, commute radius)
# MAGIC rather than a `profile_id`, because that's how the cases were
# MAGIC authored (task 17.1). The deployed Matching Agent's `predict()`
# MAGIC contract takes `{"profile_id": ...}` and reads the profile from
# MAGIC `gold.user_profiles` via the `get_user_profile` UC Function (Req
# MAGIC 7.2). This notebook therefore upserts one `gold.user_profiles` row
# MAGIC per evaluation case — keyed by `case_id` used as `profile_id` — as a
# MAGIC pre-step, resolving each case's location text against
# MAGIC `ops.geocode_lookup` the same way the app does (task 12.1).
# MAGIC
# MAGIC **Prerequisite**: `notebooks/00_setup_catalog.py`,
# MAGIC `notebooks/05a_seed_evaluation_dataset.py`, and
# MAGIC `notebooks/07_register_deploy_agent.py` (or an equivalent manual
# MAGIC registration) must have already run so that `ops.evaluation_dataset`
# MAGIC is populated and the target `model_version` of
# MAGIC `job_agent.gold.matching_agent` exists in Unity Catalog.
# MAGIC
# MAGIC Requirements: 9.3, 9.4

# COMMAND ----------

import sys
import os

# Allow `from src....` imports when this notebook is run from the
# Databricks Workspace (repo root is not automatically on sys.path there).
_NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_REPO_ROOT = os.path.abspath(os.path.join(_NOTEBOOK_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets

# COMMAND ----------

dbutils.widgets.text("catalog", "job_agent", "Catalog name")
dbutils.widgets.text("model_version", "", "Registered matching_agent version to evaluate (required)")
dbutils.widgets.text("registered_model_name", "gold.matching_agent", "UC registered model name (schema.model, catalog prepended)")
dbutils.widgets.text("judge_endpoint", "databricks-meta-llama-3-3-70b-instruct", "Foundation Model APIs chat endpoint used as the LLM judge")
dbutils.widgets.text("warehouse_id", "", "SQL warehouse ID used to call the draft_application UC Function")

CATALOG = dbutils.widgets.get("catalog")
_model_version_raw = dbutils.widgets.get("model_version").strip()
if not _model_version_raw:
    raise ValueError(
        "The 'model_version' widget is required: pass the registered "
        f"{CATALOG}.gold.matching_agent version to evaluate, e.g. from the "
        "output of notebooks/07_register_deploy_agent.py."
    )
MODEL_VERSION = int(_model_version_raw)
REGISTERED_MODEL_NAME = f"{CATALOG}.{dbutils.widgets.get('registered_model_name')}"
JUDGE_ENDPOINT = dbutils.widgets.get("judge_endpoint")
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id").strip()

EVAL_DATASET_TABLE = f"{CATALOG}.ops.evaluation_dataset"
EVAL_RESULTS_TABLE = f"{CATALOG}.ops.evaluation_results"
USER_PROFILES_TABLE = f"{CATALOG}.gold.user_profiles"

print(f"Catalog:                {CATALOG}")
print(f"Registered model name:  {REGISTERED_MODEL_NAME}")
print(f"Model version:          {MODEL_VERSION}")
print(f"Judge endpoint:         {JUDGE_ENDPOINT}")
print(f"Evaluation dataset:     {EVAL_DATASET_TABLE}")
print(f"Evaluation results:     {EVAL_RESULTS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Load the labelled evaluation cases (Req 9 AC2)

# COMMAND ----------

import json

eval_dataset_df = spark.table(EVAL_DATASET_TABLE)
eval_cases_count = eval_dataset_df.count()

assert eval_cases_count >= 20, (
    f"Expected >= 20 rows in {EVAL_DATASET_TABLE}, found {eval_cases_count}. "
    "Run notebooks/05a_seed_evaluation_dataset.py first."
)

eval_cases_pdf = eval_dataset_df.toPandas()
eval_cases_pdf["user_profile_dict"] = eval_cases_pdf["user_profile"].apply(json.loads)

print(f"Loaded {len(eval_cases_pdf)} labelled evaluation cases from {EVAL_DATASET_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Bridge cases into `gold.user_profiles`
# MAGIC
# MAGIC Each case's `case_id` becomes the `profile_id` the deployed agent
# MAGIC looks up via `get_user_profile`. Location text is resolved against
# MAGIC `ops.geocode_lookup` exactly as the app's location resolver does
# MAGIC (Req 6.5); cases whose location text has no match are skipped (with
# MAGIC a warning) since the agent cannot compute a commute distance without
# MAGIC resolved coordinates.

# COMMAND ----------

from datetime import datetime, timezone

from agent.location_resolver import resolve_location  # noqa: E402
from models.schemas import GOLD_USER_PROFILES_SCHEMA  # noqa: E402

now = datetime.now(timezone.utc)
profile_rows = []
skipped_cases = []

for _, case in eval_cases_pdf.iterrows():
    profile = case["user_profile_dict"]
    location_text = profile.get("location", "")
    resolved = resolve_location(spark, location_text)
    if resolved is None:
        skipped_cases.append(case["case_id"])
        continue

    profile_rows.append(
        (
            case["case_id"],  # profile_id
            profile.get("skills") or [],
            None,  # years_of_experience: not modeled in evaluation_dataset cases
            None,  # education_history
            [],  # job_title_history
            None,  # qualifications_summary
            resolved["latitude"],
            resolved["longitude"],
            resolved["city_name"],
            int(profile["commute_radius_km"]),
            None,  # cv_file_path
            now,
            now,
        )
    )

if skipped_cases:
    print(
        f"WARNING: skipped {len(skipped_cases)} evaluation case(s) with unresolvable "
        f"location text (not found in ops.geocode_lookup): {skipped_cases}"
    )

assert profile_rows, "No evaluation case had a resolvable location; cannot proceed."

profiles_df = spark.createDataFrame(profile_rows, schema=GOLD_USER_PROFILES_SCHEMA)

profiles_df.createOrReplaceTempView("eval_profiles_staging")
spark.sql(
    f"""
    MERGE INTO {USER_PROFILES_TABLE} AS target
    USING eval_profiles_staging AS source
    ON target.profile_id = source.profile_id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
    """
)

print(f"Upserted {len(profile_rows)} evaluation profiles into {USER_PROFILES_TABLE}")

eval_cases_pdf = eval_cases_pdf[eval_cases_pdf["case_id"].isin([row[0] for row in profile_rows])].reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Draft a reference cover letter per case (grounding context)
# MAGIC
# MAGIC `cover_letter_groundedness` needs a drafted cover letter to grade.
# MAGIC The letter is generated by calling the `draft_application` UC
# MAGIC Function (Req 8) directly for each case's first `expected_listing_id`,
# MAGIC via the SQL warehouse, using the case's own profile fields as the
# MAGIC drafting inputs — the same tool the deployed Matching Agent would
# MAGIC invoke once it has selected a listing.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

if not WAREHOUSE_ID:
    raise ValueError(
        "The 'warehouse_id' widget is required to call the draft_application UC "
        "Function (the Free Edition 2X-Small SQL warehouse ID)."
    )

w = WorkspaceClient()


def _sql_literal_array(values):
    escaped = ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)
    return f"array({escaped})"


def draft_cover_letter(listing_id: str, skills, job_title_history, qualifications_summary, years_of_experience) -> str:
    """Call `{catalog}.gold.draft_application` via the SQL warehouse and return the letter text."""
    query = f"""
    SELECT {CATALOG}.gold.draft_application(
        '{listing_id}',
        {_sql_literal_array(skills or [])},
        {_sql_literal_array(job_title_history or [])},
        '{(qualifications_summary or "").replace("'", "''")}',
        {int(years_of_experience) if years_of_experience is not None else 0}
    ) AS cover_letter
    """
    statement = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=query, wait_timeout="30s"
    )
    if statement.status.state != StatementState.SUCCEEDED:
        return ""
    rows = statement.result.data_array or []
    return rows[0][0] if rows and rows[0] else ""


cover_letters = []
for _, case in eval_cases_pdf.iterrows():
    profile = case["user_profile_dict"]
    expected_ids = case["expected_listing_ids"]
    # expected_ids may be a numpy array (from pandas), so avoid `if expected_ids`
    # which raises "truth value of an array ... is ambiguous"; check length.
    target_listing_id = (
        expected_ids[0] if expected_ids is not None and len(expected_ids) > 0 else None
    )
    if target_listing_id is None:
        cover_letters.append("")
        continue
    letter = draft_cover_letter(
        target_listing_id,
        profile.get("skills"),
        profile.get("job_title_history", []),
        profile.get("qualifications_summary", ""),
        profile.get("years_of_experience"),
    )
    cover_letters.append(letter)

eval_cases_pdf["cover_letter"] = cover_letters
eval_cases_pdf["target_listing_id"] = eval_cases_pdf["expected_listing_ids"].apply(
    lambda ids: ids[0] if ids is not None and len(ids) > 0 else None
)

print(f"Drafted {sum(1 for c in cover_letters if c)} of {len(cover_letters)} reference cover letters")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Build the `mlflow.evaluate()` input frame
# MAGIC
# MAGIC `inputs` is the request the registered agent's `predict()` expects
# MAGIC (`{"profile_id": ...}`); `user_profile_json` and `cover_letter` are
# MAGIC additional grading-context columns consumed by the two genai metrics
# MAGIC below.

# COMMAND ----------

import pandas as pd

evaluation_dataset = pd.DataFrame(
    {
        "inputs": eval_cases_pdf["case_id"].apply(lambda case_id: {"profile_id": case_id}),
        "case_id": eval_cases_pdf["case_id"],
        "user_profile_json": eval_cases_pdf["user_profile"],
        "cover_letter": eval_cases_pdf["cover_letter"],
        "target_listing_id": eval_cases_pdf["target_listing_id"],
    }
)

print(f"Built evaluation frame with {len(evaluation_dataset)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Define the LLM-as-judge scorers (Req 9 AC3)
# MAGIC
# MAGIC Both scorers use `mlflow.metrics.genai.make_genai_metric` with a 1-5
# MAGIC grading scale, `model=f"endpoints:/{JUDGE_ENDPOINT}"` as the judge,
# MAGIC and `grading_context_columns` to pull in the case's `user_profile_json`
# MAGIC and (for groundedness) the drafted `cover_letter`.

# COMMAND ----------

from mlflow.metrics.genai import EvaluationExample, make_genai_metric

match_relevance_scorer = make_genai_metric(
    name="match_relevance",
    definition=(
        "Match relevance measures how well the job listings returned by the "
        "Matching Agent align with the skills, job title history, and "
        "qualifications described in the user's profile."
    ),
    grading_prompt=(
        "Match relevance: Given the user profile and the job matches the "
        "agent returned, score from 1 to 5 how relevant the returned matches "
        "are to the profile.\n"
        "- Score 1: none of the returned matches relate to the profile's "
        "skills or experience.\n"
        "- Score 3: some returned matches are loosely related to the "
        "profile.\n"
        "- Score 5: all returned matches are strongly aligned with the "
        "profile's skills, experience, and job title history."
    ),
    examples=[
        EvaluationExample(
            input="Profile: skills=[Python, Spark, SQL], titles=[Data Engineer]",
            output="Returned: Senior Data Engineer (Spark, Delta Lake); Data Platform Engineer (Python, SQL)",
            score=5,
            justification="Both returned matches directly align with the profile's skills and title history.",
            grading_context={
                "user_profile_json": '{"skills": ["Python", "Spark", "SQL"], "job_title_history": ["Data Engineer"]}'
            },
        ),
        EvaluationExample(
            input="Profile: skills=[Python, Spark, SQL], titles=[Data Engineer]",
            output="Returned: Retail Store Manager; Graphic Designer",
            score=1,
            justification="Neither returned match relates to the profile's data engineering background.",
            grading_context={
                "user_profile_json": '{"skills": ["Python", "Spark", "SQL"], "job_title_history": ["Data Engineer"]}'
            },
        ),
    ],
    version="v1",
    model=f"endpoints:/{JUDGE_ENDPOINT}",
    parameters={"temperature": 0.0},
    grading_context_columns=["user_profile_json"],
    greater_is_better=True,
)

groundedness_scorer = make_genai_metric(
    name="cover_letter_groundedness",
    definition=(
        "Cover letter groundedness measures whether the drafted cover "
        "letter's claims are actually supported by the user's profile and "
        "the target job listing, without fabricated details."
    ),
    grading_prompt=(
        "Cover letter groundedness: Given the user profile, the target "
        "listing, and the drafted cover letter, score from 1 to 5 how well "
        "grounded the letter is.\n"
        "- Score 1: the letter fabricates skills or experience not present "
        "in the profile, or makes claims unrelated to the listing.\n"
        "- Score 3: the letter is mostly grounded but includes some vague "
        "or generic claims.\n"
        "- Score 5: every claim in the letter is directly traceable to the "
        "profile's skills/experience or the listing's requirements."
    ),
    examples=[
        EvaluationExample(
            input="Profile: skills=[Python, Spark]; Listing: Senior Data Engineer requiring Python, Spark",
            output=(
                "I have hands-on experience with Python and Apache Spark, which "
                "directly matches the Senior Data Engineer role's requirements."
            ),
            score=5,
            justification="Every claim in the letter maps directly to a skill in the profile and the listing.",
            grading_context={
                "user_profile_json": '{"skills": ["Python", "Spark"]}',
                "cover_letter": (
                    "I have hands-on experience with Python and Apache Spark, which "
                    "directly matches the Senior Data Engineer role's requirements."
                ),
            },
        ),
        EvaluationExample(
            input="Profile: skills=[Python, Spark]; Listing: Senior Data Engineer requiring Python, Spark",
            output="I am an expert in nuclear physics and have led international diplomacy initiatives.",
            score=1,
            justification="None of the claims are supported by the profile or the listing.",
            grading_context={
                "user_profile_json": '{"skills": ["Python", "Spark"]}',
                "cover_letter": "I am an expert in nuclear physics and have led international diplomacy initiatives.",
            },
        ),
    ],
    version="v1",
    model=f"endpoints:/{JUDGE_ENDPOINT}",
    parameters={"temperature": 0.0},
    grading_context_columns=["user_profile_json", "cover_letter"],
    greater_is_better=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Run `mlflow.evaluate()` against the registered agent version (Req 9 AC1, AC3)

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name=f"evaluate_matching_agent_v{MODEL_VERSION}") as run:
    results = mlflow.evaluate(
        model=f"models:/{REGISTERED_MODEL_NAME}/{MODEL_VERSION}",
        data=evaluation_dataset,
        model_type="question-answering",
        evaluators="default",
        extra_metrics=[match_relevance_scorer, groundedness_scorer],
    )

eval_run_id = run.info.run_id
print(f"Evaluation run: {eval_run_id}")
print(f"Metrics: {results.metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Record per-scorer aggregate results (Req 9 AC4)
# MAGIC
# MAGIC `make_genai_metric` reports the aggregate mean under
# MAGIC `"{name}/{version}/mean"` in `results.metrics`. Both keys are read
# MAGIC defensively (falling back to the unversioned key) since the exact
# MAGIC suffix depends on the MLflow version installed in the workspace.

# COMMAND ----------


def _extract_mean(metrics: dict, metric_name: str, metric_version: str = "v1") -> float:
    for key in (f"{metric_name}/{metric_version}/mean", f"{metric_name}/mean"):
        if key in metrics:
            return float(metrics[key])
    raise KeyError(
        f"Could not find an aggregate mean for '{metric_name}' in evaluation "
        f"results.metrics: {sorted(metrics.keys())}"
    )


match_relevance_mean = _extract_mean(results.metrics, "match_relevance")
groundedness_mean = _extract_mean(results.metrics, "cover_letter_groundedness")

print(f"match_relevance_mean:     {match_relevance_mean}")
print(f"groundedness_mean:        {groundedness_mean}")

# COMMAND ----------

from models.schemas import OPS_EVALUATION_RESULTS_SCHEMA  # noqa: E402

eval_timestamp = datetime.now(timezone.utc)

result_row = [
    (
        eval_run_id,
        MODEL_VERSION,
        match_relevance_mean,
        groundedness_mean,
        eval_timestamp,
    )
]

result_df = spark.createDataFrame(result_row, schema=OPS_EVALUATION_RESULTS_SCHEMA)
result_df.write.format("delta").mode("append").saveAsTable(EVAL_RESULTS_TABLE)

print(f"Recorded evaluation results for {REGISTERED_MODEL_NAME} version {MODEL_VERSION} to {EVAL_RESULTS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

summary = {
    "registered_model_name": REGISTERED_MODEL_NAME,
    "model_version": MODEL_VERSION,
    "eval_run_id": eval_run_id,
    "cases_evaluated": len(evaluation_dataset),
    "cases_skipped_unresolvable_location": len(skipped_cases),
    "match_relevance_mean": match_relevance_mean,
    "groundedness_mean": groundedness_mean,
}
print(summary)
