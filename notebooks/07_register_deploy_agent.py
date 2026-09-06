# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Agent Registration and Deployment
# MAGIC
# MAGIC Logs the Matching Agent (`src/agent/matching_agent.py`) to MLflow as
# MAGIC a `ResponsesAgent`, registers it in Unity Catalog as
# MAGIC `job_agent.gold.matching_agent`, and deploys the registered version to
# MAGIC the Model Serving endpoint `job-agent-matching` (Req 7 AC1, 9 AC1).
# MAGIC
# MAGIC **Deployment gate (Req 9 AC5-6)**: each registered model version must
# MAGIC have recorded evaluation results in `ops.evaluation_results` (written
# MAGIC by `notebooks/05_run_evaluation.py`) *before* it is promoted to the
# MAGIC Model Serving endpoint. This notebook always logs and registers the
# MAGIC new version — registration itself carries no risk, since an
# MAGIC unevaluated version simply sits at stage `None` in Unity Catalog and
# MAGIC is not served. Deployment to the endpoint is gated by
# MAGIC `src/utils/deployment_gate.py::check_deployment_gate(model_version)`
# MAGIC (task 17.3). Because task 17.3 has not been implemented yet as of
# MAGIC this notebook's authoring, the gate import is wrapped in a
# MAGIC `try/except ImportError`: if the module is missing, deployment is
# MAGIC skipped with a warning rather than the notebook failing outright.
# MAGIC **Once task 17.3 lands, re-run this notebook (or re-run just the
# MAGIC deployment cell) to pick up the real gate check.**
# MAGIC
# MAGIC **Prerequisite**: `notebooks/00_setup_catalog.py` and
# MAGIC `notebooks/06_create_uc_functions.py` must have already run so the
# MAGIC `gold` schema and the 4 UC Function tools exist.
# MAGIC
# MAGIC Requirements: 7.1, 9.1, 9.5

# COMMAND ----------

import sys
import os

# Allow `from src.agent.matching_agent import ...` when this notebook is run
# from the Databricks Workspace (repo root is not automatically on
# sys.path there, unlike when running via `databricks bundle` sync or repos).
_NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_REPO_ROOT = os.path.abspath(os.path.join(_NOTEBOOK_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.agent.matching_agent import (  # noqa: E402
    LLM_ENDPOINT,
    UC_FUNCTION_NAMES,
    MatchingAgent,
    build_agent,
    make_pyfunc_model,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets

# COMMAND ----------

dbutils.widgets.text("catalog", "job_agent", "Catalog name")
dbutils.widgets.text("registered_model_name", "gold.matching_agent", "UC registered model name (schema.model, catalog prepended)")
dbutils.widgets.text("serving_endpoint_name", "job-agent-matching", "Model Serving endpoint name")
dbutils.widgets.text("serving_endpoint_workload_size", "Small", "Model Serving endpoint workload size")
dbutils.widgets.text("warehouse_id", "", "SQL warehouse ID the agent tools use to call the gold.* UC functions")
dbutils.widgets.dropdown("require_evaluation_gate", "true", ["true", "false"], "Require evaluation results before deploying to serving")

CATALOG = dbutils.widgets.get("catalog")
REGISTERED_MODEL_NAME = f"{CATALOG}.{dbutils.widgets.get('registered_model_name')}"
SERVING_ENDPOINT_NAME = dbutils.widgets.get("serving_endpoint_name")
SERVING_ENDPOINT_WORKLOAD_SIZE = dbutils.widgets.get("serving_endpoint_workload_size")
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id").strip()
REQUIRE_EVALUATION_GATE = dbutils.widgets.get("require_evaluation_gate").strip().lower() == "true"

print(f"Catalog:                  {CATALOG}")
print(f"Registered model name:    {REGISTERED_MODEL_NAME}")
print(f"Serving endpoint name:    {SERVING_ENDPOINT_NAME}")
print(f"Serving endpoint size:    {SERVING_ENDPOINT_WORKLOAD_SIZE}")
print(f"Warehouse ID:             {WAREHOUSE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Log the Matching Agent to MLflow and register in Unity Catalog (Req 7.1, 9.1)
# MAGIC
# MAGIC `mlflow.set_registry_uri("databricks-uc")` targets Unity Catalog as the
# MAGIC model registry so `registered_model_name` resolves to the three-level
# MAGIC UC name `job_agent.gold.matching_agent` rather than the legacy
# MAGIC workspace registry (Req 11 AC2).
# MAGIC
# MAGIC The agent is logged as a `pyfunc` model with the `MatchingAgent`
# MAGIC `ResponsesAgent` instance as `python_model`, following the Agent
# MAGIC Framework's documented pattern for `ResponsesAgent` subclasses. Its UC
# MAGIC Function tool dependencies are declared via `resources` so Model
# MAGIC Serving automatically provisions the credentials the agent needs to
# MAGIC call `search_listings`, `compute_commute_distance`, `get_user_profile`,
# MAGIC and `draft_application` at inference time.

# COMMAND ----------

import mlflow
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint

mlflow.set_registry_uri("databricks-uc")

# The UC Function tool wiring happens inside build_agent(), which returns a
# dict of callables (get_user_profile, search_listings,
# compute_commute_distance) that execute the job_agent.gold.* UC functions on
# the SQL warehouse. The pyfunc adapter (make_pyfunc_model) constructs those
# tools at serving time and delegates matching to MatchingAgent.
pyfunc_model = make_pyfunc_model()

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT)] + [
    DatabricksFunction(function_name=fn) for fn in UC_FUNCTION_NAMES
]

# Ship the src/ package with the model so `from src.agent.matching_agent
# import ...` resolves at serving time, and give MLflow a concrete
# input/output example to infer the signature.
input_example = {"profile_id": "example-profile-id"}

with mlflow.start_run(run_name="matching_agent_registration") as run:
    logged_model_info = mlflow.pyfunc.log_model(
        artifact_path="matching_agent",
        python_model=pyfunc_model,
        code_paths=[os.path.join(_REPO_ROOT, "src")],
        input_example=input_example,
        resources=resources,
        registered_model_name=REGISTERED_MODEL_NAME,
    )

model_version = logged_model_info.registered_model_version
print(f"Logged run:      {run.info.run_id}")
print(f"Registered as:   {REGISTERED_MODEL_NAME} version {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Deployment gate check (Req 9.5, 9.6)
# MAGIC
# MAGIC Before deploying the newly registered version to the Model Serving
# MAGIC endpoint, verify it has recorded evaluation results. The gate is
# MAGIC implemented by `src/utils/deployment_gate.py::check_deployment_gate`
# MAGIC (task 17.3). If that module is not yet present in this checkout,
# MAGIC deployment is skipped with a warning — logging and registration above
# MAGIC still succeed, so this notebook can be re-run (or just the deployment
# MAGIC cell re-invoked) once task 17.3 exists.

# COMMAND ----------

deploy_allowed = False
gate_reason = "src.utils.deployment_gate is not available yet (task 17.3 not implemented)."

if not REQUIRE_EVALUATION_GATE:
    # Explicit bypass (e.g. the showcase pipeline) — deploy the freshly
    # registered version without requiring recorded evaluation results.
    deploy_allowed = True
    gate_reason = "evaluation gate bypassed (require_evaluation_gate=false)."
    print(f"Deployment gate bypassed for version {model_version}.")
else:
    try:
        from src.utils.deployment_gate import check_deployment_gate

        blocked, reason = check_deployment_gate(model_version)
        deploy_allowed = not blocked
        gate_reason = reason
    except ImportError:
        print(
            "WARNING: src.utils.deployment_gate could not be imported "
            f"({gate_reason}) Skipping deployment to '{SERVING_ENDPOINT_NAME}'. "
            f"Re-run this notebook once task 17.3 is implemented to deploy "
            f"version {model_version}."
        )

if deploy_allowed:
    print(f"Deployment gate passed for version {model_version}: {gate_reason}")
elif "gate is not available" not in gate_reason:
    print(
        f"Deployment gate BLOCKED promotion of version {model_version}: {gate_reason}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Deploy to Model Serving (Req 7.1), idempotent
# MAGIC
# MAGIC Only runs if the gate check above passed. Checks whether
# MAGIC `SERVING_ENDPOINT_NAME` already exists: if not, creates it pointing at
# MAGIC `REGISTERED_MODEL_NAME` version `model_version`; if it does, updates
# MAGIC its served-entities config to the new version. Either path uses the
# MAGIC `*_and_wait` SDK calls so the notebook blocks until the endpoint
# MAGIC reaches a ready state.
# MAGIC
# MAGIC Promotion through `None` -> `Staging` -> `Production` model version
# MAGIC stages (per the design's model registry section) is expected to be
# MAGIC driven by the evaluation/gate workflow (tasks 17.x) once implemented;
# MAGIC this notebook only performs the Model Serving deployment step itself.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

if not deploy_allowed:
    print(
        f"Skipping deployment to '{SERVING_ENDPOINT_NAME}': deployment gate did not pass "
        f"({gate_reason})."
    )
else:
    w = WorkspaceClient()

    served_entities = [
        ServedEntityInput(
            entity_name=REGISTERED_MODEL_NAME,
            entity_version=str(model_version),
            workload_size=SERVING_ENDPOINT_WORKLOAD_SIZE,
            scale_to_zero_enabled=True,
        )
    ]

    def endpoint_exists(endpoint_name: str) -> bool:
        try:
            w.serving_endpoints.get(endpoint_name)
            return True
        except Exception:  # noqa: BLE001 - any lookup failure means "not found"
            return False

    if endpoint_exists(SERVING_ENDPOINT_NAME):
        print(f"Endpoint '{SERVING_ENDPOINT_NAME}' already exists, updating served entity to version {model_version}")
        w.serving_endpoints.update_config_and_wait(
            name=SERVING_ENDPOINT_NAME,
            served_entities=served_entities,
        )
    else:
        print(f"Creating endpoint '{SERVING_ENDPOINT_NAME}' serving version {model_version}")
        w.serving_endpoints.create_and_wait(
            name=SERVING_ENDPOINT_NAME,
            config=EndpointCoreConfigInput(served_entities=served_entities),
        )

    print(f"Deployed {REGISTERED_MODEL_NAME} version {model_version} to endpoint '{SERVING_ENDPOINT_NAME}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification

# COMMAND ----------

summary = {
    "registered_model_name": REGISTERED_MODEL_NAME,
    "model_version": model_version,
    "serving_endpoint_name": SERVING_ENDPOINT_NAME,
    "deployment_gate_passed": deploy_allowed,
    "deployment_gate_reason": gate_reason,
}
print(summary)
