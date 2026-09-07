# Databricks notebook source
# Diagnostic: load the registered matching_agent model as pyfunc and call
# predict, printing the full traceback so we can see why serving returns null.
import os
import sys
import traceback

import mlflow

_NB = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_ROOT = os.path.abspath(os.path.join(_NB, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

dbutils.widgets.text("catalog", "job_agent", "Catalog")
dbutils.widgets.text("warehouse_id", "", "Warehouse id")
CATALOG = dbutils.widgets.get("catalog")
os.environ["SQL_WAREHOUSE_ID"] = dbutils.widgets.get("warehouse_id").strip()

mlflow.set_registry_uri("databricks-uc")
model_uri = f"models:/{CATALOG}.gold.matching_agent@none"

# Resolve latest version explicitly.
from mlflow import MlflowClient

c = MlflowClient(registry_uri="databricks-uc")
versions = c.search_model_versions(f"name='{CATALOG}.gold.matching_agent'")
latest = max(int(v.version) for v in versions)
model_uri = f"models:/{CATALOG}.gold.matching_agent/{latest}"
print("Loading", model_uri)

# COMMAND ----------

import pandas as pd

_result = ""
try:
    from src.agent.matching_agent import MatchingAgentModel, _extract_profile_id
    df = pd.DataFrame([{"profile_id": "demo-1"}])
    pid = _extract_profile_id(df)
    m = MatchingAgentModel()
    m.load_context(None)
    out = m.predict(None, df)
    _result = (
        f"extracted_pid={pid!r} || predict_type={type(out).__name__} || "
        f"predict_out={out!r}"
    )[:2400]
except Exception as e:
    _result = "RAISED: " + repr(e) + " || " + traceback.format_exc()[-2000:]

dbutils.notebook.exit(_result)
