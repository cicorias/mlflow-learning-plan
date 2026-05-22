# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 Generative AI — Environment Setup Check
# MAGIC
# MAGIC Verifies packages, API credentials, and MLflow connectivity before starting
# MAGIC the GenAI learning track.

# COMMAND ----------

import importlib
import os

from dotenv import load_dotenv
load_dotenv()  # auto-load .env if present
import sys

required = {
    "mlflow": "3.0",
    "openai": "1.0",
    "anthropic": "0.20",
    "langchain": "0.1",
}

print(f"Python {sys.version}")
print()

all_ok = True
for pkg, min_ver in required.items():
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  OK   {pkg}=={ver}  (need >={min_ver})")
    except ImportError:
        print(f"  MISSING  {pkg}")
        all_ok = False

if not all_ok:
    raise RuntimeError("Missing packages — run `uv sync` and retry.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check API credentials
# MAGIC
# MAGIC On Databricks, retrieve secrets with `dbutils.secrets.get(scope="...", key="...")`
# MAGIC rather than environment variables.

# COMMAND ----------

def _check_key(env_var: str) -> bool:
    val = os.getenv(env_var, "")
    if val:
        print(f"  OK   {env_var} is set ({len(val)} chars)")
        return True
    else:
        print(f"  WARN {env_var} is not set — notebooks that use this provider will fail")
        return False

print("API keys:")
_check_key("OPENAI_API_KEY")
_check_key("ANTHROPIC_API_KEY")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow tracking URI

# COMMAND ----------

import mlflow

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
mlflow.set_tracking_uri(tracking_uri)
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write a test trace
# MAGIC
# MAGIC Confirms the tracking server accepts traces (not just runs).

# COMMAND ----------

mlflow.set_experiment("02-generative-ai-setup-check")

@mlflow.trace
def _test_span(x: int) -> int:
    return x * 2

with mlflow.start_run(run_name="env-check"):
    result = _test_span(21)
    mlflow.log_param("mlflow_version", mlflow.__version__)
    mlflow.log_metric("setup_ok", 1.0)
    mlflow.set_tag("notebook", "02-generative-ai/00-setup")
    run_id = mlflow.active_run().info.run_id

print(f"\nRun logged: {run_id}")
print(f"Test span result: {result}")
print("Open MLflow UI -> 02-generative-ai-setup-check -> Traces tab to confirm.")
