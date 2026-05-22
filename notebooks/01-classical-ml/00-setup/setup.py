# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 Classical ML — Environment Setup Check
# MAGIC
# MAGIC Verifies that all required packages are installed and the MLflow tracking server
# MAGIC is reachable before starting the Classic ML learning track.

# COMMAND ----------

import importlib
import os

from dotenv import load_dotenv
load_dotenv()  # auto-load .env if present
import sys

required = {
    "mlflow": "3.0",
    "sklearn": "1.0",
    "pandas": "2.0",
    "numpy": "1.24",
}

print(f"Python {sys.version}")
print()

all_ok = True
for pkg, min_ver in required.items():
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "unknown")
        status = "OK" if ver != "unknown" else "WARN"
        print(f"  {status}  {pkg}=={ver}  (need >={min_ver})")
    except ImportError:
        print(f"  MISSING  {pkg}")
        all_ok = False

if not all_ok:
    raise RuntimeError("Missing packages — run `uv sync` and retry.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow tracking URI
# MAGIC
# MAGIC - **Local**: set `MLFLOW_TRACKING_URI=http://localhost:5000` in your shell, then run `uv run mlflow ui`
# MAGIC - **Databricks**: leave unset; the workspace sets it automatically

# COMMAND ----------

import mlflow

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
mlflow.set_tracking_uri(tracking_uri)
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write a test run
# MAGIC
# MAGIC Confirms the tracking server is reachable and we can log to it.

# COMMAND ----------

mlflow.set_experiment("01-classical-ml-setup-check")

with mlflow.start_run(run_name="env-check"):
    mlflow.log_param("python_version", sys.version.split()[0])
    mlflow.log_param("mlflow_version", mlflow.__version__)
    mlflow.log_metric("setup_ok", 1.0)
    mlflow.set_tag("notebook", "01-classical-ml/00-setup")
    run_id = mlflow.active_run().info.run_id

print(f"\nRun logged: {run_id}")
print("Open the MLflow UI -> Experiments -> 01-classical-ml-setup-check to confirm.")
