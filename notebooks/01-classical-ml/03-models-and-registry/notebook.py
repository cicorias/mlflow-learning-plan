# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 — Models & the Model Registry
# MAGIC
# MAGIC **Goal:** Move from "I logged a model" to "I have a versioned, governed model I can serve."
# MAGIC
# MAGIC Key concepts:
# MAGIC - **MLmodel file**: YAML that describes flavors (e.g., `sklearn` + `python_function`)
# MAGIC - **pyfunc**: the universal loader interface — any MLflow model can be loaded and called
# MAGIC   the same way regardless of the underlying framework
# MAGIC - **Registry**: named models with integer versions and string aliases
# MAGIC - **Aliases** (`@champion`, `@challenger`): pointers that can be reassigned; the modern
# MAGIC   lifecycle mechanism (string Stages are deprecated in MLflow 3)

# COMMAND ----------

import os

from dotenv import load_dotenv
load_dotenv()  # auto-load .env if present
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("01-classical-ml-03-models-registry")

iris = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

MODEL_NAME = "iris-classifier"
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and register Version 1 — Logistic Regression

# COMMAND ----------

with mlflow.start_run(run_name="lr-v1") as run:
    model_v1 = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    model_v1.fit(X_train, y_train)

    acc_v1 = accuracy_score(y_test, model_v1.predict(X_test))
    mlflow.log_params({"C": 1.0, "max_iter": 200, "model_type": "LogisticRegression"})
    mlflow.log_metric("accuracy", acc_v1)

    # registered_model_name registers the model in one step
    mlflow.sklearn.log_model(
        model_v1,
        name="model",
        registered_model_name=MODEL_NAME,
    )
    run_id_v1 = run.info.run_id

print(f"V1 accuracy: {acc_v1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and register Version 2 — Random Forest

# COMMAND ----------

with mlflow.start_run(run_name="rf-v2") as run:
    model_v2 = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model_v2.fit(X_train, y_train)

    acc_v2 = accuracy_score(y_test, model_v2.predict(X_test))
    mlflow.log_params({"n_estimators": 100, "max_depth": 4, "model_type": "RandomForest"})
    mlflow.log_metric("accuracy", acc_v2)

    mlflow.sklearn.log_model(
        model_v2,
        name="model",
        registered_model_name=MODEL_NAME,
    )
    run_id_v2 = run.info.run_id

print(f"V2 accuracy: {acc_v2:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assign aliases
# MAGIC
# MAGIC `@champion` points to whichever version performed best.
# MAGIC Aliases can be reassigned at any time without creating a new version.

# COMMAND ----------

# Determine which version won
champion_version = "2" if acc_v2 >= acc_v1 else "1"
challenger_version = "1" if champion_version == "2" else "2"

client.set_registered_model_alias(MODEL_NAME, "champion", champion_version)
client.set_registered_model_alias(MODEL_NAME, "challenger", challenger_version)

print(f"@champion  -> version {champion_version}")
print(f"@challenger -> version {challenger_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load by alias
# MAGIC
# MAGIC Application code references the alias URI — the underlying version can be swapped
# MAGIC by reassigning the alias without touching the application.

# COMMAND ----------

champion_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
sample = X_test.head(5)
preds = champion_model.predict(sample)
print("Predictions (champion model):", preds)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve the model locally (local execution only)
# MAGIC
# MAGIC On Databricks, use Model Serving instead. The shell command below is for local runs.
# MAGIC
# MAGIC ```bash
# MAGIC # In a separate terminal:
# MAGIC mlflow models serve \
# MAGIC   -m "models:/iris-classifier@champion" \
# MAGIC   -p 5001 \
# MAGIC   --env-manager local
# MAGIC
# MAGIC # Then send a prediction request:
# MAGIC curl -X POST http://127.0.0.1:5001/invocations \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   -d '{"inputs": [[5.1, 3.5, 1.4, 0.2]]}'
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checkpoint
# MAGIC
# MAGIC Open MLflow UI → **Models** tab → `iris-classifier`.
# MAGIC
# MAGIC - Two versions listed with their run details
# MAGIC - `@champion` and `@challenger` aliases shown on their respective versions
# MAGIC - Click a version → **Artifacts** tab to inspect the MLmodel YAML and see both flavors
# MAGIC
# MAGIC Next: [04-projects-reproducibility](../04-projects-reproducibility/notebook.py)
