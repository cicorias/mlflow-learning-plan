# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 — Foundations & Setup
# MAGIC
# MAGIC **Goal:** Understand the four MLflow components and complete one full end-to-end run.
# MAGIC
# MAGIC The four components:
# MAGIC | Component | One-sentence purpose |
# MAGIC |---|---|
# MAGIC | **Tracking** | Log parameters, metrics, code, and artifacts from every run |
# MAGIC | **Models** | A standard packaging format that works across frameworks |
# MAGIC | **Model Registry** | Versioning, aliases, and lifecycle management for trained models |
# MAGIC | **Projects** | A reproducible packaging format for the code that produced a model |

# COMMAND ----------

import os

from dotenv import load_dotenv
load_dotenv()  # auto-load .env if present
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("01-classical-ml-01-foundations")

print(f"MLflow {mlflow.__version__} | tracking: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data and split

# COMMAND ----------

iris = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and log the first run
# MAGIC
# MAGIC `mlflow.start_run()` opens a run context. Everything logged inside the `with` block
# MAGIC is attached to that run. Use it as a context manager so the run closes cleanly even
# MAGIC if training raises an exception.

# COMMAND ----------

params = {
    "C": 1.0,
    "max_iter": 200,
    "solver": "lbfgs",
}

with mlflow.start_run(run_name="logistic-regression-baseline") as run:
    # Parameters — logged once, immutable after the run closes
    mlflow.log_params(params)
    mlflow.set_tag("dataset", "iris")
    mlflow.set_tag("notebook", "01-classical-ml/01-foundations")

    # Train
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # Metrics — scalar values describing model quality
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", f1)

    # Model — saved in MLflow's portable format
    mlflow.sklearn.log_model(model, name="model")

    run_id = run.info.run_id
    print(f"Run ID : {run_id}")
    print(f"Accuracy: {acc:.4f}   F1: {f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the model back from the run
# MAGIC
# MAGIC `runs:/<RUN_ID>/model` is the URI format to reference a logged model.
# MAGIC The pyfunc loader works regardless of which framework saved the model.

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
sample = X_test.head(5)
predictions = loaded.predict(sample)
print("Sample predictions:", predictions)
print("Actual labels:     ", y_test.head(5).tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checkpoint
# MAGIC
# MAGIC Open the MLflow UI → Experiments → **01-classical-ml-01-foundations** → click the run.
# MAGIC
# MAGIC You should see:
# MAGIC - **Parameters** tab: C, max_iter, solver, multi_class
# MAGIC - **Metrics** tab: accuracy, f1_weighted
# MAGIC - **Artifacts** tab: model/ directory containing MLmodel file and serialized weights
# MAGIC
# MAGIC Next: [02-experiment-tracking](../02-experiment-tracking/notebook.py) — instrument any
# MAGIC training script and compare multiple runs.
