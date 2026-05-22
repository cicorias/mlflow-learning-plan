# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 — Experiment Tracking
# MAGIC
# MAGIC **Goal:** Instrument any training script with MLflow tracking and compare runs.
# MAGIC
# MAGIC The six functions that cover ~90% of real tracking usage:
# MAGIC
# MAGIC | Function | Purpose |
# MAGIC |---|---|
# MAGIC | `mlflow.start_run()` | Open a run context |
# MAGIC | `mlflow.log_param/s()` | Record hyperparameters (immutable once logged) |
# MAGIC | `mlflow.log_metric/s()` | Record numeric metrics; can be logged repeatedly across steps |
# MAGIC | `mlflow.log_artifact/s()` | Save any file (plots, configs, data samples) |
# MAGIC | `mlflow.<flavor>.log_model()` | Save a trained model in MLflow's portable format |
# MAGIC | `mlflow.set_tag()` | Attach searchable metadata to a run |

# COMMAND ----------

import os

from dotenv import load_dotenv
load_dotenv()  # auto-load .env if present
import subprocess
import tempfile
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for both local and Databricks
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("01-classical-ml-02-experiment-tracking")

iris = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Autologging — the one-line shortcut
# MAGIC
# MAGIC `mlflow.sklearn.autolog()` captures params, CV metrics, feature importances, and the
# MAGIC model automatically. Enable it once before training; it applies to all subsequent fits
# MAGIC in the session.
# MAGIC
# MAGIC Autolog covers the boilerplate. Add manual calls for anything domain-specific.

# COMMAND ----------

mlflow.sklearn.autolog(log_models=False)  # we'll log the model manually

with mlflow.start_run(run_name="autolog-demo"):
    mlflow.set_tag("approach", "autolog")
    model = LogisticRegression(C=0.5, max_iter=300, random_state=42)
    model.fit(X_train, y_train)
    # autolog logged params and cross-val metrics automatically

mlflow.sklearn.autolog(disable=True)  # turn off for the manual examples below

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manual tracking — three hyperparameter sweeps
# MAGIC
# MAGIC Run three variants and compare them in the UI using **Compare** → Parallel Coordinates.

# COMMAND ----------

def train_and_log(C: float, max_iter: int, solver: str) -> str:
    """Train a logistic regression, log everything, return the run_id."""

    with mlflow.start_run(run_name=f"lr-C{C}-{solver}") as run:
        # --- params ---
        mlflow.log_params({"C": C, "max_iter": max_iter, "solver": solver})
        mlflow.set_tag("dataset", "iris")
        mlflow.set_tag("notebook", "01-classical-ml/02-experiment-tracking")

        # --- train ---
        model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=42)
        model.fit(X_train, y_train)

        # --- metrics logged once (scalar) ---
        preds = model.predict(X_test)
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, preds),
            "f1_weighted": f1_score(y_test, preds, average="weighted"),
        })

        # --- metrics logged per CV fold (step = fold index) ---
        cv_scores = cross_val_score(model, iris.data, iris.target, cv=5)
        for fold, score in enumerate(cv_scores):
            mlflow.log_metric("cv_accuracy", score, step=fold)

        # --- artifact: confusion matrix plot ---
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, preds,
            display_labels=iris.target_names,
            ax=ax,
        )
        fig.tight_layout()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig.savefig(f.name, dpi=100)
            mlflow.log_artifact(f.name, artifact_path="plots")
        plt.close(fig)

        # --- model ---
        mlflow.sklearn.log_model(model, name="model")

        return run.info.run_id


run_ids = []
for C, max_iter, solver in [(0.1, 200, "lbfgs"), (1.0, 200, "lbfgs"), (10.0, 500, "saga")]:
    rid = train_and_log(C, max_iter, solver)
    run_ids.append(rid)
    print(f"Logged run {rid}  (C={C}, solver={solver})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checkpoint
# MAGIC
# MAGIC Open MLflow UI → **01-classical-ml-02-experiment-tracking** → select all three runs → **Compare**.
# MAGIC
# MAGIC - **Parallel Coordinates** view: see how C and solver relate to accuracy
# MAGIC - **Scatter Plot** view: accuracy vs f1_weighted
# MAGIC - Click any run → **Metrics** tab → click `cv_accuracy` to see the per-fold curve
# MAGIC - Click any run → **Artifacts** tab → open plots/ → view the confusion matrix
# MAGIC
# MAGIC Next: [03-models-and-registry](../03-models-and-registry/notebook.py)
