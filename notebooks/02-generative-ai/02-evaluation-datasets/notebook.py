# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 — Evaluation Datasets and Ground Truth
# MAGIC
# MAGIC **Goal:** Build, version, and use evaluation datasets — the foundation of any serious
# MAGIC evaluation effort.
# MAGIC
# MAGIC An evaluation dataset is your **test database**. Every quality decision —
# MAGIC "is this prompt better?", "did this model regress?", "is this agent ready to ship?" —
# MAGIC answers itself against this dataset.
# MAGIC
# MAGIC A dataset row has:
# MAGIC - **inputs** — what gets sent to your application
# MAGIC - **expectations** — ground truth: expected facts, guidelines, refusal conditions

# COMMAND ----------

import os

from dotenv import load_dotenv
load_dotenv()  # auto-load .env if present
import json
import mlflow
import pandas as pd

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("02-generative-ai-02-eval-datasets")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build a hand-curated evaluation dataset
# MAGIC
# MAGIC Start small. Twenty well-chosen examples beats 2,000 noisy ones.
# MAGIC Cover: happy path, edge cases, and known failure modes.

# COMMAND ----------

eval_dataset = [
    {
        "inputs": {"question": "What is MLflow Tracking?"},
        "expectations": {
            "expected_facts": [
                "MLflow Tracking logs parameters, metrics, and artifacts from training runs.",
                "It provides an API and UI to compare runs across experiments.",
                "Runs can be logged locally or to a remote tracking server.",
            ],
        },
    },
    {
        "inputs": {"question": "How do I register a model in MLflow?"},
        "expectations": {
            "expected_facts": [
                "Use mlflow.<flavor>.log_model with registered_model_name to register in one step.",
                "Alternatively use MlflowClient.create_registered_model and create_model_version.",
                "Registered models have integer versions that increment with each new version.",
            ],
        },
    },
    {
        "inputs": {"question": "What are MLflow model aliases?"},
        "expectations": {
            "expected_facts": [
                "Aliases are named pointers to specific model versions (e.g., @champion, @production).",
                "They can be reassigned to a different version without changing application code.",
                "String Stages (Staging, Production) are deprecated in MLflow 3 in favor of aliases.",
            ],
        },
    },
    {
        "inputs": {"question": "How does MLflow Tracing work?"},
        "expectations": {
            "expected_facts": [
                "MLflow Tracing captures every step of an LLM application call as spans.",
                "It supports one-line autologging for OpenAI, Anthropic, LangChain, and many others.",
                "It is OpenTelemetry-compatible.",
                "Manual spans can be added with the @mlflow.trace decorator.",
            ],
        },
    },
    {
        "inputs": {"question": "What is the pyfunc model flavor?"},
        "expectations": {
            "expected_facts": [
                "pyfunc is MLflow's universal loading interface.",
                "Any MLflow model can be loaded with mlflow.pyfunc.load_model regardless of framework.",
                "It exposes a predict() method that works the same way for all model types.",
            ],
        },
    },
    {
        "inputs": {"question": "How do I log metrics over training steps?"},
        "expectations": {
            "expected_facts": [
                "Use mlflow.log_metric with the step argument to log a metric at each step.",
                "Step-logged metrics appear as curves in the MLflow UI.",
                "Parameters are logged once; metrics can be logged many times.",
            ],
        },
    },
    {
        "inputs": {"question": "What is the Prompt Registry in MLflow?"},
        "expectations": {
            "expected_facts": [
                "The Prompt Registry versions prompts the same way the Model Registry versions models.",
                "Prompts use double-brace syntax for variables: {{variable}}.",
                "Aliases can be set on prompt versions (e.g., @production) for stable references.",
            ],
        },
    },
    {
        "inputs": {"question": "What does mlflow.genai.evaluate() do?"},
        "expectations": {
            "expected_facts": [
                "mlflow.genai.evaluate() runs your application against an evaluation dataset.",
                "It accepts scorer objects that produce per-row quality scores.",
                "Results are logged as a run and viewable in the MLflow UI.",
            ],
        },
    },
    {
        "inputs": {"question": "What is an LLM judge in MLflow?"},
        "expectations": {
            "expected_facts": [
                "An LLM judge is a scorer that uses a language model to evaluate quality.",
                "It can assess dimensions that exact matching cannot, like semantic correctness or tone.",
                "MLflow provides built-in judges: Correctness, RelevanceToQuery, Guidelines, Safety.",
            ],
        },
    },
    {
        "inputs": {"question": "What is autologging in MLflow?"},
        "expectations": {
            "expected_facts": [
                "Autologging captures parameters, metrics, and models automatically from supported frameworks.",
                "Enable it with mlflow.<framework>.autolog() before training.",
                "Supported frameworks include scikit-learn, PyTorch, XGBoost, LightGBM, and others.",
            ],
        },
    },
    {
        "inputs": {"question": "How do I compare runs in the MLflow UI?"},
        "expectations": {
            "expected_facts": [
                "Select multiple runs in the Experiments view and click Compare.",
                "The parallel coordinates plot shows relationships between params and metrics.",
                "The scatter plot view lets you plot any two metrics against each other.",
            ],
        },
    },
    {
        "inputs": {"question": "What is an MLflow Project?"},
        "expectations": {
            "expected_facts": [
                "An MLflow Project is a directory with an MLproject YAML file.",
                "It declares entry points, parameters, and a dependency specification.",
                "Anyone can run it with mlflow run <repo-url> -P param=value.",
            ],
        },
    },
    {
        "inputs": {"question": "What is the difference between backend store and artifact store?"},
        "expectations": {
            "expected_facts": [
                "The backend store holds run metadata: params, metrics, tags (typically a database).",
                "The artifact store holds large files: models, plots, data (typically object storage).",
                "Databricks Managed MLflow handles both automatically.",
            ],
        },
    },
    {
        "inputs": {"question": "How do I attach human feedback to a trace?"},
        "expectations": {
            "expected_facts": [
                "Use mlflow.log_feedback with a trace_id, name, and value.",
                "AssessmentSource specifies whether the feedback came from a human or an LLM judge.",
                "Feedback is stored with the annotator's identity and timestamp.",
            ],
        },
    },
    {
        "inputs": {"question": "What does mlflow.set_tag do?"},
        "expectations": {
            "expected_facts": [
                "mlflow.set_tag attaches a key-value string label to a run.",
                "Tags are searchable and filterable in the MLflow UI.",
                "Common uses: git commit hash, dataset version, owner, experiment branch.",
            ],
        },
    },
    {
        "inputs": {"question": "Can MLflow serve a model as a REST endpoint?"},
        "expectations": {
            "expected_facts": [
                "Yes, using mlflow models serve -m <model-uri> -p <port>.",
                "The endpoint accepts POST requests to /invocations with JSON input.",
                "On Databricks, Model Serving provides a managed REST endpoint.",
            ],
        },
    },
    {
        "inputs": {"question": "How does mlflow run reproduce a training pipeline?"},
        "expectations": {
            "expected_facts": [
                "mlflow run executes the entry point defined in MLproject.",
                "It installs the declared dependencies before running.",
                "Parameters can be overridden on the command line with -P key=value.",
            ],
        },
    },
    {
        "inputs": {"question": "What is a RAG application?"},
        "expectations": {
            "expected_facts": [
                "RAG stands for Retrieval-Augmented Generation.",
                "It retrieves relevant documents from a knowledge base and passes them as context to an LLM.",
                "It reduces hallucinations by grounding responses in retrieved facts.",
            ],
        },
    },
    {
        "inputs": {"question": "What does @mlflow.trace do?"},
        "expectations": {
            "expected_facts": [
                "@mlflow.trace is a decorator that creates a span for the decorated function.",
                "The span captures the function's inputs, outputs, and execution time.",
                "SpanType can be set to RETRIEVER, CHAIN, LLM, or TOOL for semantic tagging.",
            ],
        },
    },
    {
        "inputs": {"question": "How should I version my evaluation dataset?"},
        "expectations": {
            "expected_facts": [
                "Log the dataset as an MLflow artifact using mlflow.log_table.",
                "Tag the run with dataset_version so evaluation runs reference the exact dataset used.",
                "When you add examples, log it as a new version (v2, v3, etc.).",
            ],
        },
    },
]

print(f"Dataset size: {len(eval_dataset)} examples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Version the dataset as an MLflow artifact
# MAGIC
# MAGIC `mlflow.log_table` saves a DataFrame as a JSON artifact and registers it in the
# MAGIC Artifact Store. Tag the run so every future eval run can reference the exact dataset version.

# COMMAND ----------

df = pd.DataFrame([
    {
        "question": row["inputs"]["question"],
        "expected_facts": json.dumps(row["expectations"]["expected_facts"]),
    }
    for row in eval_dataset
])

with mlflow.start_run(run_name="eval-dataset-v1"):
    mlflow.log_table(df, artifact_file="eval_dataset_v1.json")
    mlflow.set_tag("dataset_version", "v1")
    mlflow.set_tag("dataset_size", str(len(df)))
    mlflow.set_tag("notebook", "02-generative-ai/02-evaluation-datasets")
    dataset_run_id = mlflow.active_run().info.run_id

print(f"Dataset logged in run: {dataset_run_id}")
print(f"Tag dataset_version=v1 on all downstream eval runs to link them to this artifact.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checkpoint
# MAGIC
# MAGIC Open MLflow UI → **02-generative-ai-02-eval-datasets** → click `eval-dataset-v1` run →
# MAGIC **Artifacts** tab → `eval_dataset_v1.json`.
# MAGIC
# MAGIC - [ ] Dataset has 20 examples with inputs and expected_facts
# MAGIC - [ ] Dataset logged and tagged with dataset_version=v1
# MAGIC - [ ] You can explain why versioning matters for reproducible comparisons
# MAGIC
# MAGIC Next: [03-llm-judges-scorers](../03-llm-judges-scorers/notebook.py)
