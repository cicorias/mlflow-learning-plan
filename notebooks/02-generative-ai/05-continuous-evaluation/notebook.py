# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 05 — Continuous Evaluation in Production (Stretch)
# MAGIC
# MAGIC **Goal:** Make evaluation a continuous, automated process — not a one-time pre-ship check.
# MAGIC
# MAGIC The production loop:
# MAGIC 1. User gives thumbs-down on a trace
# MAGIC 2. Trace is tagged for expert review
# MAGIC 3. Expert adds the corrected example to the eval dataset
# MAGIC 4. Next eval run catches the same failure mode automatically
# MAGIC 5. No code change ships unless evaluation passes

# COMMAND ----------

import os

from dotenv import load_dotenv
load_dotenv()  # auto-load .env if present
import json
import mlflow
import openai
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import AssessmentSource, AssessmentSourceType
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
mlflow.set_tracking_uri(tracking_uri)
JUDGE_MODEL = os.getenv("MLFLOW_JUDGE_MODEL", "openai:/gpt-5.2")

mlflow.set_experiment("02-generative-ai-05-continuous-eval")

oai_client = openai.OpenAI()
mlflow.openai.autolog()
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate production traffic
# MAGIC
# MAGIC In real deployments, your application logs traces automatically as users interact with it.
# MAGIC Here we simulate a batch of production calls.

# COMMAND ----------

@mlflow.trace
def production_app(question: str) -> str:
    response = oai_client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful MLflow technical assistant.",
            },
            {"role": "user", "content": question},
        ],
        max_completion_tokens=150,
    )
    return response.choices[0].message.content


production_questions = [
    "What is MLflow Tracking?",
    "How do I register a model?",
    "What are model aliases?",
    "How does tracing work in MLflow?",
    "What is the pyfunc flavor?",
]

trace_ids = []
with mlflow.start_run(run_name="production-traffic-sample"):
    mlflow.set_tag("traffic_type", "production_simulation")
    for question in production_questions:
        answer = production_app(question)
        tid = mlflow.get_last_active_trace_id()
        if tid:
            trace_ids.append(tid)

# Flush async trace queue before attaching feedback below.
mlflow.flush_trace_async_logging()

print(f"Simulated {len(production_questions)} production traces")
print(f"Collected {len(trace_ids)} trace IDs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Attach user feedback to production traces
# MAGIC
# MAGIC This is called by your application backend when a user clicks thumbs up/down.
# MAGIC Negative feedback marks traces for expert review.

# COMMAND ----------

# Simulate: 2 of 5 traces received thumbs-down
negative_trace_indices = [1, 3]

for i, trace_id in enumerate(trace_ids):
    value = "down" if i in negative_trace_indices else "up"
    mlflow.log_feedback(
        trace_id=trace_id,
        name="user_thumbs",
        value=value,
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id=f"user_sim_{i:04d}",
        ),
    )

negative_trace_ids = [trace_ids[i] for i in negative_trace_indices]
print(f"Feedback logged. Negative traces: {negative_trace_ids}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feed production failures back into the eval dataset
# MAGIC
# MAGIC Export thumbs-down traces, have an expert add ground-truth expectations,
# MAGIC then append them to the versioned eval dataset.

# COMMAND ----------

# In practice: query traces with negative feedback from the MLflow API
# mlflow.search_traces(filter_string="feedback.user_thumbs = 'down'")
#
# Here we simulate the expert-reviewed additions:

new_examples_from_production = [
    {
        "inputs": {"question": "How do I register a model?"},
        "expectations": {
            "expected_facts": [
                "Use mlflow.<flavor>.log_model with registered_model_name to register in one step.",
                "Or use MlflowClient.create_model_version after logging the model.",
                "Each registration increments the integer version number.",
            ],
        },
        "source": "production_thumbs_down",
    },
    {
        "inputs": {"question": "How does tracing work in MLflow?"},
        "expectations": {
            "expected_facts": [
                "MLflow Tracing captures every step of an LLM application call as spans.",
                "Enable with mlflow.<provider>.autolog() for automatic instrumentation.",
                "Add custom spans with @mlflow.trace for your own functions.",
            ],
        },
        "source": "production_thumbs_down",
    },
]

# Append to existing dataset and log as v2
existing_v1 = pd.DataFrame([
    {"question": "What is MLflow Tracking?", "expected_facts": "[]", "source": "manual"},
])

new_rows = pd.DataFrame([
    {
        "question": ex["inputs"]["question"],
        "expected_facts": json.dumps(ex["expectations"]["expected_facts"]),
        "source": ex.get("source", "manual"),
    }
    for ex in new_examples_from_production
])

dataset_v2 = pd.concat([existing_v1, new_rows], ignore_index=True)

with mlflow.start_run(run_name="eval-dataset-v2"):
    mlflow.log_table(dataset_v2, artifact_file="eval_dataset_v2.json")
    mlflow.set_tag("dataset_version", "v2")
    mlflow.set_tag("dataset_size", str(len(dataset_v2)))
    mlflow.set_tag("added_from", "production_thumbs_down")

print(f"Dataset v2 logged: {len(dataset_v2)} examples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regression-gate pattern
# MAGIC
# MAGIC Before any prompt or model change ships, evaluate it against the latest dataset
# MAGIC and compare against the stored production baseline. If any judge drops below the
# MAGIC threshold, block the promotion.

# COMMAND ----------

CORRECTNESS_THRESHOLD = 0.80
RELEVANCE_THRESHOLD = 0.80

eval_dataset = [
    {
        "inputs": {"question": "What is MLflow Tracking?"},
        "expectations": {
            "expected_facts": [
                "MLflow Tracking logs parameters, metrics, and artifacts.",
                "It provides an API and UI to compare runs.",
            ],
        },
    },
    {
        "inputs": {"question": "What are model aliases?"},
        "expectations": {
            "expected_facts": [
                "Aliases are named pointers to specific model versions.",
                "They can be reassigned without changing application code.",
            ],
        },
    },
]

candidate_prompt = mlflow.genai.load_prompt("prompts:/mlflow-qa-assistant@production")

def candidate_app(question: str):
    rendered = candidate_prompt.format(question=question)
    response = oai_client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": rendered}],
        max_completion_tokens=200,
    )
    return {"response": response.choices[0].message.content}


with mlflow.start_run(run_name="regression-gate-check") as run:
    mlflow.set_tag("gate_type", "pre-promotion")
    mlflow.set_tag("dataset_version", "v2")

    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=candidate_app,
        scorers=[Correctness(model=JUDGE_MODEL), RelevanceToQuery(model=JUDGE_MODEL)],
    )

    candidate_run_id = run.info.run_id

# Retrieve aggregate metrics from the completed run
gate_run = client.get_run(candidate_run_id)
metrics = gate_run.data.metrics

correctness = metrics.get("correctness/score", 0.0)
relevance = metrics.get("relevance_to_query/score", 0.0)

print(f"\nGate results:")
print(f"  Correctness:  {correctness:.2f}  (threshold: {CORRECTNESS_THRESHOLD})")
print(f"  Relevance:    {relevance:.2f}  (threshold: {RELEVANCE_THRESHOLD})")

if correctness >= CORRECTNESS_THRESHOLD and relevance >= RELEVANCE_THRESHOLD:
    print("\nPASSED — candidate can be promoted to @production.")
    # mlflow.genai.set_prompt_alias("mlflow-qa-assistant", "production", new_version)
else:
    print("\nBLOCKED — candidate does not meet quality thresholds.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checkpoint — Complete Self-Check
# MAGIC
# MAGIC You have finished the GenAI learning track. Check off each item:
# MAGIC
# MAGIC - [ ] Enable automatic tracing for any supported LLM library in one line
# MAGIC - [ ] Add manual spans with `@mlflow.trace` for custom logic
# MAGIC - [ ] Have a versioned evaluation dataset of at least 20 examples with expected facts
# MAGIC - [ ] Run `mlflow.genai.evaluate()` and interpret results in the UI
# MAGIC - [ ] Write a custom Guidelines judge encoding domain rules in English
# MAGIC - [ ] Measure judge-human agreement rate on a sample
# MAGIC - [ ] Register a prompt, set an alias, and load it by alias in application code
# MAGIC - [ ] Compare two prompt versions side-by-side on the same dataset
# MAGIC - [ ] Describe the loop: production feedback → eval dataset → regression-gated deployment
# MAGIC
# MAGIC If yes to all nine, you have a real evaluation practice, not just a vibe.
