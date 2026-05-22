# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 — LLM-as-a-Judge and Built-in Scorers
# MAGIC
# MAGIC **Goal:** Run automated evaluations using built-in and custom LLM judges.
# MAGIC
# MAGIC Two types of scorers:
# MAGIC - **Code-based**: deterministic, cheap, fast (regex, exact match, latency threshold)
# MAGIC - **LLM-based (judges)**: semantic, flexible, expensive — for dimensions code can't measure
# MAGIC
# MAGIC The insight: exact matching can't tell that "give me healthy food options" and
# MAGIC "food to keep me fit" are the same answer. An LLM judge can.

# COMMAND ----------

import os

from dotenv import load_dotenv
load_dotenv()  # auto-load .env if present
import re
import json
import mlflow
import openai
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines, scorer

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
mlflow.set_tracking_uri(tracking_uri)
JUDGE_MODEL = os.getenv("MLFLOW_JUDGE_MODEL", "openai:/gpt-5.2")

mlflow.set_experiment("02-generative-ai-03-llm-judges")

oai_client = openai.OpenAI()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Application under test
# MAGIC
# MAGIC A simple Q&A function that calls the LLM. This is what the scorers will evaluate.

# COMMAND ----------

mlflow.openai.autolog()

def my_app(question: str) -> dict:
    """Q&A application — predict_fn for mlflow.genai.evaluate.

    Parameter names must match the keys in the dataset's `inputs` dict.
    """
    response = oai_client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful technical assistant specializing in MLflow. "
                    "Answer concisely and accurately."
                ),
            },
            {"role": "user", "content": question},
        ],
        max_completion_tokens=200,
    )
    return {"response": response.choices[0].message.content}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation dataset (from notebook 02)
# MAGIC
# MAGIC Using a small subset here for cost/speed. In practice use the full versioned dataset.

# COMMAND ----------

eval_dataset = [
    {
        "inputs": {"question": "What is MLflow Tracking?"},
        "expectations": {
            "expected_facts": [
                "MLflow Tracking logs parameters, metrics, and artifacts from training runs.",
                "It provides an API and UI to compare runs across experiments.",
            ],
        },
    },
    {
        "inputs": {"question": "What are MLflow model aliases?"},
        "expectations": {
            "expected_facts": [
                "Aliases are named pointers to specific model versions.",
                "They can be reassigned without changing application code.",
                "String Stages are deprecated in MLflow 3 in favor of aliases.",
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
            ],
        },
    },
    {
        "inputs": {"question": "What is the pyfunc model flavor?"},
        "expectations": {
            "expected_facts": [
                "pyfunc is MLflow's universal loading interface.",
                "Any MLflow model can be loaded with mlflow.pyfunc.load_model regardless of framework.",
            ],
        },
    },
    {
        "inputs": {"question": "What does mlflow.genai.evaluate() do?"},
        "expectations": {
            "expected_facts": [
                "mlflow.genai.evaluate() runs your application against an evaluation dataset.",
                "It accepts scorer objects that produce per-row quality scores.",
            ],
        },
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run evaluation with built-in judges
# MAGIC
# MAGIC - **Correctness**: does the response match the expected facts?
# MAGIC - **RelevanceToQuery**: does the response address the user's question?

# COMMAND ----------

with mlflow.start_run(run_name="eval-builtin-judges"):
    mlflow.set_tag("dataset_version", "v1-subset")
    mlflow.set_tag("notebook", "02-generative-ai/03-llm-judges-scorers")

    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=my_app,
        scorers=[
            Correctness(model=JUDGE_MODEL),
            RelevanceToQuery(model=JUDGE_MODEL),
        ],
    )

print("Evaluation complete. Open the run in the UI to see per-row scores.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom Guidelines judge
# MAGIC
# MAGIC Encode any domain rule in plain English. The judge uses an LLM to check compliance.

# COMMAND ----------

conciseness_judge = Guidelines(
    model=JUDGE_MODEL,
    name="concise_and_technical",
    guidelines=(
        "The response must be concise — no more than four sentences. "
        "It must use precise technical terminology without excessive jargon. "
        "It must not start with phrases like 'Great question!' or 'Certainly!'."
    ),
)

with mlflow.start_run(run_name="eval-guidelines-judge"):
    mlflow.set_tag("dataset_version", "v1-subset")

    results_guidelines = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=my_app,
        scorers=[
            Correctness(model=JUDGE_MODEL),
            RelevanceToQuery(model=JUDGE_MODEL),
            conciseness_judge,
        ],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Code-based scorer
# MAGIC
# MAGIC Deterministic, free, runs in milliseconds. Use for structural checks that don't need
# MAGIC semantic understanding.

# COMMAND ----------

@scorer
def response_length_scorer(outputs) -> float:
    """Pass if the response is between 20 and 300 words."""
    response = outputs.get("response", "")
    word_count = len(response.split())
    return 1.0 if 20 <= word_count <= 300 else 0.0


@scorer
def no_apology_opener_scorer(outputs) -> float:
    """Fail if the response starts with a common filler phrase."""
    response = outputs.get("response", "").strip().lower()
    bad_openers = ["certainly", "great question", "of course", "sure!", "absolutely"]
    for opener in bad_openers:
        if response.startswith(opener):
            return 0.0
    return 1.0


with mlflow.start_run(run_name="eval-mixed-scorers"):
    mlflow.set_tag("dataset_version", "v1-subset")

    results_mixed = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=my_app,
        scorers=[
            Correctness(model=JUDGE_MODEL),
            RelevanceToQuery(model=JUDGE_MODEL),
            conciseness_judge,
            response_length_scorer,
            no_apology_opener_scorer,
        ],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checkpoint
# MAGIC
# MAGIC Open MLflow UI → **02-generative-ai-03-llm-judges** → click `eval-mixed-scorers` run.
# MAGIC
# MAGIC - **Metrics** tab: aggregate scores per judge across all rows
# MAGIC - **Artifacts** tab → `eval_results_table.json`: per-row breakdown — find the three
# MAGIC   worst-performing examples
# MAGIC - Compare the three runs side-by-side to see how adding scorers changes results
# MAGIC
# MAGIC - [ ] Ran evaluation with built-in judges
# MAGIC - [ ] Wrote a custom Guidelines judge encoding a domain rule
# MAGIC - [ ] Added code-based scorers for structure checks
# MAGIC - [ ] Identified the worst-performing examples in the results table
# MAGIC
# MAGIC Next: [04-human-feedback-prompts](../04-human-feedback-prompts/notebook.py)
