# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 — Human Feedback, Judge Validation, and Prompt Management
# MAGIC
# MAGIC **Goal:** Close the loop between automated evaluation, human judgment, and prompt iteration.
# MAGIC
# MAGIC This lesson covers:
# MAGIC 1. Attaching human feedback to traces programmatically
# MAGIC 2. Validating an LLM judge against human labels (agreement rate)
# MAGIC 3. Versioning prompts in the Prompt Registry
# MAGIC 4. Running A/B evaluation across two prompt versions

# COMMAND ----------

import os

from dotenv import load_dotenv
load_dotenv()  # auto-load .env if present
import random
import mlflow
import openai
from mlflow import MlflowClient
from mlflow.entities import AssessmentSource, AssessmentSourceType
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
mlflow.set_tracking_uri(tracking_uri)
JUDGE_MODEL = os.getenv("MLFLOW_JUDGE_MODEL", "openai:/gpt-5.2")

mlflow.set_experiment("02-generative-ai-04-feedback-prompts")

oai_client = openai.OpenAI()
mlflow.openai.autolog()
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Collect human feedback on a trace
# MAGIC
# MAGIC `mlflow.log_feedback` attaches a named assessment to a trace.
# MAGIC In a real application, this is called when a user clicks thumbs up/down in the UI.

# COMMAND ----------

# Generate a trace to attach feedback to
@mlflow.trace
def run_qa(question: str) -> str:
    response = oai_client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "You are a helpful MLflow technical assistant."},
            {"role": "user", "content": question},
        ],
        max_completion_tokens=150,
    )
    return response.choices[0].message.content


with mlflow.start_run(run_name="feedback-demo"):
    answer = run_qa("What is MLflow Tracking?")
    trace_id = mlflow.get_last_active_trace_id()

# Trace logging is async — flush before attaching feedback or the
# log_feedback call may race the trace insert and fail with a FK error.
mlflow.flush_trace_async_logging()

print(f"Trace ID: {trace_id}")
print(f"Answer: {answer[:100]}...")

# COMMAND ----------

# Attach thumbs-up feedback from a simulated user
mlflow.log_feedback(
    trace_id=trace_id,
    name="user_thumbs",
    value="up",
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="user_demo_001",
    ),
    rationale="Clear and accurate description of MLflow Tracking.",
)

print(f"Feedback logged on trace {trace_id}")
print("Open UI -> Traces tab -> click the trace -> Assessments panel to see it.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Validate a judge against human labels
# MAGIC
# MAGIC An unvalidated LLM judge is a confident liar. Measure agreement with human labels
# MAGIC on a sample before trusting the judge's aggregate scores.
# MAGIC
# MAGIC Workflow:
# MAGIC 1. Sample traces (simulated here with synthetic labels)
# MAGIC 2. Collect human labels for each trace
# MAGIC 3. Run the judge on the same traces
# MAGIC 4. Compute raw agreement rate (or Cohen's kappa for rigour)

# COMMAND ----------

# Simulate: 20 traces each with a human label and a judge label
# In practice: export real traces from the Traces tab, have a human label them,
# run the judge on those same traces, compare.

random.seed(42)

human_labels = [random.choice(["pass", "fail"]) for _ in range(20)]

# Simulate a judge that agrees ~85% of the time
judge_labels = []
for label in human_labels:
    if random.random() < 0.85:
        judge_labels.append(label)
    else:
        judge_labels.append("fail" if label == "pass" else "pass")

agreements = sum(h == j for h, j in zip(human_labels, judge_labels))
agreement_rate = agreements / len(human_labels)

print(f"Human labels:  {human_labels}")
print(f"Judge labels:  {judge_labels}")
print(f"Agreement rate: {agreement_rate:.1%}  ({agreements}/{len(human_labels)})")

if agreement_rate >= 0.90:
    print("Judge is trustworthy (>=90% agreement with humans).")
elif agreement_rate >= 0.75:
    print("Judge needs improvement — iterate on its instructions.")
else:
    print("Judge is unreliable (<75% agreement) — do not trust its aggregate scores.")

with mlflow.start_run(run_name="judge-validation"):
    mlflow.log_metric("judge_human_agreement_rate", agreement_rate)
    mlflow.log_metric("sample_size", len(human_labels))
    mlflow.set_tag("judge", "Correctness")
    mlflow.set_tag("notebook", "02-generative-ai/04-human-feedback-prompts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Register and version prompts
# MAGIC
# MAGIC The Prompt Registry versions prompts the same way the Model Registry versions models.
# MAGIC Use `{{variable}}` double-brace syntax for template variables.

# COMMAND ----------

# Register Version 1 — minimal instruction
prompt_v1 = mlflow.genai.register_prompt(
    name="mlflow-qa-assistant",
    template=(
        "You are a helpful MLflow technical assistant. "
        "Answer the following question accurately.\n\n"
        "Question: {{question}}"
    ),
    commit_message="Initial version — minimal system prompt",
)

print(f"Registered prompt v{prompt_v1.version}: {prompt_v1.name}")

# COMMAND ----------

# Register Version 2 — improved with explicit constraints
prompt_v2 = mlflow.genai.register_prompt(
    name="mlflow-qa-assistant",
    template=(
        "You are a concise MLflow technical assistant. "
        "Answer in at most three sentences using precise technical language. "
        "If the answer is not about MLflow, decline politely.\n\n"
        "Question: {{question}}"
    ),
    commit_message="Add conciseness constraint and refusal instruction to reduce verbosity",
)

print(f"Registered prompt v{prompt_v2.version}: {prompt_v2.name}")

# COMMAND ----------

# Set @production alias on v1 initially
mlflow.genai.set_prompt_alias(
    name="mlflow-qa-assistant",
    alias="production",
    version=prompt_v1.version,
)
print(f"@production -> version {prompt_v1.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — A/B evaluation across prompt versions
# MAGIC
# MAGIC Run both prompt versions against the same dataset and compare side-by-side in the UI.

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

conciseness_judge = Guidelines(
    model=JUDGE_MODEL,
    name="concise_response",
    guidelines="The response must be no more than three sentences and avoid filler phrases.",
)

for version in [prompt_v1.version, prompt_v2.version]:
    prompt = mlflow.genai.load_prompt(f"prompts:/mlflow-qa-assistant/{version}")

    def app_with_prompt(question: str, _prompt=prompt):
        rendered = _prompt.format(question=question)
        response = oai_client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "user", "content": rendered}],
            max_completion_tokens=200,
        )
        return {"response": response.choices[0].message.content}

    with mlflow.start_run(run_name=f"eval-prompt-v{version}"):
        mlflow.set_tag("prompt_version", str(version))
        mlflow.set_tag("prompt_name", "mlflow-qa-assistant")
        mlflow.set_tag("dataset_version", "v1-subset")

        mlflow.genai.evaluate(
            data=eval_dataset,
            predict_fn=app_with_prompt,
            scorers=[Correctness(model=JUDGE_MODEL), RelevanceToQuery(model=JUDGE_MODEL), conciseness_judge],
        )

    print(f"Evaluated prompt v{version}")

# COMMAND ----------

# Promote v2 to @production if it performed better (manual decision after reviewing UI)
# mlflow.genai.set_prompt_alias(name="mlflow-qa-assistant", alias="production", version=prompt_v2.version)
# print("@production promoted to v2")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checkpoint
# MAGIC
# MAGIC Open MLflow UI → **02-generative-ai-04-feedback-prompts**.
# MAGIC
# MAGIC - Select `eval-prompt-v1` and `eval-prompt-v2` → **Compare** to see side-by-side scores
# MAGIC - Did v2 improve Correctness? Did it regress on any examples?
# MAGIC - Open the Traces tab → click a trace from the feedback-demo run → Assessments panel
# MAGIC
# MAGIC - [ ] Attached human feedback to a trace with `mlflow.log_feedback`
# MAGIC - [ ] Computed judge-human agreement rate on a 20-trace sample
# MAGIC - [ ] Registered two prompt versions and set `@production` alias
# MAGIC - [ ] Compared two prompt versions side-by-side in the UI
# MAGIC
# MAGIC Next: [05-continuous-evaluation](../05-continuous-evaluation/notebook.py)
