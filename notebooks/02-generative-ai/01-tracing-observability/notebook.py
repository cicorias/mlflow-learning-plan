# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 — Tracing and Observability
# MAGIC
# MAGIC **Goal:** Instrument any LLM application so you can see exactly what happens inside
# MAGIC every request — from the user query to the final response.
# MAGIC
# MAGIC Core concepts:
# MAGIC - A **trace** is the full record of one application call
# MAGIC - A trace contains **spans** — individual operations (LLM call, retrieval, tool use)
# MAGIC - MLflow Tracing is OpenTelemetry-compatible

# COMMAND ----------

import os

from dotenv import load_dotenv
load_dotenv()  # auto-load .env if present
import mlflow
from mlflow.entities import SpanType

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("02-generative-ai-01-tracing")

print(f"MLflow {mlflow.__version__} | tracking: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Autologging — OpenAI
# MAGIC
# MAGIC One line enables tracing for every OpenAI call in the session.
# MAGIC Captured automatically: model, messages, response, token counts, latency.

# COMMAND ----------

import openai

mlflow.openai.autolog()

client = openai.OpenAI()  # reads OPENAI_API_KEY from environment

response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "Explain MLflow Tracing in one sentence."}],
    max_completion_tokens=100,
)
print("OpenAI response:", response.choices[0].message.content)

# Open MLflow UI -> Traces tab to see this trace

# COMMAND ----------

# MAGIC %md
# MAGIC ## Autologging — Anthropic

# COMMAND ----------

import anthropic

mlflow.anthropic.autolog()

if os.getenv("ANTHROPIC_API_KEY"):
    anth_client = anthropic.Anthropic()
    message = anth_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_completion_tokens=100,
        messages=[{"role": "user", "content": "Explain MLflow Tracing in one sentence."}],
    )
    print("Anthropic response:", message.content[0].text)
else:
    print("Skipping Anthropic example — ANTHROPIC_API_KEY not set in .env")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Autologging — LangChain

# COMMAND ----------

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

mlflow.langchain.autolog()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise technical assistant."),
    ("human", "{question}"),
])
chain = prompt | ChatOpenAI(model="gpt-5.2", max_completion_tokens=100)

result = chain.invoke({"question": "What is MLflow Tracking used for?"})
print("LangChain response:", result.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manual tracing with `@mlflow.trace`
# MAGIC
# MAGIC Autologging captures third-party calls. Use `@mlflow.trace` to trace *your own*
# MAGIC functions — preprocessing, retrieval, business logic.
# MAGIC
# MAGIC Decorated functions appear as nested spans inside the parent trace.

# COMMAND ----------

# Minimal RAG application with full observability

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_docs(query: str) -> list[str]:
    """Simulate a retrieval step."""
    # Replace with real vector search (e.g., Azure AI Search, Chroma, FAISS)
    knowledge_base = [
        "MLflow Tracking logs parameters, metrics, and artifacts for every run.",
        "MLflow Models provides a standard packaging format across frameworks.",
        "MLflow Model Registry manages versioning and aliases for trained models.",
        "MLflow Projects packages training code for reproducible execution.",
        "MLflow 3 added tracing, evaluation, and prompt management for GenAI.",
    ]
    # Naive keyword match — replace with embedding similarity in practice
    return [doc for doc in knowledge_base if any(w in doc.lower() for w in query.lower().split())]


@mlflow.trace(span_type=SpanType.CHAIN)
def answer_question(query: str) -> str:
    """Retrieve context then call the LLM."""
    docs = retrieve_docs(query)
    context = "\n".join(docs) if docs else "No relevant documents found."

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the provided context. "
                    "If the context does not contain the answer, say so."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        max_completion_tokens=150,
    )
    return response.choices[0].message.content


# Run two queries — each produces a complete trace in the UI
q1 = "What does MLflow Tracking do?"
q2 = "How does MLflow handle model versioning?"

for question in [q1, q2]:
    answer = answer_question(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect traces in the UI
# MAGIC
# MAGIC Open MLflow UI → Experiments → **02-generative-ai-01-tracing** → **Traces** tab.
# MAGIC
# MAGIC For each trace:
# MAGIC - Left panel: span tree — `answer_question` → `retrieve_docs` + OpenAI call
# MAGIC - Right panel: inputs, outputs, token counts, latency per span
# MAGIC - Compare two traces: select both → **Compare** to diff them side-by-side
# MAGIC
# MAGIC ## Checkpoint
# MAGIC
# MAGIC - [ ] Enabled autologging for OpenAI, Anthropic, and LangChain
# MAGIC - [ ] Added `@mlflow.trace` spans for retrieval and chain functions
# MAGIC - [ ] Inspected a complete trace in the Traces tab
# MAGIC
# MAGIC Next: [02-evaluation-datasets](../02-evaluation-datasets/notebook.py)
