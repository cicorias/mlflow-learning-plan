# 00 — Setup (GenAI Track)

Verify your environment and API credentials before starting the GenAI track.

## What this lesson covers

- Confirm MLflow 3.x, openai, anthropic, langchain package versions
- Check that `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are set
- Set the MLflow tracking URI for local or Databricks Managed MLflow
- Write a test trace to the tracking server and confirm it appears under the Traces tab

## Run it

```bash
# Local — start the server first in a separate terminal:
# uv run mlflow server --host 127.0.0.1 --port 5000
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run python setup.py

# Databricks: import setup.py into a Repo and run as a notebook
```

## Before you start

| Context | What to do |
|---|---|
| Local | Set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` in your shell |
| Databricks | Store API keys as Databricks Secrets; retrieve with `dbutils.secrets.get(...)` |

## Expected output

Versions printed, a single test trace visible in the MLflow UI under `02-generative-ai-setup-check` → Traces tab.
