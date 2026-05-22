# Learning Documentation

Background reading for the two MLflow tracks in this repository. Each background doc explains the concepts and links to the runnable notebook in this repo that exercises them.

## How to use these docs

1. **Read** the background doc for the track you're starting (classical ML or GenAI). Each section maps a topic to its corresponding lesson notebook.
2. **Run** the linked notebook from the project root using `uv run python <path>`. The notebooks are Databricks-flavored Python (`.py`) — they execute as scripts locally and import as notebooks on Databricks.
3. **Inspect** the MLflow UI at <http://127.0.0.1:5000> (local) or your Databricks workspace.

## Background docs

| Track | Document | Notebook root |
|---|---|---|
| Classical ML | [01-classical-ml-background.md](01-classical-ml-background.md) | [`notebooks/01-classical-ml/`](../notebooks/01-classical-ml/) |
| Generative AI | [02-generative-ai-background.md](02-generative-ai-background.md) | [`notebooks/02-generative-ai/`](../notebooks/02-generative-ai/) |

## Original learning-plan source

The condensed plans these background docs are derived from live in [`docs/start/`](start/). Those documents are the "what to learn and in what order" syllabus; the background docs add concepts, context, and links to the runnable code in this repo.

## Prerequisites

Before starting either track:

- `mise install` to get Python 3.14, uv, Azure CLI, Databricks CLI
- `uv sync` to install Python dependencies
- `cp .env.example .env` and fill in `OPENAI_API_KEY` (for GenAI) and `MLFLOW_TRACKING_URI`
- Either:
  - **Local**: start `uv run mlflow server --backend-store-uri sqlite:///.mlflow-local/mlflow.db --default-artifact-root ./.mlflow-local/mlartifacts --host 127.0.0.1 --port 5000`
  - **Databricks**: `az login` then `databricks auth login --host https://<workspace>.azuredatabricks.net`

Full details in the top-level [README.md](../README.md) and [CLAUDE.md](../CLAUDE.md).
