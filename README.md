# MLflow Learning Plan

Hands-on learning sandbox for MLflow — classic ML and GenAI workflows — designed to run both **locally** and on **Databricks on Azure** without code changes.

## Learning Tracks

| Track | Document | Topics |
|---|---|---|
| Classic ML | `docs/start/mlflow-learning-plan-01.md` | Tracking, Models, Model Registry, Projects |
| GenAI | `docs/start/mlflow-learning-plan-02.md` | Tracing, evaluation, LLM-as-a-judge, Prompt Registry |

## Setup

### Prerequisites

- [mise](https://mise.jdx.dev/) — manages Python, uv, Azure CLI, Databricks CLI

```bash
mise install        # installs all tools declared in mise.toml
mise run setup      # syncs Python deps and creates .env from template
# Then edit .env and set OPENAI_API_KEY for the GenAI track
```

All post-clone steps are encoded as mise tasks. Run `mise tasks ls` to see them. The most common:

| Task | What it does |
|---|---|
| `mise run setup` | Sync deps + create `.env` (run once after `mise install`) |
| `mise run mlflow-server` | Start the local MLflow tracking server on port 5000 |
| `mise run verify` | Smoke-test both 00-setup notebooks against the running server |
| `mise run run-classical` | Execute every classical ML lesson in order |
| `mise run run-genai` | Execute every GenAI lesson in order (needs `OPENAI_API_KEY`) |
| `mise run clean` | Remove local MLflow state and generated artifacts |

### Authentication (required for Databricks)

Auth uses Azure CLI federated credentials — no PATs needed:

```bash
az login
databricks auth login --host https://<your-workspace>.azuredatabricks.net
databricks current-user me    # verify
```

## Running Samples

All samples are in `scripts/` (standalone Python) and `notebooks/` (Databricks-flavored `.py` files).

**Locally:**
```bash
uv run python scripts/<name>.py
```

**On Databricks:** sync the `notebooks/` directory to a Databricks Repo or upload via the CLI. The `.py` notebook files import directly as Databricks notebooks.

### MLflow Tracking URI

```bash
# Local dev (offline)
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run mlflow ui    # start local server

# Databricks Managed MLflow (default)
unset MLFLOW_TRACKING_URI   # or set to "databricks"
```

On Databricks, the tracking URI is configured automatically.

## Dependencies

- `mlflow>=3.x` — tracking, registry, GenAI evaluation
- `anthropic` — Anthropic SDK (Claude)
- `openai` — OpenAI SDK
- `langchain` / `langchain-openai` — LangChain integrations
