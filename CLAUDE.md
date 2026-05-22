# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

MLflow learning sandbox covering two tracks, each designed to run both **locally** and on **Databricks on Azure** without code changes:

- **Classic ML track** (`notebooks/01-classical-ml/`, background in `docs/start/mlflow-learning-plan-01.md`): Tracking, Models, Model Registry, Projects
- **GenAI track** (`notebooks/02-generative-ai/`, background in `docs/start/mlflow-learning-plan-02.md`): Tracing, evaluation datasets, LLM-as-a-judge, Prompt Registry

## Toolchain

All tools are managed by [mise](https://mise.jdx.dev/). After cloning:

```bash
mise install        # installs Python 3.14, uv, Azure CLI, Databricks CLI
mise run setup      # syncs Python deps via uv and creates .env from template
```

Tools declared in `mise.toml`: `python`, `uv`, `azure-cli`, `databricks-cli` (aliased so the `databricks` executable resolves).

Post-clone workflow is encoded as mise tasks (`mise tasks ls`):

| Task | Purpose |
|---|---|
| `setup` | `sync` + `env` — one-shot bootstrap after `mise install` |
| `sync` | `uv sync` — install Python deps |
| `env` | Copy `.env.example` to `.env` if missing (never overwrites) |
| `mlflow-server` | Start local SQLite-backed tracking server on `:5000` |
| `mlflow-stop` | Kill the local tracking server |
| `verify` | Run both 00-setup notebooks against the configured tracking URI |
| `run-classical` / `run-genai` | Execute every lesson in a track end-to-end |
| `clean` | Wipe `.mlflow-local/`, `mlruns/`, `sample_project/`, screenshots |

### Scripts convention — important

**Never embed shell code inline in `mise.toml`.** Every multi-line task body lives as an executable script in `./scripts/<name>.sh`; `mise.toml` only contains the one-line `run = "./scripts/<name>.sh"` reference.

Why: inline TOML strings are hard to lint, hard to edit in a normal editor with shell highlighting/shellcheck, hard to test in isolation, and force `mise run <task>` to be the only way to execute them. Plain `.sh` files are run directly (`./scripts/verify.sh`), get full editor/shellcheck support, and keep `mise.toml` to a readable index of available work.

Rules:
- Every script starts with `#!/usr/bin/env bash` and `set -euo pipefail`
- Every script is `chmod +x` (committed with execute bit set)
- A one-line single command (e.g., `run = "uv sync"`) can stay inline — the rule applies to multi-line bodies
- When adding a new task: write `scripts/<name>.sh`, `chmod +x`, then add a one-line entry in `mise.toml`

## Secrets and Environment

Use **`python-dotenv`** for local-only secrets. The `.env` file is gitignored; `.env.example` is committed as the template.

```bash
cp .env.example .env
# edit .env and fill in real values
```

Every notebook calls `load_dotenv()` early so `os.getenv(...)` returns the values from `.env` without any shell setup. On Databricks, use **Databricks Secrets** instead — never put real keys in a notebook or `.env` that ships to the workspace.

Required keys (per track):

| Variable | Used by | Notes |
|---|---|---|
| `MLFLOW_TRACKING_URI` | All notebooks | `http://127.0.0.1:5000` local, `databricks` for managed |
| `OPENAI_API_KEY` | GenAI track | Required for tracing, eval, judges |
| `ANTHROPIC_API_KEY` | GenAI track (optional) | Only used in `01-tracing-observability`; gracefully skipped if absent |
| `MLFLOW_JUDGE_MODEL` | GenAI track (optional) | Defaults to `openai:/gpt-5.2`; set if the OpenAI project supports a different model |

## Authentication (Databricks)

**The only supported auth path is Azure CLI federated credentials — no PATs.**

```bash
# 1. Authenticate with Azure
az login

# 2. Authenticate the Databricks CLI using those Azure credentials
databricks auth login --host https://<your-workspace>.azuredatabricks.net

# Config is written to ~/.databrickscfg
databricks current-user me    # verify
```

When `~/.databrickscfg` is configured, both the Databricks CLI and the MLflow client (via `mlflow.set_tracking_uri("databricks")`) use it automatically.

## MLflow Tracking — Dual Mode

| Context | Tracking URI |
|---|---|
| On Databricks | Set automatically; no action needed |
| Local → Databricks Managed MLflow | `mlflow.set_tracking_uri("databricks")` — reads `~/.databrickscfg` |
| Local only (dev/offline) | `mlflow.set_tracking_uri("http://127.0.0.1:5000")` after starting a local server |

Every notebook uses this pattern so the same file works in both contexts:

```python
import os
from dotenv import load_dotenv
load_dotenv()
import mlflow

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
mlflow.set_tracking_uri(tracking_uri)
```

## Common Commands

```bash
# Local MLflow server (persistent — survives restarts)
uv run mlflow server \
  --backend-store-uri sqlite:///.mlflow-local/mlflow.db \
  --default-artifact-root ./.mlflow-local/mlartifacts \
  --host 127.0.0.1 --port 5000

# Run any notebook locally (load_dotenv pulls .env automatically)
uv run python notebooks/01-classical-ml/01-foundations-setup/notebook.py

# Databricks CLI
databricks workspace list /
databricks workspace import-dir notebooks/01-classical-ml /Workspace/Users/<email>/mlflow-learning/01-classical-ml --overwrite
```

## Notebook / Script Format

All samples are written as **Databricks-flavored Python files** (`.py`) — plain Python with Databricks magic-comment markers. The same file runs as a script locally and imports as a notebook on Databricks (via Databricks Repos or `databricks workspace import-dir`).

```python
# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section heading

# COMMAND ----------

import mlflow
print("works locally and on Databricks")
```

Rules:
- First line must be `# Databricks notebook source`
- Cells are separated by `# COMMAND ----------`
- Markdown cells use `# MAGIC %md` followed by `# MAGIC ...` lines
- Do **not** use `.ipynb` format — keep `.py` as the source of truth
- Always `load_dotenv()` near the top of the imports cell

## Project Layout

```
mlflow-learning-plan/
├── .env.example                  # template (commit this)
├── .env                          # real secrets (gitignored)
├── notebooks/
│   ├── 01-classical-ml/          # Classic ML lessons (00-setup + 4 lessons)
│   └── 02-generative-ai/         # GenAI lessons (00-setup + 5 lessons)
├── docs/start/                   # full learning-plan background docs
├── mise.toml                     # tool versions
├── pyproject.toml                # Python deps (managed by uv)
└── .mlflow-local/                # local server backing store (gitignored)
```

## Key MLflow Conventions

- MLflow 3.x is installed — use `mlflow.genai.evaluate()`, not the deprecated `mlflow.evaluate()`, for GenAI evals
- Use aliases (`@champion`, `@production`) not string Stages (deprecated)
- Tag every run with `git_commit`, `dataset_version`, and any other slice-able dimension
- For Prompt Registry: reference prompts by alias in app code (`prompts:/name@production`); update the alias to promote versions
- **Built-in judges (`Correctness`, `RelevanceToQuery`, `Guidelines`) require a `model=` arg** — without it they call a default that may not be authorized on your OpenAI project. Always pass `model=os.getenv("MLFLOW_JUDGE_MODEL", "openai:/gpt-5.2")` or similar.
- **GPT-5 family uses `max_completion_tokens`, not `max_tokens`** — older notebooks need this replaced
- **Custom code-based scorers must be decorated with `@mlflow.genai.scorers.scorer`** — passing a bare function fails validation
- **`predict_fn` parameter names must match the keys in the dataset's `inputs` dict** — e.g. `def app(question: str)` for `{"inputs": {"question": "..."}}`
- **Async tracing**: call `mlflow.flush_trace_async_logging()` between creating a trace and attaching `log_feedback`, or the feedback insert can race the trace insert and fail with a FK error
