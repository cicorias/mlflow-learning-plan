# TODO

Open work for this learning sandbox. Items are roughly in priority order within each section.

---

## Setup the Databricks-hosted walkthrough

**Goal:** the same 11 notebooks that work locally should run on a Databricks workspace on Azure, with the only difference being where keys come from and where the tracking server lives. Right now the local path is fully verified; the Databricks path is documented but not exercised end-to-end.

### Prerequisites

- [ ] Azure subscription with an Azure Databricks workspace (Premium tier recommended for full Model Registry features)
- [ ] Workspace URL noted: `https://<workspace-name>.azuredatabricks.net`
- [ ] User account has: **Workspace Access**, **MLflow Experiment** create, **Cluster** create-or-attach, **Model Registry** write
- [ ] Decide on the **Databricks Runtime** version — must include MLflow ≥ 3.0 (Databricks Runtime 15.x or later)

### One-time authentication

- [ ] `az login` — pick the right tenant if you have multiple
- [ ] `az account show` — confirm the active subscription
- [ ] `databricks auth login --host https://<workspace>.azuredatabricks.net` — uses the Azure CLI tokens, no PAT
- [ ] `databricks current-user me` — verify connection
- [ ] Inspect `~/.databrickscfg` and confirm the default profile points at the right workspace

### Cluster

- [ ] Create (or identify) a cluster with Databricks Runtime 15.x ML or later
- [ ] Confirm `mlflow` reports ≥ 3.0 on the cluster — run `import mlflow; print(mlflow.__version__)` in a scratch notebook
- [ ] Note the cluster ID for CLI commands: `databricks clusters list --output json | jq '.[] | {name: .cluster_name, id: .cluster_id, state: .state}'`

### Sync the notebooks to the workspace

- [ ] Decide on a workspace path (e.g. `/Workspace/Users/<your-email>/mlflow-learning/`)
- [ ] Import classical ML notebooks:
  ```bash
  databricks workspace import-dir notebooks/01-classical-ml \
    /Workspace/Users/<your-email>/mlflow-learning/01-classical-ml --overwrite
  ```
- [ ] Import GenAI notebooks:
  ```bash
  databricks workspace import-dir notebooks/02-generative-ai \
    /Workspace/Users/<your-email>/mlflow-learning/02-generative-ai --overwrite
  ```
- [ ] Open the workspace UI and confirm the notebooks rendered with cells (not as raw text)

### Secrets — replace `.env` with Databricks Secrets

`.env` and `python-dotenv` should never run on Databricks. Replace with Databricks Secrets.

- [ ] Create a secret scope: `databricks secrets create-scope genai`
- [ ] Add the OpenAI key: `databricks secrets put-secret genai openai-api-key`
- [ ] (Optional) Add the Anthropic key: `databricks secrets put-secret genai anthropic-api-key`
- [ ] Decide on a pattern for notebook code to switch between `.env` (local) and `dbutils.secrets` (Databricks). Options:
  - A boolean check on `os.getenv("DATABRICKS_RUNTIME_VERSION")` near `load_dotenv()`
  - A small helper module imported by every notebook
  - Workspace-side replacement of the `load_dotenv()` cell with a `dbutils.secrets.get(...)` cell

  Recommended pattern (to be added to every notebook):
  ```python
  import os
  if os.getenv("DATABRICKS_RUNTIME_VERSION"):
      os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="genai", key="openai-api-key")
  else:
      from dotenv import load_dotenv
      load_dotenv()
  ```

### Tracking URI and experiments

- [ ] Confirm `MLFLOW_TRACKING_URI` is unset (or `databricks`) on the cluster — Databricks sets it automatically
- [ ] Decide on the workspace experiment path convention. Recommended: `/Shared/mlflow-learning/<lesson-name>`
- [ ] Update `mlflow.set_experiment(...)` calls in every notebook to use workspace paths when running on Databricks. Pattern:
  ```python
  EXPERIMENT_ROOT = "/Shared/mlflow-learning" if os.getenv("DATABRICKS_RUNTIME_VERSION") else ""
  mlflow.set_experiment(f"{EXPERIMENT_ROOT}/01-classical-ml-01-foundations" if EXPERIMENT_ROOT else "01-classical-ml-01-foundations")
  ```

### Per-notebook adjustments needed

| Notebook | Adjustment |
|---|---|
| `01-classical-ml/04-projects-reproducibility` | The `mlflow run` subprocess call won't work the same way — port to `dbutils.notebook.run` or document as local-only |
| `01-classical-ml/03-models-and-registry` | Swap the `mlflow models serve` shell example for Databricks Model Serving instructions |
| `02-generative-ai/03-llm-judges-scorers` | Confirm the `Safety` and `RetrievalRelevance` judges work — they were Databricks-managed-first features |
| `02-generative-ai/05-continuous-evaluation` | Production traffic simulation is the same; consider showing how to query traces via the Databricks SQL warehouse |

### Verification checklist

Run each notebook on the Databricks cluster and confirm:

- [ ] `01-classical-ml/00-setup` — env-check experiment + run appear in workspace
- [ ] `01-classical-ml/01-foundations-setup` — sklearn run logged with model artifact
- [ ] `01-classical-ml/02-experiment-tracking` — autolog + manual runs, comparable in UI
- [ ] `01-classical-ml/03-models-and-registry` — `iris-classifier` registered, aliases set
- [ ] `01-classical-ml/04-projects-reproducibility` — either runs or has been adapted/skipped per above
- [ ] `02-generative-ai/00-setup` — test trace appears under Traces tab
- [ ] `02-generative-ai/01-tracing-observability` — OpenAI + LangChain spans visible, custom `@mlflow.trace` spans nested correctly
- [ ] `02-generative-ai/02-evaluation-datasets` — `eval-dataset-v1` artifact created
- [ ] `02-generative-ai/03-llm-judges-scorers` — three evaluation runs with per-scorer metrics
- [ ] `02-generative-ai/04-human-feedback-prompts` — feedback attached to trace, prompt v1+v2 registered, A/B eval visible
- [ ] `02-generative-ai/05-continuous-evaluation` — production traces with thumbs-up/down feedback, dataset v2 logged, regression-gate runs

### Deliverables

- [ ] `docs/03-databricks-walkthrough.md` — a reading companion equivalent to the two existing background docs but with Databricks-specific notes (Secrets, workspace paths, Model Serving, Repos sync, runtime versions)
- [ ] Update `CLAUDE.md` "Common Commands" section with the actual `databricks workspace import-dir` invocations and the verified workspace experiment path convention
- [ ] Add a `mise run sync-to-databricks` task that pushes the notebooks (parameterized by `DATABRICKS_USER_PATH`)
- [ ] Add a `mise run verify-databricks` task that runs the 00-setup notebooks via `databricks jobs submit` against a cluster

### Open questions to resolve before starting

- Are users expected to share one workspace, or each have their own? (Affects experiment-path convention.)
- Will the same Azure subscription pay for the GenAI inference (via OpenAI on Azure / Databricks Foundation Model APIs) or stay on the user's OpenAI account? If the former, the `gpt-5.2` references in the notebooks need to change to a deployment name and the OpenAI client config needs to point at the Azure endpoint.
- Is Databricks Model Serving available in the workspace's region, or do users need a different deployment target (Azure ML, container apps)?

---

## Other open items

### Nice-to-haves

- [ ] Add a real `RAG` example in `02-generative-ai/01-tracing-observability` that uses an actual vector store (FAISS or Chroma for local, Databricks Vector Search for hosted)
- [ ] Add a `pytest`-based regression suite that runs the 00-setup notebooks and asserts the experiments/runs exist
- [ ] Add a GitHub Actions workflow that runs the classical ML track in CI (the GenAI track needs a billable key, so skip in CI by default)

### Documentation polish

- [ ] Embed screenshots from `.verify-screenshots/` (when regenerated, currently gitignored) into the background docs so readers know what to expect in the UI
- [ ] Cross-link from each notebook's `README.md` back to the relevant section in `docs/0X-*-background.md`
