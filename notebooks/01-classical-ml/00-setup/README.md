# 00 — Setup (Classic ML Track)

Verify your environment before starting the four-week classic ML track.

## What this lesson covers

- Confirm Python, MLflow, and scikit-learn versions
- Set the MLflow tracking URI for local or Databricks execution
- Launch the MLflow UI (local mode only)
- Confirm you can write a run to the tracking server

## Run it

```bash
# Local
uv run python setup.py

# Databricks: import setup.py into a Repo and run as a notebook
```

## Before you start

| Context | What to do |
|---|---|
| Local | `export MLFLOW_TRACKING_URI=http://localhost:5000` then `uv run mlflow ui` in a separate terminal |
| Databricks | `az login` then `databricks auth login --host <workspace-url>`; tracking URI is set automatically |

## Expected output

The script prints versions and writes one test run. Open the MLflow UI and confirm the `01-classical-ml-setup-check` experiment appears with one run named `env-check`.
