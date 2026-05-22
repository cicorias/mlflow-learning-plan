#!/usr/bin/env bash
# Start the local MLflow tracking server with SQLite backend + filesystem artifacts.
# Runs in the foreground; Ctrl-C to stop.
set -euo pipefail

mkdir -p .mlflow-local

echo "MLflow UI will be at http://127.0.0.1:5000 — Ctrl-C to stop"

exec uv run mlflow server \
  --backend-store-uri sqlite:///.mlflow-local/mlflow.db \
  --default-artifact-root ./.mlflow-local/mlartifacts \
  --host 127.0.0.1 --port 5000
