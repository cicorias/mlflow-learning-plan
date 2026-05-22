#!/usr/bin/env bash
# Run both 00-setup notebooks against the configured MLflow tracking server.
set -euo pipefail

export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}
echo "Using MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"

echo ""
echo "--- Classical ML setup check ---"
uv run python notebooks/01-classical-ml/00-setup/setup.py

echo ""
echo "--- GenAI setup check ---"
uv run python notebooks/02-generative-ai/00-setup/setup.py
