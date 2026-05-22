#!/usr/bin/env bash
# Execute every lesson in the classical ML track in order.
set -euo pipefail

export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}

for nb in notebooks/01-classical-ml/*/setup.py notebooks/01-classical-ml/*/notebook.py; do
  [ -f "$nb" ] || continue
  echo ""
  echo "=== Running: $nb ==="
  uv run python "$nb"
done
