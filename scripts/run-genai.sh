#!/usr/bin/env bash
# Execute every lesson in the GenAI track in order. Requires OPENAI_API_KEY in .env.
set -euo pipefail

export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}

for nb in notebooks/02-generative-ai/*/setup.py notebooks/02-generative-ai/*/notebook.py; do
  [ -f "$nb" ] || continue
  echo ""
  echo "=== Running: $nb ==="
  uv run python "$nb"
done
