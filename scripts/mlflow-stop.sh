#!/usr/bin/env bash
# Kill any local MLflow server started by `mise run mlflow-server`.
set -euo pipefail

pids=$(pgrep -f "mlflow server.*--port 5000" || true)
if [ -n "$pids" ]; then
  echo "Stopping MLflow server PIDs: $pids"
  # shellcheck disable=SC2086
  kill $pids
else
  echo "No local MLflow server is running."
fi
