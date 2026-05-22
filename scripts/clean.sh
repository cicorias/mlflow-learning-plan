#!/usr/bin/env bash
# Remove local MLflow state and generated artifacts. Preserves .env.
set -euo pipefail

rm -rf .mlflow-local mlruns mlartifacts sample_project .verify-screenshots
echo "Cleaned local MLflow state."
