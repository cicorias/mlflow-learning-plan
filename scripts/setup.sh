#!/usr/bin/env bash
# Print next-step guidance after `mise run setup` completes.
# Actual install work is done by the `sync` and `env` task dependencies.
set -euo pipefail

echo ""
echo "Setup complete. Next steps:"
echo "  1. Edit .env and set OPENAI_API_KEY (needed for the GenAI track)"
echo "  2. Authenticate Databricks (optional):  az login  &&  databricks auth login --host <workspace-url>"
echo "  3. Start the local MLflow server:       mise run mlflow-server"
echo "  4. In another terminal, smoke-test:     mise run verify"
