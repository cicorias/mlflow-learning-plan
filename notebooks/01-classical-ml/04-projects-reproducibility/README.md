# 04 — Projects, Reproducibility & What's Next (Week 4)

Package your pipeline so anyone can reproduce it with one command, and understand production tracking server options.

## Goal

Create an MLflow Project and understand backend/artifact store architectures.

## What this notebook covers

- The `MLproject` YAML format: entry points, parameters, environment declaration
- `mlflow run` from a local directory or remote Git URL
- The two-layer tracking server architecture: backend store (database) + artifact store (object storage)
- Standing up a local SQLite-backed server vs. Databricks Managed MLflow
- What MLflow 3 adds for GenAI (tracing, Prompt Registry, evaluation) — conceptual overview

## Checkpoint

After running this notebook you should be able to:
- [ ] Package a training pipeline so a colleague can rerun it with `mlflow run`
- [ ] Explain the difference between the backend store and the artifact store
- [ ] Describe what's in MLflow 3's GenAI layer at a high level

## Local tracking server (for reference)

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000
```

## References

- [MLflow Projects guide](https://mlflow.org/docs/latest/projects/)
- [Tracking server setup](https://mlflow.org/docs/latest/tracking/server/)
- [MLflow 3 GenAI overview](https://mlflow.org/docs/latest/genai/)
