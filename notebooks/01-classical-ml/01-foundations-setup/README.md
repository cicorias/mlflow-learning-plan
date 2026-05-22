# 01 — Foundations & Setup (Week 1)

First end-to-end MLflow example: log a real scikit-learn run and navigate the UI.

## Goal

Understand the four MLflow components and complete one full tracking run.

## What this notebook covers

- The four MLflow components: Tracking, Models, Model Registry, Projects
- `mlflow.start_run()` as a context manager
- `log_param()`, `log_metric()`, `log_artifact()`, `log_model()`
- Navigating the Experiments page in the UI

## Checkpoint

After running this notebook you should be able to:
- [ ] Explain the four components to a colleague in one sentence each
- [ ] Find your run in the MLflow UI and inspect its params, metrics, and artifacts
- [ ] Load the logged model back from a `runs:/` URI

## References

- [MLflow 5-minute Tracking Quickstart](https://mlflow.org/docs/latest/getting-started/intro-quickstart/)
- [Logging your first MLflow Model](https://mlflow.org/docs/latest/getting-started/logging-first-model/)
- [What is MLflow?](https://mlflow.org/docs/latest/introduction/)
