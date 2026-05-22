# 02 — Experiment Tracking (Week 2)

Instrument any training script and compare runs intelligently — the core MLflow skill.

## Goal

Be able to add MLflow tracking to an arbitrary training script and use the UI to compare multiple runs.

## What this notebook covers

- The six core tracking functions (`start_run`, `log_param/s`, `log_metric/s`, `log_artifact/s`, `log_model`, `set_tag`)
- `mlflow.sklearn.autolog()` and when autolog falls short
- Logging metrics across training steps to draw curves in the UI
- Saving plot artifacts (confusion matrix, feature importance)
- Running three hyperparameter variations and comparing them with the UI Compare view

## Checkpoint

After running this notebook you should be able to:
- [ ] Instrument any training script without copy-pasting
- [ ] Explain the difference between parameters (set once) and metrics (logged over time)
- [ ] Navigate the MLflow UI to compare three or more runs using the parallel coordinates and scatter plot views

## References

- [mlflow.tracking API reference](https://mlflow.org/docs/latest/python_api/mlflow.html)
- [Tracking concepts guide](https://mlflow.org/docs/latest/tracking/)
- [Automatic Logging with MLflow Tracking](https://mlflow.org/docs/latest/tracking/autolog/)
