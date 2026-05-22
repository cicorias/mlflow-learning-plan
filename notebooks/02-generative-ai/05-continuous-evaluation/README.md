# 05 — Continuous Evaluation in Production (Week 5 — Stretch)

Make evaluation a continuous, automated process — not a one-time pre-ship check.

## Goal

Wire evaluation into a loop: production traces → automatic scoring → feedback into dataset → regression-gated deployments.

## What this notebook covers

- Automatic evaluation: judges that run on traces as they're logged
- Configuring a sample rate and routing judge results to a dashboard
- The production improvement loop: thumbs-down → tagged trace → expert review → eval dataset update
- Regression-gating: evaluate a candidate against the baseline before promoting
- Structuring a CI script that runs `mlflow.genai.evaluate()` and compares to a stored baseline run

## Checkpoint

After running this notebook you should be able to:
- [ ] Describe the loop from production feedback → eval dataset → regression-gated deployment
- [ ] Configure automatic evaluation on a sample of production traces
- [ ] Write a CI step that blocks a promotion if any judge regresses beyond a threshold

## Note on Databricks-managed features

`Safety` and `RetrievalRelevance` judges and some annotation UIs launched in Databricks Managed MLflow before reaching open-source. Verify current availability in the docs if you are self-hosting.

## References

- [Automatic Evaluation documentation](https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/)
- [MLflow GenAI Documentation](https://mlflow.org/docs/latest/genai/)
