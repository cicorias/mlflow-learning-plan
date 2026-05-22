# 02 — Evaluation Datasets and Ground Truth (Week 2)

Build, version, and use evaluation datasets — the foundation of any serious evaluation effort.

## Goal

Produce a versioned evaluation dataset of at least 20 examples with explicit ground-truth expectations.

## What this notebook covers

- Why evaluation datasets are your "test database"
- Dataset row structure: `inputs` + `expectations` (expected facts, guidelines)
- Hand-curating 20 examples covering main use cases and known failure modes
- Sourcing ground truth from production traces and synthetic generation
- `mlflow.log_table()` to version a dataset as a first-class artifact
- Tagging runs with `dataset_version` for reproducible comparisons

## Checkpoint

After running this notebook you should be able to:
- [ ] Create a structured eval dataset with inputs and expected facts
- [ ] Log and version the dataset as an MLflow artifact
- [ ] Explain why versioning the dataset is non-negotiable for trustworthy comparisons

## Deliverable

A versioned evaluation dataset of at least 20 examples logged to MLflow, tagged with `dataset_version=v1`.

## References

- [Evaluation Datasets documentation](https://mlflow.org/docs/latest/genai/eval-monitor/)
- [MLflow continuous improvement workflow](https://mlflow.org/docs/latest/genai/)
