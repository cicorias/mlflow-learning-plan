# 03 — Models & the Model Registry (Week 3)

Move from "I logged a model" to "I have a versioned, governed model I can serve."

## Goal

Register a model, manage it with aliases, load it, and serve it as a REST endpoint.

## What this notebook covers

- The `MLmodel` file format and what "flavors" mean
- Why pyfunc is the universal loading interface
- `mlflow.sklearn.log_model()` and `mlflow.pyfunc.load_model()`
- Registering a model: `MlflowClient.create_registered_model()` and `log_model(registered_model_name=...)`
- Aliases (`@champion`, `@challenger`) vs deprecated string Stages
- Loading by alias URI: `models:/iris-classifier@champion`
- Serving with `mlflow models serve` (local only)

## Checkpoint

After running this notebook you should be able to:
- [ ] Save, load, and serve a model in three different ways
- [ ] Explain why pyfunc is the universal interface
- [ ] Register a model, add a second version, and assign the `@champion` alias to the best one

## References

- [MLflow Models guide](https://mlflow.org/docs/latest/models/)
- [Model Registry guide](https://mlflow.org/docs/latest/model-registry/)
- [Deploy MLflow Model as a Local Inference Server](https://mlflow.org/docs/latest/deployment/deploy-model-locally/)
