# Classical ML with MLflow — Background

> Concepts, references, and a topic-to-notebook map for the classical ML track.

This is the **reading companion** to the notebooks in [`notebooks/01-classical-ml/`](../notebooks/01-classical-ml/). It is not a tutorial; it's a structured set of explanations and links so you have the why behind each notebook before you run it.

For the condensed week-by-week syllabus, see [`docs/start/mlflow-learning-plan-01.md`](start/mlflow-learning-plan-01.md).

---

## Why MLflow exists

Training a model is the easy part. What breaks projects is the bookkeeping:

- Which dataset and which hyperparameters produced *that* model checkpoint last Tuesday?
- Which version of the preprocessing code is in production right now?
- If we retrain, do we know we'll get the same result?

MLflow's answer is **four cooperating components** — each addressing one piece of the bookkeeping problem:

| Component | Solves | Covered in |
|---|---|---|
| Tracking | Per-run logging of params, metrics, artifacts, code | [`01-foundations-setup`](../notebooks/01-classical-ml/01-foundations-setup/), [`02-experiment-tracking`](../notebooks/01-classical-ml/02-experiment-tracking/) |
| Models | A portable packaging format that loads the same way regardless of framework | [`03-models-and-registry`](../notebooks/01-classical-ml/03-models-and-registry/) |
| Model Registry | Versioning, aliases, and lifecycle management for trained models | [`03-models-and-registry`](../notebooks/01-classical-ml/03-models-and-registry/) |
| Projects | A reproducible packaging format for the *code* that produced the model | [`04-projects-reproducibility`](../notebooks/01-classical-ml/04-projects-reproducibility/) |

**Primary references**

- [MLflow Documentation home](https://mlflow.org/docs/latest/) — the canonical reference
- [What is MLflow?](https://mlflow.org/docs/latest/introduction/) — conceptual intro
- [MLflow on GitHub](https://github.com/mlflow/mlflow) — source + the `examples/` directory

---

## Topic 1 — Tracking: log everything about a run

**Concepts**

A **run** is one execution of training code. Inside a run you log:
- **Parameters** (`mlflow.log_param/s`) — immutable after the run closes. Hyperparameters belong here.
- **Metrics** (`mlflow.log_metric/s`) — can be logged repeatedly with a `step` argument to draw curves.
- **Artifacts** (`mlflow.log_artifact/s`) — any file: plots, configs, sample predictions.
- **Tags** (`mlflow.set_tag`) — searchable string labels. Use them for dimensions you'll filter on: git commit, dataset version, owner.
- **Models** (`mlflow.<flavor>.log_model`) — serialized model in MLflow's portable format.

Runs are grouped into **experiments** — typically one experiment per task ("fraud-detection-v2", "iris-baseline"). The MLflow UI shows runs by experiment and lets you select multiple to compare.

**Autologging** is the one-line shortcut. `mlflow.sklearn.autolog()` (or `.pytorch.autolog()`, `.xgboost.autolog()`, etc.) captures params, metrics, and the model automatically. Autologging covers boilerplate; add manual `log_metric` / `log_artifact` calls for things specific to your problem.

**Where in this repo**

- [`01-foundations-setup/notebook.py`](../notebooks/01-classical-ml/01-foundations-setup/notebook.py) — first end-to-end run; log params, metrics, model; load back by `runs:/` URI
- [`02-experiment-tracking/notebook.py`](../notebooks/01-classical-ml/02-experiment-tracking/notebook.py) — three hyperparameter sweeps, autologging vs manual, step-logged CV metrics, plot artifacts, the **Compare** view

**External references**

- [Tracking concepts guide](https://mlflow.org/docs/latest/tracking/) — the right place to start
- [`mlflow.tracking` API reference](https://mlflow.org/docs/latest/python_api/mlflow.html)
- [Automatic Logging with MLflow Tracking](https://mlflow.org/docs/latest/tracking/autolog/) — every supported framework
- [5-minute Tracking Quickstart](https://mlflow.org/docs/latest/getting-started/intro-quickstart/) — the official walkthrough

**Practical tips**

- **Anchor on Tracking.** Roughly 70% of real MLflow usage is the tracking API. Get fluent here before worrying about the Registry or Projects.
- **Tag for filtering, not narrative.** Don't write a paragraph in a tag value. Use short strings you'll want to filter on (`owner=jane`, `git_commit=abc1234`, `dataset_version=v3`).
- **Difference between params and metrics matters.** Params are set-once and immutable inside a run; metrics can be logged many times across `step`.

---

## Topic 2 — Models: portable packaging

**Concepts**

Every model MLflow saves is a directory containing an `MLmodel` YAML file plus serialized weights. The YAML declares one or more **flavors** — for example, a scikit-learn pipeline is typically saved with both an `sklearn` flavor (native loading) and a `python_function` (pyfunc) flavor (universal loading).

The **pyfunc** flavor is the key abstraction. Any model saved with it loads and predicts the same way regardless of framework:

```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("runs:/<RUN_ID>/model")
preds = model.predict(X_test)
```

This is what makes MLflow deployment-agnostic. The serving stack doesn't need to know whether the underlying model is sklearn, XGBoost, or a custom pyfunc class.

**Where in this repo**

- [`01-foundations-setup/notebook.py`](../notebooks/01-classical-ml/01-foundations-setup/notebook.py) — `mlflow.sklearn.log_model()` and `mlflow.pyfunc.load_model()`
- [`03-models-and-registry/notebook.py`](../notebooks/01-classical-ml/03-models-and-registry/notebook.py) — both flavors, REST serving via `mlflow models serve`

**External references**

- [MLflow Models guide](https://mlflow.org/docs/latest/models/) — the conceptual overview
- [Built-in model flavors](https://mlflow.org/docs/latest/models/#built-in-model-flavors)
- [Custom Python models with pyfunc](https://mlflow.org/docs/latest/models/#custom-python-models) — for when no built-in flavor fits
- [Deploy MLflow Model as a Local Inference Server](https://mlflow.org/docs/latest/deployment/deploy-model-locally/)

**Practical tips**

- **Default to pyfunc for serving code.** Even if you saved an sklearn flavor, write your client to load via pyfunc — the same code works if you later swap the model for a custom pipeline.
- **Inspect `MLmodel`.** Always open the YAML at least once to see what flavors got saved. If your custom preprocessing isn't captured, autologging isn't going to magically save it.

---

## Topic 3 — Model Registry: versions and aliases

**Concepts**

The Registry sits on top of the Tracking server and gives you:

- **Named, versioned models** — register "fraud-detector" once; each new version increments the integer.
- **Aliases** — pointers like `@champion`, `@challenger`, `@production` that can be reassigned to any version. Application code references the alias URI (`models:/fraud-detector@production`) and never has to change when you promote a new version.
- **Tags** — arbitrary key/value labels per version (`validated_by=jane`, `dataset_version=v3`).

> **String "Stages" (Staging, Production, Archived) are deprecated** in MLflow 3. Use aliases instead. They're more flexible (any name you want) and let you promote/demote without losing history.

**Where in this repo**

- [`03-models-and-registry/notebook.py`](../notebooks/01-classical-ml/03-models-and-registry/notebook.py) — register two model versions, assign `@champion` / `@challenger`, load by alias, serve

**External references**

- [Model Registry guide](https://mlflow.org/docs/latest/model-registry/)
- [Deploy aliases and tags tutorial](https://mlflow.org/docs/latest/model-registry/#deploy-and-organize-models-with-aliases-and-tags)
- [MlflowClient API reference](https://mlflow.org/docs/latest/python_api/mlflow.client.html)

**Practical tips**

- **Application code should reference an alias, never a version number.** That's the whole point — you change which version `@production` points to without touching the app.
- **Tag the validation status.** When QA signs off on a version, tag it (`tag.qa_approved=true`). Aliases say what's deployed; tags say what's *true* about a version.

---

## Topic 4 — Projects: reproducible execution

**Concepts**

An MLflow Project is a directory with an `MLproject` YAML file that declares:

- **Entry points** — named commands with their parameter specs
- **Environment** — a conda env, pip requirements file, or Docker image

Anyone with `mlflow run` can execute your pipeline:

```bash
mlflow run https://github.com/your-org/your-repo.git -P alpha=0.5
```

This is the lightest possible answer to "can you reproduce my results?" — it pins both the code (via Git URL or local path) and the dependencies (via the env declaration).

**Where in this repo**

- [`04-projects-reproducibility/notebook.py`](../notebooks/01-classical-ml/04-projects-reproducibility/notebook.py) — generates a complete MLproject + python_env.yaml + train.py, then runs it with `mlflow run`

**External references**

- [MLflow Projects guide](https://mlflow.org/docs/latest/projects/)
- [MLproject file reference](https://mlflow.org/docs/latest/projects/#mlproject-file)

**Practical tips**

- **Use `--env-manager=local`** when your environment is already provisioned (e.g., after `uv sync`). It skips conda creation and reuses the active environment, which is much faster.
- **Pass parameters explicitly.** `mlflow run` won't fall back to defaults silently — always pass `-P key=value` for every parameter you care about, even if `MLproject` declares a default.

---

## Topic 5 — Tracking server architecture

**Concepts**

The default tracking store is the filesystem: everything goes to `./mlruns/`. That's fine for solo work but breaks down on a team. The production architecture has two pieces:

- **Backend store** — a database (SQLite for dev, PostgreSQL or MySQL for production) that holds runs, params, metrics, and registry metadata.
- **Artifact store** — object storage (local filesystem, S3, GCS, Azure Blob) that holds the actual model files, plots, and data.

This repository uses a local SQLite-backed server during development:

```bash
mlflow server \
  --backend-store-uri sqlite:///.mlflow-local/mlflow.db \
  --default-artifact-root ./.mlflow-local/mlartifacts \
  --host 127.0.0.1 --port 5000
```

In production, the natural targets on Azure are:

- **Databricks Managed MLflow** — workspace-hosted, no server to run; tracking URI is `databricks` and reads `~/.databrickscfg`
- **Azure ML Managed MLflow** — integrated with the Azure ML workspace

**Where in this repo**

- [`04-projects-reproducibility/notebook.py`](../notebooks/01-classical-ml/04-projects-reproducibility/notebook.py) — explains both store types and the recommended SQLite + local filesystem dev setup
- [`CLAUDE.md`](../CLAUDE.md) → **MLflow Tracking — Dual Mode** — the env-var-driven pattern that makes the same code work locally and on Databricks

**External references**

- [Tracking server setup](https://mlflow.org/docs/latest/tracking/server/)
- [Backend stores](https://mlflow.org/docs/latest/tracking/backend-stores/)
- [Artifact stores](https://mlflow.org/docs/latest/tracking/artifact-stores/)
- [Databricks Managed MLflow](https://learn.microsoft.com/en-us/azure/databricks/mlflow/)
- [Azure ML MLflow](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow)

---

## Practical principles (cross-cutting)

These show up across all four topics. Internalizing them turns MLflow from "the tool that logs my runs" into a real reproducibility practice.

- **Reproducibility is a habit, not a feature.** MLflow makes reproduction *possible*; it doesn't make it *automatic*. Make a rule: every run logs its git commit hash (`mlflow.set_tag("git_commit", ...)`) and its data version.
- **Don't over-tag, don't under-tag.** Tags are searchable and free; use them for things you'll want to filter on later. Skip the philosophical novel in a tag value.
- **Pick your tracking architecture early.** If you're working solo, file-based is fine for months. If you're on a team, the question of self-hosted vs. managed (Databricks, Azure ML, AWS SageMaker) needs an answer in week one.
- **Read the release notes.** MLflow ships frequently. The 2.x → 3.x bump had breaking changes (notably `log_model`'s `artifact_path` parameter renamed to `name`).

---

## Going further

After working through this track, the natural directions are:

- **Production deployment** — [Deployment overview](https://mlflow.org/docs/latest/deployment/) covers Kubernetes, SageMaker, Azure ML, and Databricks Model Serving
- **CI/CD integration** — wire the Model Registry into a webhook-driven pipeline so promoting `@production` triggers a deployment
- **Custom flavors and pyfunc** — for ensembles or hybrid pipelines that don't fit a single built-in flavor
- **The GenAI track** — once your work shifts toward LLMs, see [`02-generative-ai-background.md`](02-generative-ai-background.md)

### Recommended supplementary reading

- [Databricks Academy free MLflow courses](https://www.databricks.com/learn/training/catalog) — Databricks created MLflow and the intro course is well-paced
- *Machine Learning Engineering with MLflow* by Natu Lauchande (Packt, 2021) — older but the most thorough end-to-end book; note the Registry chapter pre-dates aliases
- *Practical MLOps* by Noah Gift & Alfredo Deza (O'Reilly, 2021) — broader MLOps context with MLflow as one chapter
- [MLflow on Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow) — most common-pitfall questions are already answered
- [MLflow GitHub Discussions](https://github.com/mlflow/mlflow/discussions) — for non-bug questions; maintainers do respond
