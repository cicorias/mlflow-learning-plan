
# MLflow: A Four-Week Starter Learning Plan

## Introduction

MLflow is the de facto open-source standard for managing the machine learning lifecycle. Originally created at Databricks in 2018, it has grown into a platform with over 30 million monthly downloads and contributions from hundreds of developers, and it now spans both classic ML and modern GenAI workflows.

The reason it caught on is simple: training models is the easy part. Keeping track of *which* model was trained with *which* data and *which* hyperparameters, then handing it off to production in a reproducible way, is where most ML projects break down. MLflow gives you four cooperating pieces to solve that:

1. **Tracking** — log parameters, metrics, code, and artifacts from every run.
2. **Models** — a standard packaging format that works across frameworks.
3. **Model Registry** — versioning, aliases, and lifecycle management for trained models.
4. **Projects** — a reproducible packaging format for the code that produced a model.

MLflow 3 (released mid-2025, with a steady stream of 3.x point releases since) expanded this scope significantly into GenAI: tracing, prompt management, LLM-as-a-judge evaluation, and human feedback collection. For a classic ML learner, MLflow 3 is still backward-compatible with the 2.x mental model — you just get more on top.

This plan assumes you already know Python and have trained a model before (scikit-learn, PyTorch, XGBoost, anything). It will take you from "never installed it" to "comfortable using it on real projects" in roughly four weeks of part-time effort (~5-8 hours per week).

---

## Week 1 — Foundations & Setup

**Goal:** Understand what MLflow is, install it, and complete one full end-to-end example.

### What to read first

Before writing any code, build a mental model of the four components. The official site has a concise overview:

- [MLflow homepage](https://mlflow.org/) — read the front page; it summarizes the four components in plain language.
- [MLflow Documentation home](https://mlflow.org/docs/latest/) — bookmark this; you'll return constantly. The docs are split into a **classic ML** track and a **GenAI** track. Stay on the classic track for now.
- [What is MLflow?](https://mlflow.org/docs/latest/introduction/) — the canonical conceptual intro.

### Install and launch the UI

MLflow installs as a single pip package and requires Python 3.9+. Step by step:

1. Create a fresh virtual environment so you don't pollute your global Python:
   ```bash
   python -m venv mlflow-env
   source mlflow-env/bin/activate   # Windows: mlflow-env\Scripts\activate
   ```
2. Install MLflow plus the basics you'll use:
   ```bash
   pip install mlflow scikit-learn pandas numpy
   ```
3. Confirm the install:
   ```bash
   mlflow --version
   ```
4. Launch the tracking UI from any directory:
   ```bash
   mlflow ui
   ```
   This starts a local server at <http://localhost:5000>. The first time you open it, you'll see an empty experiment list. That's expected.

### Run the quickstart unmodified

Don't try to be clever yet — run the official tutorial *exactly as written*, then explore.

- [MLflow 5-minute Tracking Quickstart](https://mlflow.org/docs/latest/getting-started/intro-quickstart/) — the recommended starting point. It walks through logging a scikit-learn run and viewing it in the UI.
- [Logging your first MLflow Model](https://mlflow.org/docs/latest/getting-started/logging-first-model/) — natural follow-up; introduces the `log_model` API.

### Week 1 checkpoint

By the end of Week 1, you should be able to:

- Explain the four MLflow components to a colleague in one sentence each.
- Run `mlflow ui` and navigate to the experiments page.
- Execute a training script that logs at least one parameter, one metric, and one model to MLflow.

---

## Week 2 — Experiment Tracking (the workhorse)

**Goal:** Be able to instrument *any* training script with MLflow tracking, and compare runs intelligently.

This is the single most-used feature of MLflow. If you only ever learn one component, learn this one well.

### The core API

There are six functions that cover ~90% of real usage:

| Function | Purpose |
|---|---|
| `mlflow.start_run()` | Open a run context (use as a `with` block). |
| `mlflow.log_param()` / `log_params()` | Record hyperparameters (immutable once logged). |
| `mlflow.log_metric()` / `log_metrics()` | Record numeric metrics; can be logged repeatedly across steps. |
| `mlflow.log_artifact()` / `log_artifacts()` | Save any file (plots, configs, data samples). |
| `mlflow.<flavor>.log_model()` | Save a trained model in MLflow's portable format. |
| `mlflow.set_tag()` | Attach searchable metadata to a run. |

Reference docs:

- [mlflow.tracking API reference](https://mlflow.org/docs/latest/python_api/mlflow.html)
- [Tracking concepts guide](https://mlflow.org/docs/latest/tracking/)

### Autologging — the shortcut

Most popular ML frameworks have a one-line autologger that captures parameters, metrics, and the model automatically:

```python
import mlflow
mlflow.sklearn.autolog()   # or mlflow.pytorch.autolog(), mlflow.xgboost.autolog(), etc.
```

Read [Automatic Logging with MLflow Tracking](https://mlflow.org/docs/latest/tracking/autolog/) for the full list of supported frameworks (scikit-learn, PyTorch, TensorFlow/Keras, XGBoost, LightGBM, Spark, statsmodels, fastai, and others).

Autologging is great for getting started, but it has blind spots — it won't log custom business metrics or domain-specific artifacts. The right workflow in practice is: autolog the boilerplate, then add manual `log_metric`/`log_artifact` calls for the things specific to your problem.

### Hands-on exercise

Take a real project you've worked on (or any Kaggle notebook you've cloned) and retrofit MLflow tracking onto it. At minimum:

1. Wrap the training code in `with mlflow.start_run():`.
2. Log all hyperparameters with `log_params()`.
3. Log training and validation metrics — if you train for multiple epochs, log them with the `step` argument so the UI draws a curve.
4. Save the trained model with `mlflow.<flavor>.log_model()`.
5. Save at least one artifact — a confusion matrix plot, a feature-importance bar chart, or a sample of your test predictions.
6. Run the script three times with different hyperparameters.
7. Open the UI and use the **Compare** button to view the runs side by side. Try the parallel coordinates plot and the scatter plot views.

### Week 2 checkpoint

- You can instrument an arbitrary training script without copy-pasting.
- You understand the difference between parameters (set once) and metrics (logged over time).
- You can navigate the MLflow UI to compare three or more runs.

---

## Week 3 — Models & the Model Registry

**Goal:** Move from "I logged a model" to "I have a versioned, governed model I can serve."

### The MLmodel format and flavors

Every model MLflow saves is written as a directory containing a `MLmodel` YAML file plus the underlying serialized weights. The YAML describes one or more **flavors** — for example, a scikit-learn pipeline might be saved with both an `sklearn` flavor (for native loading) and a `python_function` (pyfunc) flavor (a universal loader interface).

The pyfunc flavor is the key abstraction: anything saved with it can be loaded and served the same way, regardless of the underlying framework. This is what makes MLflow deployment-agnostic.

Read:

- [MLflow Models guide](https://mlflow.org/docs/latest/models/) — the conceptual overview.
- [Built-in model flavors](https://mlflow.org/docs/latest/models/#built-in-model-flavors) — the full list of supported frameworks.
- [Custom Python models with pyfunc](https://mlflow.org/docs/latest/models/#custom-python-models) — when none of the built-in flavors fit, you write a pyfunc class.

### Loading and serving a model locally

Once a model is logged, you can load it from any Python environment:

```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("runs:/<RUN_ID>/model")
preds = model.predict(X_test)
```

Or you can serve it as a REST endpoint without writing any web framework code:

```bash
mlflow models serve -m runs:/<RUN_ID>/model -p 5000 --env-manager local
```

Then hit it with curl:

```bash
curl -X POST -H "Content-Type:application/json" \
  --data '{"inputs": [[5.1, 3.5, 1.4, 0.2]]}' \
  http://127.0.0.1:5000/invocations
```

Reference: [Deploy MLflow Model as a Local Inference Server](https://mlflow.org/docs/latest/deployment/deploy-model-locally/).

### The Model Registry

The Registry sits on top of the Tracking server and gives you:

- **Named, versioned models** — register "fraud-detector" once, then keep adding versions over time.
- **Aliases** — pointers like `@champion`, `@challenger`, `@production` that can be reassigned to any version. This is the modern way to manage lifecycle; the older string-based "Stages" (Staging/Production/Archived) are deprecated.
- **Tags** — arbitrary key/value labels per version, useful for things like `validated_by=jane` or `dataset_version=v3`.

Walk through:

- [Model Registry guide](https://mlflow.org/docs/latest/model-registry/) — start here.
- [Deploy aliases and tags tutorial](https://mlflow.org/docs/latest/model-registry/#deploy-and-organize-models-with-aliases-and-tags) — the modern (post-2.9) workflow.

### Hands-on exercise

1. Take a model you logged in Week 2 and register it under a name like `iris-classifier`.
2. Train a second version with different hyperparameters and register that too.
3. Assign the alias `@champion` to whichever version performed best.
4. Load the model in a fresh script using the alias URI: `models:/iris-classifier@champion`.
5. Serve that model with `mlflow models serve` and send it a prediction request.

### Week 3 checkpoint

- You can save, load, and serve a model in three different ways.
- You understand why pyfunc is the universal interface.
- You can register a model, version it, and manage it with aliases.

---

## Week 4 — Projects, Reproducibility & What's Next

**Goal:** Make your work reproducible by others, and understand where MLflow fits in a real production stack.

### MLflow Projects

A Project is a directory with an `MLproject` YAML file that declares dependencies (conda env or Docker), entry points, and parameters. Anyone can then run your training pipeline with one command:

```bash
mlflow run https://github.com/your-org/your-repo.git -P alpha=0.5
```

This is the lightest possible answer to "can you reproduce my results?"

Read: [MLflow Projects guide](https://mlflow.org/docs/latest/projects/).

### Tracking server architectures

So far you've been using the default file-based tracking store, where everything lands in a local `mlruns/` directory. That's fine for solo work but breaks down on a team. The real architecture has two pieces:

- **Backend store** — a database (SQLite for dev, PostgreSQL or MySQL for production) that holds runs, params, metrics, and registry metadata.
- **Artifact store** — object storage (local filesystem, S3, GCS, Azure Blob, etc.) that holds the actual model files, plots, and data.

A reasonable stepping-stone setup using SQLite + local artifacts:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000
```

Point your training scripts at it with:

```python
mlflow.set_tracking_uri("http://localhost:5000")
```

Reference: [Tracking server setup](https://mlflow.org/docs/latest/tracking/server/).

### What's new (and worth knowing about)

MLflow 3 added a parallel set of features for GenAI applications. You don't need these for classic ML, but knowing they exist saves you from picking a different tool later if your work shifts toward LLMs:

- **Tracing** — automatic instrumentation for 20+ LLM frameworks (OpenAI, LangChain, LlamaIndex, Anthropic, and others) that captures every prompt, response, latency, and token count.
- **Prompt Registry** — versioning and optimization for prompts, treated as first-class artifacts.
- **LLM-as-a-judge evaluation** — `mlflow.genai.evaluate()` runs built-in or custom judges for correctness, relevance, and safety.
- **Human feedback collection** — a review UI for domain experts to label model outputs.

Skim the [MLflow 3 GenAI overview](https://mlflow.org/docs/latest/genai/) to know what's there. Come back to it when you need it.

### Hands-on exercise

1. Convert one of your training scripts from Week 2 into a proper MLflow Project with an `MLproject` file and a `conda.yaml`. Push it to GitHub.
2. From a different machine (or a fresh clone), run it with `mlflow run <your-repo-url>` and confirm it produces the same results.
3. Stand up a local tracking server backed by SQLite. Configure your project to log to it instead of the default file store.

### Week 4 checkpoint

- You can package a training pipeline so a colleague can rerun it with one command.
- You understand the difference between the backend store and the artifact store.
- You know what's in MLflow 3's GenAI layer at a high level.

---

## Recommended Resources

### Official

- [MLflow Documentation](https://mlflow.org/docs/latest/) — start here for any new topic.
- [MLflow GitHub repository](https://github.com/mlflow/mlflow) — the `examples/` folder is genuinely useful, not just toy code.
- [MLflow release notes](https://mlflow.org/releases/) — quick way to see what's changed.
- [MLflow API reference](https://mlflow.org/docs/latest/python_api/) — the authoritative function-level docs.

### Tutorials and courses

- [Databricks Academy free MLflow courses](https://www.databricks.com/learn/training/catalog) — Databricks created MLflow and offers free training; the introductory MLflow course is well-paced.
- [MLflow tutorials index](https://mlflow.org/docs/latest/getting-started/) — official walkthroughs for the major frameworks.

### Community

- [MLflow on Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow) — most common-pitfall questions are already answered here.
- [MLflow GitHub Discussions](https://github.com/mlflow/mlflow/discussions) — for non-bug questions; the maintainers do respond.

### Books

- *Machine Learning Engineering with MLflow* by Natu Lauchande (Packt, 2021) — older but still the most thorough end-to-end book. Note that the Model Registry chapter pre-dates aliases.
- *Practical MLOps* by Noah Gift and Alfredo Deza (O'Reilly, 2021) — broader MLOps context with MLflow as one chapter; useful for understanding where it fits.

---

## Practical Tips

A few hard-won lessons from people who've used MLflow on real projects:

- **Anchor on Tracking.** Roughly 70% of real MLflow usage is the tracking API. Get fluent there before worrying about the Registry or Projects.
- **Work on a real project by Week 2.** Toy datasets don't surface the messy parts — like what to do when your training data is too big to log as an artifact, or how to track preprocessing alongside the model.
- **Pick your tracking server architecture early.** If you'll be working on a team, the question of self-hosted vs. managed (Databricks, Azure ML, AWS SageMaker with managed MLflow) affects your setup. Solo work? File-based is fine for months.
- **Don't over-tag, don't under-tag.** Tags are searchable and free; use them for things you'll want to filter on later (dataset version, git branch, owner). Skip the philosophical novel.
- **Reproducibility is a habit, not a feature.** MLflow makes it possible to reproduce a run; it doesn't make it automatic. Make a rule: every run logs its git commit hash (`mlflow.set_tag("git_commit", ...)`) and its data version.
- **Read the release notes when you upgrade.** MLflow ships frequently, and 2.x → 3.x had a few breaking changes (notably around `log_model`'s `artifact_path` parameter being renamed to `name`).

---

## Next Steps After This Plan

Once the four weeks are behind you, the natural directions are:

- **Production deployment** — explore [MLflow deployment to Kubernetes, SageMaker, or Azure ML](https://mlflow.org/docs/latest/deployment/).
- **CI/CD integration** — wire the Model Registry into a webhook-driven pipeline so promoting a model to `@production` triggers a deployment.
- **Custom flavors and pyfunc models** — when your model isn't a standard framework output (e.g., an ensemble or a hybrid pipeline with preprocessing).
- **GenAI workflows** — if your work is heading toward LLMs, the MLflow 3 GenAI track is the natural extension of what you've learned.

Happy tracking.