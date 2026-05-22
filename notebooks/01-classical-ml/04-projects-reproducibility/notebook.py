# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 — Projects, Reproducibility & What's Next
# MAGIC
# MAGIC **Goal:** Package a training pipeline so anyone can reproduce it with one command,
# MAGIC and understand the production tracking server architecture.
# MAGIC
# MAGIC ## MLflow Projects
# MAGIC
# MAGIC A Project is a directory with an `MLproject` YAML that declares:
# MAGIC - **entry_points** — named commands with their parameter specs
# MAGIC - **dependencies** — a conda env, pip requirements file, or Docker image
# MAGIC
# MAGIC Anyone with `mlflow run` can rerun your pipeline:
# MAGIC ```bash
# MAGIC mlflow run https://github.com/your-org/your-repo.git -P alpha=0.5
# MAGIC ```

# COMMAND ----------

import os

from dotenv import load_dotenv
load_dotenv()  # auto-load .env if present
import pathlib
import textwrap
import mlflow

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("01-classical-ml-04-projects")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write a minimal MLproject structure
# MAGIC
# MAGIC We'll create the files in a temp directory to demonstrate the format.
# MAGIC In practice this lives at the root of your Git repository.

# COMMAND ----------

project_dir = pathlib.Path("./sample_project")
project_dir.mkdir(exist_ok=True)

# MLproject — entry point and parameter spec
(project_dir / "MLproject").write_text(textwrap.dedent("""\
    name: iris-training

    python_env: python_env.yaml

    entry_points:
      main:
        parameters:
          C:
            type: float
            default: 1.0
          max_iter:
            type: int
            default: 200
        command: "python train.py --C {C} --max_iter {max_iter}"
"""))

# python_env.yaml — reproducible Python environment declaration
(project_dir / "python_env.yaml").write_text(textwrap.dedent("""\
    python: "3.14"
    build_dependencies:
      - pip
    dependencies:
      - mlflow>=3.0
      - scikit-learn>=1.0
      - pandas>=2.0
"""))

# train.py — the actual training script
(project_dir / "train.py").write_text(textwrap.dedent("""\
    import argparse, subprocess, mlflow, mlflow.sklearn
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=200)
    args = parser.parse_args()

    iris = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        mlflow.log_params({"C": args.C, "max_iter": args.max_iter})
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], text=True
            ).strip()
            mlflow.set_tag("git_commit", git_hash)
        except Exception:
            pass

        model = LogisticRegression(C=args.C, max_iter=args.max_iter, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, name="model")
        print(f"accuracy={acc:.4f}")
"""))

print("Project files written:")
for f in sorted(project_dir.iterdir()):
    print(f"  {f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the project locally
# MAGIC
# MAGIC `mlflow run` respects the `MLproject` entry point and passes parameters through.
# MAGIC The `--env-manager=local` flag skips conda/virtualenv creation and uses the current
# MAGIC environment — convenient in a repo that already has `uv sync` run.

# COMMAND ----------

import subprocess

result = subprocess.run(
    [
        "mlflow", "run", str(project_dir),
        "--env-manager=local",
        "-P", "C=0.5",
        "-P", "max_iter=300",
    ],
    capture_output=True,
    text=True,
)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tracking server architectures
# MAGIC
# MAGIC | Setup | Backend store | Artifact store | Good for |
# MAGIC |---|---|---|---|
# MAGIC | Default (no server) | `./mlruns/` filesystem | same | Solo dev, offline |
# MAGIC | Local SQLite server | `sqlite:///mlflow.db` | `./mlruns/` | Small team, local |
# MAGIC | **Databricks Managed** | Workspace DB | DBFS / cloud storage | Teams, production |
# MAGIC
# MAGIC ### Local SQLite server (reference — run in terminal, not this notebook)
# MAGIC
# MAGIC ```bash
# MAGIC mlflow server \
# MAGIC   --backend-store-uri sqlite:///mlflow.db \
# MAGIC   --default-artifact-root ./mlruns \
# MAGIC   --host 0.0.0.0 --port 5000
# MAGIC ```
# MAGIC
# MAGIC Then point scripts at it:
# MAGIC ```python
# MAGIC mlflow.set_tracking_uri("http://localhost:5000")
# MAGIC ```
# MAGIC
# MAGIC On Databricks, `MLFLOW_TRACKING_URI` is pre-set to the workspace server — no action needed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What's next: MLflow 3 GenAI layer
# MAGIC
# MAGIC MLflow 3 added parallel capabilities for LLM applications that you don't need for
# MAGIC classic ML but will want when your work shifts toward LLMs:
# MAGIC
# MAGIC | Feature | What it does |
# MAGIC |---|---|
# MAGIC | **Tracing** | Auto-instrument 20+ LLM frameworks; every prompt/response captured |
# MAGIC | **Prompt Registry** | Version prompts the same way you version models |
# MAGIC | **`mlflow.genai.evaluate()`** | LLM-as-a-judge evaluation with built-in and custom scorers |
# MAGIC | **Human feedback** | Collect and record expert labels on traces |
# MAGIC
# MAGIC When you're ready: start [02-generative-ai → 00-setup](../../02-generative-ai/00-setup/setup.py)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checkpoint
# MAGIC
# MAGIC - [ ] You wrote an `MLproject` file with entry points and parameters
# MAGIC - [ ] `mlflow run` executed the project and logged a run
# MAGIC - [ ] You can explain backend store vs artifact store
# MAGIC - [ ] You know what the MLflow 3 GenAI features are at a conceptual level
