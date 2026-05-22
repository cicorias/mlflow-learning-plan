# Generative AI with MLflow — Background

> Concepts, references, and a topic-to-notebook map for the GenAI track.

This is the **reading companion** to the notebooks in [`notebooks/02-generative-ai/`](../notebooks/02-generative-ai/). It is not a tutorial; it's a structured set of explanations and links so you have the why behind each notebook before you run it.

For the condensed week-by-week syllabus, see [`docs/start/mlflow-learning-plan-02.md`](start/mlflow-learning-plan-02.md).

---

## Why this matters

Building a GenAI application is easy. Knowing whether it actually works is hard. Unlike classical ML, you can't compute one accuracy number and call it a day:

- Outputs are free-form text, not class labels.
- Failure modes are subtle (hallucinations, off-topic responses, ignored instructions, retrieval misses).
- Ground truth often doesn't exist until you create it.
- Whatever evaluation system you build today has to keep working as you swap models, change prompts, and deploy to production.

MLflow 3 added a parallel set of capabilities for this work. The six pieces that matter:

| Capability | Solves | Covered in |
|---|---|---|
| Tracing | Capture every step of every request so you can debug what your app actually did | [`01-tracing-observability`](../notebooks/02-generative-ai/01-tracing-observability/) |
| Evaluation datasets | Build the "test database" that anchors quality work | [`02-evaluation-datasets`](../notebooks/02-generative-ai/02-evaluation-datasets/) |
| LLM-as-a-judge | Automate quality scoring with built-in and custom judges | [`03-llm-judges-scorers`](../notebooks/02-generative-ai/03-llm-judges-scorers/) |
| Human feedback | Collect, manage, and use labels from real experts | [`04-human-feedback-prompts`](../notebooks/02-generative-ai/04-human-feedback-prompts/) |
| Prompt management | Version prompts the way you version code | [`04-human-feedback-prompts`](../notebooks/02-generative-ai/04-human-feedback-prompts/) |
| Continuous evaluation | Close the loop between production and your eval suite | [`05-continuous-evaluation`](../notebooks/02-generative-ai/05-continuous-evaluation/) |

**Primary reference**: [MLflow GenAI Documentation](https://mlflow.org/docs/latest/genai/) — return to this throughout.

This repo's notebooks use **MLflow 3.x** APIs (`mlflow.genai.evaluate()`, etc.). If you find docs referring to `mlflow.evaluate()` for LLM tasks, those are the older 2.x patterns — superseded.

---

## Topic 1 — Tracing: see what your app actually did

**Concepts**

You can't evaluate what you can't see. Tracing is the foundation that everything else builds on.

- A **trace** is the full record of one application call (e.g., one user query end-to-end).
- A trace contains **spans** — individual operations (LLM call, retrieval, tool use, custom logic).
- MLflow Tracing is **OpenTelemetry-compatible**, so it integrates with existing observability stacks (Jaeger, Grafana Tempo, etc.).

Two ways to instrument:

1. **Autologging** — one line per provider, captures every LLM call automatically.
   ```python
   mlflow.openai.autolog()
   mlflow.anthropic.autolog()
   mlflow.langchain.autolog()
   ```
2. **`@mlflow.trace`** decorator — for your own functions (preprocessing, retrieval, business logic). Decorated functions appear as nested spans in the trace.

The minimum useful instrumentation for a RAG application is: autolog the LLM provider + `@mlflow.trace` your retriever and your chain function. That's enough to see what was retrieved, what context was passed to the LLM, and how long each step took.

**Where in this repo**

- [`01-tracing-observability/notebook.py`](../notebooks/02-generative-ai/01-tracing-observability/notebook.py) — OpenAI autolog, Anthropic autolog (gracefully skipped if no key), LangChain autolog, a minimal RAG application with `@mlflow.trace`-decorated retriever and chain

**External references**

- [MLflow Tracing documentation](https://mlflow.org/docs/latest/genai/tracing/) — the canonical reference
- [Automatic Tracing guide](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/automatic/)
- [Tracing Integrations](https://mlflow.org/docs/latest/genai/tracing/integrations/) — every supported library (OpenAI, Anthropic, LangChain, LangGraph, LlamaIndex, DSPy, CrewAI, AutoGen, Bedrock, ...)
- [Practical AI Observability with MLflow Tracing](https://mlflow.org/blog/ai-observability-mlflow-tracing/) — official blog walkthrough

**Gotchas this repo's notebooks document**

- Trace logging is **async by default**. If you immediately try to attach feedback to a trace you just created, the `log_feedback` call can race the trace insert and fail with a FK error. Call `mlflow.flush_trace_async_logging()` between trace creation and feedback. (See [`04-human-feedback-prompts/notebook.py`](../notebooks/02-generative-ai/04-human-feedback-prompts/notebook.py).)
- The trace ID is fetched with `mlflow.get_last_active_trace_id()` (MLflow 3.10+) — earlier docs may show `get_last_active_trace().info.trace_id` which no longer exists.

---

## Topic 2 — Evaluation datasets: your test database

**Concepts**

An evaluation dataset is **your test database**. Every quality decision — "is this prompt better?", "did this model regress?", "is this agent ready to ship?" — answers itself against this dataset. Without one, you're vibing.

A row typically has:
- **inputs** — what gets sent to your application (the user query plus any context)
- **expectations** — ground truth: facts the answer must contain, guidelines it must follow, refusal conditions

```python
{
    "inputs": {"question": "What is MLflow Tracing?"},
    "expectations": {
        "expected_facts": [
            "MLflow Tracing captures inputs, outputs, and metadata of LLM applications.",
            "It supports automatic tracing for many GenAI libraries.",
            "It is OpenTelemetry-compatible.",
        ],
    },
}
```

Start small. **Twenty hand-curated examples beats two thousand noisy ones.** Cover the happy path, edge cases, and known failure modes.

Two complementary sources scale you past hand-writing:
1. **Production traces with negative feedback** — once users start clicking thumbs-down, those failures are gold; promote them into the eval set with corrected expectations.
2. **Synthetic generation from a strong model** — have a frontier model generate candidate Q&A pairs from your docs, then have a domain expert review and correct.

**Where in this repo**

- [`02-evaluation-datasets/notebook.py`](../notebooks/02-generative-ai/02-evaluation-datasets/notebook.py) — 20 hand-curated examples, versioned via `mlflow.log_table`, tagged with `dataset_version=v1`
- [`05-continuous-evaluation/notebook.py`](../notebooks/02-generative-ai/05-continuous-evaluation/notebook.py) — production thumbs-down → expert review → dataset v2 loop

**External references**

- [Evaluation and Monitoring documentation](https://mlflow.org/docs/latest/genai/eval-monitor/)

**Practical tips**

- **Version the dataset.** Log every revision with `mlflow.log_table`, tag the run with `dataset_version=v1`, `v2`, etc. Every evaluation run downstream should tag itself with the dataset version it used. Non-negotiable for trustworthy comparisons across time.
- **Build the dataset before the app — or at least in parallel.** If you can't articulate what "good" looks like in 20 concrete examples, you don't know what you're building yet.

---

## Topic 3 — LLM-as-a-judge: automated quality scoring

**Concepts**

- A **scorer** takes a trace (or its inputs/outputs/expectations) and produces a score.
- A **judge** is an LLM-based scorer. It uses a model to evaluate quality on dimensions exact matching can't capture (semantic correctness, tone, faithfulness to context).
- Code-based scorers are cheap, deterministic, and fast. Use them for structural checks (regex, exact match, length thresholds, latency limits).
- Combine code and LLM scorers in the same eval run for balanced coverage.

**Built-in judges** (most useful first):
- **`Correctness`** — does the response match the expected facts?
- **`RelevanceToQuery`** — does it actually address the user's question?
- **`RetrievalRelevance`** — for RAG: were the retrieved docs relevant?
- **`Safety`** — does it avoid harmful content?
- **`Guidelines`** — does it follow your custom natural-language rules?

`Guidelines` is the workhorse for domain rules. Anything you can describe in writing, you can evaluate:

```python
Guidelines(
    name="professional_tone",
    guidelines=(
        "The response must maintain a professional, formal tone. "
        "It must not use slang or emojis. "
        "It must address the user with 'you' and avoid first-person plural ('we')."
    ),
    model="openai:/gpt-5.2",
)
```

For fully custom dimensions, use `make_judge` to write the rubric from scratch.

**Where in this repo**

- [`03-llm-judges-scorers/notebook.py`](../notebooks/02-generative-ai/03-llm-judges-scorers/notebook.py) — built-in judges, custom `Guidelines` judge, code-based scorers (`@scorer`), three eval runs comparing scorer combinations

**External references**

- [LLM Judges and Scorers documentation](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/)
- [Built-in LLM Judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined)
- [Guidelines judge guide](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/guidelines/)
- [RAG relevance judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/relevance/)

**Gotchas this repo's notebooks document**

- **Pass `model=` to every built-in judge.** Without it, they call a default that may not be authorized on your OpenAI project. The notebooks read `MLFLOW_JUDGE_MODEL` from `.env` (defaults to `openai:/gpt-5.2`).
- **Custom code-based scorers must be decorated with `@mlflow.genai.scorers.scorer`.** Passing a bare function fails validation with "The `scorers` argument must be a list of scorers."
- **`predict_fn` parameter names must match the dataset's `inputs` keys.** For data `{"inputs": {"question": "..."}}`, use `def my_app(question: str)` — not `def my_app(inputs: dict)`.
- **GPT-5 family uses `max_completion_tokens`, not `max_tokens`.** The OpenAI API rejects `max_tokens` with a 400 for gpt-5.x models.

---

## Topic 4 — Human feedback and judge validation

**Concepts**

Three patterns for human feedback:

- **End-user feedback** — thumbs up/down attached programmatically when users react in your app's UI
- **Expert annotation** — domain experts label correctness, relevance, or any custom dimension directly in the MLflow UI
- **Annotation campaigns** — structured projects where reviewers grind through a queue of traces

```python
mlflow.log_feedback(
    trace_id=trace_id,
    name="user_thumbs",
    value="up",
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="user_12345",
    ),
    rationale="Clear and accurate.",
)
```

**Judge validation is the step most teams skip — and the one that determines whether your eval system is real or theatrical.** An LLM judge is a hypothesis: "this model scores quality the way a human would." Test that hypothesis:

1. Sample 50–100 traces
2. Have a human expert label each on the dimension your judge measures
3. Run your judge against the same traces
4. Compute agreement (raw % is fine to start; Cohen's kappa if rigorous)
5. If agreement is poor, iterate on the judge's instructions and try again

> A judge that agrees 90%+ with humans is trustworthy. A judge that agrees 60% is noise. You won't know which one you have until you measure.

**Where in this repo**

- [`04-human-feedback-prompts/notebook.py`](../notebooks/02-generative-ai/04-human-feedback-prompts/notebook.py) — `log_feedback`, simulated judge validation with agreement rate calculation

**External references**

- [Human feedback documentation](https://mlflow.org/docs/latest/genai/eval-monitor/)
- [Cohen's kappa (Wikipedia)](https://en.wikipedia.org/wiki/Cohen%27s_kappa) — for when raw agreement isn't enough

---

## Topic 5 — Prompt management

**Concepts**

The **Prompt Registry** versions prompts the same way the Model Registry versions models. Register your prompts with a template that uses `{{variable}}` syntax for variables:

```python
mlflow.genai.register_prompt(
    name="rag_answer_v1",
    template=(
        "Use only the following context to answer. "
        "If the answer is not in the context, say so.\n\n"
        "Context: {{context}}\n\n"
        "Question: {{question}}"
    ),
    commit_message="Initial version with explicit refusal instruction",
)
```

Each registration creates a new immutable version. Set an alias for stable references:

```python
mlflow.genai.set_prompt_alias(name="rag_answer_v1", alias="production", version=2)
```

Then your application code references `prompts:/rag_answer_v1@production` — and you change which version `@production` points to without touching the app. Same alias pattern as the Model Registry.

**A/B evaluation pattern**:

```python
for version in [1, 2]:
    prompt = mlflow.genai.load_prompt(f"prompts:/rag_answer_v1/{version}")
    def app_with_prompt(question, _prompt=prompt):
        ...
    with mlflow.start_run(run_name=f"eval-prompt-v{version}"):
        mlflow.set_tag("prompt_version", version)
        mlflow.genai.evaluate(data=eval_dataset, predict_fn=app_with_prompt, scorers=[...])
```

Compare the two runs side-by-side in the UI: did v2 actually improve? On which dimensions? Did it regress on any examples?

**Where in this repo**

- [`04-human-feedback-prompts/notebook.py`](../notebooks/02-generative-ai/04-human-feedback-prompts/notebook.py) — register two prompt versions, set `@production` alias, A/B eval

**External references**

- [Prompt Registry documentation](https://mlflow.org/docs/latest/genai/prompt-registry/)
- [Evaluating Prompts guide](https://mlflow.org/docs/latest/genai/prompt-registry/evaluate-prompts/)

---

## Topic 6 — Continuous evaluation

**Concepts**

Evaluation has to keep working after you ship. The full production loop:

1. User gives thumbs-down on a trace in production
2. Trace is automatically tagged for review
3. An expert reviews and adds the corrected example to your evaluation dataset
4. Next eval run catches the same failure mode automatically
5. No code or prompt change ships unless a regression-gate evaluation passes

**Automatic evaluation** — judges that run on traces as they're logged, without you invoking an eval script. Configure a sample rate, dashboard the results, alert on regressions.

**Regression-gating** — before any candidate change ships:

1. The candidate is evaluated against the latest dataset version
2. Results are compared to the stored production baseline
3. If any judge drops below the threshold, the change is blocked

**Where in this repo**

- [`05-continuous-evaluation/notebook.py`](../notebooks/02-generative-ai/05-continuous-evaluation/notebook.py) — simulated production traffic, thumbs-down feedback collection, dataset v2 from production failures, regression-gate pattern

**External references**

- [Automatic Evaluation documentation](https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/)

**Note on Databricks-managed-first features**

Some capabilities (notably `Safety` and `RetrievalRelevance` judges, and some annotation UIs) launched in **Databricks Managed MLflow** before reaching open-source. If you're self-hosting and a feature isn't behaving, check the docs for current availability and consider running against the Databricks workspace via `MLFLOW_TRACKING_URI=databricks`.

---

## Cross-cutting principles

These show up across the whole GenAI track. Internalizing them is the difference between a real evaluation practice and one that looks impressive but tells you nothing.

- **Validate your judges before trusting them.** An unvalidated LLM judge is a confident liar. Measure agreement with humans on a sample, or your scores mean nothing.
- **Mix judge types.** LLM judges for quality, code scorers for structure, human review for the hardest cases. No single layer catches everything.
- **Don't over-engineer scorers.** Three good scorers beat fifteen mediocre ones. Each scorer adds latency and cost on every eval run.
- **Tag everything.** Tag traces with user ID, prompt version, model name, app version, dataset version. Without these you can't slice when something regresses.
- **Watch for evaluation data leakage.** If a frontier model wrote your eval dataset and the same model is your judge, you have a problem. Use a different model family for judge vs generator when you can.

---

## Self-check

By the end of this track you should be able to answer "yes" to all of these:

- [ ] I can enable automatic tracing for any supported LLM library in one line
- [ ] I can add manual spans with `@mlflow.trace` for my application's custom logic
- [ ] I have a versioned evaluation dataset of at least 20 examples with expected facts
- [ ] I can run `mlflow.genai.evaluate()` and interpret the results in the UI
- [ ] I can write a custom Guidelines judge that encodes domain rules in English
- [ ] I have measured at least one of my judges against human labels and know its agreement rate
- [ ] I can register a prompt, set an alias, and load it by alias in my application code
- [ ] I can compare two prompt versions side-by-side on the same dataset
- [ ] I can describe the loop from production feedback → eval dataset → regression-gated deployment

If yes to all nine, you have a real evaluation practice, not just a vibe.

---

## Recommended supplementary reading

- [MLflow GenAI examples on GitHub](https://github.com/mlflow/mlflow/tree/master/examples) — real working code
- [Anthropic's "Building effective agents"](https://www.anthropic.com/research/building-effective-agents) — vendor-agnostic principles you'll want to evaluate against
- [Eugene Yan's writing on LLM evaluation](https://eugeneyan.com/) — pragmatic, field-tested guidance
- [Databricks Managed MLflow](https://learn.microsoft.com/en-us/azure/databricks/mlflow/) — when you're ready to move from self-hosted to managed on Azure
