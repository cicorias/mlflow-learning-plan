
# MLflow for Generative AI — A Learning Plan for Evaluation, Tracing, and Ground Truth

> A focused, practitioner-oriented onboarding to MLflow's generative AI capabilities: tracing, evaluation, LLM-as-a-judge, ground-truth datasets, human feedback, and prompt management. This plan assumes you've built at least one LLM app or agent (with the OpenAI SDK, LangChain, LlamaIndex, or similar) and understand basic ML concepts.

---

## Why This Plan Exists

Building a GenAI application is easy. Knowing whether it actually works is hard.

Unlike traditional ML, you can't compute a single accuracy number and call it a day. Outputs are free-form text. Failure modes are subtle (hallucinations, off-topic responses, ignored instructions, retrieval misses). Ground truth often doesn't exist until you create it. And whatever evaluation system you build today has to keep working as you swap models, change prompts, and deploy to production.

MLflow has grown into one of the most complete open-source platforms for this work. The plan below covers the pieces that matter:

1. **Tracing** — capture every step of every request so you can debug what your app actually did.
2. **Evaluation datasets** — build the "test database" that anchors quality work.
3. **LLM-as-a-judge** — automate quality scoring with built-in and custom judges.
4. **Ground truth and human feedback** — collect, manage, and use labels from real experts.
5. **Prompt management** — version prompts the way you version code.
6. **Continuous evaluation** — close the loop between production and your eval suite.

**Anchor reference throughout:** the [MLflow GenAI documentation](https://mlflow.org/docs/latest/genai/).

This plan assumes MLflow 3.x. If you're stuck on 2.x, upgrade first — the GenAI APIs in 3.x are substantially better and the older `mlflow.evaluate()` patterns have been superseded by `mlflow.genai.evaluate()`.

---

## Week 1 — Tracing and Observability

**Goal:** Instrument any LLM application so you can see exactly what happens inside every request, from the user query to the final response.

You can't evaluate what you can't see. Tracing is the foundation everything else builds on.

### Step 1.1 — Set up your environment

In a clean virtual environment:

```bash
pip install --upgrade mlflow openai anthropic langchain langchain-openai
mlflow server --host 127.0.0.1 --port 5000
```

Open [http://localhost:5000](http://localhost:5000) in your browser. In a separate terminal or notebook:

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("genai-tracing-week1")
```

### Step 1.2 — Read the tracing overview

Spend 30 minutes with the [MLflow Tracing documentation](https://mlflow.org/docs/latest/genai/tracing/) and the [Practical AI Observability blog post](https://mlflow.org/blog/ai-observability-mlflow-tracing/). The core ideas to leave with:

- A **trace** is the full record of one application call.
- A trace contains **spans** — individual operations (LLM call, retrieval, tool use).
- MLflow Tracing is OpenTelemetry-compatible, so it integrates with existing observability stacks.

### Step 1.3 — One-line autologging across providers

Read the [Automatic Tracing guide](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/automatic/), then try each of these:

```python
# OpenAI
import mlflow, openai
mlflow.openai.autolog()
client = openai.OpenAI()
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain MLflow Tracing in one sentence."}]
)
```

```python
# Anthropic
import mlflow, anthropic
mlflow.anthropic.autolog()
client = anthropic.Anthropic()
client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain MLflow Tracing in one sentence."}]
)
```

```python
# LangChain / LangGraph
import mlflow
mlflow.langchain.autolog()
# Now any chain.invoke() / graph.invoke() call is traced automatically
```

Each call should produce a trace visible in the UI under the **Traces** tab. Browse them, expand the spans, and notice what's captured: inputs, outputs, token counts, latencies, errors.

The full list of supported integrations (OpenAI, Anthropic, LangChain, LangGraph, LlamaIndex, DSPy, CrewAI, AutoGen, Bedrock, and many more) is on the [Tracing Integrations page](https://mlflow.org/docs/latest/genai/tracing/integrations/).

### Step 1.4 — Manual tracing with the `@mlflow.trace` decorator

Autologging captures third-party library calls. To trace your own functions — preprocessing, retrieval, custom logic — use the decorator:

```python
import mlflow
from mlflow.entities import SpanType

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_docs(query: str) -> list[str]:
    # your retrieval logic
    return ["doc 1", "doc 2"]

@mlflow.trace(span_type=SpanType.CHAIN)
def answer_question(query: str) -> str:
    docs = retrieve_docs(query)
    # ... your LLM call here (autolog will capture it)
    return "answer"
```

Decorated functions appear as nested spans in the trace. This is how you build a RAG application that's fully observable end-to-end.

### Step 1.5 — Inspect traces in the UI

In the MLflow UI:

- **Traces tab** — chronological list of recent traces, sortable by latency, status, and tags.
- **Click any trace** — see the full span tree on the left, the selected span's inputs/outputs on the right.
- **Trace comparison** — select two traces to diff them side-by-side (useful for debugging regressions).

### Week 1 deliverable

A small RAG or agent application of your own, instrumented with autologging plus at least two `@mlflow.trace`-decorated functions, producing complete traces visible in the MLflow UI.

---

## Week 2 — Evaluation Datasets and Ground Truth

**Goal:** Build, version, and use evaluation datasets — the foundation of any serious evaluation effort.

### Step 2.1 — Understand the role of evaluation datasets

Read the [Evaluation Datasets documentation](https://mlflow.org/docs/latest/genai/eval-monitor/) and absorb this framing: an evaluation dataset is your test database. Every quality decision — "is this prompt better?", "did this model regress?", "is this agent ready to ship?" — answers itself against this dataset. Without it, you're just vibing.

A row in an MLflow eval dataset typically has:

- **inputs** — what gets sent to your application (the user query, plus any context).
- **expectations** — ground truth: what should the output contain, what facts must be present, what guidelines must be followed.

### Step 2.2 — Build your first dataset

Start small. Twenty hand-curated examples beats 2,000 noisy ones. For each example, write:

- The input.
- The expected facts the answer must contain.
- Any specific guidelines (tone, format, refusal conditions).

```python
eval_dataset = [
    {
        "inputs": {"question": "What is MLflow Tracing?"},
        "expectations": {
            "expected_facts": [
                "MLflow Tracing captures inputs, outputs, and metadata of LLM applications.",
                "It supports automatic tracing for many GenAI libraries.",
                "It is OpenTelemetry-compatible.",
            ],
        },
    },
    {
        "inputs": {"question": "How do I register a prompt in MLflow?"},
        "expectations": {
            "expected_facts": [
                "Use mlflow.genai.register_prompt() with a name and template.",
                "Templates use double-brace syntax for variables.",
                "Each registration creates a new immutable version.",
            ],
        },
    },
    # ... at least 18 more
]
```

### Step 2.3 — Source ground truth from real conversations

Hand-writing every example is expensive. Two complementary sources of higher-quality ground truth:

1. **Production traces with negative feedback.** Once you start collecting user thumbs-down (Week 3), export those traces into an eval dataset. These are real failures.
2. **Synthetic generation from a strong model.** Use a frontier model to generate candidate Q&A pairs from your documentation, then have a domain expert review and correct them. This gets you to 100+ examples fast.

Read about the [continuous improvement workflow](https://mlflow.org/docs/latest/genai/) to see how this loop is meant to operate.

### Step 2.4 — Version your dataset

Treat the dataset as a first-class artifact. Log it to MLflow so every evaluation run is reproducible:

```python
import mlflow
import pandas as pd

df = pd.DataFrame(eval_dataset)

with mlflow.start_run(run_name="eval-dataset-v1"):
    mlflow.log_table(df, artifact_file="eval_dataset_v1.json")
    mlflow.set_tag("dataset_version", "v1")
    mlflow.set_tag("dataset_size", len(df))
```

When you add examples later, log it as v2. When you run an evaluation, tag the run with the dataset version you used. This is non-negotiable for trustworthy comparisons across time.

### Week 2 deliverable

A versioned evaluation dataset of at least 20 examples covering the main use cases (and failure modes) of your application, with explicit ground-truth expectations.

---

## Week 3 — LLM-as-a-Judge and Built-in Scorers

**Goal:** Run automated evaluations using both built-in and custom LLM judges.

### Step 3.1 — Understand judges and scorers

Read the [LLM Judges and Scorers documentation](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/). Two key concepts:

- A **scorer** is anything that takes a trace (or its inputs/outputs) and produces a score. Scorers can be code-based (regex match, exact string match) or LLM-based.
- A **judge** is an LLM-based scorer. It uses a model to evaluate quality on dimensions that exact matching can't capture — semantic correctness, tone, helpfulness, faithfulness to retrieved context.

The fundamental insight: exact string matching can't tell that "give me healthy food options" and "food to keep me fit" are the same answer. An LLM judge can.

### Step 3.2 — Use built-in judges

Read [Built-in LLM Judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined). The most useful ones to start:

- **`Correctness`** — does the response match the expected facts?
- **`RelevanceToQuery`** — does the response actually address the user's question?
- **`RetrievalRelevance`** — for RAG: were the retrieved documents relevant? (See the [RAG relevance judges guide](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/relevance/).)
- **`Safety`** — does the response avoid harmful content?
- **`Guidelines`** — does the response follow your custom natural-language rules?

Run your first evaluation:

```python
import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_app,        # your application function
    scorers=[
        Correctness(),
        RelevanceToQuery(),
    ],
)
```

In the UI, navigate to your experiment and open the resulting evaluation run. You'll see per-row scores, aggregate metrics, and the ability to click into individual traces to see exactly why a row passed or failed.

### Step 3.3 — Write a custom Guidelines judge

The [Guidelines judge guide](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/guidelines/) lets you encode domain rules in plain English:

```python
from mlflow.genai.scorers import Guidelines

tone_judge = Guidelines(
    name="professional_tone",
    guidelines=(
        "The response must maintain a professional, formal tone. "
        "It must not use slang, exclamation points, or emojis. "
        "It must address the user with 'you' and avoid first-person plural ('we')."
    ),
)

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_app,
    scorers=[tone_judge],
)
```

This is enormously powerful: anything you can describe in writing, you can evaluate.

### Step 3.4 — Write a fully custom scorer with `make_judge`

For more complex evaluations, build a judge from scratch:

```python
import mlflow
from mlflow.genai.judges import make_judge

faithfulness_judge = make_judge(
    name="faithfulness_to_retrieved_context",
    instructions=(
        "Given the user's question, the retrieved context, and the model's answer, "
        "determine whether every factual claim in the answer is supported by the context. "
        "Return a score of 'pass' only if no claim contradicts or goes beyond the context."
    ),
    model="openai:/gpt-4o",
)
```

This pattern — write the rubric in English, point at a strong evaluator model — is how teams scale evaluation to dozens of dimensions without writing dozens of bespoke metrics.

### Step 3.5 — Add code-based scorers

Not everything should be LLM-judged. Code is cheaper, deterministic, and faster:

- Exact-match scorers for structured outputs (JSON schemas, classifications).
- Regex scorers for required substrings or forbidden phrases.
- Latency and cost scorers (every trace records both).

Combine code and LLM scorers in the same run for a balanced eval.

### Week 3 deliverable

An evaluation run on your Week 2 dataset that combines: at least one built-in judge, at least one custom Guidelines judge, and at least one code-based scorer. Open the evaluation results in the UI and identify the three worst-performing examples.

---

## Week 4 — Human Feedback, Judge Validation, and Prompt Management

**Goal:** Close the loop between automated evaluation, human judgment, and prompt iteration.

### Step 4.1 — Collect human feedback on traces

Read the [human feedback documentation](https://mlflow.org/docs/latest/genai/eval-monitor/). Feedback is attached to traces and recorded with the annotator's identity, timestamp, and revision history.

The patterns to know:

- **End-user feedback** — thumbs up/down attached programmatically when users react in your app's UI.
- **Expert annotation** — domain experts label correctness, relevance, or any custom dimension directly in the MLflow UI.
- **Annotation campaigns** — structured projects where reviewers grind through a queue of traces.

Programmatically attach feedback to a trace:

```python
import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

mlflow.log_feedback(
    trace_id=trace_id,
    name="user_thumbs",
    value="up",                  # or "down"
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="user_12345",
    ),
)
```

### Step 4.2 — Validate your LLM judges against humans

This is the step most teams skip — and the one that determines whether your eval system is real or theatrical. An LLM judge is a hypothesis: "this model can score quality the way a human would." You should test that hypothesis.

The workflow:

1. Sample 50–100 traces.
2. Have a human expert label each on the dimension your judge measures.
3. Run your judge against the same traces.
4. Compare. Compute agreement (raw % agreement is fine to start; Cohen's kappa if you want to be rigorous).
5. If agreement is poor, iterate on the judge's instructions and try again.

A judge that agrees with humans 90%+ of the time is trustworthy. A judge that agrees 60% of the time is noise. You won't know which one you have until you measure.

### Step 4.3 — Manage prompts with the Prompt Registry

Read the [Prompt Registry documentation](https://mlflow.org/docs/latest/genai/prompt-registry/) and the [Evaluating Prompts guide](https://mlflow.org/docs/latest/genai/prompt-registry/evaluate-prompts/).

Register your prompts so they're versioned, comparable, and reusable:

```python
import mlflow

prompt_v1 = mlflow.genai.register_prompt(
    name="rag_answer_v1",
    template=(
        "You are a helpful assistant. Use only the following context to answer "
        "the question. If the answer is not in the context, say so explicitly.\n\n"
        "Context: {{context}}\n\n"
        "Question: {{question}}"
    ),
    commit_message="Initial version with explicit refusal instruction",
)

# Later, register an improvement
prompt_v2 = mlflow.genai.register_prompt(
    name="rag_answer_v1",
    template="...improved template...",
    commit_message="Add chain-of-thought instruction to reduce hallucinations",
)

# Set an alias for your production code to reference
from mlflow import MlflowClient
client = MlflowClient()
mlflow.genai.set_prompt_alias(name="rag_answer_v1", alias="production", version=2)

# In your app
prompt = mlflow.genai.load_prompt("prompts:/rag_answer_v1@production")
rendered = prompt.format(context="...", question="...")
```

The big idea: your application code references `prompts:/rag_answer_v1@production`, and you change which version `@production` points to. Same alias pattern as the Model Registry.

### Step 4.4 — Run A/B evaluation across prompt versions

Wire it all together by evaluating two prompt versions against the same dataset:

```python
for version in [1, 2]:
    prompt = mlflow.genai.load_prompt(f"prompts:/rag_answer_v1/{version}")

    def app_with_prompt(inputs):
        rendered = prompt.format(**inputs)
        # ... call your LLM with `rendered` ...
        return {"response": llm_response}

    with mlflow.start_run(run_name=f"eval-prompt-v{version}"):
        mlflow.set_tag("prompt_version", version)
        mlflow.genai.evaluate(
            data=eval_dataset,
            predict_fn=app_with_prompt,
            scorers=[Correctness(), RelevanceToQuery(), tone_judge],
        )
```

In the UI, compare the two runs side-by-side. Did v2 actually improve? On which dimensions? Did it regress on any examples?

### Week 4 deliverable

A side-by-side evaluation comparing two versions of a prompt on your dataset, with at least one judge whose agreement with human labels you've measured on a 50-trace sample.

---

## Week 5 (Stretch) — Continuous Evaluation in Production

**Goal:** Make evaluation a continuous, automated process — not a thing you do once before shipping.

### Step 5.1 — Automatic evaluation on production traces

Read the [Automatic Evaluation documentation](https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/). The idea: judges run automatically on traces as they're logged, without you needing to invoke an eval script. This turns the MLflow server into a continuous quality monitor.

Configure judges to run on a sample of production traces, dashboard the results, and alert on regressions.

### Step 5.2 — Feed production failures back into your eval dataset

Production is your richest source of evaluation data. The continuous improvement loop:

1. User gives thumbs-down on a trace in production.
2. Trace is automatically tagged for review.
3. An expert reviews and adds the corrected example to your evaluation dataset (with the right ground-truth expectations).
4. Next time you run eval, that case is in the test set.

This is how eval datasets stay relevant as your traffic and use cases evolve.

### Step 5.3 — Regression-gate your deployments

Hook MLflow evaluation into CI. Before any prompt or model change can ship:

1. The candidate is evaluated against the latest version of your dataset.
2. Results are compared against the current production baseline.
3. If any judge regresses beyond a threshold, the change is blocked.

This turns evaluation from an after-the-fact check into a deployment gate.

---

## Recommended Resources

### Primary

- **[MLflow GenAI Documentation](https://mlflow.org/docs/latest/genai/)** — the canonical reference, updated continuously.
- **[MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/)** — observability for LLM applications.
- **[Evaluation and Monitoring](https://mlflow.org/docs/latest/genai/eval-monitor/)** — judges, datasets, scorers.
- **[Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/)** — versioned prompt management.
- **[Tracing Integrations List](https://mlflow.org/docs/latest/genai/tracing/integrations/)** — every supported library and how to enable it.

### Supplementary

- **[Practical AI Observability with MLflow Tracing](https://mlflow.org/blog/ai-observability-mlflow-tracing/)** — official MLflow blog walkthrough.
- **[MLflow GenAI examples on GitHub](https://github.com/mlflow/mlflow/tree/master/examples)** — real working code.
- **Anthropic's "Building effective agents"** — vendor-agnostic principles you'll want to evaluate against.
- **Eugene Yan's writing on LLM evaluation** — pragmatic, field-tested guidance ([eugeneyan.com](https://eugeneyan.com)).

---

## Tips and Pitfalls

- **Build the dataset before the app.** Or at least in parallel. If you can't articulate what "good" looks like in 20 concrete examples, you don't know what you're building yet.
- **Validate your judges before trusting them.** An unvalidated LLM judge is a confident liar. Measure agreement with humans on a sample, or your eval scores mean nothing.
- **Mix judge types.** LLM judges for quality, code scorers for structure, human review for the hardest cases. No single layer catches everything.
- **Don't over-engineer scorers.** Three good scorers beat fifteen mediocre ones. Each scorer adds latency and cost on every evaluation run.
- **Tag everything.** Tag traces with user ID, prompt version, model name, app version, and dataset version. Without these, you can't slice and dice when something regresses.
- **Watch out for evaluation data leakage.** If a frontier model wrote your eval dataset and the same model is your judge, you have a problem. Use a different model family as judge than as generator when you can.
- **Some features are Databricks-managed first.** A few capabilities (notably `Safety` and `RetrievalRelevance` judges, and some human-annotation UIs) launched in Databricks Managed MLflow before reaching open-source. Check the docs for current availability if you're self-hosting.

---

## Self-Check: Are You Done?

By the end of this plan you should be able to answer "yes" to all of these:

- [ ] I can enable automatic tracing for any supported LLM library in one line.
- [ ] I can add manual spans with `@mlflow.trace` for my application's custom logic.
- [ ] I have a versioned evaluation dataset of at least 20 examples with expected facts.
- [ ] I can run `mlflow.genai.evaluate()` and interpret the results in the UI.
- [ ] I can write a custom Guidelines judge that encodes domain rules in English.
- [ ] I have measured at least one of my judges against human labels and know its agreement rate.
- [ ] I can register a prompt, set an alias, and load it by alias in my application code.
- [ ] I can compare two prompt versions side-by-side on the same dataset.
- [ ] I can describe the loop from production feedback → eval dataset → regression-gated deployment.

If yes to all nine, you have a real evaluation practice, not just a vibe.