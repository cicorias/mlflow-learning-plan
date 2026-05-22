# 01 — Tracing and Observability (Week 1)

Instrument any LLM application so you can see exactly what happens inside every request.

## Goal

Produce complete traces for OpenAI, Anthropic, and LangChain calls — plus your own functions — visible in the MLflow UI.

## What this notebook covers

- Tracing concepts: trace, span, OpenTelemetry compatibility
- One-line autologging: `mlflow.openai.autolog()`, `mlflow.anthropic.autolog()`, `mlflow.langchain.autolog()`
- The `@mlflow.trace` decorator with `SpanType` for custom functions
- Building a minimal RAG application that is fully observable end-to-end
- Navigating the Traces tab: span tree, inputs/outputs, latency, token counts

## Checkpoint

After running this notebook you should be able to:
- [ ] Enable automatic tracing for any supported LLM library in one line
- [ ] Add manual spans with `@mlflow.trace` for your application's custom logic
- [ ] Inspect a complete trace in the UI and identify where latency is spent

## Deliverable

A RAG application instrumented with autologging and at least two `@mlflow.trace`-decorated functions, producing complete traces in the MLflow UI.

## References

- [MLflow Tracing documentation](https://mlflow.org/docs/latest/genai/tracing/)
- [Automatic Tracing guide](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/automatic/)
- [Tracing Integrations](https://mlflow.org/docs/latest/genai/tracing/integrations/)
