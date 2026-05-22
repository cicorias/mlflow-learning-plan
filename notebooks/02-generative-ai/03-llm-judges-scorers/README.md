# 03 — LLM-as-a-Judge and Built-in Scorers (Week 3)

Run automated evaluations using built-in and custom LLM judges.

## Goal

Produce an evaluation run combining built-in judges, a custom Guidelines judge, and a code-based scorer.

## What this notebook covers

- Scorers vs judges: code-based (deterministic, cheap) vs LLM-based (semantic, flexible)
- Built-in judges: `Correctness`, `RelevanceToQuery`, `RetrievalRelevance`, `Safety`, `Guidelines`
- `mlflow.genai.evaluate()` with `predict_fn` and `scorers`
- Writing a custom `Guidelines` judge that encodes domain rules in plain English
- Building a custom judge from scratch with `make_judge`
- Code-based scorers: exact match, regex, latency thresholds
- Reading per-row scores and aggregate metrics in the UI

## Checkpoint

After running this notebook you should be able to:
- [ ] Run `mlflow.genai.evaluate()` and interpret the results in the UI
- [ ] Write a custom Guidelines judge that encodes domain rules in English
- [ ] Identify the three worst-performing examples in an evaluation run

## Deliverable

An evaluation run on your Week 2 dataset combining at least one built-in judge, one custom Guidelines judge, and one code-based scorer.

## References

- [LLM Judges and Scorers documentation](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/)
- [Built-in LLM Judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined)
- [Guidelines judge guide](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/guidelines/)
