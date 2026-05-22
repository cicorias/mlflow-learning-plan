# 04 — Human Feedback, Judge Validation, and Prompt Management (Week 4)

Close the loop between automated evaluation, human judgment, and prompt iteration.

## Goal

Compare two prompt versions side-by-side on your dataset, with at least one judge validated against human labels.

## What this notebook covers

- `mlflow.log_feedback()` with `AssessmentSource` for end-user and expert feedback
- The judge validation workflow: sample traces → human labels → judge labels → agreement rate
- Cohen's kappa for measuring judge reliability
- `mlflow.genai.register_prompt()` with double-brace templates `{{variable}}`
- Prompt versioning and `set_prompt_alias()` for `@production` pointer
- Loading prompts by alias URI: `prompts:/name@production`
- A/B evaluation: same dataset, two prompt versions, side-by-side comparison in the UI

## Checkpoint

After running this notebook you should be able to:
- [ ] Attach human feedback to a trace programmatically
- [ ] Register a prompt, set an alias, and load it by alias in application code
- [ ] Compare two prompt versions side-by-side on the same dataset
- [ ] Describe what judge agreement rate means and why it matters

## Deliverable

A side-by-side evaluation of two prompt versions, with agreement rate computed for at least one judge on a 20+ trace sample.

## References

- [Human feedback documentation](https://mlflow.org/docs/latest/genai/eval-monitor/)
- [Prompt Registry documentation](https://mlflow.org/docs/latest/genai/prompt-registry/)
- [Evaluating Prompts guide](https://mlflow.org/docs/latest/genai/prompt-registry/evaluate-prompts/)
