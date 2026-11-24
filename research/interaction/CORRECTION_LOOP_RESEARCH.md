# Correction-Driven Response Research

## Response Metrics

Exact wording and semantic correctness are separate outcomes. A response may communicate the right proposition without repeating the question's vocabulary. Score meaning first against a task-specific semantic label or independent verifier; score response form separately for requested precision, brevity, tone, and required details. Do not count a paraphrase as a reasoning error solely because it differs from an expected string.

For constrained structural tasks, exact state equality remains appropriate for transition verification. For user-facing answers, use semantic correctness as the baseline and exactness as a secondary realization metric when the request asks for exact wording, a specific value, or a complete record.

## Correction Loop

1. Save the original prompt and response only with the caller's explicit private-persistence policy.
2. Ask whether the correction changes the meaning or only the expression. These labels remain separate.
3. Store the correction as an immutable source event and an append-only `Learn` experience episode. Do not write it directly into concept facts or model weights.
4. Collect multiple distinct episodes into a candidate training dataset through the existing `Learn.trial` path.
5. Evaluate on independent inputs and task families. Promote a dataset/checkpoint only after the existing independent review and model-transfer gates pass.
6. Keep user-specific style preferences scoped to that user. A correction to phrasing must not silently revise factual baselines.

The capture prototype is `correction.py`. It labels meaning and expression corrections distinctly, records the prompt/response/correction as source evidence, and appends an episode. It does not train or promote anything. Corrections require the explicit private persistence policy already used by the memory system.

## Current Evidence

The concept fixture has a six-item manually labeled semantic response baseline. All six variants express the expected yes/no meaning, while only 2/6 exactly match the short reference string. This demonstrates how exact string scoring can undercount equivalent meanings; it does not measure neural capability. The deterministic resolver returns the requested source evidence. The Foundation checkpoint scored 0/2 on direct concept prompts and repeated `to`; this is a failure of the current model, not a reason to equate exact string match with meaning.

The follow-up semantic correction pilot used a 56-world benchmark and compared 24 correction updates with a 96-update dose extension from identical starting weights. Semantic accuracy and coverage both stayed at 0/80 in both corrected runs; the four-pass training loss fell, but generation stayed empty or repetitive. All synthetic drafts were semantically unparseable, so this did not test style-only corrections. See [`SEMANTIC_CORRECTION_PILOT.md`](SEMANTIC_CORRECTION_PILOT.md) for hashes, metrics and the no-go decision.

The correction-capture tests verify that a wording-only correction is recorded with `meaning_confirmed=true`, a meaning correction is distinct, private persistence is required, the original response and correction are preserved, and no fact is silently promoted. There is no evidence yet that these episodes improve model behavior. Do not start a larger fine-tune until a paraphrase-tolerant semantic benchmark, corrected-response realization set, and independent transfer gate are in place.

Reproduce the lifecycle tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/batch6-venv/bin/python -m unittest research.interaction.test_correction
```
