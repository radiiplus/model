# Evaluation Specification (Batch 1)

## Status and scope

This is a small, offline evaluator for checking scoring behavior before a model exists. The fixtures are hand-authored synthetic examples in `../experiments/fixtures/tiny_eval.json`; they contain no external benchmark records. No package or model was installed or downloaded. The harness uses Python's standard library and accepts a backend through a JSONL prediction adapter. A future backend can implement `generate(case)` directly or export rows keyed by fixture `id`.

Run from the workspace root:

```sh
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 research/experiments/eval_harness.py --baseline abstain --seed 1234 \
  --output research/experiments/results/baseline_abstain.json
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python3 -m unittest -v research/experiments/test_eval_harness.py
```

The harness checks `MemAvailable` before running and exits if below 2,560 MiB. The external `timeout` is the wall-time guard. Code evaluation has a 2-second subprocess timeout. The interpreter runs a fixed, local toy fixture only; subprocess execution is not a security sandbox and must not be used for untrusted code submissions.

## Evaluation tracks

Track scores are reported separately and are never averaged into a general capability score.

| Track | Current cases | Scoring now | Later extension |
|---|---:|---|---|
| Base next-token prediction | 0 | Not implemented; requires token log probabilities and a versioned held-out text fixture | Mean negative log likelihood, bits/token, perplexity with tokenizer, boundaries and corpus license pinned |
| Instruction following | 1 | Exact normalized output match for strict format instruction | Constraint satisfaction per instruction, including structured output/schema checks |
| Reasoning | 2 | Exact answer match on arithmetic and a small relational deduction | Pre-registered task families, answer-only and process diagnostics separated; no chain-of-thought reward |
| Coding | 1 | Execute candidate code against deterministic unit assertions | Multiple hidden/public tests, runtime, timeout, syntax/error categories, exact environment pin |
| Retrieval-grounded QA | 2 | Exact answer plus atomic claim and evidence-ID match | Larger licensed corpus and human audited claim/evidence labels |
| Hallucination resistance | 3 | Abstention classification on unavailable answer plus correct contradiction handling and evidence metrics where labeled | Stress tests for distractors, source conflicts, paraphrase, missing evidence and false premises |

Current examples are smoke tests of evaluator mechanics, not stable estimates of model ability. Case count, expected answer, evidence IDs, source notes and scoring version must be frozen before comparing model variants.

## Output contract

Each prediction is keyed by case `id` and may contain:

```json
{"id":"fact_city","answer":"Sample City","claims":[{"text":"Exampleland's capital is Sample City.","evidence_ids":["e1"]}],"abstained":false,"confidence":0.8,"tokens_generated":4}
```

`answer` is required for answer tasks; `code` is used for the coding task; `claims` contains atomic claims and source IDs; `abstained` is a boolean; `confidence` is optional and must be in [0,1]. `tokens_generated` is optional. The harness uses reported count if present, otherwise emits a whitespace count over answer/code text. That estimate is not a tokenizer count and should not be compared across tokenizers.

## Metrics and scoring

- **Exact match:** stripped, whitespace-collapsed and case-insensitive string equality by default. Strict `format_constraint` cases preserve case. No semantic equivalence, stemming or fuzzy credit. Code is excluded from text EM.
- **Task accuracy:** correct task behavior over all examples. Answerable tasks require a non-abstained exact answer; unanswerable tasks count a declared abstention as correct. Coding requires all fixture assertions to pass. This makes abstention behavior visible without conflating it with answer-string EM.
- **Track accuracy:** the same correctness rule, grouped by track. Small denominators are included in result JSON and must be shown alongside rates.
- **Evidence-supported claim precision:** number of predicted claims whose normalized text exactly matches a gold atomic claim and whose cited evidence ID intersects the gold IDs, divided by predicted claims. Null when no claims are emitted.
- **Evidence-supported claim recall:** those supported predictions divided by all gold atomic claims. A missing claim scores as a miss. A repeated claim can increase denominator without increasing credit.
- **Unsupported-claim rate:** predicted claims that fail the exact text/source check divided by predicted claims. Null when no claims are emitted. Empty claims therefore do not earn a perfect precision score.
- **Abstention precision/recall:** abstention is the positive class; gold positive means the fixture is unanswerable from supplied evidence. Thus false abstentions on answerable tasks lower precision, and answered unsupported questions lower recall.
- **Calibration:** when confidence is supplied, Brier score and 5-bin expected calibration error (ECE) are computed against task correctness, including correct abstention as correct behavior. Calibration is omitted (`n=0`) when no confidence values exist. Tiny-bin estimates are highly unstable.
- **Latency:** per-case wall time around backend generation, summarized by mean, median and nearest-rank p95. It excludes external service setup unless the adapter includes it.
- **Tokens:** total and per-case generated count, supplied by backend where possible; fallback is whitespace count and explicitly approximate.
- **Peak RSS:** Linux `ru_maxrss` for the evaluator plus the maximum child-process RSS reported by `resource`; this sum is a conservative approximation, not a sampled simultaneous process-tree maximum. Runtime and interpreter overhead are included.
- **Compute per correct answer:** evaluator process CPU seconds divided by correct task count. It excludes child CPU usage and backend work outside this process; currently only a rough local proxy, not FLOPs, energy or cloud cost. Null if no correct task.

The JSON retains each raw prediction, per-case correctness, latency and token count to support audit. No overall score is defined. Before a real comparison, report per-track numerators/denominators, confidence intervals or repeated seeds, latency and resource cost, and uncertainty from fixture sampling.

## Evaluator validation

`test_eval_harness.py` checks correct and incorrect exact answers, abstention on unanswerable and hallucinated answers, exact evidence/source matching, unit-tested code, and oracle self-check metrics. Baselines:

- `abstain`: deterministic abstain-all behavior, a safety/coverage reference.
- `random`: seeded sampling from a fixed answer pool without reading gold labels.
- `oracle`: reads fixture labels and emits reference claims/code. This is only a scorer self-check and must never be compared as a model.

These are sanity anchors, not capability baselines. The seeded random result is fixture dependent and not a meaningful chance level for a task suite with mixed answer formats.

## Known validity limits

1. The entire suite has nine synthetic examples, one code exercise, and no population validity. It does not select an architecture or establish general intelligence.
2. Exact string matching is brittle to valid paraphrase and easy to game. Human-reviewed equivalence rubrics or deterministic parsers are needed for broader tasks.
3. Claim support is a strict fixture lookup, not semantic entailment. It does not detect extra unsupported details inside a single claim; atomicity and source annotations need human review.
4. Self-authored cases are not contaminated by a public benchmark release, but future public benchmark scores may reflect training exposure. Keep evaluation splits private where appropriate, document data overlap, and use newly authored held-out cases for selection.
5. Goodharting is likely if repeated architecture or prompt selection optimizes this tiny suite. Freeze a development set, rotate a separately controlled test set, and include external task families only after license and reproducibility review.
6. Model-based judges can inherit prompt sensitivity, preference bias, verbosity bias, correlated errors and target-model bias. They add cost and nondeterminism. Prefer deterministic checks for exact arithmetic, code tests and evidence IDs; if a judge is necessary, calibrate against blinded human labels and report agreement and judge version.
7. Contradictions are represented by an expected conflict response; real source authority, time, provenance and confidence are not modeled.
8. No base next-token predictor is implemented because there is no trained model or licensed held-out corpus. No model architecture, benchmark suite or final capability claim is selected here.
