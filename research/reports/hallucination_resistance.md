# Batch 9: Hallucination Resistance, Verification and Abstention

Audited 2026-11-23. The environment recheck found 8,327,024,640 bytes host RAM, 5,753,774,080 bytes available at audit, no swap, 2 visible logical CPUs, no NVIDIA device, Python 3.14.2, 18 GiB free on `/`, and 37 GiB free on `/tmp`. The experiment began with 5,508 MiB available and peaked at 18,575,360 bytes RSS. System Python has no Torch/NumPy/Transformers; an isolated CPU PyTorch venv exists but was not needed. No model, corpus, or package was installed or downloaded.

## Failure taxonomy

"Hallucination" is not one label. The evaluation must separate:

| Failure class | What failed | Useful check |
|---|---|---|
| Unsupported fabrication | Claim has no adequate evidence in the available scope | Evidence coverage, answerability labels, abstention recall |
| Contradiction | Claim conflicts with current authoritative evidence | Claim-level contradiction check; preserve both sources |
| Staleness | Claim was once supported but is no longer valid | `as_of`, `valid_from`, `valid_to`, source version and freshness policy |
| Retrieval failure | Needed evidence was absent or ranked too low | Recall@k and evidence coverage separate from answer verification |
| Unfaithful answer | Answer adds content not entailed by supplied context, even if it might be true elsewhere | Faithfulness/claim entailment against provided context |
| Citation failure | Citation is missing, irrelevant, stale, or does not support the attached claim | Citation precision/recall at claim and span level |
| Overclaim / partial support | Compound sentence contains one supported clause plus unsupported detail | Atomicity audit, supported-claim precision and recall |
| Reasoning/execution error | Arithmetic, logical inference, code, or tool result is wrong | Deterministic calculation, schema validator, tests, or trusted tool result |
| Ambiguity / insufficient evidence | Sources conflict, are low authority, or do not determine an answer | Three-way/graded status; abstain or qualify, do not convert unknown into false |
| Miscalibration / over-refusal | Confidence is not predictive, or answerable questions are rejected | Brier/ECE plus risk-coverage and false-rejection rate |
| Evidence/tool injection | Retrieved text attempts to change policy or cause tool actions | Treat source as data; constrained tool schema and access boundary; adversarial fixtures |

Factual truth, support from supplied evidence, citation correctness, reasoning validity, and confidence calibration are distinct outcomes. A true answer with a bad citation can be ungrounded; an answer contradicted by a limited context may still be true in the world. Report these dimensions separately.

## Source and method landscape

Paper findings below are author-reported for their data/model/judge setups. Repository observations are source-level only. No upstream benchmark or model was run for this batch.

| Candidate | Evidence and source observation | Cost, failure mode, fit and disposition |
|---|---|---|
| TruthfulQA | Lin et al. construct adversarial questions around common human false beliefs. Existing audit of `truthfulqa/evaluate.py`, `metrics.py`, and `models.py` found mechanically-scored multiple choice alongside a generation path using GPT-3 judging/lexical metrics. [Paper](https://arxiv.org/abs/2109.07958); [pinned source](https://github.com/sylinrl/TruthfulQA/tree/d71c110897f5d31c5d7f309e7bc316c152f6f031). | Targets a narrow truthfulness construct; exposed public questions risk contamination. Judge and package dependencies burden a local offline test. Use as a later stress-test candidate, not the single factuality score. Apache-2.0 code; data terms require separate review. |
| HaluEval | Existing source audit of `evaluation/evaluate.py`: repository evaluation randomizes answer order and calls a model judge; the repo describes QA, dialogue and summarization tasks. [Paper](https://arxiv.org/abs/2305.11747); [pinned source](https://github.com/RUCAIBox/HaluEval/tree/b7253db3cdaa0ab2c382f92b26b390109174f77e). | Generated examples and borrowed task sources complicate provenance; unseeded selection and judge errors reduce repeatability. MIT code does not determine the terms for bundled/source datasets. Taxonomy reference only. |
| FActScore | Min et al. decompose long-form answers into atomic facts, retrieve support and score supported-fact precision. Source path `factscore/factscorer.py::FactScorer.get_score`; existing audit records its Wikipedia-scale index and multi-model dependencies. [Paper](https://arxiv.org/abs/2305.14251); [pinned source](https://github.com/shmsw25/FActScore/tree/f28272deffcf33efc1f1117d5479c10bb75221a9). | Claim extraction, retrieval and verifier errors compound; decomposition misses/merges claims. Source/data/index costs exceed this bench. MIT code; evidence corpus/model/data terms are separate. Useful claim-level design reference. |
| Ragas | Source `src/ragas/metrics/_faithfulness.py` generates statements then asks an LLM to judge whether context supports them. [Paper](https://aclanthology.org/2024.eacl-demo.16/); [pinned source](https://github.com/vibrantlabsai/ragas/tree/8c63f10d500c44efbf24fdb9e2920de2104c321c). | Adds extraction and judge passes, model/prompt bias, dependencies and nondeterminism; does not establish external truth or retrieval coverage. Apache-2.0 code. Defer until calibrated against human labels. |
| SelfCheckGPT | Manakul et al. sample multiple answer passages and estimate sentence inconsistency; `modeling_selfcheck.py` includes sampling-based MQAG, NLI/BERTScore and n-gram paths. The prompt variant uses an LLM to compare sentence/context, while NLI/BERTScore need separate models. [Paper](https://arxiv.org/abs/2303.08896); [pinned source](https://github.com/potsawee/selfcheckgpt/tree/19b492a2a380931bf1ed0ca94a9565c9aa7b03e1). | The original paper reports zero-resource black-box detection on its WikiBio setting, not guaranteed factuality. Sampling spends several generations and can repeat correlated errors; agreement is not source evidence. MIT. A distinct model/checkpoint is a dependency for learned detectors. Candidate for later calibrated study only. |
| FactBench / VERIFY | The ACL 2025 FactBench paper and canonical `launchnlp/FactBench` repo describe supported/unsupported/undecidable units and in-the-wild evaluation. At pinned source, `VERIFY/factuality_evaluation.py::FactEval.get_scores` runs atomic-unit extraction, decontextualization, evidence search, relevance and fact rating; `rate_atomic_fact_.py` includes search/model calls and retry paths. [Paper](https://aclanthology.org/2025.acl-long.1587/); [pinned repo](https://github.com/launchnlp/FactBench/tree/530d5710eb81851a79c04f36c78c1b4d8831dd3b). | Better distinction between unverifiable and false; retrieval and model labels remain noisy. Code calls external search/judging services in its default pipeline. At audited revision no top-level LICENSE file was present; do not reuse code/data pending explicit license review. |
| FaStFact | Wan et al. (Findings EMNLP 2025) report claim extraction with confidence-based pre-verification and document-level evidence retrieval; benchmark is 400 annotated long-form responses. Source `pipeline/score_metrics.py` computes precision and informativeness/claim-count-sensitive score. [Paper](https://aclanthology.org/2025.findings-emnlp.1295/); [pinned repo](https://github.com/Yingjia-Wan/FastFact/tree/a852d43f6ea465fea3732fbf2435aeb28f80e56f). | Addresses claim omission/excess and evidence quality at long-form scale, but extraction/search/verification can still fail and benchmark prompts/outputs have source-model provenance. Apache-2.0 code; benchmark constituents and annotations require separate terms review. No local reproduction. |
| Verify with Caution | Godbole and Jia re-evaluate factuality metrics across 11 datasets and report disagreement, paraphrase sensitivity, and sensitivity to evidence distance. [Paper](https://aclanthology.org/2025.findings-acl.1175/). | Strong warning against treating a single automatic judge as gold. Paper-level evidence only in this audit; no implementation or weights used. |
| Selective prediction / conformal abstention | Conformal Language Modeling proposes calibrated sampling/rejection for generated answer sets ([paper](https://arxiv.org/abs/2306.10193)). Newer work studies model-specific failures of entropy-only abstention ([Entropy Alone is Insufficient for Safe Selective Prediction in LLMs](https://arxiv.org/abs/2603.21172)) and representation-informed conformal abstention ([paper](https://arxiv.org/abs/2604.27914)). | Risk guarantees require exchangeability and calibration/test discipline; distribution shifts, adaptive retrieval, model changes, or policy changes can invalidate coverage/risk claims. Confidence is not evidence. Paper leads only; implementation/license/activity not verified here, so not adopted. |
| Deterministic verifiers | Arithmetic libraries, parsers, schema validators, exact source IDs, and unit tests can check bounded predicates without an LLM judge. Batch 1/8 local harnesses already include arithmetic, schema-like exact outputs, test execution and evidence IDs. | High precision only inside a known contract; cannot prove unrestricted factual entailment, tool correctness or source truth. Prefer for arithmetic/code/structured output, with OS sandbox before untrusted code execution. |

Existing detailed source records for TruthfulQA, HaluEval, FActScore, Ragas, Inspect AI, HELM, `lm-evaluation-harness`, RAGTruth and AbstentionBench remain in [`REPOSITORY_LEDGER.csv`](../REPOSITORY_LEDGER.csv). This batch adds pinned source records for SelfCheckGPT, FactBench, FaStFact, and paper-only calibration leads. Current upstream commit activity is recorded where checked; no clone, dependency install, benchmark corpus or model download was performed. `UniFact` was inspected as a recent paper/code lead but its repository did not expose a top-level license at the audited revision; it is not adopted.

## Local controlled fixture

[`claim_verification_tiny.json`](../experiments/fixtures/claim_verification_tiny.json) contains ten CC0 fictional cases: 2 supported, 2 unsupported (one irrelevant retrieved source), 3 contradicted (one includes obsolete and current evidence), 1 stale-only, and 2 ambiguous (conflicting records and low-authority-only). One maliciously worded source says to ignore policy and reveal a fictional secret; the verifier treats it strictly as text. Each case includes explicit claim keys, evidence relations, source validity dates and authority values.

[`claim_verification.py`](../experiments/claim_verification.py) runs four policy controls over the same candidate claims:

- `single_pass`: emits all claims at confidence threshold 0.
- `retrieval_only`: emits when any citation/source was returned; it does not check support.
- `echo_review`: one simulated no-new-evidence review that repeats the retrieval decision. This is a failure-control, **not a model or second-judge measurement**.
- `deterministic_verifier`: checks typed support/contradiction relations, date validity and authority; emits only supported claims, otherwise abstains.

The deterministic checker consumes structured annotations. It does not read arbitrary prose to infer entailment. Therefore 10/10 answer behavior validates the policy implementation against the fixture, not natural-language fact checking.

| Policy | Emitted / coverage | Supported precision | Citation correctness | Unsupported assertions | Supported recall | Correct abstentions | False rejection | Answer behavior accuracy | Whitespace token proxy / supported claim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single pass | 10 / 1.00 | .20 | .20 | 8 | 1.00 | 0 | 0 | .20 | 35 / 2 |
| Retrieval only | 9 / .90 | .22 | .22 | 7 | 1.00 | 1 | 0 | .30 | 31 / 2 |
| Echo-review control | 9 / .90 | .22 | .22 | 7 | 1.00 | 1 | 0 | .30 | 31 / 2 |
| Structured deterministic verifier | 2 / .20 | 1.00 | 1.00 | 0 | 1.00 | 8 | 0 | 1.00 | 13 / 2 |

Confidence-only risk/coverage was not monotonic in risk on this tiny fixture. Thresholds 0.0, 0.7, 0.8, 0.85, 0.9 and 0.95 produced coverage 1.00, .90, .60, .40, .20 and 0.00, while risk among emitted claims was .80, .78, .83, .75, .50 and undefined. At threshold .95, both supported claims were falsely rejected. The candidate confidence Brier score was 0.540 for all policy rows because each policy uses the same candidate scores. The fixture deliberately contains high-confidence wrong claims; it demonstrates why threshold selection needs held-out calibration and why Brier/ECE or confidence scores alone do not guarantee a useful policy. This curve is descriptive, not a conformal risk guarantee.

The structured verifier used 3.54 microseconds mean policy time per claim (p95 5.94 microseconds) in the final recorded run, and 4.57 microseconds of policy time per supported claim; the output claim used 6.5 whitespace tokens per supported claim on average. Process RSS was 18.6 MB. These timings exclude language generation, retrieval/index construction, annotation cost and external tools. The claim text whitespace count is a token proxy. The experiment uses one deterministic synthetic suite and one pass; latency/resource differences are not meaningful model comparisons.

Tests in [`test_claim_verification.py`](../experiments/test_claim_verification.py) cover status labels, irrelevant and injected evidence, stale/current conflict, false rejection, retrieval-only acceptance, echo pass, and risk-coverage monotonicity. The initial test invocation failed because the test did not add its sibling module path; the import was corrected and the failed run retained in `EXPERIMENT_LOG.csv`.

## Bounded verification and abstention design

Use a claim ledger between generation and response:

1. Split the draft into atomic, checkable claims and preserve their original text/spans. If decomposition confidence is low, treat the whole statement as unverified.
2. Classify each claim separately as `supported`, `contradicted`, `stale`, `insufficient`, or `ambiguous`. Retain source IDs, spans, authority, validity interval, retrieval timestamp and verifier/version.
3. For arithmetic/code/schema claims, route only typed inputs to deterministic computation, parser/schema validation or sandboxed tests; bind the output to the exact tool invocation.
4. Emit supported claims with citation spans; qualify partial evidence; reject contradicted/ambiguous claims; abstain on missing or stale evidence unless a safe as-of clarification resolves it.
5. Bound claim count, retrieval calls, reviewer calls, generated tokens and wall-clock. A model review may request another source or identify a missing claim but cannot make evidence true. Any adaptive evidence-acquisition policy must be included in calibration/evaluation.
6. Track selective risk (unsupported or incorrect claims among emitted claims) against coverage, plus false rejection, claim recall, calibration, citation precision/recall, latency and token/call cost. Select thresholds on a calibration split; report a held-out set and confidence intervals.

For decisions that use model confidence, separately calibrate answer correctness and evidence sufficiency. Do not equate verbal confidence, token entropy, self-consistency, retrieval rank or verifier probability with truth. If abstention cost differs by task, set an explicit loss/cost matrix and show the resulting curve. Excess abstention is visible as false-rejection and coverage loss, not hidden inside a high precision number.

## Reproduction

```sh
timeout 60s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v research/experiments/test_claim_verification.py
timeout 60s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 research/experiments/claim_verification.py --output research/experiments/results/claim_verification_tiny.json
```

The script requires at least 2,560 MiB `MemAvailable`; wrap execution in the external 60-second timeout. All values and the fixture SHA should be version-pinned before a model comparison. No external judge was available or invoked. No claim here is that a system can be hallucination-free.
