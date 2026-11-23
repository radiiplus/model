# Falsifiable Ablation Roadmap

All items are proposals; no result is implied. Keep the Batch 4 model/config/checkpoint as baseline and version every changed artifact. The listed thresholds are project decision rules to register in advance, not established universal standards.

## Before Any Ablation

- Replace synthetic-only capability evidence with a small, rights-cleared, representative corpus and disjoint train/dev/test splits. Group near duplicates and source/problem families before splitting.
- Freeze model/tokenizer/config, prompts, task versions, scoring code, evidence labels and decoding settings. Keep language modeling, instruction, reasoning, coding, grounded QA, hallucination/abstention, and resource tracks separate.
- Record exact environment, seed, token/byte budget, wall time, peak RSS, artifact hash and failure. Run at least 3 seeds for a major model decision when the bounded runtime permits. Do not tune on the held-out test set.
- For this Codespace: one process, 1-2 threads, one data worker, memory guard at 2,560 MiB available, 120-second hard timeout for each small run; stop sooner on memory pressure.

## Ordered Work

| Order | Experiment | Isolated change | Budget / metrics | Promotion rule |
|---|---|---|---|---|
| 0 | Freeze and repeat baseline | No model change; establish run-to-run spread on the fixed text/task suite | Tiny sequence lengths 32-128, batch 1-4, <=120 s per run; train/validation NLL in bits/byte, per-task accuracy, RSS, tokens/s | Do not compare modifications until baseline and evaluator are reproducible and data licenses are reviewed. |
| 1 | Tokenizer ablation | Byte IDs vs one byte-level BPE vs one Unigram; same raw content, model body/optimizer/initialization policy | First fixed raw-byte budget, then fixed measured compute; 3 seeds if feasible; separate vocab parameters, sequence length, BPB, Unicode/code/math exactness, task scores and latency | Keep compression only if held-out task quality is noninferior and end-to-end compute or inference cost improves. Exact round-trip alone is insufficient. |
| 2 | Transformer KV path / GQA follow-up | Add KV cache to unchanged baseline first; then change MHA KV-head count only | Prefill and one-token decode at T=32/64/128, cache bytes, parity error, latency, exact recall. Compare same checkpoint and CPU threads | Keep cache if parity holds and steady decode improves at target lengths. Consider GQA only if memory/latency gain is measured without >2-point protected-task regression. |
| 3 | One recurrent-mixer replication | Replace only the sequence mixer; match total parameters, tokenizer, optimizer tuning budget, total target tokens and evaluation | Use a canonical CPU-capable source or independent recurrence; three seeds; copy/retrieval at delays 1-128, prefill/decode, train time, state bytes, perplexity/task tracks | Do not promote on constant state alone. Require >=5-point mean gain on a preselected long-range task or a target-device latency/memory improvement with capability noninferiority. |
| 4 | Retrieval gate hypothesis H1 | No retrieval vs always-BM25 vs question/evidence-gated BM25+verifier; same frozen generator and same evidence corpus | 100-300 licensed/human-labeled claims with paraphrase, distractor, stale, conflict and no-answer groups; <=2 retrieval/2 verification calls; measure supported precision/recall, unsupported rate, false abstention, answer accuracy, context tokens, p50/p95 and RSS | Require >=20-point absolute unsupported-rate reduction, <=3-point answerability accuracy loss, <=5% false-abstention, >=30% context reduction vs full context, and p95 lexical/controller overhead <=50 ms at the tested fixture scale. If labels are too small, stay preliminary. |
| 5 | Answerability-head hypothesis H2 | Add a small auxiliary answerable/unanswerable head to the frozen/identical LM representation; compare with no head and deterministic evidence gate | Use disjoint paraphrase/source splits; three seeds; measure risk-coverage, selective risk, ECE/Brier, unsupported claims, false abstention, calibration transfer; <=120 s per toy run | Keep only if at >=80% answerable coverage selective error is >=20% lower relative to baseline, false-abstention <=5%, and no task accuracy loss >2 points. Must beat the deterministic policy on a held-out set; otherwise remove the head. |
| 6 | Concise SFT transfer | Add assistant-only-masked concise SFT; hold example family and active target-token budget fixed. Test worked format separately, not together. | 3 seeds; held-out prompt/problem families; arithmetic plus code and instruction controls; active tokens, pass rate, regression and abstention | Keep SFT only if it improves held-out instruction/task behavior by >=5 points without >2-point regression elsewhere. No preference or RL phase until a verified residual gap remains. |
| 7 | Export/quantization gate | Export exact baseline then compare FP32/BF16/Q8/Q5/Q4 where supported; one format at a time | Target-device cold/warm start, peak RSS, file size, prefill, steady decode, cache bytes, logits, each task track and calibration slice | Require parity contract and noninferiority under the project thresholds. Stop if unsupported format, exporter drift, or any critical track exceeds regression budget. |

Experiments 0-3 test the trainable model. H1 and H2 test reliability/system behavior; retrieval index success must not be counted as model intelligence. Experiments 6-7 are post-training/execution interventions, not architecture changes.

## Patch 6 Result

The three-seed generated-task ablation is recorded in [`../recursive/ablation.md`](../recursive/ablation.md), with full metrics and hashes in [`../recursive/out.json`](../recursive/out.json). The predeclared gate returned **no-go**: neither graph extrapolation nor graph-to-chain extrapolation improved by 5 points over the same dual-state checkpoint at K=1 in any seed. Exact match was 0% across the measured task conditions. Keep the plain TinyGPT baseline; do not add recurrent components to rescue this result. The run used only four updates and eight sampled examples per seed, so it rejects promotion under this fixture budget rather than recurrence in general.

## Hypotheses

### H1: Selective evidence retrieval improves supported answering per context token

**Hypothesis:** An explicit question/evidence-sufficiency gate that invokes BM25 only when external evidence is requested or needed, followed by a source/claim verifier, will reduce unsupported assertions while using less evidence context than always passing all documents. This is a project hypothesis, not a novelty claim; selective retrieval, RAG, attribution and abstention already have substantial literature.

- **Baseline:** identical frozen generator with no retrieval; secondary control is always-BM25 top-k; full-context condition as context-cost reference.
- **Changed variable:** controller trigger only. Same retrieval index, evidence snippets, generator, verifier, prompt, and budgets.
- **Success:** on a pre-registered, source-versioned test set, unsupported assertion rate falls by >=20 percentage points; answerable exact/verified accuracy falls by <=3 points; false-abstention <=5%; mean evidence context falls by >=30% vs full context; at most 2 retrieval + 2 verification calls.
- **Failure:** retrieval precision falls below the fixed baseline, unsupported rate does not improve, answerable accuracy drops >3 points, or false abstention exceeds 5%. Controller is removed or restricted to explicit evidence requests.
- **Compute budget:** CPU-only lexical index, one process/thread, <=120 s per experiment, <=2,560 MiB available-memory stop guard. Current 10-record/9-query fixture is only a smoke test and cannot pass the evidence gate.

### H2: A small answerability head improves selective risk beyond explicit evidence rules

**Hypothesis:** A separately calibrated answerable/unanswerable signal on the model's pooled representation may improve the risk-coverage tradeoff when natural-language evidence sufficiency is ambiguous. It must not replace source citations or deterministic checks.

- **Baseline:** same LM without a head, plus a deterministic evidence-present/conflict/staleness policy.
- **Changed variable:** auxiliary classification head and its loss; keep generator weights/architecture/training examples otherwise fixed in the first probe.
- **Success:** on held-out paraphrase/source families and at >=80% answerable coverage, selective error is at least 20% lower than the no-head model, false abstention <=5%, and answer accuracy declines by no more than 2 points. Calibration is measured on a separate calibration split and evaluated on test.
- **Failure:** no stable gain across 3 seeds, calibration shifts across source/domain/time groups, or the deterministic policy is equal/better. Remove the head; keep the explicit abstention path.
- **Compute budget:** a small head (<=100K additional parameters), tiny fixed-length examples, 3 seeds, <=120 s per training run and the normal memory stop condition. No separate confidence-model checkpoint.

Neither hypothesis is assumed novel. The literature review and source checks must be extended before publication or a novelty claim. A hypothesis can fail and still be useful evidence.

## Do Not Implement Yet

- Do not replace the Transformer with an SSM/retention hybrid based on linear-time or fixed-state claims alone.
- Do not combine GQA, RoPE, RMSNorm, SwiGLU, unusual attention, memory tokens, auxiliary heads, and alternative losses in one model run; isolate one variable.
- Do not download embeddings, install FAISS/vector DB/server stacks, or use dense/late interaction before BM25 fails on a licensed paraphrase fixture.
- Do not automatically write model outputs into external memory; no provenance-safe learned write/delete policy has been validated.
- Do not add RLM/recursive prompting, self-judging passes, repeated sampling, or speculative decoding without a model-backed cost/quality win and an execution sandbox.
- Do not add DPO/KTO/PPO/GRPO or chain-of-thought training until a real held-out behavior gap and reliable preference/verifier labels exist.
- Do not choose a corpus mixture, curriculum, quality classifier, dedup threshold, production tokenizer, low-bit runtime, or consumer target from synthetic measurements.
- Do not treat benchmark aggregate score, lower training loss, smaller checkpoint, or higher tokens/s as a capability score.

## Integrated Prototype Gate

Proceed only with an offline, reversible wrapper around the unchanged baseline and existing lexical/provenance interface. Before calling it an integrated evaluation prototype, require (1) adapter tests for direct, retrieved, conflict, stale, no-answer, and abstain paths; (2) fixed, independently adjudicated evidence labels; (3) bounded call/context/time budget; (4) no untrusted code execution or automatic memory writes; and (5) reproducible end-to-end latency/RSS with baseline, full-context, and retrieved conditions. If these are missing, keep components as isolated prototypes. No architecture or quality claim follows from integration passing.
