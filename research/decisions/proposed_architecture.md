# Proposed Architecture and Integrated Prototype Gate

Status: proposed research baseline, not final model selection. This keeps the smallest trainable, debuggable model in view and treats memory and orchestration as optional system components.

## Core Model

Keep the Batch 4 reference unchanged for controlled comparisons:

- Decoder-only causal Transformer; pre-LayerNorm; learned absolute positions; causal MHA; one layer in the current smoke config; 4x GELU MLP; residual paths; tied token embedding/output; next-token cross entropy.
- FP32 and AdamW for the tiny CPU correctness baseline. Do not promote Muon, mixed precision, low-bit weights, auxiliary losses, or a recurrent mixer by default.
- Preserve exact checkpoint, config, tokenizer, split, and seed manifests. The current vocabulary of 32 is an artificial synthetic harness, not a usable natural-language tokenizer.
- For the first text-bearing prototype, UTF-8 byte IDs are the lossless reference; BPE and SentencePiece Unigram remain candidates. Do not freeze a production vocabulary until a matched raw-byte/compute model study includes vocabulary parameters, Unicode/code slices, and decode behavior.
- Pretraining is the base capability stage. Add concise assistant-only SFT only if instruction behavior is missing and a broader held-out result justifies its data/compute. The toy SFT does not justify chain-of-thought traces, DPO, KTO, PPO, or GRPO.

This is deliberately a compact *reference*, not the eventual trained size. The Codespace data cannot answer the target parameter count, context, or full-training compute question.

## Optional Inference-System Layer

Keep the model stateless across requests by default. Add an opt-in, in-process lexical retrieval adapter for requests that need supplied or current evidence:

1. Store immutable source/chunk IDs, source URI, license/terms, content hash, publisher/authority, acquisition/version date, valid-time interval, duplicate lineage, and deletion/update version.
2. Use BM25/lexical search only to propose candidate evidence. Return compact snippets plus IDs and offsets; retrieval rank is not truth or entailment.
3. Require claim-to-cited-source checks, validity-time checks, and conflict detection before returning a grounded factual answer. Abstain or report the conflict when evidence is absent, stale, contradictory, or unverifiable.
4. Bound the controller to at most 2 retrieval and 2 verification calls, max 64 evidence-text tokens on the tiny fixture, max 16 output tokens, and an explicit wall-time deadline. Make budgets configurable per actual workload; these fixture settings are not deployment defaults.
5. Do not automatically write model-generated claims to persistent memory. Updates/deletes must be explicit and invalidate derived indexes and answer/KV caches. Keep user data isolated by access scope.
6. Use deterministic calculation/schema/unit checks where possible. Do not execute model-generated code without a separately designed OS sandbox; current subprocess timeouts are not a sandbox.

This controller is external retrieval/orchestration. It is not a recurrent neural state, learned test-time memory, or a model architecture change. The existing retrieval fixture validates plumbing only; it does not validate a natural-language evidence checker. Keep model output and evidence provenance separately visible to the caller.

## Inference and Deployment Stance

- Keep the canonical PyTorch checkpoint as the parity reference. Implement or select a runtime only after the actual model architecture, tokenizer, position encoding, tied weights, and KV representation are fixed.
- GGUF/llama.cpp is the leading CPU deployment candidate for later export work, not a selected or locally tested runtime. Validate logits, greedy IDs, tokenizer/chat serialization, cache equivalence, and exact math/code tasks before profiling.
- Do not select quantization from file size alone. Dynamic INT8 on the toy reduced the export 52.6% but ran slower and left embeddings, tied output and KV state FP32. Require a representative, held-out quality and memory/latency comparison before choosing Q8/Q5/Q4 or a different format.
- KV caching is a measured execution optimization for the toy and matched the full forward within 1.43e-6. Retain correctness tests; profile with larger contexts and target hardware before claiming speed benefit.
- Speculative decoding, early exit, GPU-serving stacks and Apple-Silicon branches remain future device-specific work. They do not improve model knowledge by themselves.

## Decision and Go/No-Go Criteria

**GO:** build a small offline adapter prototype that connects the frozen baseline interface to the existing lexical retriever and structured verifier, provided it remains reversible, dependency-free, bounded, and does not claim general QA. Keep direct-generation and abstain controls. The current fixture/controller can be used for interface tests only.

**NO-GO:** do not call the integrated system capability-improving, select a trainable Transformer/SSM hybrid, or choose production tokenizer/runtime/quantization until all of these gates pass:

- Fixed, provenance-checked, license-cleared training and held-out evaluation data; no train/eval duplicate leakage; human-reviewed claim/evidence labels for factuality tests.
- Baseline repeatability on a pre-registered task suite with at least 3 seeds where training is stochastic; report denominators, per-track scores, calibration/coverage, latency and memory.
- For any architecture or training change: mean gain of at least 5 percentage points on the target held-out capability metric, no seed regresses by more than 2 points, and no protected track (language, coding, evidence support, abstention) loses more than 2 points. These are project decision thresholds, not literature-established constants. If the effect is smaller, uncertain, or only a loss/latency gain, retain the simpler baseline.
- For a retrieval/verification layer: reduce unsupported-claim rate by at least 20 percentage points against the same model without verification, keep answerable-task accuracy within 3 points, false-abstention rate at or below 5%, and lower evidence-context use by at least 30% versus full-context input on a pre-registered retrieval workload. A high retrieval recall alone is not enough.
- Resource gates for this Codespace: one process, one worker, one or two threads; check at least 2,560 MiB `MemAvailable` before start; stop if it falls below 2,560 MiB or if RSS grows unexpectedly; hard timeout at 120 seconds per tiny experiment. Reserve >=2 GiB for the OS/editor/Codex. Do not use this as a full-training budget.
- For an inference runtime/quantization choice: exact exporter/runtime versions pinned; parity test passes; held-out task quality clears the same noninferiority gate; report single-request latency separately from batch throughput, peak RSS, cold/warm start, KV bytes and model-file size.

Thresholds must be registered before looking at final test outcomes. If a task is too small for a stable estimate, mark it preliminary and do not advance a production claim.

## Fallback

If tokenizer, SFT, recurrence, retrieval, verifier, or runtime experiments fail, retain the unchanged tiny causal Transformer with AdamW, standard next-token loss, no persistent learned memory, and direct generation behind the evaluator. For factual requests, the system may abstain; it must not invent support. This fallback is the comparison anchor, not a claim that the tiny model is useful.
