# Decisions

## Batch 1: Evaluation before architecture

- **Decision:** implement a small model-agnostic local evaluator before choosing any model or architecture.
- **Reason:** scorer behavior, abstention, evidence use, code checks, resource costs and per-track reporting should be tested independently of model quality.
- **Implementation:** Python standard library only; hand-authored synthetic fixtures; prediction JSONL adapter; deterministic abstain, seeded random and label-aware oracle self-check.
- **Not decided:** final benchmark suite, architecture, training data, judge model, or deployment backend.
- **Validity:** nine cases are evaluator smoke fixtures only. Do not report their score as intelligence, general capability or comparative evidence.
- **Next gate:** complete source/license review, expand scoring tests, define protected held-out sets, then compare model backends only with repeated seeds and fixed budgets.

## Batch 2: Data curation and mixture

- **Decision:** keep the raw synthetic fixture plus metadata; compare normalized exact hashing, small MinHash, and SimHash, and preserve reason-coded outputs.
- **Measured observation:** at selected tiny-fixture thresholds, exact hash removed 2 duplicate records; MinHash and SimHash each removed 3 records including one punctuation-edit near duplicate. No labeled unique item was removed at those thresholds. Looser thresholds quickly overmerged rare facts and boilerplate-heavy notes.
- **Quality filter:** only the deliberately repeated-token noise item was filtered. The heuristic score is diagnostic and is not a training-quality label.
- **Not decided:** source corpus, production dedup thresholds, language/PII/safety filters, final mixture shares, or curriculum.
- **Next gate:** clear each source and constituent license, then run equal-token/compute ablations with per-domain held-out metrics and at least three seeds where practical.

## Batch 3: Tokenizer and input representation

- **Decision:** retain UTF-8 byte IDs as a coverage/correctness baseline and keep byte-level BPE plus SentencePiece Unigram as candidates for later controlled model tests.
- **Measured observation:** on 7 synthetic held-out records (629 UTF-8 bytes), byte baseline used 629 IDs; trained byte-level BPE variants used 520–532 IDs and all six tested methods round-tripped exactly. The attention-pair proxy was 0.716–0.747 for the learned BPE methods. This is only compression and smoke evidence.
- **Not decided:** tokenizer family, vocabulary size, normalization, regex, special-token inventory, or final chat serialization.
- **Reason:** compression does not demonstrate capability, training stability, or consumer inference speed. The data are too small to produce stable merges or an unbiased quality comparison.
- **Next gate:** freeze realistic licensed raw-text splits and compare byte/BPE/Unigram under equal raw-byte and equal-compute training budgets after a baseline model exists.

## Batch 4: Decoder baseline and architecture anatomy

- **Decision:** establish a clean one-layer, decoder-only Transformer as the fixed baseline: learned token/position embeddings, pre-LayerNorm, causal MHA, 4x GELU MLP, residual connections, tied embedding/output weights and next-token cross entropy.
- **Measured observation:** the 14,816-parameter CPU model reduced loss on one repeated 12-token synthetic sequence from 3.5358 to 0.000487 nats/token in 250 AdamW steps. It generated in-range token IDs; checkpoint loss round-trip was exact.
- **Resource observation:** CPU-only PyTorch `2.15.0+cpu` was installed into an isolated 954 MiB venv after the user authorized it. The bounded run peaked at about 327 MiB RSS and took about 0.58 s with one Torch thread.
- **Not decided:** tokenizer, learned-position versus RoPE/ALiBi, MHA versus GQA/MQA, LayerNorm versus RMSNorm, GELU versus gated MLP, tying versus untied head, or any memory/retrieval/RL/runtime optimization.
- **Validity:** this is a training-mechanics and memorization smoke test, not language modeling capability. No held-out or generalization claim.
- **Next gate:** freeze a licensed tiny train/validation fixture and evaluate this unchanged baseline before running one-factor architecture ablations.

## Batch 5: Alternative sequence mixers and memory

- **Decision:** preserve the Batch 4 Transformer unchanged. Keep an independently written normalized retention recurrence as an experimental candidate only; do not replace the baseline.
- **Measured observation:** at exactly 17,888 parameters, both Transformer and retention learned a short periodic next-token pattern. The 80-step retention model remained at chance on exact-copy probes. In a separate 300-step follow-up, Transformer reached 0.857 mean exact-copy accuracy at delays 4-125 across three seeds, with per-seed scores 0.571/1.0/1.0; retention remained at chance. This result applies to this simplified recurrence and training recipe only.
- **Resource observation:** retention persistent state was 1,040 bytes for the tested one-layer width, independent of sequence length. Its eager recurrent prefill was substantially slower on CPU than Transformer prefill; its one-step decode was faster than the uncached Transformer prefix-recompute path, not a KV-cached/fused Transformer.
- **Not decided:** Mamba/RWKV/RetNet/xLSTM/S4/Hyena/linear attention/GQA/MQA/sliding-window architecture choice. No paper headline or asymptotic complexity is treated as a local performance result.
- **Next gate:** make a retrieval-capable recurrent baseline under matched training budget and validate on held-out exact-recall/adversarial tasks before considering longer-context or memory-compression tradeoffs.

## Batch 6: Training engine and optimizer

- **Decision:** keep PyTorch AdamW as the fixed training baseline. Retain the PyTorch Muon+AdamW hybrid as a candidate for a separate, tuned comparison; do not replace the baseline from this toy result.
- **Measured observation:** on a fixed 14,816-parameter periodic task, three-seed mean final training loss after 200 updates was 0.004315 for AdamW and 0.004518 for Muon hybrid. Mean runtime was 0.614s versus 1.166s; measured optimizer tensor state was 118,592 versus 69,424 bytes. Rates differed (AdamW 0.003, Muon 0.02), so this does not prove an optimizer is generally better.
- **Training-loop checks:** batch-4 gradients versus four microbatches matched after one AdamW update within 2.98e-7 maximum parameter difference. File-serialized resume at step 100 reproduced the uninterrupted step-200 model state hash and loss exactly for both methods.
- **Precision observation:** CPU BF16 and FP16 autocast produced finite losses but were slower than FP32 on this tiny model (4.73ms and 6.27ms/update versus 3.07ms). No general CPU speed or large-model stability claim.
- **Not decided:** optimizer learning rates/schedules, Sophia, 8-bit CPU optimizers, mixed-precision training policy, activation checkpointing or large-scale training engine.
- **Next gate:** compare tuned optimizers on a small varied licensed train/validation set at both matched token and wall-time budgets, with at least three seeds and source/license-checked implementations.

## Batch 7: Capability development

- **Decision:** keep next-token pretraining as the base stage and treat SFT as an optional, evidence-gated stage for instruction/task behavior. Do not add DPO/KTO or PPO/GRPO by default.
- **Measured observation:** a tiny assistant-only-masked arithmetic SFT improved held-out exact answer accuracy from 0 at random initialization to 0.583 mean for concise outputs and 0.367 for step-labelled outputs on a 20-pair held-out split (three seeds). Step-labelled outputs used 580 vs 180 supervised completion tokens per epoch and did not show an accuracy gain. This is synthetic token-ID arithmetic, not natural-language reasoning.
- **Abstention:** missing-input recall varied substantially by seed (0.567 concise, 0.633 step-labelled average; ranges included zero); false-abstention rate was zero on the held-out answerable pairs. Learned uncertainty is not yet robust; deterministic verification can enforce missing inputs.
- **Data audit:** the initial 80/20 split held out the only example of answer class zero. That first result is preserved but excluded from the final comparison. Corrected split keeps every answer value in training and has zero pair overlap.
- **Not decided:** final instruction corpus, solution style, preference optimization, reward model, online RL, or reasoning curriculum.
- **Next gate:** use larger diverse licensed tasks with held-out generator/template/problem families; compare concise/worked formats at matched examples, tokens and measured compute, then add preference/RL only if a verified held-out gap justifies the cost.

## Batch 8: External memory and retrieval

- **Decision:** keep external lexical retrieval as an optional system component candidate; do not add an embedding model, vector server, recursive controller or learned test-time memory by default. Preserve the language model as the source-independent baseline.
- **Measured observation:** on 9 synthetic structured queries, no context answered 3/9 (the three correct abstentions); full context, overlap top-3, BM25 top-3 and bounded controller each answered 9/9 under the exact structured verifier. Mean text context was 105, 28.9, 28.9 and 14.8 tokens respectively. Both lexical methods reached recall@3 1.0 and fixed-denominator precision@3 0.381 over 7 evidence-bearing queries.
- **Control/provenance:** bounded controller used at most 2 retrieval and 2 verifier calls, max 22/64 context-text tokens, max 16 output tokens, and remained under its 50 ms timeout. The verifier rejects stale evidence and contradictory sources and validates cited content, not merely search success.
- **Evidence limits:** all queries are synthetic and structured; the answer resolver is not an LM. No embedding, RAG model, external server, Titans code or recursive LM was run. The result demonstrates fixture/controller mechanics only.
- **Not decided:** lexical versus dense/late-interaction retrieval at realistic scale; embedding encoder; chunking/index server; automatic memory write policy; learned memory or RLM use.
- **Next gate:** create a licensed paraphrase/no-answer/conflict corpus, run with a fixed model backend and protected held-out evidence, and compare recall, citation support, abstention, context, latency and CPU memory before adding a model/server dependency.

## Batch 9: Hallucination resistance and abstention

- **Decision:** keep claim-level evidence checks and selective abstention as system-level candidates; use deterministic verification for exact arithmetic, schemas and tested code. Do not claim that a second pass, self-consistency, retrieval, confidence threshold or prompt alone proves truth.
- **Measured observation:** on 10 synthetic claims, the structured relation/date/authority verifier emitted 2 supported claims with 1.00 precision and recall, abstained on 8 unsupported/contradicted/stale/ambiguous cases, and falsely rejected 0 supported cases. Retrieval-only emitted 9 claims at 0.22 precision and 0.78 unsupported rate; a one-step echo control was identical. Confidence-only thresholds had nonmonotone empirical risk; at threshold .95 coverage was zero and both supported claims were falsely rejected.
- **Evidence limits:** the verifier consumed hand-authored structured relations, not natural-language entailment. No language-model backend or reviewer was present; echo control does not estimate model self-review. Latency excludes generation/retrieval. This is a policy-mechanics test only.
- **Not decided:** claim segmentation, open-domain verifier, confidence feature, calibration method, risk target, domain thresholds, or final hallucination benchmark.
- **Next gate:** evaluate a fixed model on a larger human-adjudicated, source-versioned claim set; separate factual truth, context faithfulness, citation correctness, reasoning validity and abstention cost. Calibrate and select thresholds on separate data, then report held-out risk/coverage and subgroup false rejection.

## Batch 10: Inference runtime and consumer profiling

- **Decision:** preserve the PyTorch TinyGPT as the correctness reference. Select no serving runtime yet; use GGUF/llama.cpp as the leading CPU deployment candidate only after an architecture exporter and held-out parity evaluation exist.
- **Measured observation:** dynamic INT8 quantized the four Linear modules only, leaving embeddings/tied head and KV state FP32. Inference export shrank from 66,107 B to 31,323 B (52.6%). On 61 synthetic targets, argmax agreement was 1.00 and accuracy delta 0; logit MAE was .0154 and max .2051. Manual FP32 KV output matched full-forward output within 1.43e-6 and used 256 B per cached token on this one-layer model. Cached FP32 decode-only transitions took 0.26-0.33 ms/token versus 0.48-0.61 ms/token with full-prefix recomputation in this eager prototype. This does not predict optimized runtimes.
- **Performance/quality limits:** INT8 was slower than FP32 on these tiny operators, PyTorch's dynamic quant API is deprecated, and the quality probe is synthetic and not held out. A fresh Python+PyTorch import proxy took 1.592 s with warm filesystem cache; true cold-disk startup, GPU/Apple/ARM, long context, concurrency and energy were not measured.
- **Not decided:** GGUF format conversion, Q4/Q5/Q8 target, quant calibration, cache dtype, target device/backend, batching/serving policy, speculative/early-exit use.
- **Next gate:** fix target consumer hardware and architecture/tokenizer export; compare native reference against pinned runtime/format builds and quant levels using held-out next-token, coding, reasoning, grounded QA, abstention, RSS, mmap/startup, prefill/decode, cache, and one-user vs batched profiles.

## Batch 11: Synthesis and architecture options

- **Decision:** retain the unchanged decoder-only Transformer as the research baseline and fallback. Treat external lexical retrieval, provenance checks, deterministic verification and abstention as optional system-level components. Keep the tested recurrence as a separate research branch; do not replace the Transformer or assemble a multi-technique hybrid.
- **Reason:** the local evidence validates mechanics on small synthetic fixtures, not broad capability. The one recurrent recipe remained at seven-way chance on exact copy; retrieval and claim verification used structured hand-authored facts without an LM; SFT and quantization studies are synthetic and narrow.
- **Prototype gate:** GO only for a reversible offline wrapper around the frozen baseline and existing lexical/provenance interface, with bounded budgets and adapter tests. NO-GO for production capability claims, learned memory, an integrated trainable hybrid, selected tokenizer/runtime/quantization, or RL until the preregistered evidence gates in `decisions/proposed_architecture.md` and `decisions/ablation_roadmap.md` pass.
- **Proposed hypotheses:** H1 compares no retrieval, always-BM25 and evidence-gated BM25 plus verification on source-versioned held-out claims. H2 compares a small calibrated answerability head against the same model and deterministic evidence policy. Both include explicit success and failure thresholds; neither is a novelty claim.
- **Evidence limits:** no licensed representative language corpus, model-backed natural-language retrieval/factuality evaluation, code-task model result, full canonical SSM comparison, target consumer hardware profile, or held-out quantization-quality result exists. Project decision thresholds are provisional rules, not literature constants.
- **Next gate:** establish the held-out data/evaluation contract and repeat the unchanged baseline before any architecture or training change; only then run the ablation roadmap one variable at a time.

## Batch 12: Integrated tiny prototype and reproducible ablations

- **Decision:** keep the byte-tokenized one-layer Transformer/AdamW text path as a separate smoke baseline; keep explicit data-split checks, deterministic byte metrics and resumable checkpoint code. Keep BM25 plus the structured verifier and bounded call budget as a system-mechanics candidate. Reject retrieval without verification as a default.
- **Measured observation:** across 3 seeds, 60 steps each, validation mean fell from 8.001 to 6.079 bits/byte and held-out test mean from 7.989 to 6.879 bits/byte; test next-byte accuracy rose from 3.61% to 8.11%. Parameter count was 21,984; latest step times were 3.38-3.79 ms; process peak RSS was 344,686,592 bytes. This is an 18-document synthetic byte fixture, not language capability evidence.
- **System result:** on 9 structured synthetic queries, no-context abstention scored 3/9; full context and verified BM25 scored 9/9; unverified BM25 top-1 scored 5/9 and had 44.4% incorrect emitted answers; bounded BM25 scored 9/9 at mean 14.8 context-text tokens and <=2 retrieval/verification calls. The resolver is structured and is not LM generation or natural-language entailment.
- **Engineering checks:** train/validation/test IDs partitioned the 18 documents 11/3/4 with no exact or normalized text hash crossing splits. A regression test verifies cross-split duplicate rejection. Cooperative interruption at step 10 and resume to 20 exactly matched uninterrupted model state. Deterministic inference and inference export roundtrip passed. 9/9 tests passed. Swap activity and 2,560 MiB available-memory guards were enforced; no swap exists on the measured host.
- **Keep/revert:** keep byte encoding as a lossless reference and modular interfaces; keep bounded retrieval/verifier for offline plumbing tests only; reject unverified retrieval; defer architecture changes, answerability head, SFT, dense retrieval, quantization and runtime selection.
- **Next gate:** acquire a small licensed source-family-disjoint text/evidence set with human claim labels. Connect a real model generator to the evidence-system adapter only after its output contract is testable; then measure unsupported claims and false abstention against no-retrieval and full-context controls. No general capability claim follows from this prototype.

## Batch 13: Final training specification and acquisition plan

- **Decision:** do not begin genuine pretraining or rent expensive compute yet. Retain the unchanged causal Transformer and AdamW as the reference. The first genuine pilot after data/evaluation/tokenizer/hardware gates is 13,783,296 parameters and at most 100M tokens; checkpoint at 10M and require a review before continuing.
- **Resource plan:** the editable calculator covers 13.8M, 51.2M and 99.2M parameter configurations. Wall scenarios use assumed sustained useful FLOP/s, not hardware benchmarks. Provision a single 16-24 GiB GPU, 64 GiB RAM and 2 TB NVMe only after a 1,000-5,000-step target-hardware smoke passes memory, throughput and restore gates.
- **Data plan:** use the provisional Batch 2 mixture only as a controlled starting matrix, with 120M distinct usable tokens for the 100M pilot plus reserve and full rights/provenance review. No corpus or full training run was started.
- **Status:** not ready for genuine pretraining. Blockers are source-level license/provenance clearance, realistic frozen and contamination-audited evaluation, model-level tokenizer choice, production-capable packing/training/checkpoint code, and measured target-hardware throughput/memory. A small CPU mechanics smoke remains appropriate.
- **Caveat:** 10M/100M-token checkpoints are process and learning-curve experiments, not promises of broad language, reasoning, code or factual capability. Token/parameter ratios and GPU-time scenarios are planning heuristics only.

## Patch 1: CPU memory and lazy windows

- **Decision:** sample byte windows lazily from document arrays and prefix offsets; keep training explicitly on CPU. Use the minimum of host `MemAvailable` and finite cgroup headroom, inspect v1/v2 memory controllers, and fail closed when a finite limit lacks readable usage.
- **Verification:** Batch 12 tests passed 9/9, TinyGPT tests 4/4, and the dependency-free baseline tests 3/3. A seeded sampler smoke selected 8 windows of length 33 from 99,968 possible windows in 0.000692 s; the loader stores a `range` index rather than per-window tensors. Simulated 1 MiB available memory was refused before training.
- **Environment snapshot:** host available 6,750,887,936 B; cgroup v2 `/sys/fs/cgroup/init` current 3,119,828,992 B and parent `/sys/fs/cgroup` current 3,156,475,904 B; both limits were `max`, so effective available memory was host-limited at 6,750,887,936 B. Peak RSS was 241,909,760 B; the 2,560 MiB guard remained active. Full snapshot and source hashes are in [`resource.json`](experiments/results/resource.json).
- **Limits:** the sampler smoke uses a generated integer sequence, performs no model training, and makes no quality claim. This Codespace exposed cgroup v2 only; v1 unlimited/finite parsing is unit-covered but was not exercised against a live v1 mount. The host had no finite container memory cap during this run.
