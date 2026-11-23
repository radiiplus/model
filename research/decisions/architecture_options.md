# Architecture Options and Evidence Matrix

Status: Batch 11 synthesis, 2026-11-23. This is a research design decision, not a capability claim or a claim that any design is generally superior. Paper/repository evidence is described in the earlier reports; numbers below are local measurements and remain preliminary.

## Evidence Base and Gaps

The research artifacts cover evaluation design, data curation, tokenization, a trainable decoder baseline, a simplified recurrent retention mixer, training/optimizer mechanics, synthetic SFT, lexical retrieval, claim verification, and CPU inference. The decision trail is in [`DECISIONS.md`](../DECISIONS.md), and run-level results are in [`EXPERIMENT_LOG.csv`](../EXPERIMENT_LOG.csv).

What local evidence supports:

- A 14,816-parameter, one-layer causal Transformer can train, checkpoint, reload, and memorize a repeated 12-token sequence. Training loss fell from 3.5358 to 0.000487 in 250 steps; this validates the loop and not general language ability.
- At 17,888 matched parameters on a narrow exact-copy setup, one Transformer recipe reached mean accuracy 0.857 over delays 4-125 across three seeds after 300 steps; the independently implemented normalized-retention recipe stayed at 0.143 (seven-way chance). This is one mixer and recipe, not a comparison of Transformer against all recurrent/SSM architectures.
- Byte-level BPE and Unigram fixtures round-tripped. Learned tokenizers shortened the synthetic held-out sequence, but no model was trained with them, so no capability or training-efficiency ranking follows.
- AdamW remains the fixed baseline. The three-seed Muon-hybrid run used different learning rates and was slower on the toy task; optimizer selection is unresolved.
- Concise synthetic arithmetic SFT averaged 0.583 held-out answer accuracy over three seeds; step-labelled output averaged 0.367. The abstention recall was variable. This does not measure natural-language reasoning or coding.
- A 10-document synthetic retrieval fixture showed bounded lexical retrieval, freshness/conflict checks, and abstention mechanics. It had 9/9 structured task success and 14.8 mean evidence-text tokens in the bounded-controller condition, but no language model generated or judged the answers.
- The structured verifier caught annotated stale/conflicting/unsupported cases; it did not perform natural-language entailment. Confidence-only thresholds were unreliable on the tiny fixture.
- Native PyTorch CPU profiling established KV parity and a small-file INT8 serialization probe. INT8 was slower on this model; its 61 synthetic probe tokens are not a quality holdout.

Missing evidence that blocks a production architecture choice:

- No realistic, license-cleared train/validation/test corpus or held-out model capability result for language, reasoning, coding, multilingual coverage, factuality, or abstention.
- No tokenizer comparison at matched parameter, raw-byte, or measured-compute budgets; vocabulary, normalization, and chat serialization remain open.
- No fair comparison among current Transformer variants and Mamba, RWKV, full RetNet, xLSTM, S4, Hyena, or another canonical sequence model. The local recurrence is educational and eager; the Transformer mixer run had no KV-cache decode, unlike Batch 10's later one-layer cache probe.
- No natural-language retrieval benchmark with paraphrases, source coverage labels, human-adjudicated claim support, or an LM backend; no combined model-plus-index RSS/latency result.
- No robust answerability calibration, independent factual judge, real abstention cost curve, secure generated-code tool, or adversarially validated reward/verifier.
- No quality result for DPO/KTO/RL, no optimizer comparison on varied data at tuned/equal wall-time settings, no web-scale data filtering/mixture result, and no measured curriculum gain.
- No supported GGUF export of this custom model, Q4/Q5/Q8 held-out quality study, true cold disk-start measurement, target consumer device, or Apple/ARM/GPU profile. Repository and paper performance claims have not been reproduced here.

## Three System Designs

| Design | Core | Possible capability strengths | Local quality evidence | Main cost/risk | Status |
|---|---|---|---|---|---|
| **A. Compact decoder Transformer** | One causal decoder, pretrained next-token objective; optional later SFT | General language modeling is the direct objective. Attention can use content-addressable token history; code/reasoning behavior would have to be learned from suitable data and measured. | Train-loop memorization smoke; synthetic copy test in the separate 17,888-parameter comparison. No local language, code, or general reasoning score. | Attention compute; KV grows with layers/context/batch. Simple to train and debug. | **Keep as reference and fallback.** |
| **B. Recurrent/SSM or hybrid mixer** | Fixed-state recurrent update or a canonical selective SSM, optionally mixed with attention | Bounded recurrent state can reduce state bytes as context grows; may make streaming decode attractive. | One simplified retention prototype: state 1,040 B across T=32/64/128; exact copy remained at chance after 300 steps. No canonical Mamba/RWKV/RetNet/xLSTM model was trained. | Sequential updates can make prefill/training slower; state compression can lose exact details. CPU support and optimized kernels vary by implementation/license. Hybrid adds a second set of failure modes. | **Defer replacement.** Keep one-mixer research branch only. |
| **C. Transformer plus selective retrieval/verification controller** | Design A plus optional external lexical index, provenance-aware evidence assembly, deterministic checks, and bounded abstention | Can provide current/source-backed facts without storing all knowledge in weights; exact math/schema/code tasks can be checked by suitable deterministic tools. Retrieval and model memory remain distinct. | Synthetic structured lookup only: top-3 recall 1.00, fixed-k precision 0.381; controller 9/9 on 9 structured cases, 14.8 mean text tokens, 12.3 ms total fixture run and 23.3 MB retrieval-process peak RSS. Claim verifier fixture: retrieval-only unsupported rate 0.778 versus annotated-rule verifier 0 on 10 cases. No LM answer generation or natural-language entailment. | Index/source maintenance, stale/conflicting evidence, false abstention, privacy and tool boundary; adds calls and context. Combined model/index resource cost not measured. | **Optional system layer; build an offline interface prototype only.** |

Design C is not a third neural architecture. It wraps A with external records and orchestration. A recurrent model's hidden state is not a searchable, attributable, editable document store. KV caching and quantization are execution/representation techniques, not evidence of more intelligence.

## Capability and Cost Matrix

`Not measured` means the local artifacts do not answer the question; upstream model/repository claims are not substituted. Resource results are from different micro-experiments and are not directly comparable unless noted.

| Dimension | A. Transformer | B. Recurrent/SSM candidate | C. Transformer + retrieval/controller |
|---|---|---|---|
| Language modeling | Mechanism is the baseline next-token objective; only one-pattern memorization measured. Held-out NLL/BPB: not measured. | Not measured on text; toy periodic task learned. | Same base-model capability as A; retrieval does not improve base-model next-token quality. |
| Reasoning | Not measured beyond tiny synthetic exact-copy task. | Not demonstrated; no broad reasoning task. | Not measured with an LM. A verifier can check only explicitly encoded constraints. |
| Coding | Not measured. No untrusted code execution integration. | Not measured. | Not measured; any execution service needs an OS-level sandbox before use. |
| Long-context information use | Copy accuracy 0.857 mean at delays 4-125 in a separate 17,888-parameter/3-seed toy run; not an actual long-context deployment test. KV grows linearly. | Exact-copy stayed at chance for the tested recurrence; fixed state can forget exact details. | External index can retrieve beyond prompt context in principle; fixture has 10 records only. Natural-language recall/coverage is unknown. |
| Factuality / abstention | Not measured on a real model. | Not measured. | Structured synthetic support checks worked; natural-language claim entailment and real false-abstention rate unknown. |
| Training RAM / method | 342,876,160 B peak RSS for one tiny 250-step run; one thread; 4,719 target tokens/s on one repeated sequence. Not a scale estimate. | Sequence study peak 366,149,632 B for a multi-run harness; not per-model allocation. Eager loop training took ~2.6-2.7 s per 80-step task run versus ~0.21-0.22 s for the Transformer in that implementation. | Model training is the same as A. Retrieval fixture is inference-only; no retriever training. |
| Inference RAM | Batch 10 process with FP32 and INT8 copies peaked at 345,866,240 B. FP32 export 66,107 B. Single tiny model/cache only. | 1,040 B recurrent state measured, excluding weights/activations/runtime. | Retrieval process alone peaked at 23,330,816 B. Combined model + index + caches not measured; index grows with corpus. |
| Latency / throughput | Train profile: 4,719 target tokens/s on T=12 synthetic sequence. Batch 10 cached FP32 decode: ~0.26-0.33 ms/token at prefix 8-24 (about 3,000-3,900 steps/s) in eager PyTorch. Not production throughput. | Prefill 7.15/16.72/30.01 ms at T=32/64/128. Recurrent decode 0.19-0.30 ms/token. Not a fair comparison against cached/fused Transformer at the same lengths. | Retrieval-only controller took 12.3 ms over 9 warm synthetic queries; excludes model generation. End-to-end latency not measured. |
| Parameter count | 14,816 baseline; separate matched sequence-mixer comparison used 17,888. Eventual size not chosen. | 17,888 in the matched educational comparison. Full SSM size not assessed. | Same model parameter count as A; index bytes are not model parameters. |
| State/cache growth | KV bytes follow `2 * batch * layers * tokens * kv_heads * head_dim * bytes_per_element`. Toy baseline: 256 B/token, FP32. | Tested recurrence state fixed at 1,040 B for sequence lengths 32-128; no claim for other models or exact-retention quality. | Model KV as A plus external index/cache. Index growth and invalidation cost not measured at scale. |
| Implementation burden | Low for toy research; medium for production tokenizer/export/kernel work. Existing PyTorch dependency is isolated and ~954 MiB. | High for a canonical fast SSM; medium for educational recurrence that was slow on prefill. | Medium for in-process BM25 + evidence schema; high for robust natural-language verifier, security, freshness and deployment. Avoid vector DB initially. |
| Portability / license | Local model code is project-authored; PyTorch is used under its own license. CPU path exists; consumer runtime/export not selected. | Independent prototype is project-authored. Candidate source licenses differ (e.g. Mamba Apache-2.0, xLSTM community license); review exact implementation before reuse. | Prototype can remain Python standard library. Data rights and source terms attach to each record; do not infer rights from index software. |

No single scalar score is appropriate. Reasoning, coding, evidence use, factuality and abstention remain separate tracks from language-model loss, latency, and index recall.

## Source-Backed Context

Earlier source surveys inspect and pin candidate code in the [architecture report](../reports/architecture_anatomy.md), [sequence-mixer report](../reports/sequence_mixer_landscape.md), [training report](../reports/training_efficiency.md), [memory report](../reports/memory_retrieval_landscape.md), and [runtime report](../reports/inference_runtime_landscape.md). These include Mamba selective-scan CPU caveats, RWKV and RetNet source paths, attention variants, and the serving/quantization systems. They are source maps, not proof that their published model results reproduce on this Codespace or on our eventual model.

The measurements above justify a baseline, diagnostic checks, and low-cost candidate experiments. They do not justify a claim of frontier capability, an optimum corpus/tokenizer/architecture, an “intelligence per parameter” ranking, or a deployment recommendation for a device not profiled.
