# Final Training Specification and Resource Acquisition Plan

Status: planning specification, 2026-11-23. No full corpus acquisition or full training was started. The recommendation is a staged, single-GPU research pilot; it is not a capability promise. Evidence classes remain separate: upstream paper/repository claims are cited in prior reports, local measurements are cited here as such, and larger-scale values are calculated estimates.

## Verdict

**Not ready for the first rights-cleared pretraining run.** Ready for bounded code and synthetic-data mechanics experiments. Blockers are a reviewed training corpus and its license/provenance records, a realistic frozen evaluation suite with held-out source families, tokenizer selection under a model-level comparison, a production-capable trainer/data packer, and the actual training/deployment machine. Batch 12's trainer is a 60-step synthetic byte-window harness, not a pretraining engine: it does not implement a scalable multi-shard sampler/packer, selected subword tokenizer, long-context kernel, or production checkpoint rotation. Do not rent a GPU or begin paid pretraining before those gates pass.

The first genuine pilot after the gates should use the preserved decoder-only Transformer family and AdamW, with the 13.8M configuration below. Train a first 10M-token checkpoint for pipeline validation, then continue to 100M tokens only if data, loss curves, evaluation and resource measurements pass. This is a pilot with about 7.3 training tokens per parameter, below the roughly 20-token/parameter compute-optimal heuristic reported by Hoffmann et al.; that ratio is not a universal prescription, particularly for a small, narrow-data run. An extension toward 275M tokens would reach that heuristic for this pilot, but is not approved in advance. The paper's 70B-scale result does not establish an optimum for this project. [Hoffmann et al.](https://arxiv.org/abs/2203.15556)

## Current Research Host

Rechecked 2026-11-23: `/home/codespace` has 8,131,860 kB RAM, 5,576,516 kB available, no swap, two visible logical CPUs (Intel Xeon Platinum 8370C), and no available `nvidia-smi`. Root has 18 GiB free and `/tmp` 37 GiB free. This is still a CPU-only Codespace; it is suitable for bounded preprocessing and tiny mechanics checks, not the proposed 100M-token training host. Recheck the acquired machine before every run.

## Evidence and Decisions

- **Local measurement:** the Batch 4 14,816-parameter model memorized one periodic sequence. The Batch 12 21,984-parameter byte model reduced held-out synthetic bits/byte across three seeds, but had only four held-out documents. Neither is a broad language or capability result. See [architecture anatomy](../reports/architecture_anatomy.md) and [integrated profile](../reports/integrated_tiny_prototype.md).
- **Local measurement:** AdamW is the only retained optimizer baseline; Muon hybrid was slower on the small synthetic task and used a differently tuned rate. BF16/FP16 autocast was slower on that CPU fixture. Neither result predicts GPU training. See [training efficiency](../reports/training_efficiency.md).
- **Local measurement:** byte IDs round-trip; learned BPE/Unigram compressed the tiny fixture. No tokenizer was trained into a model. Keep byte IDs as coverage control and compare one byte-level BPE and one Unigram tokenizer before full pilot data packing. See [tokenizer spec](../reports/tokenizer_spec.md).
- **Local measurement:** structured retrieval and verifier checks passed synthetic fixtures only; they did not involve LM-generated claims. Keep retrieval, evidence verification and abstention as a separate optional inference system. Do not include them in pretraining compute or score them as model capability.
- **Decision:** unchanged causal Transformer, next-token cross entropy, AdamW, fixed data order/sampler, FP32 reference implementation. For the first GPU pilot, BF16 autocast with FP32 master parameters and FP32 Adam moments is the preferred candidate only after a short hardware smoke and numerical parity check. Keep FP32 as the correctness fallback. No Muon, RL, learned memory, SSM replacement, auxiliary loss, quantization-aware training or speculative decoding in this run.

## Model Configurations

All configurations use tied input/output token embeddings, learned absolute positions, pre-LayerNorm, causal MHA, a 4x GELU MLP with biases, standard next-token cross entropy, and the Batch 4 implementation's parameterization. These are sizing configurations, not a claim that the tokenizer or learned positions are final. Require `d_model % n_heads == 0`; the listed head widths are 64. At context beyond the configured maximum, the current reference implementation must be changed and retested before use.

| Option | Purpose | Vocabulary | Context | Layers | Width | Heads | MLP width | Exact parameters |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Pilot | First genuine run | 16,384 | 1,024 | 4 | 384 | 6 | 1,536 | 13,783,296 |
| Target candidate | Scale only after pilot gates | 16,384 | 2,048 | 8 | 640 | 10 | 2,560 | 51,185,920 |
| Stretch candidate | Requires new VRAM/profile evidence | 16,384 | 2,048 | 12 | 768 | 12 | 3,072 | 99,211,776 |

The exact count for the current tied, biased, learned-position block is `V*d + C*d + L*(4*d^2 + 2*d*d_ff + d_ff + 9*d) + 2*d`. It includes token and position embeddings, QKV/output projections, both MLP matrices and biases, two per-block LayerNorms, and final LayerNorm; the tied output matrix is counted once. Change the formula if attention, normalization, MLP, position scheme, or tying changes. A production tokenizer with a different vocabulary changes parameter count materially: the 16,384-token embedding alone is 6.29M / 10.49M / 12.58M parameters in the three options.

Reproduce/edit the planning sheet (no packages required):

```sh
python3 research/prototype/resource_estimator.py --output research/decisions/training_resource_scenarios.csv
python3 research/prototype/resource_estimator.py --model pilot --tokens 275000000 --sustained-tflops 1.0
```

The first command emits all three configurations across the three assumed hardware-rate scenarios. Override token count and useful sustained TFLOP/s on the command line; edit `CONFIGS` and `PROFILES` in the calculator when changing dimensions or scenario assumptions.

## Tokens and Mixture

### Run targets

| Stage | Model | Tokens | Purpose | Decision after stage |
|---|---|---:|---|---|
| Software smoke | Existing 22K model | 10K-100K | Validate data order, loss masks, checkpoint/resume, memory guard and export; synthetic data only | No capability conclusion; fix pipeline defects |
| First genuine pilot | 13.78M | 10M, then 100M cumulative | Measure real held-out learning curves and per-domain metrics; about 0.73 then 7.3 tokens/parameter | Stop at 10M on any data/eval/resource issue; extend only at the pre-registered gate |
| Optional pilot extension | 13.78M | Up to 275M cumulative | Roughly 20 tokens/parameter heuristic, only if validation still improves and data is not being repeated excessively | New approval at 100M checkpoint; not automatic |
| Later scale test | 51.19M | 300M initial ceiling | About 5.9 tokens/parameter; tests scaling/throughput, not compute-optimal training | Requires a fresh resource and quality review |
| Later target study | 51.19M | Up to 1B | About 19.5 tokens/parameter | Do not schedule until 300M study and downstream checks justify it |

The token ratios are planning points, not capability thresholds. The 10M checkpoint is primarily a data/optimizer smoke. The 100M run is a research pilot that can reveal whether training is stable and whether held-out loss moves; it cannot establish broad reasoning, coding or factuality. Public benchmark exposure, data contamination, tokenizer effects and the tiny pilot model all limit interpretation.

### Mixture starting matrix

Use the Batch 2 provisional proportions only as an initial pilot matrix, sampled by tokens after quality and rights review. At 100M tokens the planning allocation is 35M general web/reference prose, 15M books/long-form writing, 15M code/technical documentation, 15M math/science/educational text, 10M dialogue/forums and 10M multilingual/low-resource text. This is **not** an approved source list or evidence-backed optimum. It is drawn from the [data pipeline spec](../reports/data_pipeline_spec.md).

For the initial corpus target, acquire at least 100M distinct usable tokenizer tokens plus a 20% reserve (120M total) so the pilot does not silently repeat examples. Keep domain-specific reserve and validation material separate. Each source must have a reviewed license expression, attribution, allowed-use decision, source ID/revision/date, raw hash, transform version and duplicate-component ID. Do not infer training permission from the data-processing repository's software license. Remove benchmark/test overlap before packing. Keep a reversible reject manifest with reason codes. Use an independently versioned validation/test set; never train on the evaluation set.

Storage planning assumptions (not measurements): average 3-5 UTF-8 bytes per BPE token is unknown until the tokenizer and corpus are fixed. For 120M tokens this is 360-600 MB raw text before metadata; reserve 2-4x for JSONL/metadata, cleaned shards, temporary dedup indexes and manifests. Packed 16-bit token IDs at 120M tokens are about 240 MB before shard/index overhead. Use a 2 TB local NVMe working disk to allow raw/derived copies and future growth; retain an encrypted, access-controlled independent backup of raw licensed data and manifests where license terms allow. Do not put non-redistributable raw data in public artifacts.

## Memory and Checkpoint Accounting

The [resource calculator](../prototype/resource_estimator.py) emits editable CSV assumptions and run estimates. Its estimates exclude framework/library overhead, fragmentation, dataloader buffers and implementation-specific workspaces; add them through the explicit reserve below. Convert decimal GB and binary GiB carefully.

### Persistent training tensors

- **FP32 AdamW reference:** weights `4N`, gradients `4N`, first and second moments `8N`, total persistent floor `16N` bytes. The training checkpoint usually contains weights plus optimizer moments (`12N`), not gradients. An atomic save can temporarily keep both the old and new checkpoint.
- **BF16 autocast with FP32 master weights and moments:** typical mixed-precision path still has roughly `4N` master weights + `4N` gradients + `8N` Adam states = `16N`; BF16 compute casts/temporary buffers add extra memory. If the implementation keeps BF16 parameters and BF16 gradients plus FP32 master weights and FP32 moments, budget `16N`; if gradients are FP32 too, budget `18N`. Confirm actual optimizer tensor dtypes from the selected framework.
- **FP16:** same accounting as BF16 when master weights and FP32 moments are kept, plus loss scaler state and temporary casts. FP16 needs a dynamic loss scaler and finite-gradient checks. BF16 is preferred when the selected GPU supports it well; do not assume CPU autocast is faster.
- **Muon:** not selected. Its matrix momentum and AdamW state depend on parameter grouping; do not use a blanket bytes/parameter estimate. Recalculate from actual tensor dtypes/shapes and include orthogonalization workspaces.
- **Activation memory:** scales with `layers * microbatch * context * width`, attention intermediates can scale with `layers * microbatch * context^2`, and logits can cost `microbatch * context * vocabulary * dtype_bytes`. Checkpointing/rematerialization trades memory for recompute. The calculator reports a rough activation proxy (12 saved `B*C*d`-sized tensors per layer) and full FP32 logits; treat as a lower-order planning estimate, not an allocator measurement.
- **Inference KV cache:** MHA estimate `2 * layers * batch * context * width * bytes_per_element`; pilot FP16 cache is 8 MiB at context 1,024, target 32 MiB at 2,048, stretch 48 MiB at 2,048 for batch one. This is separate from weights and grows with batch/context. GQA would change it but is not part of the approved config.

### Approximate peak planning memory

For BF16 compute + FP32 Adam state with microbatch 1, full configured context, 16-bit activations, FP32 logits, the calculator estimates model+optimizer+gradient persistent bytes plus activation and logits proxies. Reserve at least 4 GiB for CUDA context, framework, allocator fragmentation, kernels, dataset buffers, evaluation and transient copies. The planning table deliberately rounds upward:

| Option | FP32 persistent floor | BF16/FP32-state floor | FP32 training checkpoint floor (`12N`) | Est. proxy activations+logits | Minimum acquisition profile |
|---|---:|---:|---:|---:|---|
| Pilot 13.8M | 210 MiB | 210-237 MiB | 158 MiB | ~100 MiB | 16 GiB VRAM, 64 GiB RAM |
| Target 51.2M | 781 MiB | 781-879 MiB | 586 MiB | ~368 MiB | 24 GiB VRAM, 64-128 GiB RAM |
| Stretch 99.2M | 1,514 MiB | 1,514-1,703 MiB | 1,135 MiB | ~560 MiB | 24-48 GiB VRAM, 128 GiB RAM |

FP32 parameter/gradient/moment persistent floor equals 16N. The BF16 range reflects implementation-specific gradient/master precision; use the conservative high end until measured. Add 4 GiB runtime/temporary allowance and 25% headroom to the estimated peak. VRAM recommendations are acquisition minima, not proof the run fits: validate actual peak allocated and reserved memory with the exact framework, kernel, batch, context, logits path and checkpoint mode. The target configuration should use gradient accumulation with microbatch 1-2 and should not increase batch/context until a measured profile permits it.

Checkpoint storage should reserve at least `12N` for FP32 weights plus Adam moments, plus scheduler/scaler/RNG/sampler metadata and a temporary duplicate during atomic write. Save model-only FP32/BF16 exports separately. At 100M tokens, storage is dominated by data/checkpoint history rather than the 14M model; preserve best, latest and milestone checkpoints, and prune only after verifying the backup and manifest.

## Compute and Wall-Time Scenarios

Approximate dense training work is `6*N*T + 12*L*C*d*T` FLOPs for this full-attention decoder, with `N` parameters, `T` training tokens, layer count `L`, context `C`, width `d`. The `6NT` term is a common dense matmul estimate. The added term is a conservative full-context attention forward/backward proxy; exact causal kernels may perform less work, while padding, vocab logits, optimizer, input pipeline, evaluation and checkpointing add overhead. It is not a measured FLOP count. The script adds 25% wall-time overhead after dividing by assumed sustained useful FLOP/s.

| Profile | Illustrative acquisition, not a required SKU | Assumed sustained useful training rate | Pilot 13.8M / 100M tokens | Target 51.2M / 1B tokens | System |
|---|---|---:|---:|---:|---|
| Conservative | One supported 16-24 GiB GPU; single host | 0.25 TFLOP/s | ~14.1 h | ~25.1 days | 64 GiB RAM, 1-2 TB NVMe |
| Moderate | One current consumer GPU with 24-32 GiB VRAM | 1.5 TFLOP/s | ~2.35 h | ~4.2 days | 64-128 GiB RAM, 2 TB NVMe |
| Ambitious | One well-supported high-end 32-48 GiB GPU with efficient BF16 kernels | 8 TFLOP/s | ~26 min | ~18.8 h | 128 GiB RAM, 2 TB NVMe |

These are **extrapolations**, not published or local benchmark measurements. The assumed sustained rates are deliberately scenario inputs, not the card's advertised peak. The only local throughput result is about 17-19K target bytes/s for a 21,984-parameter, context-32 CPU run; it is not useful for predicting the proposed model. No hardware-training measurement from the actual acquisition target exists. Measure 1,000-5,000 optimizer steps on the exact config and a representative licensed shard, with warmup and profiler off/on comparisons, before authorizing the remaining tokens. Recompute wall time from measured tokens/s and include tokenizer/preprocessing and evaluation separately.

## Bill of Materials and Acquisition Stages

| Stage | Acquire | Purpose / gate |
|---|---|---|
| 0: no-cost preparation | Rights-cleared <=100 MiB representative sample; independent held-out evaluation fixture; tokenizer candidate packages in isolated environment | Validate parsing, filtering, split/contamination checks, tokenization, packing, deterministic sampler, loss masking and evaluator before renting compute. |
| 1: local run host | Linux x86_64 host, 64 GiB system RAM, 16-24 GiB supported VRAM, 12+ CPU cores, 2 TB NVMe scratch, stable cooling/PSU, 1-2 TB independent backup | First 13.8M/10M smoke then 100M pilot. CPU cores are for preprocessing; use one data-loader worker until profiling says otherwise. GPU must support the selected PyTorch/CUDA/BF16 stack. |
| 2: target upgrade | 24-32 GiB VRAM, 128 GiB RAM, 2 TB NVMe, separate backup; optional second GPU only if single-GPU profiling shows need | 51M-parameter scale test. Do not buy a second GPU before proving the run is VRAM/throughput limited and measuring communication overhead. |
| 3: stretch | 48 GiB or multiple well-supported GPUs, >=128 GiB RAM, high-throughput local storage | Only for the ~100M candidate after target-scale data and capability evidence; requires distributed checkpoint/resume tests and a new budget. |

For an example of current VRAM classes, NVIDIA lists 24 GB for the RTX 4090 and 32 GB for the RTX 5090. These official memory specifications indicate capacity only, not training throughput, compatibility with our software stack, price or value: [RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/), [RTX 5090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/).

For rented hardware, require a pinned instance image, exact GPU model/VRAM, CPU/RAM/storage throughput, attached-volume durability and egress costs. Never place credentials or private raw data in logs. Use encrypted persistent storage and an independent checkpoint backup; validate download/upload checksums. Software: Linux, Python version supported by pinned PyTorch, PyTorch CUDA build matched to driver, tokenizer library pinned, standard-library preprocessing where practical, and the project evaluator. Lock dependencies, capture driver/runtime/library versions, and avoid installing training-stack dependencies on the 8 GiB Codespace.

## First-Run Runbook

### Preflight checklist

1. Freeze a dataset snapshot and SHA-256 manifest; source-level license/provenance audit signed off; no private or excluded data; document counts by stratum and language.
2. Build duplicate connected components before splits. Keep train, validation and test source families disjoint. Decontaminate every public benchmark/test fixture and record the procedure.
3. Freeze one tokenizer candidate and all special-token/chat serialization IDs. Confirm exact round-trip for multilingual, code, numbers and malformed UTF-8 policies. Retain byte-tokenized comparison only as a separate later ablation.
4. Freeze config, seed list, effective batch tokens, sampler order, learning-rate schedule, clipping, precision, loss-mask behavior and all evaluation checkpoints. First optimizer is AdamW; do not tune on the final test.
5. Run CPU smoke: train 10K-100K tokens, force an interruption, resume from checkpoint and compare state/loss against uninterrupted training. Verify greedy inference determinism and model export/load parity.
6. On the acquired GPU, run 1,000-5,000 warmup/measurement steps at pilot context and microbatch. Capture step time, tokens/s, GPU peak allocated/reserved, host RSS, dataloader wait, gradient norms, loss, non-finite counts, power if available, and checkpoint write/read time.
7. Estimate complete run duration and storage from measured throughput. Go only if estimated wall time fits a defined rental/window budget with 30% headroom, memory has >=20% free VRAM at peak, and the backup/checkpoint restore drill succeeds.

### Training cadence and stop/rollback rules

- Pilot maximum: 100M tokens without a new go/no-go review. Save checkpoints at 10M, 25M, 50M, 75M and 100M tokens, plus latest every <=30 minutes or 1% of total tokens, whichever comes first. Keep best validation checkpoint and last known-good checkpoint.
- Run held-out next-token evaluation per mixture stratum every 1M tokens for the first 10M, then every 5M tokens. Run the frozen small task suite at 10M, 25M, 50M and 100M; show each track and denominator separately. No test-set-driven hyperparameter tuning.
- Stop on OOM, persistent swap, non-finite loss/gradients, data/license/provenance failure, train/validation leakage, disk below two checkpoint sizes plus 20% reserve, hardware errors, or measured throughput making the approved budget infeasible. Pause on a >3 consecutive-evaluation-step validation degradation, unexpected per-domain collapse, or instability; inspect before resume.
- Roll back to the last verified checkpoint if resume hash/state checks fail, a data shard hash differs, a loss mask/sampler/config changes, or validation/task regressions breach the pre-registered threshold. Never overwrite the source checkpoint; version config/data/code together.
- Stop scaling if held-out loss plateaus and the lower confidence bound of per-track task gain is below the predefined minimum, if the next token block mostly repeats data, or if compute per verified improvement exceeds the acquisition budget. Extend to 275M only after the 100M review; do not treat a falling train loss as a go signal.

### CPU versus GPU

CPU is appropriate for tokenizer training on capped samples, corpus preprocessing/dedup, evaluator development, data loader tests, 10^4-10^6 parameter model mechanics, and short profiler/reference runs. The Codespace's 8 GiB/CPU-only setup is not a sensible host for the 13.8M/100M-token pilot: its context-32 tiny-model throughput cannot be extrapolated to this config, and its memory reserve must remain intact. A GPU is not mathematically required to train a language model, but a CUDA-capable GPU with ample VRAM is practically required for the planned pilot within a bounded schedule. Full-corpus dedup and tokenizer/model training also need sufficient disk and RAM; they do not require a GPU by definition.

## Remaining Risks and Go/No-Go

Key risks are training-data rights/provenance, source/domain imbalance, tokenizer-induced sequence and embedding cost, benchmark contamination, synthetic evaluation overfit, small-scale scaling-law mismatch, training instability at context 1,024, unmeasured optimizer/kernel memory, and deployment hardware mismatch. Retrieval or deterministic tools can improve system behavior but do not improve the pretrained model's next-token capability. A tiny training run may establish mechanics and learning curves only; parameter count does not predict an intelligence level.

**Go for acquisition only after** (a) all data sources are cleared and versioned, (b) tokenizer/model/evaluation hashes are frozen, (c) leakage checks pass, (d) CPU interruption/resume and export parity pass, (e) exact GPU smoke meets memory/throughput and restore gates, and (f) the 100M-token budget has an identified source, runtime window and backup plan. Until then, verdict remains **not ready for genuine pretraining**.
