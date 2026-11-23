# Batch 6: Training Engine, Loss, Optimizer and Compute

Checked 2026-11-23. The Batch 4 Transformer, configuration, checkpoint and result remain preserved. Environment at batch start: Python 3.14.2, host RAM 8,327,024,640 bytes, 5,786,624,000 bytes `MemAvailable`, no swap, 2 allowed logical CPUs (affinity 0-1), no NVIDIA device, 37 GiB free on `/tmp`. System Python has no PyTorch; reused the existing isolated `/tmp/b4-torch-venv` (954 MiB, PyTorch 2.15.0+cpu, CUDA unavailable), with no installs. Batch experiment peak RSS was 351,035,392 bytes.

## Source map

Upstream source observations below were read at pinned revisions and are not local benchmark results. Paper performance claims remain author-reported. Revision, license, paths and CPU caveats are also in [`REPOSITORY_LEDGER.csv`](../REPOSITORY_LEDGER.csv).

| Candidate | Source mechanics and memory | CPU, license and relevance |
|---|---|---|
| PyTorch AdamW | [`adamw.py`](https://github.com/pytorch/pytorch/blob/a7c9e15b4c4b205c87c765a0a799c39dfcfe5418/torch/optim/adamw.py) delegates to Adam with decoupled decay. Per parameter it keeps first and second moments (2 state values); AMSGrad adds a third. Update uses exponential moment averages, bias correction and decoupled `p *= 1-lr*wd`. | CPU reference path available; BSD-style project license. Foreach/fused variants change implementation/memory behavior. `foreach` can require an additional parameter-sized temporary allocation according to docs. This is the baseline optimizer. |
| Muon / PyTorch hybrid | PyTorch [`_muon.py`](https://github.com/pytorch/pytorch/blob/a7c9e15b4c4b205c87c765a0a799c39dfcfe5418/torch/optim/_muon.py) uses momentum, optional Nesterov, Newton-Schulz orthogonalization and shape-based LR adjustment. Documentation explicitly says to use a conventional optimizer for embeddings, output heads and non-matrix parameters. Hybrid state is one momentum tensor for eligible matrices plus Adam moments elsewhere. KellerJordan's [`muon.py`](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py) shows the standalone update. | Installed PyTorch built-in Muon executed on CPU in this batch. KellerJordan/Muon is MIT; PyTorch BSD-style. PyTorch implementation and version are recorded separately from upstream git SHA. nanochat's hybrid has additional NorMuon variance reduction, row equilibration, Muon+ renormalization and cautious decay; it is not plain Muon. |
| Sophia | [`SophiaG`](https://github.com/Liuhong99/Sophia/blob/a7e157229b71d58cf995d32854f1be15c265b350/sophia.py) stores first moment plus diagonal Hessian estimate (2 state tensors/parameter), requires scheduled `update_hessian()` calls, and clips a normalized sign update. | MIT, CPU Torch ops exist. The update cadence and effective batch must be reproduced faithfully; extra curvature work and optimizer recipe are not benchmarked here. Paper claims lower compute at target LM quality; not locally verified. |
| bitsandbytes 8-bit optimizers | [`AdamW8bit`](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/a616886cacf86701514f0cb1d6be760f967481c0/bitsandbytes/optim/adamw.py) quantizes optimizer state in blocks and by default skips tensors below `min_8bit_size=4096`. | MIT. The audited support matrix does not establish CPU support for 8-bit optimizers at this revision; no installation or run was attempted. Deferred rather than assuming CPU support. |
| nanochat | [`MuonAdamW`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/optim.py) and [`base_train.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/base_train.py) combine compiled/fused AdamW and modified Muon; training has warmup/warmdown, momentum/weight-decay schedules, resume and distributed concerns. The source explicitly treats transfer of AdamW batch/depth scaling assumptions to Muon as unverified. | MIT; CPU device path exists, but default compile, fused operations, shapes and training recipe are not a small CPU setup. Inspect for design; do not vendor or claim its performance. |
| LitGPT | [`pretrain.py`](https://github.com/Lightning-AI/litgpt/blob/5341568a65e918a85cb284fac038de2157eebf3f/litgpt/pretrain.py) has AdamW, gradient accumulation, loss scaling by accumulation count, no-sync, clipping, LR warmup/schedules and checkpoint resume paths. | Hardware can include CPU, but dependency/framework footprint exceeds the local harness. Root license was not confirmed at this pinned path; verify before reuse. |
| TorchTitan | [Checkpoint guide](https://github.com/pytorch/torchtitan/blob/692c9bb5b26f606e5fb739acd55effb6f8850bc9/docs/checkpoint.md) distinguishes model-only and full training state; system covers optimizer and distributed/checkpoint operations. | BSD-3-Clause; a distributed/parallel training system, not a justified dependency for this one-process CPU bench. |
| llm.c | [`configure_optimizers`](https://github.com/karpathy/llm.c/blob/f1e2ace651495b74ae22d45d1723443fd00ecd3a/train_gpt2.py) separates decay/no-decay parameters, uses AdamW and fused mode where available, and has a zero-stage option. | MIT; CPU reference branch exists but the repository targets GPT-2 scale/checkpoints/data. Read as an implementation reference; no build or run. |

### Evidence level

The AdamW decoupled-weight-decay argument is from Loshchilov and Hutter's [paper](https://arxiv.org/abs/1711.05101); it motivates the update form but is not an optimizer ranking. Muon's orthogonalized momentum update and scaling are from the [KellerJordan implementation](https://github.com/KellerJordan/Muon/tree/f98f1cacc0263b04290753e32be8d498c1efc806) and the authors' [scaling study](https://arxiv.org/abs/2502.16982). Sophia's diagonal curvature estimator and reported compute-to-quality results are from the [paper](https://arxiv.org/abs/2305.14342). 8-bit optimizer-state quantization is described in the [bitsandbytes paper](https://arxiv.org/abs/2110.02861). These are paper/repository claims and equations, not results reproduced here. Our results are limited to the synthetic fixture below.

## Local comparison

The new [`training_efficiency.py`](../experiments/training_efficiency.py) uses the unchanged one-layer TinyGPT (14,816 parameters, context capacity 32). Every method and seed starts from the same seeded initial weights and sees the same four fixed, phase-shifted repetitions of a six-token synthetic cycle for 200 updates. It uses batch 4, sequence length 24, float32, cross entropy, global gradient norm clipping at 1.0 and no schedule/weight decay. Seeds: 17, 29, 43. AdamW LR=0.003; hybrid Muon LR=0.02 for hidden matrix weights and AdamW LR=0.003 for all other parameters. Rates are not tuned or intrinsically comparable, so this is an implementation/recipe comparison, not a claim that either optimizer is generally better.

| Metric | AdamW | Muon + AdamW hybrid |
|---|---:|---:|
| Parameters | 14,816 | 14,816 |
| Final train loss (mean of 3 seeds) | 0.004315 | 0.004518 |
| Loss at update 50 (per seed) | 0.0676, 0.0791, 0.0748 | 0.0937, 0.0958, 0.0877 |
| Training time for 200 updates (mean) | 0.614 s | 1.166 s |
| Target tokens/s (mean; 4x23 targets/update) | 29,995 | 15,791 |
| Optimizer tensor-state bytes | 118,592 | 69,424 |

The full six-run comparison took under 10 seconds, peak process RSS 351,035,392 bytes and started with 5,280 MiB available. Losses are on one repeated pattern and do not measure generalization or useful language capability. Muon reduced optimizer-state storage by about 41% for this parameter grouping but was roughly 1.9x slower here. A learning-rate/tuning study and a varied held-out task would be needed before selecting it.

Gradient accumulation was checked for one update using four microbatches of size 1 versus one batch of size 4, with the same examples, mean loss scaling and AdamW. Maximum parameter difference was `2.98e-7`, within the `1e-6` tolerance. Dropout is zero. With stochastic layers, mismatched order, non-additive batch-dependent layers, or loss scaling errors, equivalence need not hold.

Both optimizer resume checks serialized model, optimizer and RNG state to a temporary file at update 100, then continued to update 200. Final loss and SHA256 of every model tensor matched the uninterrupted run exactly for AdamW and hybrid Muon. The fixed dataset needs no iterator state; a real data pipeline must also checkpoint sampler/shuffle position, data cursor, scheduler/scaler state and global update count.

## CPU precision probe

[`training_precision_probe.py`](../experiments/training_precision_probe.py) ran 10 measured AdamW updates plus 2 warmups for each mode on the same CPU, one thread, same initial weights and tiny fixture. Float32 took 3.07 ms/update; CPU BF16 autocast took 4.73 ms/update; CPU FP16 autocast took 6.27 ms/update. All modes completed with finite losses. These tiny shapes are dominated by dispatch/overhead and do not establish general CPU precision performance. Autocast kept FP32 master parameters and Adam state; it mainly changes eligible activation/compute dtype, not the per-parameter master+gradient+optimizer-state budget. Peak RSS was 347,197,440 bytes across the process.

## Memory and compute accounting

For FP32 parameters, gradients and AdamW moments, the lower-bound persistent per-parameter accounting is 4 + 4 + 8 = **16 bytes/parameter**, before activations, temporary buffers, allocator overhead and checkpoint copies. On 14,816 parameters that is 237,056 bytes plus scalar state; measured AdamW tensor state was 118,592 bytes (two moments plus per-parameter step scalars). For the observed Muon hybrid, parameters+gradients are 8 bytes/parameter and optimizer tensor state measured 69,424 bytes (momentum for selected matrices, Adam moments for the rest), around 12.7 bytes/parameter total at this size. Newton-Schulz matrix work uses transient buffers, excluded from persistent state bytes.

BF16/FP16 autocast with FP32 master parameters and AdamW moments retains approximately the same 16-byte persistent parameter budget; lower-precision activations can save activation memory but some operations promote to FP32. Pure low-precision weights/gradients plus low-precision optimizer states could use fewer bytes, but have different numerical/stability risks and were not tested. A full FP32 master copy plus BF16/FP16 model/gradient and FP32 Adam moments commonly approaches 16 bytes/parameter as well. Do not infer precision from a label; count actual tensors and optimizer state for the chosen implementation.

Gradient accumulation lowers activation memory roughly with microbatch size, not optimizer-state memory. Activation checkpointing trades recomputation for fewer retained activations; it is not useful for this tiny model but can matter for larger/deeper training. Optimizer sharding/offload are not useful for a 14.8k-parameter single-process CPU run and would add transfer/state-management cost.

As a rough dense Transformer training estimate, use about `6 * parameter_count * target_tokens` FLOPs for matrix multiplies, then add attention's sequence-quadratic work and normalization/softmax overhead. Here 14,816 parameters and 18,400 target tokens imply about 1.63 GFLOPs for the parameter-matrix term; attention adds roughly another 1.63 GFLOPs under the `12 * B * T^2 * d_model * updates` forward/backward approximation for B=4, T=23, d=32, 200 updates. This approximately 3.3 GFLOPs is a theoretical order estimate, not hardware-counter measurement, and ignores exact kernel/backward details. The observed wall time and tokens/s are measured only on this CPU and tiny shape.

## Loss and safe-loop specification

- Keep next-token cross entropy as the training baseline; sum valid token losses and divide by valid target-token count. Record total loss and per-domain held-out metrics separately.
- Use bits-per-byte / nats-per-byte as an evaluation normalization when comparing tokenizers: sum token NLL and divide by the original UTF-8 bytes represented. Do not silently substitute byte-normalized optimization for ordinary token CE; it changes document weighting and requires a controlled ablation.
- Add auxiliary losses only with a named hypothesis and ablation. No auxiliary objective is justified by these experiments.
- Fix seed, model initialization, token order, effective token count, batch size, dtype, clipping, LR schedule and checkpoint cadence before comparing optimizers. Log update and token counts, tokens/s, loss, gradient norm, peak RSS, optimizer-state bytes and non-finite events.
- Use one process and one worker; cap BLAS/Torch threads at one or two. Before and during each run, stop if `MemAvailable < 2560 MiB`; bound runs with an explicit timeout and finite step cap. Save to a new experiment-specific path, never over the baseline.
- Save model, optimizer, scheduler, mixed-precision scaler if used, RNGs, global update/token count and sampler/data position atomically. Verify a resumed run matches uninterrupted training on a deterministic fixture.

Reproduce the optimizer comparison:

```sh
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/b4-torch-venv/bin/python research/experiments/training_efficiency.py --config research/experiments/baseline_config.json --output research/experiments/results/training_efficiency_comparison.json --steps 200 --seeds 17 29 43 --include-muon
```

Reproduce the precision probe:

```sh
timeout 60s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/b4-torch-venv/bin/python research/experiments/training_precision_probe.py --config research/experiments/baseline_config.json --output research/experiments/results/training_precision_probe.json --steps 10
```

## Recommendation and unresolved tests

Keep AdamW as the baseline because it is simple, portable and currently the only optimizer with a preserved training result. Keep hybrid Muon as a candidate for hidden matrix weights: it ran on this CPU and saved state, but cost more time and did not improve this toy loss. Do not select Sophia, 8-bit Adam, or an engine framework before a specific need and a measured, license-reviewed CPU path. Next optimizer evidence should use a small varied licensed train/validation corpus, compare equal target-token and equal-wall-time budgets, include at least three seeds, tune each method only on a development split, and report quality, stability, state bytes and loss-versus-time. This Codespace cannot establish eventual pretraining scale or consumer deployment performance.

| Unproven assumption | Why it remains unproven | Evidence needed |
|---|---|---|
| Muon's learning-rate scaling transfers to this model and task | This run used a single LR per group and no tune budget | Small development-only LR/schedule selection, then held-out equal-token and equal-time runs across seeds |
| Fewer optimizer-state bytes improve practical fit or speed | The model is tiny; CPU Muon was slower and transient workspaces were not isolated | Larger-but-guarded model/context memory profile and end-to-end consumer CPU benchmark |
| CPU BF16/FP16 is beneficial | This tiny shape was slower; autocast keeps FP32 master weights and states | Repeated, warmed benchmarks at realistic layer/sequence sizes plus held-out loss drift/stability |
| Accumulation is identical to a large batch | Verified only for deterministic dropout-free data and one update | Resume/order tests with real sampler, dropout and all intended loss masks |
| `6 * parameters * tokens` predicts training cost | It omits explicit quadratic attention and backend effects | Profile operation counts or hardware counters at several sequence lengths; retain formula as estimate |
| Token-normalized CE is fair across tokenizers | Token count changes the weighting of documents and bytes | Compare both token CE and byte-normalized evaluation on fixed raw-byte held-out data |
