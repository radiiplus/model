# Batch 4: Decoder Baseline and Architecture Anatomy

Checked 2026-11-23. This report separates repository source observations from local measurements. No model-quality claim is made from the toy sequence run. Candidate revisions and licenses are also recorded in `../REPOSITORY_LEDGER.csv`.

## Candidate source comparison

| Candidate | Pinned source and license | Source observations | Fit and disposition |
|---|---|---|---|
| nanoGPT | [HEAD `3adf61e`](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291), MIT. [`model.py`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L18), [`train.py`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py#L36). | Clear GPT-2 style reference: learned token/position embeddings, pre-LayerNorm residual blocks, combined QKV, causal attention (SDPA or triangular fallback), 4x GELU MLP, tied embedding/output weights, scaled residual init and next-token cross entropy. The README marks the project historical/deprecated. | Best source anatomy reference for a clean baseline. The default training recipe is GPU/data oriented; we use independently written compact code and a tiny explicit config. No code copied. |
| nanochat | [HEAD `92d63d4`](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd), MIT. [`gpt.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py#L1), [`runs/runcpu.sh`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/runs/runcpu.sh#L1). | Adds RoPE, QK normalization, GQA options, parameter-free RMSNorm, untied output, no linear biases, ReLU-squared and value-residual options. Its CPU script installs a dependency, trains a tokenizer on about 2B characters and runs a roughly 30-minute Mac-tuned setup. | Useful later comparison, but intentionally not our first baseline. Do not run the project script in this Codespace. |
| LitGPT | [HEAD `5341568`](https://github.com/Lightning-AI/litgpt/tree/5341568a65e918a85cb284fac038de2157eebf3f), Apache-2.0. [`litgpt/model.py`](https://github.com/Lightning-AI/litgpt/blob/5341568a65e918a85cb284fac038de2157eebf3f/litgpt/model.py#L22). | Supports many families and compatibility paths. LLaMA path uses configurable RoPE and gated SiLU MLP; behavior/defaults depend on selected model config. | Broad source reference; framework/model ecosystem is larger than the baseline needs. |
| Transformers | [HEAD `bc7bd00`](https://github.com/huggingface/transformers/tree/bc7bd00302067106c9d4f30f09689caddb955be8), Apache-2.0. [`modeling_llama.py`](https://github.com/huggingface/transformers/blob/bc7bd00302067106c9d4f30f09689caddb955be8/src/transformers/models/llama/modeling_llama.py). | Llama path uses RoPE, RMSNorm, separate Q/K/V projections, optional GQA via repeated K/V, gated MLP and selectable attention backends. | Important compatibility reference, but not a minimal from-scratch baseline or dependency for this experiment. |
| llm.c | [HEAD `f1e2ace`](https://github.com/karpathy/llm.c/tree/f1e2ace651495b74ae22d45d1723443fd00ecd3a), MIT. [`train_gpt2.c`](https://github.com/karpathy/llm.c/blob/f1e2ace651495b74ae22d45d1723443fd00ecd3a/train_gpt2.c#L1). | Has a scalar-loop CPU reference forward/backward path separate from optimized C/CUDA paths; GPT-2 embeddings, LayerNorm and training loss are explicit. | Useful low-level cross-check. C build/debug effort is unnecessary while a CPU PyTorch wheel fits disk and memory. |
| OLMo-core | [HEAD `704625a`](https://github.com/allenai/OLMo-core/tree/704625ad94a7c8bc93f9bea7a0b865b9a0c3d44e), Apache-2.0. [`Transformer config`](https://github.com/allenai/OLMo-core/blob/704625ad94a7c8bc93f9bea7a0b865b9a0c3d44e/src/olmo_core/nn/transformer/config.py#L337), [`Transformer model`](https://github.com/allenai/OLMo-core/blob/704625ad94a7c8bc93f9bea7a0b865b9a0c3d44e/src/olmo_core/nn/transformer/model.py#L87). | Exposes numerous block, normalization, attention, initialization, tying, logits and loss choices. | Strong scalable research reference; training stack exceeds this local experiment. |
| TorchTitan | [HEAD `692c9bb`](https://github.com/pytorch/torchtitan/tree/692c9bb5b26f606e5fb739acd55effb6f8850bc9), BSD-style. [`Llama 3 model`](https://github.com/pytorch/torchtitan/blob/692c9bb5b26f606e5fb739acd55effb6f8850bc9/torchtitan/models/llama3/model.py#L14). | Llama block delegates to shared attention/decoder/MLP modules; compile regions, rematerialization and distributed training belong to its runtime layer. | Training-system reference, not needed for a single-process CPU model. |

These are repository/source observations at the pinned revisions, not independent replications of their papers or model results. Kernel backends, compilation, activation checkpointing, data loading and distributed execution are runtime/system choices. Embeddings, position scheme, attention topology, normalization, MLP, residual structure and tying are model/design choices.

## Component map

| Component | Baseline implementation | Alternatives to test separately | Main consequence |
|---|---|---|---|
| Input and embeddings | Integer token IDs; learned token embedding `V x d`; learned absolute position embedding `T x d`. | Byte/BPE/Unigram tokenizer; sinusoidal, RoPE, ALiBi; no-position variants. | Vocabulary controls embedding and output parameters; position method changes extrapolation and implementation. |
| Causal mask | Boolean upper triangle above the diagonal, masked before softmax; query position `i` attends only to keys `<= i`. | Fused SDPA causal flag, block/sparse masks. | Must prevent target/future leakage. Kernel changes should preserve the same function. |
| Attention | Standard multi-head attention (MHA), combined QKV projection with biases, scale by `sqrt(head_dim)`, softmax, output projection. | GQA/MQA reduce K/V head count and cache state; different kernels are separate implementation changes. | MHA cache grows with K and V per layer and context; this prototype has no KV cache and recomputes prefix attention at each generated token. |
| Normalization | Pre-LayerNorm before attention and MLP; final LayerNorm; epsilon `1e-5`. | RMSNorm, post-LN, parameter-free normalization. | Stability and compute/parameter effects need training ablations; no alternative is presumed better. |
| Residual path | `x + attention(LN(x))`, then `x + MLP(LN(x))`. | Parallel residual or value-residual variants. | Alters signal paths; keep the plain path as the baseline. |
| MLP | Dense `d -> 4d -> d`, exact GELU, biases. | SwiGLU, squared ReLU, other expansion ratios. | Dominant non-attention block compute; gated variants have different parameter/FLOP costs. |
| Weights and initialization | Embedding and linear weights normal std 0.02, biases zero, LayerNorm gain one/bias zero; attention and MLP output projections scaled by `0.02/sqrt(2)` for one layer. | Untied output head; alternate init scales. | Tying saves a `V x d` matrix; init is part of training recipe, not an inference kernel. |
| Output and objective | Output logits use the tied token embedding as the linear classifier; cross entropy predicts token `t+1` from prefix through `t`. | Label smoothing or alternate objectives are excluded. | Baseline loss is next-token cross entropy; quality must use held-out bytes/tasks later. |

## Local baseline

The trainable implementation is [`tiny_gpt.py`](../experiments/tiny_gpt.py); training and checkpointing are in [`train_tiny_lm.py`](../experiments/train_tiny_lm.py). The stable initial config is [`baseline_config.json`](../experiments/baseline_config.json). It has 32 token IDs, context 32, one layer, width 32, four heads of width 8, 4x MLP (`d_ff=128`), float32, tied embeddings, no dropout and one seed. Total parameters: 14,816. Batch size is 1 and the overfit input sequence has length 12.

The implementation is a readable PyTorch CPU model; the standard-library [`baseline_transformer.py`](../experiments/baseline_transformer.py) remains as a forward-only fallback/reference, not as the selected trainable baseline. That fallback has its own initialization and omits the attention projection biases, so its parameter count/timing should not be compared to the PyTorch baseline. Neither implementation is an optimized inference runtime. The trainable implementation includes no retrieval, memory module, RL, speculative decoding, fused attention or KV cache.

### Reproduction

PyTorch was absent from system Python. User authorization was received before installation; the isolated environment is `/tmp/b4-torch-venv`. The CPU wheel was 209,533,799 bytes (about 200 MiB); the installed venv is 954 MiB including dependencies. CUDA is unavailable and no GPU packages were installed. NumPy was not installed; PyTorch emits a benign warning that NumPy is unavailable, but these runs use only Torch tensors. The full resolved environment is pinned in [`requirements_cpu.lock.txt`](../experiments/requirements_cpu.lock.txt).

```sh
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/b4-torch-venv/bin/python research/experiments/train_tiny_lm.py --config research/experiments/baseline_config.json --output research/experiments/results/baseline_training.json --steps 250 --min-available-mib 2560
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/b4-torch-venv/bin/python -m unittest -v research/experiments/test_tiny_gpt.py
```

The command checks `MemAvailable >= 2560 MiB` before training and every 25 steps. It trains one deterministic periodic sequence with AdamW (`lr=0.01`, weight decay 0), runs one process, and caps Torch threads to one. Results are in `../experiments/results/baseline_training.json`; the state+optimizer checkpoint is `../experiments/results/baseline_tiny_gpt.pt`. The exact baseline source/config/checkpoint hashes are in [`baseline_manifest.json`](../experiments/baseline_manifest.json).

### Measured smoke result

| Metric | Result |
|---|---:|
| Parameters | 14,816 |
| Input sequence / batch | 12 tokens / 1 |
| Training steps | 250 |
| Initial -> final train loss | 3.5358 -> 0.000487 nats/token |
| Training wall time / time per step | 0.583 s / 2.33 ms |
| Supervised target tokens per second | 4,719 |
| Peak RSS | 342,876,160 bytes (~327 MiB) |
| MemAvailable at start | 5,320 MiB |
| Checkpoint loss roundtrip | exact |
| Generated IDs | all within vocabulary `[0, 31]` |

The result is only memorization of one periodic synthetic sequence. A low training loss here is not held-out language modeling, reasoning, factuality, code ability or generalization evidence. Throughput is for this CPU/Python/Torch environment and tiny sequence only.

The separate dependency-free forward test prototype reports ~18 MiB peak RSS but no training. Its output/result is `../experiments/results/baseline_forward.json`.

## Validation and stop conditions

Four PyTorch tests cover output shapes/parameter budget, causal no-future influence, finite next-token loss, tiny sequence overfit, valid generated IDs and checkpoint round-trip. The dependency-free forward reference also has three smoke tests. Both suites passed. The baseline must remain unchanged during later ablations; make copies/config identifiers for any one-factor change and preserve these results. Stop if `MemAvailable` falls below 2.5 GiB, if memory growth is unexpected, or when the explicit 120-second timeout expires.
