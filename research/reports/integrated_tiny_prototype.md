# Batch 12: Integrated Tiny Prototype and Ablations

Run date: 2026-11-23. This is a short CPU research prototype, not a capable general-purpose model. It uses only the already installed isolated PyTorch 2.15.0+cpu environment and project-authored synthetic fixtures; no package, model, or dataset was downloaded or installed.

## Components and Reproduction

The modular prototype lives in [`research/prototype`](../prototype/):

- `interfaces.py`: UTF-8 byte tokenizer plus tokenizer, generator, evidence-store, writer and verifier contracts.
- `model.py`: adapter around the unchanged Batch 4 `TinyGPT`; the byte model uses a separate config/checkpoint and does not overwrite the vocab-32 baseline.
- `data.py`: UTF-8 text loading, split/hash leakage checks, deterministic byte windows and byte-level NLL/BPB evaluation.
- `trainer.py`: one-thread FP32 AdamW loop, 2,560 MiB memory guard, swap-activity stop check, atomic checkpoint writes, and optimizer/data RNG resume state.
- `system.py`: optional in-process retrieval adapter using the existing BM25/versioned store and structured verifier; maximum 2 retrieval and 2 verification calls, 64 context-text tokens, 16 output tokens, and 50 ms per request on this fixture.
- `run_batch12.py`: three-seed model profile plus a separate external-system ablation.

The 18-document byte fixture is split 11 train / 3 validation / 4 test by document ID in [`splits.json`](../prototype/splits.json). It checks exact and normalized-text hashes across splits; zero collisions were found. It is a representation smoke fixture (prose, code, math, numbers, URLs and Unicode), not representative language data. The separate 10-record / 9-query knowledge-base fixture is synthetic CC0 and does not share examples with byte-model training.

Run from `/home/codespace`:

```sh
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/b4-torch-venv/bin/python -m unittest -v research.prototype.test_batch12
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/b4-torch-venv/bin/python -m research.prototype.run_batch12 \
  --output research/experiments/results/batch12_tiny.json --steps 60 --seeds 11 23 37 --repeats 5
```

The harness records full environment, fixture/split hashes, separate result JSON and CSV, training checkpoints, inference-only export sizes, test metrics, controller budgets, peak RSS and resume hashes. To keep the baseline cheap and reproducible, it uses batch 2, context 32, one layer, 32 hidden dimensions, 4 heads, 4x GELU MLP, tied embeddings, FP32, AdamW at 0.003 and 60 steps. Each run processes 3,840 next-byte targets. No code benchmark executes generated programs.

## Model Result

The byte-vocabulary model has 21,984 parameters and an inference export of 95,085 bytes. Its analytical FP32 MHA KV cache is 256 bytes/context token, or 8,192 bytes at context 32; this prototype does not implement a KV cache. The run began with 5,327 MiB available, used one CPU thread, and peaked at 344,686,592 bytes process RSS. Mean training time was about 0.214 s per 60-step seed run (about 3.56 ms/step and 18.0K target bytes/s). Total profile wall time was about 2.20 s.

| Metric (3 seeds) | Before training | After 60 updates | Scope |
|---|---:|---:|---|
| Validation bits/byte | 8.001 mean | 6.079 mean | 251 target bytes/document split; three seeds |
| Test bits/byte | 7.989 mean | 6.879 mean | 415 target bytes over four held-out documents; no content-hash overlap |
| Test next-byte accuracy | 3.61% mean | 8.11% mean | 415 target bytes; highly small-sample and fixture-specific |
| Parameters | 21,984 | 21,984 | One compact model |
| Step time / throughput | N/A | 3.38-3.79 ms / 16.9K-19.0K target bytes/s | CPU PyTorch 2.15, one thread |
| Inference export | N/A | 95,085 B | Serialized FP32 model state, per seed |
| Peak RSS | N/A | 344,686,592 B | Process high-water includes PyTorch and sequential runs |

The lower byte loss is evidence that the loop learned correlations in this tiny synthetic text fixture. It does not establish language understanding, reasoning, factuality, coding, or useful generation. Three seeds are preliminary; there is no natural-language benchmark suite with enough examples to support a capability conclusion.

Interruption/resume test stopped cooperatively at update 10, resumed to update 20, and exactly matched the uninterrupted update-20 model state SHA-256 (`d962c6...b94746`). This tests a saved checkpoint boundary, not an uncatchable process kill. Deterministic greedy generation and inference-export roundtrip passed for all three profile seeds.

## External-System Ablation

This is measured separately from the byte model. The system uses a structured resolver that reads hand-authored entity/attribute/value/time/provenance fields; it does not ask the language model to write answers and does not test natural-language entailment. Context token counts use a regex/word proxy, not the byte tokenizer.

| Mode | Correct / 9 | Answerable accuracy | Unsupported emitted answer rate | False abstention | Mean evidence-text tokens | Mean / p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| No context (abstain if unsupported) | 3/9 | 0/6 | 0/0 emitted | 100% | 0 | 0.0017 / 0.0024 ms |
| Full context + structured check | 9/9 | 6/6 | 0% | 0% | 105 | 0.059 / 0.066 ms |
| BM25 top-1, no verifier | 5/9 | 5/6 | 44.4% | 0% | 10.6 | 0.0105 / 0.030 ms |
| BM25 top-3 + structured check | 9/9 | 6/6 | 0% | 0% | 28.9 | 0.0237 / 0.043 ms |
| Bounded BM25 + structured check | 9/9 | 6/6 | 0% | 0% | 14.8 | 0.0208 / 0.039 ms |

The bounded path stayed within 2 retrieval/2 verifier calls and the 64-token context/16-token output/50 ms limits. The no-context baseline has zero unsupported emitted answers only because it abstains on every answerable case; its 100% false-abstention rate is a real cost. The unverified retrieval condition illustrates that a result in the index does not establish support. These are exactly annotated synthetic queries, so the apparent 9/9 verified score cannot be used as evidence of real-world factual reliability.

The retrieval-process-only Batch 8 measurement was 23,330,816 bytes peak RSS. Batch 12 reports a shared process high-water of 344,686,592 bytes after model training; that is not a separate system-memory estimate. End-to-end LM-plus-retrieval latency and RSS remain unmeasured.

## Keep / Defer Decisions

- **Keep:** unchanged one-layer causal Transformer architecture, byte tokenizer as a coverage reference, AdamW/F32 baseline, split-integrity checks, atomic resumable checkpoints, bounded controller interfaces, structured verifier for exact synthetic fields, deterministic direct/abstain controls.
- **Reject as a default:** top-1 retrieval without verification; 4/9 of its emitted answers were incorrect in this fixture.
- **Keep only as system-mechanics evidence:** BM25 top-3 and bounded BM25 plus structured checks; no LM factuality or natural-language evidence capability was tested.
- **Defer:** production tokenizer selection, SSM/hybrid, SFT, answerability head, learned memory, automatic memory writes, dense/vector retrieval, code execution, runtime export, low-bit quantization and speculative decoding.
- **Do not claim:** general capability, architecture superiority, calibrated abstention, source-grounded language generation, or consumer-device speed from these results.

## Limits and Next Gate

The byte fixture has only 18 synthetic documents; 3 validation and 4 test documents yield noisy estimates. No third-party data were used. The test split is disjoint by exact and normalized content, but this does not guarantee semantic independence: examples cover related representation categories. The lexical experiment has 9 structured queries, and the verifier gets entity, attribute and temporal labels directly. There is no model-backed QA, coding execution, calibrated confidence, adversarial natural-language evidence set, consumer target device, or actual KV implementation here.

Next, acquire a small rights-cleared text/evidence fixture with source-family splits and human-reviewed claim labels. Freeze this baseline, expand the evaluator, and connect a text model backend to the system API only after it can produce relevant claim structures. Run no architecture change until one-factor ablations have held-out language/task metrics and at least three seeds where practical. Do not scale training in this Codespace.
