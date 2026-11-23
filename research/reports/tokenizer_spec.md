# Tokenizer Benchmark and Input-Format Specification

Checked 2026-11-23. This is a reproducible smoke comparison, not a tokenizer quality ranking. The executable and test suite are in `../experiments/benchmark_tokenizers.py` and `../experiments/test_tokenizers.py`; raw results are in `../experiments/results/tokenizer_benchmark.json`.

## Setup and reproducibility

The fixture contains 18 authored synthetic records; 11 train the tokenizers and 7 held-out records test numbers, Rust code, Unicode equations, URLs, rare/emoji strings, mixed scripts and punctuation. The held-out fixture has 629 UTF-8 bytes. Its SHA-256 is stored in the result JSON. A fixed split and seed `20261123` are used. Each method is fit from scratch on the same 11 texts, with a requested vocabulary cap of 512. SentencePiece uses byte fallback and identity normalization; Hugging Face uses ByteLevel with no normalizer; tiktoken uses its educational Python byte-BPE trainer with the stated regex. The measured trainer configurations are serialized in the result.

Packages were installed in an isolated virtual environment, not system Python: `tokenizers==0.23.3`, `sentencepiece==0.2.3`, `tiktoken==0.14.0`. Downloaded wheels occupied 15 MiB; the installed environment occupies about 75 MiB. No pretrained tokenizer assets, model weights or datasets were downloaded. The educational tiktoken trainer is intentionally a Python reference path, so its measured throughput is not comparable to tiktoken's Rust inference engine.

Reproduce with a 120-second timeout and one-thread caps:

```sh
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/tokenizer-bench-venv/bin/python research/experiments/benchmark_tokenizers.py --repeats 20 --output research/experiments/results/tokenizer_benchmark.json
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/tokenizer-bench-venv/bin/python -m unittest -v research/experiments/test_tokenizers.py
```

The script checks `/proc/meminfo` and stops below 2,560 MiB `MemAvailable`. It records peak RSS and environment/package versions. The final measured peak RSS was 42,553,344 bytes. Runtime was below one second; the timeout remains explicit. Results are deterministic in coverage and configuration, but timings vary by load and are not stable performance estimates.

## Observations

| Representation | Vocab size | Held-out tokens / 629 bytes | Tokens/byte | Attention-pair proxy | Encode throughput | Roundtrip |
|---|---:|---:|---:|---:|---:|---|
| UTF-8 bytes | 256 | 629 | 1.000 | 1.000 | 286 MB/s | exact |
| Unicode codepoints | 1,112,064 theoretical | 532 | 0.846 | 0.684 | 129 MB/s | exact |
| HF ByteLevel BPE | 393 effective | 531 | 0.844 | 0.747 | 31.3 MB/s | exact |
| SentencePiece BPE + byte fallback | 512 | 527 | 0.838 | 0.738 | 38.2 MB/s | exact |
| SentencePiece Unigram + byte fallback | 500 | 532 | 0.846 | 0.745 | 66.9 MB/s | exact |
| tiktoken educational BPE | 512 | 520 | 0.827 | 0.716 | 4.21 MB/s | exact |

The attention-pair proxy sums squared per-document token lengths and divides by the byte baseline's corresponding sum. It estimates only the quadratic attention term for separate unpadded examples; it is not a FLOP or latency measurement. Token count reduction on this tiny fixture does not establish a capability gain. The results JSON contains per-document lengths, timing, fallback estimates, configs and environment metadata.

## Chat and special-token proposal

Keep model control tokens distinct from ordinary text. Reserve immutable IDs (and record them in the tokenizer manifest) for:

`<|pad|>`, `<|bos|>`, `<|eos|>`, `<|unk|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`, `<|channel|>`, `<|end_msg|>`, `<|end_turn|>`, `<|tool_call|>`, `<|tool_result|>`.

An assembled exchange can be represented as `<|bos|><|system|>...<|end_msg|><|user|>...<|end_msg|><|assistant|><|channel|>analysis...<|end_msg|><|assistant|><|channel|>final...<|end_turn|>`. Tool calls and results should use structured fields and explicit control IDs. Do not infer control flow from matching text in user content. The input builder must escape literal spellings or encode user text through a path that cannot introduce reserved IDs. The final exact token inventory and whether channels are tokens or ordinary text remain open until the model objective is selected.

## Serialization requirements

Every trained tokenizer artifact should ship with a versioned manifest containing schema version, library and trainer versions, vocabulary size and ordered token-to-ID mapping, normalization and pre-tokenization rules, regex engine/version, merge or piece scores, byte fallback policy, special-token IDs, decoder behavior, training fixture/source hashes, and SHA-256 hashes of all artifact files. Keep a human-readable config alongside the canonical library artifact.

- Hugging Face: preserve `tokenizer.json` and manifest; avoid relying on implicit defaults when loading.
- SentencePiece: preserve `.model` protobuf and manifest; explicitly record normalization and special-ID choices.
- tiktoken: preserve mergeable byte ranks, regex pattern and special-token map; do not depend on remote named encoding downloads at runtime.
- UTF-8 byte IDs: version the fixed mapping `byte b -> ID b` and strict decode policy. Decode only after concatenating bytes so multibyte UTF-8 sequences can cross token boundaries.
- Codepoint IDs: define valid Unicode scalar handling and model output map. A direct 1.1M-class softmax is likely costly; any hash/bucket representation requires a separate collision/quality study.

Do not use Python pickle as a portable or untrusted-loadable tokenizer format. Verify artifact checksums before use and retain the exact training config and source revision.

## Controlled downstream test

After a baseline language model exists, compare byte IDs, one byte-level BPE and SentencePiece Unigram on identical raw train/validation/test text and split hashes. Fix model body shape, optimizer, initialization policy, sequence packing policy, data order and seed. Report embedding/output parameters separately because vocabulary size changes parameter count. Run two budget views: equal raw bytes processed and equal measured training compute. Report per-track task metrics, held-out NLL in bits per UTF-8 byte, exact-copy and Unicode/code/math slices, emitted tokens per second, memory, padding and total parameters. Use at least three seeds for a major selection when affordable; otherwise label the result preliminary. A reduction in context length or per-token loss is not a capability result by itself.

## Limitations and decision state

The fixture is authored, tiny and unlike a realistic multilingual/code corpus. The short split can produce unstable merges and vocabulary underuse. We have not measured trained language-model quality, training stability, energy, full-context behavior, quantization impact, consumer deployment or semantic robustness. These methods remain candidates; no final tokenizer has been selected.
