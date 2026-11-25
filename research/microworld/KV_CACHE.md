# CPU KV Cache

## Design

The Foundation weights and causal attention are unchanged. Incremental generation now preallocates per-layer key/value tensors on CPU, runs the prompt once, and computes each later token against cached keys and values. The cache is inference-only and never enters a checkpoint or optimizer state.

For batch size 1 at context 1,024, four layers, width 256 and float32, the cache occupies 8,388,608 bytes (8 MiB). This is small relative to the measured 5.1 GiB available memory, so the training pilot uses RAM-backed tensors. An optional file-backed `mmap` mode puts the same cache in files that the OS can keep in its page cache or evict to disk. This mode is available for constrained-memory experiments; it is not selected by default because explicit disk traffic is unlikely to help when RAM is available.

## Equivalence And Timing

Using the original 800-update checkpoint and a validation prompt with a 756-token prefix, greedy generation produced identical token IDs with full-prefix recomputation, RAM cache, and mmap cache for 24 generated tokens.

| Mode | Seconds | Relative to full recomputation |
| --- | ---: | ---: |
| Full-prefix recomputation | 3.7985 | 1.0x |
| Preallocated RAM cache | 0.1997 | 19.0x faster |
| File-backed mmap cache | 0.1942 | 19.6x faster |

This is one CPU timing sample, not a general performance claim. The mmap result reflects the operating system's page cache; it does not establish that sustained disk paging improves throughput. The cache implementation has unit tests for chunked-logit equivalence, incremental greedy-token equivalence, mmap equivalence, and context bounds.
