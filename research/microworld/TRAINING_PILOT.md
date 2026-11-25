# Batch 4 Controlled Micro-World Training

## Decision

**NO-GO for additional training.** The fixed pilot reduced some teacher-forced losses, but all three trained arms had 0% free-generation coverage and semantic accuracy on every partition. All generated outputs reached the 128-token cap and contained repeated trigrams. Structural accuracy did not translate into usable answers or transferable structure recovery.

No second seed was run: no arm met the predeclared compositional-majority candidate criterion. This is not evidence for or against general reasoning beyond this small generated task and pilot budget.

## Experiment

The starting checkpoint is the original 800-update Foundation: SHA-256 `76b60e84c3b203ef1a30bd93f0fd7e8c8fbe619d08799e57e7f434a6c237a5fe`. The model has 4,470,272 parameters, 4 layers, width 256, 4 attention heads, a 1,024-token context, and tied embeddings/output weights. Tokenizer SHA-256 is `956cefc54a7f4e54761c477124d210a1068b0fdb8b0850ab7c77a275849b93fe`; dataset manifest SHA-256 is `3df845563dd5f84a29ebd60d858e1798f6f7193f13fb1eebc2e1f4cb14525d19`; objective manifest SHA-256 is `aa4e43cd4f707a668b719e9a24ebcde0cbe91a47b89c7961f661657a3880f261`.

All arms started from identical backbone weights and used seed `20261125`, the same fixed 64-example presentation order, 54,248 input tokens, effective batch size 1, AdamW (`lr=5e-5`, weight decay `0.1`), gradient clipping at 1.0, and a constant scheduler. Validation was sampled once using seed `20261126`; greedy generation used seed `20261127` and 32 examples per partition. Runs used CPU float32, one Torch thread, and the 2,560 MiB available-memory guard. No architecture or training-data changes were made between arms.

The active target counts differ by design: A optimized 54,184 language next-token targets; B optimized zero language targets and 667,116 structural token labels; C optimized both. The structural labels include outside (`O`) positions for span tasks, so the totals are not directly comparable to language target counts. A and C had 4,470,272 parameters; B and C had 115,393 additional head parameters (4,585,665 total, +2.58%).

## Training Results

| Arm | Validation language NLL | Validation objective | Training seconds | Input tokens/sec | Peak RSS | Free-generation accuracy, all partitions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: language only | 6.2569 -> 4.9580 | 6.2614 -> 4.9622 | 48.1 | 1,127.7 | 793 MiB | 0% |
| B: structure only | 6.2569 -> 6.0925 | 1.8616 -> 1.5269 | 79.0 | 686.7 | 795 MiB | 0% |
| C: combined | 6.2569 -> 4.9824 | 8.1230 -> 6.5731 | 81.6 | 665.0 | 820 MiB | 0% |

The B/C objective values include their structural objectives; they should not be compared numerically with A's language-only objective. The initial checkpoint's sampled language NLL was 6.2569 under this benchmark. B's final teacher-forced event metrics were 7.0% entity binding (95/1,354), 1.3% relation recovery (18/1,419), and 39.3% state-transition entity accuracy (92/234). C scored 0%, 1.1%, and 4.7% on those measures. Many token-level span accuracies are inflated by the large outside-label class and are not evidence of successful span recovery.

## Free Generation

| Partition | Initial Foundation | A | B | C | Majority structural baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0% | 0% | 0% | 0% | 51.7% |
| IID | 0% | 0% | 0% | 0% | 53.1% |
| Compositional | 0% | 0% | 0% | 0% | 52.4% |
| Counterfactual | 0% | 0% | 0% | 0% | 53.5% |
| Held-out grammar | 0% | 0% | 0% | 0% | 52.4% |

For the initial checkpoint and all three arms, semantic coverage, accuracy among covered outputs, overall semantic accuracy, and correct period termination were 0%; trigram repetition was 100%; mean output length was 128 tokens. No output yielded parseable answer structure. Arm A and C produced repeated fragments; B produced repeated whitespace. The parser reported no invalid entity references because outputs never reached a parseable entity-bearing form.

The deterministic oracle and template systems reached 100% exact structural accuracy over their full partitions. Exact retrieval had zero coverage. The majority baseline was about 52% and substantially exceeded every neural arm. The symbolic/template systems were evaluated separately and did not provide answers to the models.

No successful neural generation sample was observed. Semantic coverage uses the independent answer parser and structural oracle; no human-rated language-quality evaluation was performed in this pilot.

## KV Cache

The inference-only cache leaves parameters and checkpoints unchanged. It preallocates per-layer CPU keys and values, evaluates each prompt once, then evaluates only the newly generated token. At context 1,024 it uses 8 MiB per sequence. On the original checkpoint with a 756-token prompt and 24 generated tokens, cached and full-recompute greedy token IDs matched exactly: full-prefix recomputation took 3.7985 seconds, RAM caching took 0.1997 seconds (19.0x faster), and optional file-backed mmap took 0.1942 seconds. The mmap mode lets the operating system keep hot pages in memory and evict cold pages to disk; it is optional and not used in the pilot because RAM was available. See [KV_CACHE.md](KV_CACHE.md).

## Artifacts And Reproduction

The complete machine-readable manifest, per-arm reports, validation snapshots, raw generations, comparison JSON, and local checkpoints are in `runs/batch4_pilot_20261125_kvcache/`. Checkpoint SHA-256 values are:

- A: `bd922303bbcbabdf0482640b9a3fe81c4e84ea6d7b70a86f37daf0c4dee1329e`
- B: `f6e09b12b95712030c61489e94d6c243d52c0fbe78ae2bdd6b36aadd653dd86d`
- C: `299ec801079e99846ba1b0a0445a01e5fb6b4df5c298ba3858a09bd5503597ed`

The training-time runner code hash recorded in all three arm reports is `95291ff32c5bfeda397e2b6b4d45df940465b6c5d12343edbb4eb0e0459ffdbf`. The current runner hash is `5f37a438634e8a9272d923c7d23fd7846565ea1d42e35ed608508f57d8dde2e7`; after training, the runner changed only to correct target-count labels, finalize report/manifest rendering, and prevent mmap cache files from being overwritten. Training, loss, data, and greedy decoding behavior were unchanged.

Checkpoint binaries are local and ignored by Git; the JSON reports and configs record their hashes. Arm configurations are [A](configs/arm_a.json), [B](configs/arm_b.json), and [C](configs/arm_c.json). An earlier evaluation-only attempt without KV caching was stopped before any training update and is recorded separately as `runs/batch4_pilot_20261125/experiment_manifest.json`.

Smoke and pilot commands, using the tested CPU PyTorch environment:

```sh
/tmp/microworld-b3-venv/bin/python -m research.microworld.train \
  --smoke --output research/microworld/runs/batch4_smoke_kvcache_20261125.json

/tmp/microworld-b3-venv/bin/python -m research.microworld.train \
  --pilot --output research/microworld/runs/batch4_pilot_20261125_kvcache
```

The smoke passed finite forward/backward losses for all arms, evaluation, generation, memory guard checks, and exact two-update resume equality for model, optimizer, and scheduler state. Test command:

```sh
/tmp/microworld-b3-venv/bin/python -m unittest \
  research.foundation.test_kv_cache research.foundation.test \
  research.microworld.test_training research.microworld.test_microworld \
  research.microworld.test_dense
```

That suite passed 34 tests. The venv emits a non-fatal warning that NumPy is not installed; these tests do not require NumPy.

The shared TinyGPT and training-efficiency regression modules also passed separately: `research.experiments.test_tiny_gpt` plus `research.experiments.test_baseline_transformer` passed 7 tests, and `research.experiments.test_training_efficiency` passed 3 tests. They are run in separate Python processes because multiple suites set PyTorch's process-global interop-thread count.
