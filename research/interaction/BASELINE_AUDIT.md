# Baseline Audit

Audit date: 2026-11-24
Branch: `adaptive`
Starting commit: `9b6f9e6b0e59839618cc80cfa4508c6e2019d9e7`

This note reconciles the existing experiments before interaction-system work. It preserves the distinction between language-model training, recurrent smoke tests, symbolic task results, memory mechanics, and end-to-end orchestration. None of the current results establishes useful open-domain interaction capability.

## Evidence Reconciliation

| Run | Parameters | Training | Held-out result | Interpretation |
|---|---:|---|---|---|
| Batch 12 byte LM | 21,984 | 60 updates per seed, seeds 11/23/37; one-layer byte TinyGPT, context 32; 18 project-authored synthetic documents | Validation BPB 8.001 to 6.079; test BPB 7.989 to 6.879; test next-byte accuracy 3.61% to 8.11% | Learned byte correlations on a tiny synthetic text fixture. This is the only listed result that measures held-out text prediction; it is not evidence of coherent conversation, reasoning, or broad language capability. |
| Patch 5 recurrent language smoke | 23,264 | Two updates; dual-state model, K=2, deep supervision; 128 target bytes processed | Validation NLL 5.5379 nats/byte, or 7.9894 bits/byte, over 1,024 bytes; next-byte accuracy 0.39% | A two-update pipeline smoke. Do not compare this value as a meaningful improvement over Batch 12: architecture, checkpoint, training budget and evaluation procedure differ, and the run is far too small. |
| Patch 6 recurrent ablation | 37,344-75,456 depending on condition | Three seeds, four updates per condition on generated tasks | No exact-match gain or graph-to-chain transfer; held-out text BPB about 7.97-7.98; registered gate returned no-go | Small synthetic ablation. It does not show that recurrence cannot help under other data/budgets, but it gives no basis to promote the tested loop. |
| Patch 9 strategy adapter | 37,408 frozen recurrent base plus a 256-parameter adapter | 120 adapter updates from 12 synthetic episodes | Base and adapter each scored 0/120 generated held-out cases; symbolic strategy scored 100% on its selected trace slice | Adapter mechanics only; no neural gain. Adapter was not retained or selected. |
| Patch 10 orchestration | Frozen 22,432-parameter byte decoder | Bounded integration control; no training | Synthetic structured path passed; direct neural smoke abstained; checkpoint hash unchanged | Validates typed plumbing, provenance and budgets only. The verifier is exact matching on structured facts, not natural-language entailment. |

The primary Batch 12 record is [`batch12_tiny.json`](../experiments/results/batch12_tiny.json), with configuration, split, source and checkpoint hashes in [`batch12_manifest.json`](../prototype/batch12_manifest.json) and the summary in [`integrated_tiny_prototype.md`](../reports/integrated_tiny_prototype.md). The recurrent two-update record is [`training.json`](../recursive/training.json). These records must remain separate; the Batch 12 improvements are not attributable to the recurrent model.

## Reusable Training Machinery

- [`tiny_gpt.py`](../experiments/tiny_gpt.py) is the preserved decoder-only causal Transformer reference. Batch 12 wraps it with byte IDs and a separate config/checkpoint.
- [`prototype/trainer.py`](../prototype/trainer.py) provides CPU FP32/AdamW training, deterministic Torch setup, the 2,560 MiB cgroup-aware memory guard, a swap-activity stop check, atomic checkpoint replacement and exact resume for its tested sampler/RNG state.
- [`prototype/data.py`](../prototype/data.py) samples individual byte windows lazily from document token arrays and compact prefix offsets. It avoids allocating one tensor per possible window.
- [`recursive/train.py`](../recursive/train.py) supports plain, shared-loop, dual-state and independent-stack controls; target-only task masks; deep or final-stage loss; fixed K; CPU resource checks; and checkpoints that bind config, architecture, split hashes and refinement settings while saving optimizer, Python/Torch/sampler RNG, cursor and loss history. Exact-resume tests pass for the controlled fixture.
- [`recursive/model.py`](../recursive/model.py) and [`recursive/dual.py`](../recursive/dual.py) reuse one shared causal block at K=1/2/4/8. K changes compute, not unique block count.
- [`memory/store.py`](../memory/store.py) provides SQLite evidence chunks, provenance, immutable raw events, versioned facts/skills, stale/conflict handling and bounded lexical retrieval. [`orchestrator/flow.py`](../orchestrator/flow.py) connects that store to a frozen model and structured verifier with independent controls and an opt-in append-only episode queue.
- [`adaptive/loop.py`](../adaptive/loop.py) provides an auditable symbolic strategy archive and held-out promotion gate. It is not a neural learning loop.

## Gaps and Limitations

### Tokenizer and Data

- UTF-8 byte IDs are the only tokenizer used for model training. Byte/BPE/Unigram comparisons measured round-trip and compression on seven synthetic texts, not downstream model quality. There is no finalized chat token inventory or model-level tokenizer comparison.
- The repository contains only project-authored synthetic text and task fixtures; no realistic interaction corpus with a completed source/license review is checked in or locally available.
- Data ingestion is not end-to-end streamed. `load_documents()` reads the JSONL file into a string and dictionary; `load_splits()` materializes all encoded documents; `ByteWindowLoader` then keeps those arrays resident. The recursive language trainer also accepts fully materialized document lists. Window sampling is lazy, corpus loading is not.
- There is no source-family conversation split or contamination audit for a real interaction corpus. Existing split checks operate on the small synthetic document fixture.

### Model Architecture and Context

- The Batch 12 model is one Transformer block, width 32, vocabulary 256, tied byte embedding/head, context 32, and 21,984 parameters. Thirty-two bytes cannot carry an ordinary multi-turn interaction or realistic retrieved evidence.
- The shared recurrent decoder has 22,432 parameters at this config; the dual-state refiner has 23,264. The two-update recurrent language run is only a smoke test. The three-seed recurrent task ablation found no tested transfer gain.
- A separate adapter experiment extended the positional table to context 512 while keeping the source model frozen, but its adapter scored 0/120. This is not evidence of effective long-context behavior.
- The prototype has no trained, useful chat model, multi-turn objective tracking, learned memory, tested long-context generation, or model-level interaction-quality result. The recurrent decoder has no KV cache; repeated block applications cost additional inference compute.

### Checkpoints and Resume

- Batch 12 checkpoints store model, optimizer, config, seed, update, batch-generator RNG and Torch RNG. Resume checks config/seed and exactly reproduces the tested uninterrupted state. Dataset/split hashes are recorded in run manifests rather than required by the checkpoint resume signature; the trainer takes already-materialized documents. No scheduler state is needed by that trainer because it has no scheduler.
- The recursive trainer uses a stronger run signature including train/validation hashes, generator version, architecture and refinement settings, and saves optimizer, RNG, cursor and stage-loss history. Its data is still materialized, so the saved cursor is not a disk-stream offset into a large corpus.
- These tests establish deterministic resume at controlled checkpoint boundaries, not recovery from every process or filesystem failure mode. Source checkpoints and historical result files are reference artifacts and must not be overwritten.

### Problem and Memory Management

- There is no problem/session manager that stores an explicit objective, constraints, milestones, next action and unresolved questions across turns.
- The memory package stores evidence and derived facts/skills, but ingestion receives a complete Python string; it is not a streaming corpus or conversation log.
- The Patch 10 orchestrator accepts a typed request with caller-supplied retrieval query, task family and optional fact key. It does not parse ordinary language into needs. Its 32-byte model context is not usable for most retrieved chunks.
- The orchestrator's queue is opt-in and policy-gated, but stores an input hash rather than reusable conversation text and only accepts deterministic-verifier outcomes. It cannot serve as supervised interaction data and cannot promote strategies.
- There is no validated conversation compaction, rollbackable session projection, old-context retrieval workflow, correction/recovery evaluation, or mechanism that converts validated episodes into improved model behavior.

## Baseline Tests and Environment

The relevant regression command passed **69/69 tests** on this audit. The prior `/tmp/recurrent-audit-venv` was absent after the environment restart, so the pinned CPU lock was installed into `/tmp/interaction-venv`; no repository dependency file changed.

```sh
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/interaction-venv/bin/python -m unittest -v \
  research.orchestrator.test research.memory.test research.adaptive.test \
  research.recursive.test research.prototype.test_batch12 \
  research.experiments.test_tiny_gpt research.experiments.test_baseline_transformer \
  research.experiments.test_eval_harness research.experiments.test_retrieval_controller \
  research.experiments.test_claim_verification
```

At audit time the host exposed 8,131,856 kB total memory, 6,799,500 kB `MemAvailable`, no swap, cgroup v2 with no finite memory ceiling, and two visible logical CPUs. Model tests use one Torch intra-op/inter-op thread and the existing 2,560 MiB guard. The current audit changed no source, config, checkpoint or historical result. Only this audit note is new.

## Readiness

The repository has reusable CPU training, deterministic checkpointing, synthetic evaluators, bounded evidence memory and orchestration mechanics. It does not yet have realistic training data, end-to-end streaming, a useful context length, a session/objective manager, conversation compaction, or a realistic held-out interaction evaluator. Batch 12 demonstrates limited next-byte fitting on synthetic text; the Patch 5 recurrent smoke demonstrates two updates completed. Neither is evidence that the project can yet train a genuinely interaction-capable language model.
