# Next Steps

The evaluation harness and fixtures currently live in `/home/codespace/research`; this directory is not a Git checkout. Once the project repository is attached, keep these files in the repository's `research/` directory or link/copy them deliberately after checking for existing files.

1. From the repository root, rerun the scorer smoke suite:

   ```sh
   timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
     python3 -m unittest -v research/experiments/test_eval_harness.py
   ```

2. Reproduce all three references (each command writes a JSON result):

   ```sh
   timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
     python3 research/experiments/eval_harness.py --baseline abstain --seed 1234 \
     --output research/experiments/results/baseline_abstain.json
   timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
     python3 research/experiments/eval_harness.py --baseline random --seed 1234 \
     --output research/experiments/results/baseline_random_seed1234.json
   timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
     python3 research/experiments/eval_harness.py --baseline oracle --seed 1234 \
     --output research/experiments/results/oracle_scorer_selfcheck.json
   ```

3. For a backend, create a UTF-8 JSONL file with one output per fixture ID, then pass `--predictions PATH --output PATH`. Example row:

   ```json
   {"id":"arithmetic","answer":"43","abstained":false,"confidence":0.9,"tokens_generated":1}
   ```

   Evidence QA rows should also provide `claims` as atomic `{ "text": ..., "evidence_ids": [...] }` items; coding rows provide `code`. The output JSON preserves raw predictions. Do not execute untrusted generated code: current code subprocess timeout does not isolate filesystem or system calls.

4. Before using external benchmarks, review the exact pinned source and separate dataset terms in `REPOSITORY_LEDGER.csv`; run only a tiny licensed subset with a timeout and memory guard. Keep track metrics separate and preregister which are used for a comparison.

5. To make a model comparison, freeze fixture/task versions, run a deterministic baseline and at least three model seeds when runtime permits, add a held-out set not used for prompt/model selection, and report uncertainty plus CPU/RSS/token/latency costs. Do not optimize against the toy suite as a proxy for general intelligence.

## Batch 4 Baseline

The clean trainable baseline is [`tiny_gpt.py`](experiments/tiny_gpt.py) with [`baseline_config.json`](experiments/baseline_config.json). It is pinned by the config and result JSON; do not modify it during ablations. The isolated CPU PyTorch environment is reproducible from [`requirements_cpu.lock.txt`](experiments/requirements_cpu.lock.txt). Reproduce in that environment:

```sh
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/b4-torch-venv/bin/python research/experiments/train_tiny_lm.py --config research/experiments/baseline_config.json --output research/experiments/results/baseline_training.json --steps 250 --min-available-mib 2560
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/b4-torch-venv/bin/python -m unittest -v research/experiments/test_tiny_gpt.py
```

The training fixture is a single periodic token sequence and only demonstrates that the optimizer can memorize it. Before comparing architecture changes, define licensed held-out train/validation/test text, freeze tokenization and evaluation, then change one model variable per experiment.

## Batch 5 Sequence Mixers

Batch 5 leaves the Batch 4 baseline unchanged. The current result is preliminary and its retention prototype did not learn exact copy under the tested recipe. Reproduce the three-seed task comparison and the separate longer copy follow-up with the commands in [`sequence_mixer_landscape.md`](reports/sequence_mixer_landscape.md). Do not compare its decode latency against a hypothetical optimized Transformer; the preserved baseline does not implement KV caching.
