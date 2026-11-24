# Stage A Pilot Diagnosis

## Conclusion

**Restart contribution: supported in the controlled diagnostic, with low
confidence.** At identical starting weights, data order, token budget, and
evaluation, the restarted-schedule Arm C finished with higher validation
language NLL than both genuine continuation Arm A and optimizer-reset,
matched-schedule Arm B. This is evidence that the schedule restart can
contribute to the regression. It does not establish that it explains the
entire 1M-position pilot regression.

The diagnostic is deliberately small: one seed, 16 updates, 2,000 target
positions per arm, and 625 evaluated target positions for a requested
512-position validation budget (the final batch is consumed whole). The
differences are modest, and this is not a multi-seed estimate.

## Artifact Findings

The prior checkpoint [`learn.pt`](../foundation/data/learn.pt) is present
locally with SHA-256
`76b60e84c3b203ef1a30bd93f0fd7e8c8fbe619d08799e57e7f434a6c237a5fe`. Its
payload has AdamW state (52 parameter entries), scheduler state at update
800, RNG states, and a sampler cursor/buffer. The 1M-position pilot
[`a1.pt`](../foundation/data/a1.pt) is also present with SHA-256
`46d0f379d86500be14e6a9812aa254c515d7ce704bb7b17a920acd24ec2529be` and
contains pilot-end optimizer/scheduler state at update 8,000. The Stage A
report and code confirm that this latter state began from freshly initialized
optimizer and scheduler state; it cannot serve as the original state at the
pilot start.

All checkpoints and arms use the same 4,470,272-parameter architecture,
tokenizer, corpus, and train/validation split. Exact hashes are in
[`experiment_manifest.json`](experiment_manifest.json) and the generated
[`arm manifest`](pilot_arms/manifest.json).

## Controlled Comparison

Each arm starts from the exact weights in `learn.pt`. The input sequence was
continued from its saved legacy sampler cursor and prefetched buffer. The
same precomputed windows were used in every arm; their shared sequence hash
is `8852766e65074767f765dac54329ac478ce74bd9a9fb8ef4976b724731649035`.
Each consumed 16 training windows and 2,000 target positions. Validation
used the same deterministic sampler, seed 2027, and budget for all arms.

| Arm | State | Start optimizer entries | LR used, first to last | Validation language NLL | Assistant-only NLL | Greedy sample after training |
|---|---|---:|---|---:|---:|---|
| A | Restore original optimizer and scheduler | 52 | 0.000278788 to 0.000278333 | 4.714762 to 4.657043 | 4.605471 to 4.583638 | `<user> S a a a a ` |
| B | Reset optimizer; restore Arm A scheduler trajectory | 0 | 0.000278788 to 0.000278333 | 4.714762 to 4.680023 | 4.605471 to 4.599065 | `<user> a a a a a ` |
| C | Reset optimizer and scheduler as in the pilot | 0 | 0.000030 to 0.000048 | 4.714762 to 4.701696 | 4.605471 to 4.605134 | `<user> a a the the the ` |

Arm B's full learning-rate list exactly matches Arm A's. Arm C begins at the
floor-limited warmup rate used by the historical pilot. All three arms report
actual used and post-step rates, per-window hashes, complete validation
metrics, and token IDs/text for their greedy sample in their JSON reports:
[`arm_a.json`](pilot_arms/arm_a.json),
[`arm_b.json`](pilot_arms/arm_b.json), and
[`arm_c.json`](pilot_arms/arm_c.json).

The historical 1M run also reset the data cursor and began at the start of
the sequential stream. For the diagnostic, all arms instead begin at the
exact next position after `learn.pt`, as required to hold training examples
constant. Thus Arm C reproduces the pilot's optimizer/schedule reset but
deliberately does not reproduce its data-position reset. The three-arm
comparison isolates schedule and optimizer effects on a common next segment;
it does not attribute changes caused by the historical data reset.

## Training Infrastructure Fixes

`trainer.run` now starts its fit deadline after initial validation and gives
both initial and final validation independent bounded budgets. Evaluation
returns `complete` and `requested` fields; the curriculum runner refuses to
write segment metrics if either evaluation is incomplete. This fixes the
segment deadline issue that previously allowed empty final validation
metrics.

New training runs use a seeded permutation of whole training episodes at
each epoch. The sampler stores its shuffled order, episode position, buffered
tokens, epoch, and RNG state in the checkpoint. A compatibility sampler
remains available for old sequential checkpoints. The diagnostic explicitly
uses that legacy sampler to preserve the original checkpoint's exact data
position and does not silently change order in the schedule comparison.

The new sampler is tested for seeded order and state replay. The trainer's
existing uninterrupted-versus-interrupted test checks identical model,
optimizer, scheduler, sampler, Python RNG, and Torch RNG states.

## Commands And Tests

Diagnostic command:

```sh
timeout 180s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/audit-pytorch-venv/bin/python -m research.foundation.diagnose \
  --output research/structure/pilot_arms --updates 16
```

The focused regression suite and baseline test counts are recorded in
[`TEST_RESULTS.md`](TEST_RESULTS.md). The final run includes the sampler
replay test and passed 52/52 tests across Foundation, interaction,
orchestration, adaptive symbolic extraction/transfer, and recursive model
regressions.

No larger foundation training run was launched. No model architecture,
historical checkpoint, training corpus, or prior report was modified.
