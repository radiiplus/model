# Value-Driven Training Selection

## Result

**Efficiency hypothesis: unsupported by this bounded pilot.** Uniform selection matched or slightly beat the scored samplers on held-out structural NLL, while difficulty, learning-progress and RHO-style scoring added 61–64 seconds of selection work. The RHO result excludes the separate reference-model cost from its target-run wall time, so its full cost is higher still. No condition produced valid free-generated structural JSON or a semantically correct generated answer.

This is one paired seed and 24 updates. It is evidence against these sampler settings, not a general claim that data selection cannot help. Batch 4 had no positive capability signal, so there was no basis to spend more seeds on this pilot.

## Controls and Methods

All target runs used structural recovery (Model B), the same update-800 Foundation weights, restored optimizer and scheduler state, the same architecture and batch size one, the same tokenizer and dataset manifest, the same held-out evaluation data, seed `2025`, and 24 updates. Each run processed exactly **29,393 input tokens** and 24 training examples. The per-update token-cost schedule and each eight-example candidate pool were fixed before training and shared across all four modes. Candidate pools were disjoint from one another and came from the same target training corpus. Six training worlds were reserved from every target sampler for the RHO reference model.

Every sampler applies the same coverage rule: among a candidate pool, first prefer the least-used query/surface group, then the least-used world. This limits repeated focus on a narrow group. Selection scores are:

- `uniform`: random eligible candidate.
- `difficulty`: current structural loss; select randomly from the middle-to-high loss band, excluding the hardest tail when the eligible pool permits.
- `learning_progress`: structural loss under the common starting weights minus current structural loss. This measures cumulative loss reduction, not just current loss.
- `rho_loss`: current structural loss minus that candidate’s loss under a separate reference model trained on the reserved training-world holdout.

RHO-LOSS uses reducible holdout loss, the difference between target-model loss and an estimate from an irreducible-loss reference model. The selection score follows that calculation, but this is explicitly an **RHO-LOSS-style approximation**, not a full replication: the Foundation-size reference model was trained for 24 updates and was not converged. The published method and score definition are described in [Mindermann et al., 2022](https://arxiv.org/abs/2206.07137).

## Held-Out Results

Lower structural NLL is better. These are teacher-forced losses from the same independent split evaluation procedure used by the training reports.

| Sampler | IID | Compositional | Held-out patterns | Counterfactual | Validation |
|---|---:|---:|---:|---:|---:|
| Uniform | 4.0173 | **4.0102** | **3.9592** | **3.9944** | **4.0144** |
| Difficulty | 4.0196 | 4.0116 | 3.9612 | 3.9968 | 4.0157 |
| Learning progress | 4.0215 | 4.0136 | 3.9630 | 3.9954 | 4.0184 |
| RHO-LOSS-style | 4.0297 | 4.0226 | 3.9740 | 4.0054 | 4.0284 |

All methods began at the same validation structural NLL, `7.72945`. Uniform’s validation NLL gain was `3.71505`; difficulty gained `3.71378`, learning progress `3.71107`, and RHO-style `3.70107`.

| Sampler | Gain / 1,000 input tokens | Gain / update | Gain / target-run second | Gain / amortized end-to-end second | Selection seconds | Active objective tokens |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | **0.12639** | **0.15479** | **0.05033** | **0.03164** | 0.0 | 10,398 |
| Difficulty | 0.12635 | 0.15474 | 0.02700 | 0.02050 | 61.4 | 10,438 |
| Learning progress | 0.12626 | 0.15463 | 0.02678 | 0.02037 | 62.1 | 10,530 |
| RHO-LOSS-style | 0.12592 | 0.15421 | 0.02577 | 0.01374 | 64.0 | 10,479 |

The target runs each used 4,470,272 parameters and peaked at approximately 1.3 GB RSS. Target-run wall time includes train-run evaluation and selection. Common plan preparation took 174.4 seconds and is amortized evenly across the four conditions in the end-to-end column. RHO additionally took 82.2 seconds for its separate 24-update, 4,470,272-parameter reference fit and 192 candidate-loss evaluations. Its end-to-end figure includes that cost. The entire experiment took about 750 seconds.

Structural token accuracy improved from `0.2073` to `0.3042` with uniform, `0.3016` with difficulty, `0.3031` with learning progress, and `0.3042` with RHO-style sampling on validation. Held-out structural NLL improved for all methods, but the differences between samplers were small and did not translate into valid structural generation: teacher-forced JSON validity was zero for every method on every partition. Free-generated semantic answer accuracy was also zero. This repeats the Batch 4 gap between token-level loss improvements and usable structural behavior.

The sampler coverage rule reached 19–20 of the 24 query/surface groups in each run and exposed the selected rows to 22–24 underlying worlds. All runs selected 24 unique training examples. Active objective tokens varied from 10,398 to 10,530 even though full input-token budgets were exactly matched; these counts are included so this residual exposure difference is visible.

## Interpretation

The scored samplers did not improve generalization per token. Their NLL gains were nearly the same as uniform, with small regressions on each held-out partition. Their extra scoring passes cut validation-NLL gain per target-run wall second by roughly 46% for difficulty, 47% for learning progress, and 49% for RHO-style selection. The RHO-style method also requires a separately trained reference model.

The difficulty run was not a highest-loss-only sampler: it avoided the top loss tail when its eligible pool contained alternatives. This did not beat uniform. Learning-progress ranking measured actual loss reduction from the shared initial model, but showed no advantage. The RHO calculation used a separate training-only holdout and its published loss-difference form, but the short reference fit limits faithfulness and its results were weaker than uniform.

Conclusion: retain uniform sampling as the baseline. Do not promote value-based selection or spend further compute on these sampler settings without a new hypothesis and an evaluation that can detect free-generated structural capability. The Batch 4 evaluator’s small world sample also limits claims about population generalization.

## Artifacts and Reproduction

- `value_curriculum.py`: deterministic plan, reserved reference fit, and four matched target runs.
- `sampler.py`: uniform, difficulty, learning-progress and RHO-style selection with coverage balancing and exact sampler state.
- `comparison_metrics.json`: per-split NLL/accuracy, normalized gains, target-run and amortized end-to-end compute costs, matched-budget assertions and reference cost.
- `comparison.json`: hashes, per-condition checkpoints, training sequences and report names.
- `uniform.json`, `difficulty.json`, `learning_progress.json`, `rho_loss.json`: per-update training losses, selected candidate pools, checkpoints, held-out metrics, throughput and memory.
- `rho_reference_report.json`: reference-model training, data, score file, checkpoint hash and approximation limits.

Reproduce the full bounded experiment with:

```bash
timeout 1800s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/audit-pytorch-venv/bin/python -m research.structure.value_curriculum
```

Sampler and exact training-resume regression tests are in `research/structure/test_sampler.py` and `research/structure/test.py`.
