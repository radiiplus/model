# Structural Transfer Evaluation

## Decision

**NO-GO for Batch 5 and for promoting structural supervision.** The structural and combined conditions did not beat the conventional model on compositional tasks; all three produced zero semantically correct free-running answers and zero exact structural reconstructions in this audit. The hybrid verifier abstained on every neural answer. The pilot signal is not strong enough to justify another training seed or a value-driven sampling experiment.

This decision is specific to the bounded v5 pilot. It does not establish that structural supervision cannot work; it establishes that these checkpoints do not demonstrate the capability required by the promotion gate.

## Setup

The evaluator scores the three existing 12-update checkpoints (`a`, `b`, `c`) from `pilot_v5`. All have 4,470,272 parameters and share the same starting weights, tokenizer, dataset and training exposure. No model or training changes were made for this evaluation. The dataset is `dataset_v5`, manifest SHA-256 `a0aa954d5ef297b4b93b508427216e5aac02c0ee62aff1a1a0ec8ebf28d87880`.

The held-out sample is selected by underlying world ID, not by sentence. It includes the first world in each ordinary partition, both patterns in `pattern_test`, both counterfactual actions for its selected world, and all six surface forms. Every selected input is decoded freely. Semantic answer scoring normalizes case and punctuation and compares the answer content to the deterministic oracle value; it does not require matching the surrounding answer template. Structural output is parsed as JSON and compared field-by-field with the exact canonical target. Invalid JSON counts as a failed reconstruction and as a constraint violation.

The existing strategy extractor is given only the 5,760 training episodes, mapped to its episode schema with train labels. It is then called on the same held-out prompts used by the neural evaluation. The extractor proposes its existing `memorize` procedure; held-out prompts have no matching hashes. A separate explicit-record solver parses the structured surface, infers the query type from the question text, then applies the project’s deterministic state/query rules. It receives neither the hidden query label nor answer/world fields. This is an auditable symbolic ceiling, not a learned model. Neural structured-input scores use only the same structured-surface rows. The hybrid verifier returns a neural candidate only when it matches the symbolic result and otherwise abstains; the symbolic answer is never returned as the neural answer.

## Results

| Condition | Answer accuracy, IID | Answer accuracy, compositional | Answer accuracy, held-out patterns | Exact structure, compositional | Constraint violation, compositional |
|---|---:|---:|---:|---:|---:|
| A: conventional | 0/24 | 0/24 | 0/48 | 0/6 | 6/6 |
| B: structural | 0/24 | 0/24 | 0/48 | 0/6 | 6/6 |
| C: combined | 0/24 | 0/24 | 0/48 | 0/6 | 6/6 |

The same zero answer accuracy and zero exact reconstruction held on validation and counterfactual partitions. All three conditions had zero entity-binding F1, edge-recovery F1, transition accuracy, and parseable JSON. Counterfactual pair consistency was 0/6 pairs for each model. Free answers were also zero across natural, paraphrased, reordered, irrelevant, counterfactual, and explicit structured descriptions.

| Baseline | IID | Compositional | Held-out patterns | Counterfactual |
|---|---:|---:|---:|---:|
| Existing strategy extractor, exact-memory procedure | 0/24 | 0/24 | 0/48 | 0/48 |
| Explicit structured rule solver, structured surface only | 4/4 | 4/4 | 8/8 | 8/8 |

On the pooled selected structured prompts, the train-majority answer baseline scored 21.4%. A seeded random answer drawn by query type averaged 17.9% across five seeds (standard deviation 3.2 percentage points).

For the matched explicit-record subset, A, B and C each scored 0% (0/4 IID, 0/4 compositional, 0/8 held-out patterns, 0/8 counterfactual). The rule solver scored 100% on those same prompts. The hybrid verifier accepted 0 candidates and had 0% coverage for every condition and partition.

The Wilson 95% intervals are in `evaluation_metrics.json`. They are computed over sampled examples; examples within a world share structure and are correlated, so these intervals understate uncertainty about generalization to new worlds. The audit samples only one world in ordinary partitions and two in the held-out-pattern partition. It is therefore a strong failure signal for these checkpoints, not a precise estimate of population performance.

## Failure Cases

- On the structured IID influence question for `iid_test-world-000`, the oracle answer is `Larch`. A and C emit no answer and stop at EOS; B emits eight quote characters and stops under the repetition guard.
- On the corresponding structural reconstruction prompt, A and C again stop at EOS. B emits the same repeated-quote sequence. None emits parseable JSON; exact reconstruction, binding, edge, state and constraint scores all fail.
- The same generation failure appears on paraphrased and reordered variants, and on both held-out graph patterns. No representation surface produced a semantic answer from these checkpoints.

The deterministic rule solver correctly reconstructs answers on explicit structured records. The strategy extractor's zero held-out result is expected for its extracted `memorize` procedure and should not be interpreted as a failed graph algorithm: it is a task-interface-matched baseline for the existing extractor's actual capability.

## Reproduction

```bash
timeout 240s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/audit-pytorch-venv/bin/python -m research.structure.evaluate
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/audit-pytorch-venv/bin/python -m unittest -v research.structure.test_evaluate research.structure.test
```

The first command writes `evaluation_metrics.json`, including per-example predictions, per-surface results, baseline reports, confidence intervals, checkpoint hashes, and counterfactual consistency. The second command runs the evaluator tests plus the existing structural data/training regressions. Result: **16 tests passed**.

Batch 5 is not authorized by this result. A new hypothesis and a training/evaluation change that addresses the observed EOS/repetition and zero-JSON failure would be needed before reconsidering structural promotion.
