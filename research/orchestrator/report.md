# Patch 10: Offline orchestration

## Result

**Mechanics pass; capability unproven.** The end-to-end structured fixture path retrieves a provenance-bearing chunk, selects a validated strategy as data, makes an initial model call and bounded recurrent calls, verifies an exact structured fact, and returns a citation. The controls below use a deterministic probe model and test orchestration behavior, not reasoning quality.

| Condition | Answer | Abstained | Retrievals | Strategies | Refinement calls | Loop steps | Verifier calls |
|---|---|---:|---:|---:|---:|---:|---:|
| full | blue | False | 1 | 1 | 1 | 3 | 1 |
| no_retrieval | none | True | 0 | 1 | 1 | 3 | 1 |
| no_strategy | blue | False | 1 | 0 | 1 | 3 | 1 |
| no_refinement | blue | False | 1 | 1 | 0 | 1 | 1 |
| no_verifier | none | True | 1 | 1 | 1 | 3 | 0 |

## Neural Smoke

The existing frozen Batch 12 byte decoder has 22,432 parameters and context 32. On a short direct request it used 2 model calls and requested 3 shared-block steps, then returned `None` with reason `model_output_unverified`. The checkpoint hash remained unchanged. This does not test grounded answer quality; the model's 32-byte context cannot represent most ordinary evidence chunks and this checkpoint is trained only on a tiny synthetic fixture.

## Budgets

The synthetic mechanics run completed in 0.075s with 339,529,728 bytes peak RSS, one Torch thread, CPU execution, 6195 MiB available at completion and a 2,560 MiB memory guard. Retrieval and refinement caps are enforced per request; each result includes resource counters and full chunk provenance.

## Limits

Natural-language request parsing is explicit and typed: callers provide the retrieval query, task family and optional structured fact key. Retrieved text and strategy procedures remain separate data fields and are never executed. The verifier checks only exact equality against validated structured facts and their retrieved evidence links; it is not natural-language entailment. Learning is opt-in, requires matching persistence policy (including private-data permission), and appends episodes only. It cannot create or promote strategies. No web, shell, subprocess or generated-code path exists.

The machine-readable run record includes hashes, per-control counts, timings, memory, checkpoint identity and limitations in `result.json`.
