# Recurrent Candidate

## Structure

`Refiner` has the same token embedding, learned absolute positions, final
LayerNorm, and four independently parameterized `DecoderBlock`s as the plain
`Foundation`. Those four blocks form one shared stack. Every outer refinement
applies that same stack; no additional Transformer stack is allocated.

The state tensors have shape `[batch, sequence, d_model]`:

- `source` is the unchanged embedding-plus-position representation.
- `latent` is the intermediate workspace passed through the shared four-block
  stack.
- `answer` is the continuous state used by the tied output projection.

For each step, the input gates mix `source`, the prior `answer`, and prior
`latent` before the shared stack updates `latent`. Output gates then update
`answer` from `source`, the new latent state, and prior answer. All gates start
at zero. This makes the first transferred pass exactly the plain four-block
model while keeping separate latent and answer tensors. The gate table has
`limit * 4 * d_model` parameters; with limit four it adds 4,096 parameters.

The implementation adapts dual-state refinement to next-token language
modeling: `answer` remains continuous, every stage predicts next-token logits,
and assistant-only target masks apply at each supervised stage. The latent
state is not decoded or exposed as a reasoning trace. This is an architectural
hypothesis, not a reproduction of a specialized puzzle solver.

## Fair Comparison

The one-seed comparison trains three arms from the same transferred initial
weights and deterministic stream order:

- Plain K=1 and recurrent K=2 both receive the same first two batches for the
  matched-token comparison.
- Plain K=1 receives twice as many updates as recurrent K=2 for the
  matched-compute comparison. The plain arm therefore sees more tokens; its
  batches continue the same ordered stream, and the shared prefix is checked.

Each plain update applies four blocks. Each recurrent K=2 update applies the
same four blocks twice. Reports give optimizer updates, tokens, block calls,
block-token positions, wall time, and throughput separately. The one-seed
result is a bounded pilot; do not expand to three seeds unless the plain arm
shows held-out learning on the intended interaction behavior.

## Pilot Results

Run seed 2025 used 64-token batches, batch size 1, AdamW at 3e-5, and two
updates for the token-matched pair. The plain model has 4,470,272 parameters;
the recurrent model has 4,474,368, including 4,096 gates. Its four-block
shared stack has 3,159,040 unique parameters.

| Condition | Target tokens | Block calls | Block-token positions | Train seconds | Target tokens/s | Block-token positions/s | Validation language loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain K=1, token matched | 128 | 8 | 512 | 0.149 | 860 | 3,440 | 7.7629 |
| Recurrent K=2, token matched | 128 | 16 | 1,024 | 0.199 | 644 | 5,155 | 7.7625 at refinement 2 |
| Plain K=1, compute matched | 256 | 16 | 1,024 | 0.250 | 1,026 | 4,102 | 7.3787 |

Both matching checks passed: the token-matched arms saw identical ordered
presentations, and the compute-matched arms used equal block-token positions.
Before training, validation language loss was 8.3869. Recurrent K=1 after K=2
training was 7.7629; K=2 refinement losses were 7.7629 and 7.7625. The fixed
validation slice contained zero assistant answer targets, so interaction loss
and answer behavior were not measured.

The recurrent fit helper writes atomic segment checkpoints containing model,
optimizer, Python/Torch RNG, sampler cursor, run identity, and metrics. A K=2
resume test split a two-update run after update one and matched the uninterrupted
model and optimizer state exactly. The comparison pilot itself stays in memory
and writes only its report.

The full run took 1.445 seconds including evaluation, peaked at 549,072,896
bytes RSS, ran CPU FP32 with one Torch thread, and used the existing 2,560 MiB
memory guard. K=1 transfer was bitwise equal before training. This pilot does
not establish that the plain arm has learned the intended interaction
behaviors, so no three-seed expansion is justified. It is not evidence of
recurrent superiority, language quality, or interaction capability. Detailed
hashes and per-update losses are in [`pilot.json`](pilot.json).
