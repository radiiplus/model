# Patch 9: Strategy adapter experiment

## Decision

**NO-GO**: adapter must beat the symbolic strategy-memory condition.

The recurrent byte model was loaded from the existing Batch 12 checkpoint. Its parameters were frozen. A rank-4 LoRA residual was attached to `block.attn.proj`, trained from the validated Patch 8 strategy's 12 chain training episodes, and evaluated with the adapter explicitly selected. The rejected weight artifact was temporary and was deleted after measurement.

## Results

Each table cell shows exact matches per seed, with interpolation and extrapolation kept separate.

### addition

| Seed | Split | Base | Adapter | Samples |
|---:|---|---:|---:|---:|
| 911 | interpolation | 0 | 0 | 4 |
| 911 | extrapolation | 0 | 0 | 4 |
| 912 | interpolation | 0 | 0 | 4 |
| 912 | extrapolation | 0 | 0 | 4 |
| 913 | interpolation | 0 | 0 | 4 |
| 913 | extrapolation | 0 | 0 | 4 |
### bracket

| Seed | Split | Base | Adapter | Samples |
|---:|---|---:|---:|---:|
| 911 | interpolation | 0 | 0 | 4 |
| 911 | extrapolation | 0 | 0 | 4 |
| 912 | interpolation | 0 | 0 | 4 |
| 912 | extrapolation | 0 | 0 | 4 |
| 913 | interpolation | 0 | 0 | 4 |
| 913 | extrapolation | 0 | 0 | 4 |
### chain

| Seed | Split | Base | Adapter | Samples |
|---:|---|---:|---:|---:|
| 911 | interpolation | 0 | 0 | 4 |
| 911 | extrapolation | 0 | 0 | 4 |
| 912 | interpolation | 0 | 0 | 4 |
| 912 | extrapolation | 0 | 0 | 4 |
| 913 | interpolation | 0 | 0 | 4 |
| 913 | extrapolation | 0 | 0 | 4 |
### graph

| Seed | Split | Base | Adapter | Samples |
|---:|---|---:|---:|---:|
| 911 | interpolation | 0 | 0 | 4 |
| 911 | extrapolation | 0 | 0 | 4 |
| 912 | interpolation | 0 | 0 | 4 |
| 912 | extrapolation | 0 | 0 | 4 |
| 913 | interpolation | 0 | 0 | 4 |
| 913 | extrapolation | 0 | 0 | 4 |
### trace

| Seed | Split | Base | Adapter | Samples |
|---:|---|---:|---:|---:|
| 911 | interpolation | 0 | 0 | 4 |
| 911 | extrapolation | 0 | 0 | 4 |
| 912 | interpolation | 0 | 0 | 4 |
| 912 | extrapolation | 0 | 0 | 4 |
| 913 | interpolation | 0 | 0 | 4 |
| 913 | extrapolation | 0 | 0 | 4 |

## Transfer and safeguards

Patch 8's symbolic strategy scored 100.0% on the same held-out trace rows; the neural adapter scored 0.0%. It therefore did not beat the symbolic condition. The frozen foundation parameter hash remained `f8f5a8fb44cc4e067e158ca468d1a248273124c974372176ae6e845b88b2a9b4`. The recurrent foundation has 37,408 frozen parameters, including 15,360 deterministic extended position parameters and 64 recurrence controls. One rank-4 adapter has 256 trainable parameters; active total is 37,664, below the 1,000,000 limit.
Adapter fitting used 12 chain episodes, 120 updates and 240 forward block applications in 0.565s (0.0047s/update). Across 120 held-out generated cases, the adapter scored 0/120; per-family counts were addition 0/24, bracket 0/24, chain 0/24, graph 0/24, trace 0/24. The overall run took 10.888s and peaked at 361,492,480 bytes RSS.

The adapter-only checkpoint contains no foundation tensors and records strategy, strategy version, dataset, evaluator, task and foundation hashes. A prior adapter state was restored exactly. Two adapter versions were measured together as an interference probe, but combination was not activated. No adapter is automatically selected or updated during ordinary inference.

## Limits

The only available neural foundation is a 21,984-parameter model trained on a tiny synthetic text fixture. The model and all reasoning inputs remain synthetic. Context was extended from 32 to 512 with a deterministic frozen positional encoding; the first 32 learned positions were copied exactly. This makes the full task prompts fit without truncation but is an architectural change that may affect behavior. Results are a CPU mechanics experiment, not real-world reasoning or interaction quality.

Per-instance outputs, per-seed task metrics, hashes, resume data, parameter counts, interference results and resource data are in `neural.json`.
