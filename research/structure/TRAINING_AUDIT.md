# Structural Training Pilot

## Conditions

All three conditions use the 4,470,272-parameter Foundation, initialized from
`research/foundation/data/learn.pt` at update 800. The model configuration and
initial state are identical. The checkpoint optimizer and scheduler states
were restored in every arm; the scheduler continued at update 800. No auxiliary
head or other parameters were added.

Arm A is conditional language modeling over answer tokens. Arm B predicts the
canonical structural JSON target. Arm C uses the mean answer and structural
losses. All arms process the same paired input, answer, and structure
examples, but only the selected objective contributes gradients in A and B.
This puts direct gradient on the requested answer while preserving the
conventional text-generation objective. Data order is shuffled from seed 2026
(training seed 2025 plus one) and saved in the atomic checkpoint.
Training-order and starting-weight hashes match across all arms.

## Runs

Each smoke run used two updates and two examples. Each bounded pilot used the
predeclared 12 updates, batch size one, and the same 12 examples from the
corrected v5 dataset. The pilots consumed 14,643 input tokens each; target
counts were 4,807 full-sequence language tokens, 183 answer tokens, and 5,212
structural tokens. Active objective token counts were 183 for A, 5,212 for B,
and 5,395 for C. The learning rate started at
0.0002787878788 and ended at 0.0002784545455 in every arm, continuing the
existing schedule. Each per-update LR and loss is recorded in the arm JSON.

A full-size A smoke run was also interrupted after update one and resumed
from its atomic checkpoint. Its final model, optimizer, scheduler, sampler,
history, step, Python RNG, and Torch RNG states all exactly match the
uninterrupted two-update A smoke run.

| Arm | Final train objective loss | Validation answer NLL after | Validation structure NLL after | Answer greedy exact / semantic | Structural token accuracy | Teacher-forced JSON valid |
|---|---:|---:|---:|---:|---:|---:|
| A: answer language | 5.5077 | 4.8334 | 7.6415 | 0 / 0 | 0.1964 | 0 |
| B: structure | 5.7145 | 6.0565 | 5.5429 | 0 / 0 | 0.3788 | 0 |
| C: combined | 5.8416 | 5.3992 | 5.8438 | 0 / 0 | 0.2084 | 0 |

Before-training metrics are identical across arms because weights and held-out
rows match. All five held-out partitions are evaluated before and after each
run. Teacher-forced scoring uses six examples per partition, one per surface
family. The evaluator separately autoregressively generates answers for two
examples per partition, up to 32 tokens, and checks exact text and normalized
answer-value matches. `structure_teacher_forced_json_valid` only parses
per-position argmaxes conditioned on the gold prefix; it is not a free-running
JSON generation score. All free-generated answer exact and semantic matches,
and all teacher-forced structural exact/JSON-valid scores, are zero. Sampled
outputs terminate immediately at EOS in A/C and repeat quotes in B. This
small evaluation slice is diagnostic rather than a stable estimate.

Throughput, fit time, peak RSS, per-update objective losses, validation
metrics by split, selected evaluation IDs, and atomic checkpoint hashes are
in `pilot_v5/arm_a.json`, `pilot_v5/arm_b.json`, and `pilot_v5/arm_c.json`. The
machine summary and artifact hashes are in `training_manifest_v5.json`.

## Reproduction

Run the targeted regression suite:

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 /tmp/audit-pytorch-venv/bin/python -m unittest -v research.structure.test
```

The commands below write fresh outputs; choose new output paths when repeating
them because existing reports are intentionally not overwritten.

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 /tmp/audit-pytorch-venv/bin/python -m research.structure.train \
  --config research/structure/configs/smoke.json --arm a \
  --initial research/foundation/data/learn.pt --data research/structure/dataset_v5 \
  --output research/structure/smoke_v2/a.pt --report research/structure/smoke_v2/arm_a.json
```

Repeat the smoke command with `--arm b` or `--arm c` and matching output and
report paths. This is the bounded pilot command for A:

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 /tmp/audit-pytorch-venv/bin/python -m research.structure.train \
  --config research/structure/configs/arm_a.json --arm a \
  --initial research/foundation/data/learn.pt --data research/structure/dataset_v5 \
  --output research/structure/pilot_v5/a.pt \
  --report research/structure/pilot_v5/arm_a.json
```

Use `arm_b.json` and `arm_c.json` with `--arm b` and `--arm c`, changing the
output and report filenames to match.

## Interpretation

The source audit found why v1 exact-match results were misleading and weak.
Natural-language prompts omitted properties, constraints, and complete state
information present in structural targets, making full recovery
underdetermined. Some held-out ownership prose reversed edge direction. The
original evaluator called teacher-forced token argmax an exact answer score;
it did not test free generation. The 12-update exposure is also tiny for this
4.47M-parameter byte-BPE model. Its checkpoint is trained with conversational
next-token and assistant-response losses but has no structural objective or
reasoning module. The pinned environment includes NetworkX and SymPy, but the
Foundation and structural training path imports neither; the exact graph
oracle is implemented directly in the generator. v5 makes the target
observable, fixes direction wording, focuses A/C loss on answers, and adds
free-generation checks.

On v5, B improves validation structure NLL from 7.729 to 5.543 and structural
token accuracy from 0.207 to 0.379, while answer NLL worsens. C improves
structure NLL to 5.844 but has little structural token-accuracy gain. A
reduces answer NLL from 5.344 to 4.833, yet free-generated answers still
terminate at EOS or produce repeated punctuation. The data and objective
corrections improve observability and teacher-forced fitting, but this pilot
does not show usable exact reasoning or answer generation. No larger
foundation run should be started from this result alone.
