# Explicit Structural Head Pilot

## Result

**NO-GO for further training at this scale.** A fixed-slot structural head
trained for 12 updates did not recover any exact structure or exact transition
on the 12-example validation sample or on paired examples from the held-out
partitions. The JSON target baseline also emitted no valid structure in its
five bounded free-generation samples. The pilot does not support scaling either
objective.

The machine-readable controls and metrics are in
[`explicit_head_pilot/comparison.json`](explicit_head_pilot/comparison.json).
Per-arm histories and metrics are in the adjacent JSON reports. Checkpoint
files are generated locally and ignored by Git.

## Method

The Foundation now exposes its final normalized token states with
`Foundation.hidden()` while preserving the tied-weight language logits and
existing checkpoint keys. `explicit_head.py` attaches a 237,468-parameter
head to the final prompt state. It predicts six entity names and properties,
typed relation slots and endpoints, initial and resulting edge activity,
and action type. The 4,470,272-parameter backbone is unchanged; the head brings
the combined parameter count to 4,707,740, a 5.31% increase.

Three arms started from the same 800-update Foundation weights:

| Arm | Objective | Parameters |
| --- | --- | ---: |
| JSON | Autoregressive structural JSON | 4,470,272 |
| Slot structure | Explicit categorical and state outputs | 4,707,740 |
| Slot combined | Explicit outputs plus language-answer loss | 4,707,740 |

All arms used a reset AdamW optimizer, the same 12 example IDs in the same
order, 12 updates, 4,624 prompt tokens, a 2025 seed, the same tokenizer and
dataset manifest, and identical evaluation example IDs. The starting-weight
hash, training-order hash, and learning-rate sequence match across arms. The
learning rate starts at `0.0002787879` and ends at `0.0002784812`.

The objectives do not have equal output-target counts: the JSON arm trained on
5,212 structural tokens; the slot arm trained on 896 categorical/state target
elements; the combined arm used those 896 elements plus 183 language-answer
tokens. These are different supervision units, so the pilot compares the
shared input and update budget and reports target exposure separately. It does
not establish efficiency per equivalent target bit or per compute unit.

The JSON evaluator scores teacher-forced structure and greedily generates up
to 480 tokens for one example per held-out partition. The slot evaluator
decodes its outputs directly. The independent evaluator compares against
generator labels and checks predicted active edges for no-self, single-owner,
single-container, and dependency-cycle violations.

## Metrics

| Arm | Validation exact structure | Validation exact transition | Validation answer NLL | Answer semantic accuracy |
| --- | ---: | ---: | ---: | ---: |
| JSON | Not applicable to token generation | Not applicable | 5.962 | 0% |
| Slot structure | 0/12 | 0/12 | 5.228 | 0% |
| Slot combined | 0/12 | 0/12 | 4.762 | 0% |

The JSON arm had 0/5 valid free-generated structures and 0/5 exact structures;
all five reached the token limit. The slot arms had 0/5 exact structures and
0/5 exact transitions on those same paired examples. On the full 12-example
samples, both slot arms had zero exact structures and transitions in validation,
IID, compositional, held-out-pattern, and counterfactual partitions. On the
compositional partition, relation type accuracy was 31.6%; entity
identification accuracy was 2.8% for slot structure and 6.9% for slot
combined. High endpoint accuracy reflects the generator's stable relation-slot
ordering and is not evidence of unordered graph recovery.

Combined supervision lowered answer NLL relative to the JSON arm, but semantic
answer accuracy remained zero. The slot-only model also had lower answer NLL
than the JSON arm, without a correct generated answer. Loss changes alone do
not show successful language or structural transfer.

## Limits And Decision

This is a one-seed diagnostic with 12 training examples and 12 updates. The
head pools the prompt into one vector, uses fixed entity/relation slots, and
predicts action type but not every action argument. It does not predict
constraints; the evaluator checks constraints after decoding. The current
results therefore do not establish a general relational workspace, and the
small samples do not estimate population-level performance.

Do not increase the training budget from this result. A new hypothesis should
address entity binding and variable-size, unordered relations, then define a
predeclared matched-budget comparison before training. Do not add more head
complexity to rescue this pilot.

## Reproduction

Regenerate v5 data as described in [`DATASET_AUDIT.md`](DATASET_AUDIT.md), then
run from the repository root with the same CPU environment:

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 /tmp/batch6-venv/bin/python -m research.structure.train \
  --config research/structure/configs/explicit_head_pilot.json --arm b \
  --initial research/foundation/data/learn.pt \
  --output /tmp/explicit-head-repro/json_baseline.pt \
  --report /tmp/explicit-head-repro/json_baseline.json \
  --data /tmp/structure-dataset-v5

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 /tmp/batch6-venv/bin/python -m research.structure.explicit_head \
  --config research/structure/configs/explicit_head_pilot.json \
  --initial research/foundation/data/learn.pt --data /tmp/structure-dataset-v5 \
  --output /tmp/explicit-head-repro/slot_structure.pt \
  --report /tmp/explicit-head-repro/slot_structure.json --objective structure

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 /tmp/batch6-venv/bin/python -m research.structure.explicit_head \
  --config research/structure/configs/explicit_head_pilot.json \
  --initial research/foundation/data/learn.pt --data /tmp/structure-dataset-v5 \
  --output /tmp/explicit-head-repro/slot_combined.pt \
  --report /tmp/explicit-head-repro/slot_combined.json --objective combined
```

Use `research.structure.compare_explicit` with those reports and checkpoints
to verify matched controls and produce the comparison JSON:

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 /tmp/batch6-venv/bin/python -m research.structure.compare_explicit \
  --baseline /tmp/explicit-head-repro/json_baseline.json \
  --head-structure /tmp/explicit-head-repro/slot_structure.json \
  --head-combined /tmp/explicit-head-repro/slot_combined.json \
  --head-structure-checkpoint /tmp/explicit-head-repro/slot_structure.pt \
  --head-combined-checkpoint /tmp/explicit-head-repro/slot_combined.pt \
  --config research/structure/configs/explicit_head_pilot.json \
  --data /tmp/structure-dataset-v5 \
  --output /tmp/explicit-head-repro/comparison.json
```

Checkpoint files are kept locally and ignored by Git; source, configuration,
and JSON reports are tracked.
