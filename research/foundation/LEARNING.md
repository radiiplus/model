# Learning Evidence

## Fixed Run

The plain four-block Foundation was initialized from seed 2025 and trained on
the streamed Batch 1 training split for 800 AdamW updates. Each update used one
128-token sequence; 102,400 target-token positions were presented, about 2.4%
of the 4,224,016 stored training tokens. The run used learning rate 3e-4,
100-update warmup, CPU FP32, one Torch thread, the existing 2,560 MiB guard,
and a 100-second internal deadline. No validation or test targets entered the
optimizer.

The 4,470,272-parameter model completed in 68.04 seconds including initial and
final validation, with 446,746,624 bytes peak RSS. The existing 2,048-token
validation slice included 1,328 assistant targets: language loss fell from
8.320 to 4.303 nats/token, and assistant-only loss fell from 8.308 to 4.252.
The full streamed training report, checkpoint, settings and hashes are in
[`learn.json`](learn.json); the local checkpoint is ignored by Git.

## Held-Out Response Probe

Before training, the evaluator selected 96 assistant turns deterministically
from a held-out validation or test partition: 32 each from OASST1,
Taskmaster self-dialogues, and Taskmaster Wizard-of-Oz dialogues. It chose one
turn per document, excluded the project-authored synthetic examples, and
scored each answer token plus its end marker with teacher-forced next-token
prediction. The same selected turns were measured with the seed-matched random
initialization and the fixed 800-update checkpoint. Test results were produced
once after training; test targets were not used to choose the model or settings.

| Split | Assistant target tokens | Initial NLL | Trained NLL | Initial next-token accuracy | Trained next-token accuracy | Exact turns before/after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 8,579 | 8.324 | 4.550 | 0.117% | 37.64% | 0/96, 0/96 |
| Test | 10,858 | 8.319 | 4.481 | 0.055% | 38.01% | 0/96, 0/96 |

All three source groups improved in both NLL and next-token accuracy on the
test sample. Source metrics, selected-example hashes, split hashes,
checkpoint hash, latency and RSS are in [`proof.json`](proof.json) and
[`test.json`](test.json). OASST1 contributed 32 distinct recorded families in
each sample. Taskmaster's metadata records one broad family per source subtype
in validation and one to two in the test sample, so the Taskmaster rows should
not be read as cross-family transfer evidence.

## Interpretation

This establishes that the trained model learned useful conditional token
predictors for held-out assistant responses across the included data sources.
It does not establish complete answer generation: exact response match was
zero for all 96 turns, and this probe supplies the gold prefix at every token
(teacher forcing). There was no human quality review, task-success scoring,
long-horizon objective test, or free-generation evaluation. The evidence
supports basic next-token/response-pattern learning, not coherent interaction,
objective maintenance, reasoning, or frontier-level behavior.
