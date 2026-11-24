# Repository Baseline Audit

Audit date: 2026-11-24
Branch: `adaptive`
Starting HEAD: `453b5e5d7e311d5638f885216b2f42ae9dfe78dd`
Starting worktree: clean; branch matched `origin/adaptive`.

## Repository State

Recent commits were `453b5e5 Add staged curriculum pilot runner`,
`ad928b6 Implement compaction learning lifecycle`, `b3dcf50 Add persistent
problem manager and bounded context`, `b2eb515 Record held-out response
learning evidence`, `15eae45 Add multi-layer recurrent training arm`, and
`5f3c717 Add four-block neural language foundation`. The complete starting
state and audit hashes are in [`experiment_manifest.json`](experiment_manifest.json).

## Training Report And Curriculum

The latest training report is the Stage A pilot in
[`a1.json`](../foundation/a1.json), summarized in
[`curriculum.md`](../foundation/curriculum.md). It continued the 800-update
weights for 8,000 updates and 1,000,000 target positions, using batch 1,
sequence 125, AdamW, language loss only, and a new optimizer and scheduler.
The fixed validation language NLL changed from 4.296 to 4.551; the separate
96-turn assistant-response probe changed from 4.550 to 4.609 NLL. The run
therefore regressed on both recorded held-out measures. The curriculum sets
Stage A's eventual budget to 60M positions, Stage B to 16.2M positions, and
keeps later stages gated on evidence and data review.

The pilot report records six bounded segments, with target updates 1,191,
2,392, 3,591, 4,791, 5,990, 7,179, and 8,000. Segment post-training
validation was incomplete in several segments; the final segment has complete
post-training metrics. The trainer used one absolute time budget for initial
validation, fitting, and final validation. Initial evaluation consumed much
of that budget, and the final evaluation ran after it had expired, yielding
null/zero metrics. The pilot report's separate held-out probe remains the
usable final comparison.

## Checkpoint Availability And Identity

Both requested checkpoint files exist locally. Their full file hashes match
their reports; see the manifest for report hashes and tensor-payload details.

| Checkpoint | Local file | Updates | Serialized optimizer/scheduler | Result |
|---|---|---:|---|---|
| Previous response-learning checkpoint | `research/foundation/data/learn.pt` | 800 | AdamW moments and scheduler at update 800 are present | Eligible for genuine continuation from its saved sampler cursor |
| Stage A pilot checkpoint | `research/foundation/data/a1.pt` | 8,000 | Pilot-end optimizer and scheduler are present | Confirms end state only; it does not restore the pre-pilot optimizer, which the run restarted |

The files are ignored local artifacts rather than Git-tracked files. Their
presence was checked directly and not inferred from their report hashes.
Both payloads have schema 1, architecture version 1, seed 2025, and the same
model configuration: vocab 4,096, context 1,024, four independent layers,
width 256, four heads, FFN 1,024, tied token/output weights, learned absolute
positions, pre-LayerNorm, FP32. The parameter count is 4,470,272.

The dataset hash is
`9016272c1ffb57a3a9bd8dd3e86ef47c3789f5ffbde62a86b451f46a371d5078`; the
training split hash is
`e622c83f287fe699dbd717620208604a36b01b0d7e673d6bac6490753494cbdd`; the
validation split hash is
`6e01a31db20425d006237654c86fce6f9636ee37eaf507fd096180825d53351c`; and
the tokenizer hash is
`28d45ae290e761e90c42f95dbe0ae68befb32c1ba44006f01e4dd46fa9dfcff8`.
The train shards contain 4,224,016 token positions across 8,017 documents.
The associated source manifest reports an interaction-heavy mix of
OpenAssistant (Apache-2.0), Taskmaster TM-1 (CC BY 4.0), and project-authored
data.

## Relevant Code And Tests

Inspected `research/foundation/model.py`, `recur.py`, `trainer.py`,
`curriculum.py`, `research/interaction/`, and `research/orchestrator/`.
Existing symbolic extraction and transfer regression tests are in
`research/adaptive/test.py`; model-level transfer and exact-resume tests are
in `research/foundation/test.py` and `research/recursive/test.py`.

Before source changes, the targeted regression command passed 51/51 tests.
The command and environment are recorded in
[`TEST_RESULTS.md`](TEST_RESULTS.md). PyTorch was not installed in the
workspace's system interpreter, so the existing pinned CPU lock was installed
in an isolated `/tmp/audit-pytorch-venv`; repository dependencies were not
changed. The environment emitted PyTorch's expected warning that NumPy was
absent; the tested paths do not depend on NumPy.

## Initial Diagnosis

The Stage A report explicitly states that optimizer and scheduler were
reinitialized from the 800-update checkpoint. However, the source
`learn.pt` locally retains both states, so a genuine continuation arm is
possible. The checkpoint also stores the exact legacy sequential sampler
cursor and buffered tokens, allowing the comparison to begin at the exact
next training position. The corrected episode-shuffle sampler is a separate
infrastructure change and is not used in the schedule-confound arms.

No model architecture, source checkpoint, historical report, or training
data was changed during the baseline audit.
