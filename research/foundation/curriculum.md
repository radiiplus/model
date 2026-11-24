# Batch 6: Staged training curriculum

## Stage A pilot

The first bounded curriculum run continued the fixed 4,470,272-parameter,
four-block Foundation from the existing 800-update checkpoint. It consumed
1,000,000 causal next-token positions from the streamed, approved Batch 1
training shards over 8,000 AdamW updates (sequence length 125, batch 1,
accumulation 1). The source checkpoint was not modified. The optimizer and
schedule were newly initialized for this segment; the model weights were
loaded from the source checkpoint. The sampler cursor, optimizer, scheduler,
and RNG states were checkpointed atomically between bounded segments.

The local train split contains 8,017 documents and 3,877,486 audited training
tokens; the shard stream contains 4,224,016 token positions. Thus the pilot
presented about 0.24 passes over the stored stream. The available licensed
mix is OpenAssistant (Apache-2.0) and Taskmaster TM-1 (CC BY 4.0), plus
project-authored examples. It is interaction-heavy, not the broad prose/story
mixture specified for the eventual language foundation. This run is a real
training pass over approved local data, but it does not satisfy that corpus
coverage goal.

## Measurement

| Measure | Result |
|---|---:|
| Updates / target positions | 8,000 / 1,000,000 |
| Fit time / fit throughput | 604.0 s / 1,655.5 positions/s |
| Total elapsed time | 609.93 s |
| Total-wall throughput | 1,639.5 positions/s |
| Peak RSS | 530,370,560 B |
| Parameters / independent blocks | 4,470,272 / 4 |
| Device / dtype / Torch threads | CPU / FP32 / 1 |
| Memory guard | Existing 2,560 MiB guard; passed |
| Initial checkpoint SHA-256 | `76b60e84c3b203ef1a30bd93f0fd7e8c8fbe619d08799e57e7f434a6c237a5fe` |
| Pilot checkpoint SHA-256 | `46d0f379d86500be14e6a9812aa254c515d7ce704bb7b17a920acd24ec2529be` |
| Train / validation split hashes | `e622c83f287fe699dbd717620208604a36b01b0d7e673d6bac6490753494cbdd` / `6e01a31db20425d006237654c86fce6f9636ee37eaf507fd096180825d53351c` |
| Tokenizer hash | `28d45ae290e761e90c42f95dbe0ae68befb32c1ba44006f01e4dd46fa9dfcff8` |

The complete segment and per-update training log, configuration/data/code
hashes, and resource snapshots are in [`a1.json`](a1.json). The local 52 MiB
pilot checkpoint is ignored by Git and stored separately from the reference
checkpoint. The held-out validation probe is [`valid.json`](valid.json).

## Evaluation

Validation comparisons used the same 2,048-token stream slice and seed for
the original 800-update checkpoint and the pilot. Language NLL changed from
4.296 to 4.551; assistant-only NLL changed from 4.253 to 4.570. On a separate
fixed 96-turn source-balanced validation probe (8,579 assistant target
tokens), the same checkpoints scored 4.550 and 4.609 NLL respectively, with
exact full turns at 0/96 for both. Per-source pilot NLL was 4.820 on OASST1
(7,600 tokens), 2.835 on Taskmaster self-dialogues (563 tokens), and 3.152 on
Taskmaster Wizard-of-Oz (416 tokens). Taskmaster records each represent one
broad metadata family in this sample, so these source rows do not establish
transfer across task families.

The probe trained on no validation or test examples and was not used to select
a checkpoint. A separate greedy free-generation check produced repeated
common words (`"Okay you the the the to to to to to to "` over 24 tokens).
These measurements show that the pilot performed neural updates on the
approved corpus, but it did not improve these held-out measures over the
existing checkpoint and did not demonstrate coherent interaction. This is a
no-go for extending to the 5M assessment under the current data mixture and
schedule. No test split was used for this decision.

## Budget projection

At the measured total-wall rate of 1,639.5 positions/s, the proposed 60M Stage
A target would take about 10.16 hours; Stage B's 16.2M input-position target
would take about 2.75 hours; and the combined 76.2M would take about 12.92
hours (46,512 s). At fit-only throughput, the combined estimate is 12.78
hours. These are linear projections from one 1M run on this machine and
configuration, not a guarantee; they omit validation/selection runs and any
extra cost from a revised corpus or task format. Stage B must separately
report masked assistant target tokens and total input positions. Stage C has
no reviewed complete-episode memory/problem-management dataset and is not
estimated or runnable yet.

The 60M Stage A target corresponds to about 14.2 passes over the currently
stored shard tokens. Do not launch that target, Stage B, or the full program
until a better-balanced rights-cleared corpus is available and an assessment
shows held-out improvement. The initial 1M pilot is complete; 5M and 20M
milestones remain gated.
