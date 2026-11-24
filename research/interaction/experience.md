# Compaction and Experience Learning

## Design

Batch 5 extends the existing scoped SQLite memory rather than creating a second
conversation archive. Batch 4's immutable `problems` snapshots remain level 1
working state. The new `notes` and `nrefs` tables are level 2 episodic records;
each version carries a fidelity report and links back to immutable evidence
events. `Compact.original` fetches those events through the existing bounded
memory request API. A later version points to its predecessor, and rollback
creates a new version containing an earlier projection. A summary has level 2
status; it does not replace or supersede its source events.

Level 3 facts and validated skills continue to use the existing `facts` and
`skills` projections. New `trials`, `reviews`, `datasets`, and `models` tables
record proposed experience, independent validation, approved training rows,
and checkpoint lineage. These records are append-only. The orchestration queue
can feed `Learn.collect`, but collection still requires the caller's explicit
scope, owner, persistence, and private-data policy.

The structured compaction record retains objective, constraints, numbers,
decisions and reasons, evidence-backed facts, unresolved questions, failed
approaches, dependencies, and event IDs. The trigger counts serialized records
with the frozen BPE tokenizer and writes only after a caller's token budget is
crossed. The deterministic fidelity check requires every field and source link
to match. The Foundation can propose a compacted record, but malformed output,
unsupported edits, or missing sources fail the gate. Generated text is always
marked untrusted.

The learning path is:

```text
opt-in request queue -> immutable observable episode -> candidate trial
-> independent cross-family review -> approved dataset -> offline CPU fit
-> registered checkpoint
```

Validation inputs must be distinct from training inputs and include a family
not used for discovery. The stored report must show a candidate gain on a
transfer family and no regression on any reported family. A failed review is
retained and cannot produce an approved dataset. `Fit` accepts only the export
shape emitted by the approval ledger, applies loss to assistant answer tokens
and the end marker, freezes no source artifact in place, and writes a separate
atomic, resumable checkpoint. The checkpoint binds the base, approved dataset,
tokenizer, model shape, code, seed, and training settings. Registration checks
those hashes and requires a separate independent model-level transfer report
with a gain and no reported regressions; rollback adds a new registry version.
No request-time weight updates exist.

The local ledger is an integrity and workflow boundary, not an authentication
service. It does not cryptographically authenticate reviewer identity, and a
caller able to rewrite the SQLite file or fabricate an export can bypass local
policy. Production use needs a trusted review identity and signed approval
records. Private content is not collected by default.

## Neural Evidence

The four-layer FP32 Foundation has 4,470,272 parameters. Its existing fixed
800-update run learned conditional response-token prediction from the licensed
Batch 1 training split: held-out assistant NLL changed from 8.324 to 4.550 on
validation and 8.319 to 4.481 on test; next-token accuracy reached 37.64% and
38.01%. Full-turn exact generation was 0/96 on both partitions. The probe used
gold prefixes, so it measured token prediction, not free-form interaction.

A reproducible greedy generation probe, `python -m research.interaction.check --count 48 --wall 20`, loaded the frozen `learn.pt` checkpoint
(SHA-256 `76b60e84c3b203ef1a30bd93f0fd7e8c8fbe619d08799e57e7f434a6c237a5fe`)
and the frozen BPE tokenizer (SHA-256
`28d45ae290e761e90c42f95dbe0ae68befb32c1ba44006f01e4dd46fa9dfcff8`). Asked
to compact and continue a short review record, it emitted repeated `the`
tokens for its full 48-token budget. Generation took 0.683 seconds; process
peak RSS was 394,240,000 bytes. It used CPU FP32 and one Torch thread. At
measurement, host effective available memory was 6,582,386,688 bytes; cgroup
v2 reported current usage near 1.8 GB and no
finite configured limit. The existing 2,560 MiB guard remained active. This
result is a failure for generated compaction and continuation, not evidence of
useful summaries.

No model-generated record was promoted. No new production-sized candidate fit
was run: there is no independently reviewed, cross-family experience that
passes the transfer gate. The offline trainer's exact interrupted/resumed
comparison and low-memory refusal were tested on a small, project-authored
fixture; that test verifies the update path only. It is not learning or
capability evidence. The Foundation's protected checkpoint was unchanged.

## Verification and Limits

The focused interaction and foundation suites passed, as did the existing
memory, orchestrator, adaptive, recurrent, TinyGPT, Batch 12, and evaluator
regressions: 92 tests total, completing in 5.960 seconds. The
tests cover source recovery, context-token triggering, immutable compaction
versions and rollback, candidate-only output, duplicate-input rejection,
cross-family validation, approval/export, assistant-only fit masks, exact fit
resume, base-hash preservation, checkpoint registration and rollback. The
suite completed in under six seconds on CPU with one Torch thread. The existing
2,560 MiB memory guard and external 120-second test timeout were used; no new
packages, corpus, checkpoint source, or service was added.

Compaction currently preserves the structured record exactly, so it validates
fidelity but does not yet demonstrate meaningful shortening of natural
conversation histories. Context triggering is exposed through the existing
tokenizer and memory API; the orchestrator does not silently persist private
turns or automatically compact its own prompt. The successful-fit test uses a
small synthetic fixture. No general interaction, long-horizon memory,
cross-family neural transfer, or frontier capability claim is supported.
