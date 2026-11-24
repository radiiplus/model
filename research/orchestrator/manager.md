# Problem Manager and Long Context

Date: 2026-11-24
Branch: `adaptive`

## Implementation

`Manager` stores scoped, append-only problem snapshots in the existing SQLite memory database. Each snapshot contains the required objective, current state, subgoals, established facts, assumptions, failed attempts, completed and pending work, progress, completion criteria, and evidence references. Version checks use an immediate SQLite transaction; snapshots retain their predecessor hash and reject updates without matching explicit persistence policy. Reads require the matching scope and owner.

Updates use a fixed typed operation set (`set`, `append`, `fact`, `evidence`, `complete`). Only current state and progress can be set; the problem objective is caller-owned and cannot be rewritten by model proposals. Established facts require an evidence event from the same scope and owner. The model can optionally expose a `propose` method, but each proposed operation is schema checked before state changes. Unknown operations, including shell-shaped proposals, are rejected and never executed. Invalid proposals leave the prior snapshot intact.

`Prompt` serializes system instructions and the current request as required fields, then admits bounded recent turns, problem state, retrieved memory, referenced historical evidence, strategies, and draft in that order. It counts UTF-8 bytes, which are exact tokens for the existing byte tokenizer. Evidence is serialized under separate `memory` or `history` fields with an explicit `untrusted` marker; tool messages in recent turns receive the same marker. The current 32-token Batch 12 decoder cannot fit the mandatory system and request fields and now safely returns context overflow; no model checkpoint or baseline was changed. Results include the serialized prompt byte-token count.

## Verification

- `research.orchestrator.test`: 20 tests passed in 1.532 s test time; 3.441 s process wall; 318,776 KiB peak RSS.
- `research.memory.test`: 7 tests passed in 0.813 s test time; 0.916 s process wall; 26,368 KiB peak RSS.
- The broader orchestrator, memory, adaptive, recursive, Batch 12, TinyGPT, baseline Transformer, evaluation, retrieval, and verifier regression selection passed 76/76 tests in 2.448 s test time.
- CPU PyTorch 2.15.0+cpu, one Torch thread for model tests, 120-second command timeouts. The isolated `/tmp/b4-torch-venv` was created from the existing CPU lock because it was absent at the start of this patch.

Tests cover objective retention and restart recovery, referenced older evidence, conflicting source preservation, exact context-budget enforcement, serialized injection boundaries, private persistence policy, optimistic version conflicts, immutable snapshots, and rejection of an untyped model proposal.

## Limits

This validates persistence, serialization, and operation boundaries only. It does not demonstrate that the neural model can understand long context, maintain objectives from free-form conversation, resolve natural-language evidence conflicts, or improve interaction quality. The byte model's context length remains 32 and is unusable with the required system boundary. Context fitting omits lower-priority records or problem fields whole and marks partial problem state; it does not summarize or compress them. The existing Patch 10 measurement and report remain unchanged.
