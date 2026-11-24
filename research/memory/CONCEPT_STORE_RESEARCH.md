# Extensible Concept Store Research

## Prototype Decision

The persistent concept prototype is built as a typed view over the existing `Memory` evidence, candidate and fact lifecycle. It does not add a parallel database. **The schema/lifecycle mechanics pass; neural retrieval and reasoning are NO-GO.**

## Schema Proposal

Each immutable fact payload uses schema version 1:

| Field | Meaning |
|---|---|
| `concept_id` | Stable identity independent of wording |
| `kind` | `baseline`, `qualification`, or `relationship` |
| `statement`, `asserted_value` | Claim text and optional scalar value |
| `context` | Required for contextual qualifications; stored under a separate key from baseline |
| `relationship`, `related_concept` | Typed edge and endpoint for relationship claims |
| `source_refs` | Immutable evidence event IDs; observed records retain their source references |
| `origin` | `observed` or `inferred`, kept explicit |
| `confidence` | Numeric metadata from 0 to 1; never treated as a truth test |
| `verification`, `review` | Candidate or verified status and independent reviewer/method record |
| `fact_version`, `fact_id`, `rollback` | Supplied by the existing append-only fact ledger |

The baseline key is `concept:{id}:baseline`; a qualification uses `concept:{id}:context:{context}`; a relationship uses `concept:{id}:relationship:{type}:{endpoint}`. Thus a new context cannot replace the stable baseline, and one concept can have multiple relationships of the same type to different endpoints. A new conflicting claim under the same key remains visible as a conflict unless an explicit replacement is requested. Replacement and rollback append ledger versions; source evidence is not edited or deleted.

## Lifecycle and Migration

1. Keep source material in `Memory.evidence`; concept claims refer to its immutable event IDs.
2. Store a proposal in the existing unverified candidate table.
3. Promote only a candidate with a distinct independent reviewer and a passing human, deterministic, or evidence review. The resulting verified claim is appended through `Memory.fact`.
4. Let the existing fact ledger provide versions, supersession, validity intervals, conflict detection and rollback references. A confidence score never suppresses a conflict or authorizes promotion.
5. Retrieve baseline and requested context separately. If the needed relationship or contextual extension is absent, stale, disputed, or conflicting, abstain and return the evidence references.

No existing table migration is needed: the prototype encodes the versioned schema in existing candidate/fact claim payloads and uses namespaced fact keys. A future production migration should first export a read-only backup, audit existing fact keys and source references, map only human-reviewed entries, and verify count/hash parity before enabling reads. Do not rewrite old facts in place.

## Retrieval and Composition Fixture

`concept_eval.py` stores a baseline cell definition, cold/warm contextual qualifications, and a lamp-to-cell `requires` edge through the existing lifecycle. A deterministic resolver answers two answerable cases, abstains for an unknown context, and reports a contradiction instead of choosing between conflicting cold claims.

Results: 4/4 expected answer/abstention outcomes, 2/2 answerable answers, and 4/4 complete evidence sets. This tests keyed retrieval plus one hand-authored composition rule.

The report also includes six manually labeled response variants: semantic-label accuracy is 6/6, while exact-string accuracy is 2/6. This is an annotated metric fixture, not a model score; it records why semantic response quality and exact realization must be reported separately.

The existing 4.47M-parameter structural Foundation checkpoint was separately given cold and warm concept records and asked to answer both. It scored **0/2** and repeated `to` through the 48-token budget for both cases. The per-case output and checkpoint/tokenizer hashes are in `concept_model_metrics.json`. This is a tiny probe, but it gives no basis to claim that the model can retrieve and combine contextual extensions.

Tests cover baseline/context separation, observed versus inferred provenance, evidence immutability, candidate review gating, source links, conflict visibility, version history, rollback, and correct abstention. They are in `test_concepts.py`.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/batch6-venv/bin/python -m research.memory.concept_eval
PYTHONDONTWRITEBYTECODE=1 /tmp/batch6-venv/bin/python -m unittest research.memory.test_concepts
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/batch6-venv/bin/python -m research.memory.concept_model_eval
```
