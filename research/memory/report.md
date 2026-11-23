# Patch 7: Persistent external memory

## Design

`store.py` provides a local SQLite store with separate append-only tables for raw events and chunks, unverified candidate claims, versioned validated facts, and versioned reasoning skills. Raw event and chunk rows have database triggers that reject updates and deletes. Supersession and invalidation append events; readers retain and expose old records with stale, superseded, conflict, rollback, or validity status.

Evidence text is split into UTF-8-safe chunks of at most the configured byte size. Each chunk has byte offsets, a stable key, token count, and SHA-256; each event carries a hash over the complete source. Retrieval first ranks indexed rows and then fetches only selected chunk text. It returns source URI/ID, source version, acquisition time, licence, event and chunk hashes, offsets, scope, and an explicit `untrusted` marker. Search rank is only a retrieval signal.

SQLite FTS5 is detected at open time. If unavailable or disabled, a standard-library token index in SQLite provides deterministic overlap ranking. Callers must pass an explicit persistence policy matching scope and owner; private records additionally require private opt-in. Retrieved content is data only and is never executed or promoted automatically.

Request limits are capped at 32 calls, 1 MiB returned bytes, 128 chunks, 65,536 lexical tokens, 30 seconds, 4 KiB query text, and 128 unique query terms. Candidate/fact/skill lists and individual derived fields are also bounded. These are per-request/per-record safeguards, not a total database quota or OS-level filesystem quota.

## Verification

The standard-library unit suite covers forced lexical fallback, automatic FTS5 when available, UTF-8 chunk reconstruction, immutable raw rows, hash verification, restart/reopen, explicit persistence and private-content policy, untrusted instruction-like evidence, scope isolation, stale/superseded/conflicting/invalidated records, rollback links, skill promotion metadata, wall/call/byte/chunk/token bounds, write-side size limits, and indexed retrieval over a generated corpus.

The bounded scale smoke wrote 3,000 project-generated records in 4.081 seconds. A unique indexed query returned and read one chunk (108 bytes, 17 lexical tokens) from a 6,303,744-byte SQLite database. Peak process RSS was 23,330,816 bytes under a 1 GiB address-space cap and 110-second internal deadline. CPU-only, one thread; no neural model was loaded. Exact source hash and run data are in `out.json`.

## Limitations

This is a local lexical memory mechanism, not semantic retrieval, a natural-language entailment verifier, or a strategy learner. FTS/term rank does not establish truth. Fact validation is recorded as caller-supplied provenance and method; this package does not independently prove a fact. Skill validation metadata is a gate record, not a test runner. A database owner can still replace the file or disable SQLite triggers; tamper resistance against a hostile filesystem/database administrator is out of scope. No real corpus, private conversation, external dependency, or model was used. Disk growth is not globally capped; production use needs caller-managed retention and storage quotas.
