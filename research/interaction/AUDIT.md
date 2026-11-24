# Data Audit

## Sources

The acquisition manifest is [source.json](source.json). Files were retrieved on 2026-11-24, pinned to immutable revisions, and verified against their recorded byte counts and SHA-256 hashes.

- OpenAssistant OASST1 ready export and tree export, revision `694bedfd102490866caa5507d1c93b6c66484367`, declared Apache-2.0; attribution is to the OpenAssistant project and contributors. The official card describes human-generated, human-annotated conversations and includes quality, review, language, deletion, synthetic, PII, and spam metadata. [Dataset card](https://huggingface.co/datasets/OpenAssistant/oasst1)
- Taskmaster-1 self-dialogue and Wizard-of-Oz files, revision `d92cb6af3005f1dc09c39e75e7daf4a04905e00b`, CC BY 4.0; authors are Bill Byrne, Karthik Krishnamoorthi, Chinnadhurai Sankar, Arvind Neelakantan, Amit Dubey, Kyu-Young Kim, and Andy Cedilnik. Its official readme documents six service domains, collection methods, annotation gaps, and transcription limits. [Pinned readme](https://github.com/google-research-datasets/Taskmaster/blob/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019/README.md)
- Ten hand-authored seed scenarios and six project-authored templates with two substitutions each. All 22 constructed examples are marked synthetic; no model-generated targets are included.

The raw downloads occupy about 171 MB and remain local under the ignored `data/` directory. TinyStories was not added: the proposed corpus is GPT-generated and much larger, so it adds synthetic material rather than the reviewed interaction coverage this batch needs.

## Curation

The reproducible output is `data/ready7/`; its manifest is [manifest7.json](manifest7.json). Curation streamed one tree or dialogue at a time and wrote intermediate deduplication indexes to SQLite. It retained:

- 10,950 unique episodes from 10,946 source documents, containing 10,802,464 unique UTF-8 message-content bytes (role markers are excluded from this byte count).
- 224,441 turns; episode length is 2 to 64 turns, median 21 and 90th percentile 29. UTF-8 content length is 21 to 7,696 bytes, median 871 and 90th percentile 1,499.
- OASST1: 986 retained assistant-ending paths after English, review, deletion, synthetic, PII/spam, assistant rank-0, and quality >= 0.75 filters.
- Taskmaster-1 self-dialogues: 5,907; Wizard-of-Oz dialogues: 4,035. Whole dialogues were kept together. E-mail, phone, payment-number, SSN, IPv4, and street-address patterns were excluded.
- Project-authored and controlled examples: 22, all marked synthetic.

Exact, whitespace/case-normalized, and SimHash-plus-Jaccard near-duplicate checks ran before splitting. Three normalized duplicates were dropped; 13 near-match records across eight family links were retained and assigned to the same split family. There were zero exact, normalized, or near-match collisions across train, validation, and test. Near matching is an audit heuristic, not a semantic duplicate detector.

Splits are deterministic, stratified by source, and grouped by OASST prompt family, Taskmaster instruction family, or project template family. Family and template groups never cross partitions.

| Source | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| OASST1 | 788 | 99 | 99 |
| Taskmaster-1 self | 4,544 | 572 | 791 |
| Taskmaster-1 Wizard-of-Oz | 2,668 | 750 | 617 |
| Project-authored/controlled | 17 | 3 | 2 |
| **Total** | **8,017** | **1,424** | **1,509** |

The split hashes are in `manifest7.json`. Taskmaster family sizes make the proportions approximate rather than exactly 80/10/10.

## Tokenizer And Shards

The local byte-level BPE tokenizer has a 4,096-ID vocabulary: 256 byte fallbacks, eight reserved special IDs, and 3,832 learned merges. It was trained on exactly 1 MiB from the training split only. Its SHA-256 is `28d45ae290e761e90c42f95dbe0ae68befb32c1ba44006f01e4dd46fa9dfcff8`.

The tokenizer preserves UTF-8 bytes without text normalization and defines BOS, EOS, message end, system, user, assistant, and tool boundaries; one additional special ID is reserved. Chat serialization round-trips message role and text. On content-only text, BPE averages 2.107 bytes/token on train, 2.088 on validation, and 2.094 on test, compared with one byte token per byte. These are compression measurements, not language-model quality measurements.

Disk shards use little-endian uint16 IDs, per-episode indexes, assistant-only target spans, and exact `{shard, token}` cursors. The verified shards contain 5,607,476 tokens across seven files. The tokenizer used 3,981 distinct IDs. Assistant target positions are recorded, but **zero tokens have been presented to a model and zero epochs have been run**.

## Resource And Limits

The complete build took 86.599 seconds and peaked at 455,811,072 bytes RSS. It ran CPU-only with one Torch intra-op and inter-op thread, enforced a 120-second timeout, and checked the existing 2,560 MiB memory guard. Source rows and corpus rows are streamed; tokenizer-training input is capped at 1 MiB. Raw corpora, token shards, and indexes are not committed.

Sensitive-data filtering combines source PII/spam labels and conservative patterns. It cannot guarantee that all personal information is removed; Taskmaster does not provide the same PII annotations as OASST1. The synthetic seed is small and intentionally does not establish broad interaction competence. No model was trained in this batch, so there are no learning, interaction, or generalization claims.
