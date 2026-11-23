# Tiny Curation Pipeline Specification (Batch 2)

## Reproduction

From `/home/codespace`:

```sh
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest -v research/experiments/test_curate_tiny.py
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 research/experiments/curate_tiny.py --output research/experiments/results/curation_tiny.json
```

The script uses only the Python standard library and refuses to run below 2,560 MiB `MemAvailable`; `timeout` limits wall time. Threads are capped in the command. The fixture is 20 project-authored synthetic documents with CC0-1.0 metadata; no external text, model, corpus or package was acquired. Output stores a fixture SHA-256 and environment snapshot.

## Pipeline stages

1. **Read and validate records.** JSONL records have `id`, `source`, `license`, `provenance`, `domain`, `style`, `duplicate_group`, `quality_label`, `text`. Raw fixture is immutable; transformations produce annotations and manifests.
2. **Explicit boilerplate removal.** Remove only the exact known synthetic archive footer after normalization. Do not use broad regex removal based on generic terms. Record each changed document.
3. **Exact dedup.** Case-fold and whitespace-collapse, then SHA-256; keep a deterministic ID-sorted representative and preserve all component IDs.
4. **Fuzzy dedup comparisons.** Use word 3-gram shingle sets. MinHash has 64 fixed deterministic permutations and uses estimated Jaccard >=0.70. SimHash is 64-bit and uses Hamming distance <=5. Both generate pair edges and connected components; one deterministic representative is selected by ID for byte-count estimates only. Sweep lower thresholds to expose false merges.
5. **Transparent quality signals.** Emit repeated-token fraction, unique-token ratio and heuristic quality score. A single conservative removal rule flags repeated-token fraction >=0.70 on >=10 word tokens. Code and math receive different diagnostic score formulas; the hard repeated-token flag remains visible, and no low score alone deletes a record. In this fixture, only labeled noise `d08` is removed. Threshold is a toy falsification point, not a proposed production classifier.
6. **Mixture and split audit.** Count domains/styles before/after filtering. Assign duplicate components to one split using a stable SHA-256 bucket, avoiding near-duplicate leakage across train/dev/test. Current tiny split proportions are an illustrative deterministic preview and are not guaranteed to populate each split on a small set.

The local prototype intentionally omits web extraction, language identification, PII/unsafe classification, tokenizer packing and contamination scan against an external held-out set. The later pipeline should acquire only source-approved shards, preserve source URL/ID/date/license and raw hash, then emit versioned extraction/language/quality/risk/dedup annotations. Apply reviewed filtering with a reason code; split duplicate components and remove evaluation matches before tokenization/packing; then create tokenizer-specific shards and retain a manifest mapping packed token ranges back to source documents.

## Acquisition plan (later; no download in Batch 2)

1. Start with one small, explicitly CC0/public-domain or clearly training-permitted source per target stratum. Prefer a local fixture or a capped sample under 100 MiB after license review; never infer rights from a processing-repository license.
2. Before fetching, record canonical source URL, dataset/card revision or Git SHA, component/source license and attribution, allowed use, retrieval date, source ID, URL/date fields, and a checksum. Reject sources with unclear provenance or incompatible noncommercial/redistribution terms until resolved.
3. Fetch only the requested sample with a hard byte cap and a timeout. Save an unmodified raw fixture, then derive cleaned output into a separate directory. Do not include credentials, cookies, access tokens or private data in manifests.
4. Record counts and byte/token estimates after every transform, plus kept/rejected IDs and reason codes. Sample false positives for every removal rule and preserve reversible metadata.
5. Expand one source at a time only after held-out, per-domain capability changes justify its processing/storage cost. No web-scale crawl or dataset shard is planned for this Codespace.

Minimum provenance fields for later documents: `document_id`, `source_dataset`, `source_record_id`, `source_url_or_archive_path`, `retrieval_or_publication_date`, `dataset_revision_or_snapshot`, `license_expression`, `attribution`, `allowed_use_review`, `raw_content_sha256`, `transform_pipeline_version`, `language`, `language_confidence`, `domain`, `duplicate_component_id`, `filter_decisions`, and `split`.

All document records keep source, license, provenance and original text in the input fixture. The result JSON includes exact, MinHash and SimHash retained manifests with source metadata, transformed text, and component IDs. The input fixture remains untouched. For a future corpus, store retained JSONL and a separate removed-record manifest with reason codes and hashes.

## Tiny measured results

Result artifact: `../experiments/results/curation_tiny.json`.

| Stage/method | Input/retained | Removed | Unique false positives at selected threshold | Text bytes after stage |
|---|---:|---:|---:|---:|
| Fixture | 20 | 0 | 0 | 2,226 raw text bytes (JSONL fixture is larger due to metadata) |
| Exact normalized hash | 20 -> 18 before quality filtering; after filtering 19 -> 17 | 2 duplicate records (`d02`, `d10`) | 0 | 1,730 after quality + exact dedup |
| MinHash 64 x 3-gram, Jaccard estimate >=0.70 | after quality 19 -> 16 | 3 (`d02`, `d03`, `d10`) | 0 | 1,615 |
| SimHash 64 x 3-gram, Hamming <=5 | after quality 19 -> 16 | 3 (`d02`, `d03`, `d10`) | 0 | 1,615 |
| Explicit boilerplate line removal | 20 docs remain | 3 footer lines across `d05`-`d07` | 0 | body texts remain distinct |
| Repeated-token quality rule | 20 -> 19 | `d08` only | 0; code/math/rare terms retained | 1,927 pre-dedup, then counts above |

The input fixture is 6,742 bytes. These chosen fuzzy methods find the exact duplicates plus the punctuation-edit near duplicate, while preserving the unique rare fact `d04`. Threshold sweep demonstrates failure modes: MinHash at 0.20 merges `d04` with the Veloria charter cluster; at 0.10 it also merges three distinct archive notes (`d05`-`d07`) because common context/boilerplate dominates. SimHash at Hamming <=25 creates many false edges, including unrelated pairs such as `d01`/`d11` and `d01`/`d20`; transitive components then merge many documents. These are deliberate fixture counterexamples, not estimates of real-corpus false-positive rates.

Measured in the current Codespace on Python 3.14.2: the final run including the threshold sweep took 0.0995 s wall and 0.0827 s process CPU, about 201 records/s; `resource` high-water RSS 22,405,120 bytes; `tracemalloc` peak 63,080 bytes. The corpus is too small for meaningful throughput comparison or robust memory scaling. `resource` RSS includes interpreter overhead; `tracemalloc` misses native allocations.

## Mixture proposal for later ablations

Do not convert heuristic quality scores into training weights yet. A starting *experimental matrix*, subject to rights review and held-out capability measurement, is:

| Stratum | Initial mixture share for controlled pilot | Rationale / safeguards |
|---|---:|---|
| General web/reference prose with source/date/URL metadata | 35% | Broad vocabulary and factual exposition; dedup by crawl snapshot carefully and retain rare examples. |
| Books and long-form public-domain or explicitly licensed writing | 15% | Long context, narrative and stylistic coverage; verify edition/translation and jurisdiction. |
| Code and technical documentation | 15% | Preserve code-specific formatting and repositories; repository/source license filters and project-level dedup. |
| Math/science/educational material | 15% | Worked explanations and structured reasoning; do not optimize only for a model-graded “educational” score. |
| Dialogue, forums and informal language | 10% | Conversational forms, questions, disagreement and dialect/style breadth; apply privacy and safety review. |
| Multilingual and low-resource text | 10% | Preserve language coverage; measure per-language quality and tokens, not an English-first global score. |

These are provisional balancing targets, not an asserted optimum. Acquire no source until its license, provenance, privacy, source-level attribution and allowed training use are recorded. Establish per-stratum held-out likelihood and task metrics, then run fixed proportions against controlled alternatives.

## Curriculum / schedule ablation

Compare constant mixture against scheduled mixture while holding total tokens, model, optimizer, seed, data dedup state and compute fixed. Use at least three seeds if the runtime allows. Pre-register: A) constant draw at the target final proportions; B) staged easy-to-hard or domain schedule whose *overall token shares exactly match A*. Measure per-domain held-out NLL plus separate reasoning, coding, factual retrieval, instruction and multilingual evaluation tracks at the same checkpoint token budgets. Also check forgetting on early-domain held-out sets and training stability. Keep a schedule only if it produces repeatable downstream gains without material regressions at equal compute; lower loss on a proxy corpus alone is insufficient. Xie et al.'s [DoReMi paper](https://arxiv.org/abs/2305.10429) uses a 280M proxy LM with group DRO to estimate domain mixture weights and reports gains for its evaluated 8B target experiments; this is evidence for adaptive mixture search in that setup, not evidence that curriculum is generally superior. RegMix and other proxy-mixture methods are additional leads for a later controlled review. Run this ablation on a future tiny training prototype, not in this curation batch.

## Risks and limitations

- Synthetic labels are easy and partly encode expected duplicate relationships; no human-labeled real corpus is tested.
- MinHash signature quality, bucket parameters, word tokenization and threshold calibration are corpus-dependent. LSH is not implemented; the fixture directly compares every pair and is O(n^2).
- SimHash bit scoring is sensitive to short documents and feature weighting; the permissive distance threshold is intentionally shown to fail.
- Semantic dedup is not implemented because embeddings require a model, download/install and additional biases. No claim is made that lexical methods detect paraphrases.
- Quality scores are transparent heuristics with no target-quality labels beyond one deliberately poor row. They are not calibrated quality estimators.
- No language detector, classifier-based quality/safety filter, PII detector, source-specific copyright classifier, web extraction or token packing is run. These need separately licensed fixtures and audits.
- Memory guard is 2,560 MiB `MemAvailable` at start. Stop the run if availability falls below that, any process reports allocation failure/OOM, or observed RSS exceeds the configured experiment target (5.5 GiB total experiment usage); reduce input size before rerunning. The external 120-second timeout is mandatory.
- Content-hash dedup does not prevent memorization or benchmark contamination. Split by duplicate/near-duplicate connected components and decontaminate held-out tests against train data before evaluating.

To claim that a smaller corpus preserves downstream capability, we would need fixed model architecture/tokenizer/training recipe and equal token/compute comparisons over multiple seeds; broader frozen and contamination-audited test sets; per-domain quality and rare-language/rare-fact metrics; matched held-out NLL; repeated ablations on dedup and quality thresholds; uncertainty intervals; and a measured Pareto comparison of capability vs training tokens, time and peak memory. A single synthetic fixture cannot support that conclusion.
