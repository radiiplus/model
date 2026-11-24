# Semantic Baseline and Correction Pilot

## Purpose

This pilot separates semantic correctness from exact wording. The semantic label (`yes`, `no`, or `unknown`) is the primary answer score; exact response-string match and numeric-detail match are secondary. A paraphrase can be semantically correct without copying the target sentence. `unknown` is not treated as correct for either answerable class, and semantic coverage is reported so a model that emits no interpretable answer cannot hide behind conditional accuracy.

## Benchmark

`semantic_benchmark.py` generates 56 independent worlds and 384 prompts. Each world and all of its surface/action variants stay in one split. The direct-supply pattern is used for training, validation and IID; compositional test uses a two-hop supply path; counterfactual test compares no action with an insulation action. Every split has three independently worded input surfaces and a separate answer wording bank. A deterministic capacity/context/relation oracle creates the yes/no labels and target responses. The model is never used to generate labels.

| Split | Worlds | Examples | Purpose |
|---|---:|---:|---|
| Train | 24 | 144 | 24 direct-supply worlds, contexts and surfaces |
| Validation | 8 | 48 | Held-out direct worlds and wording |
| IID test | 8 | 48 | New direct worlds and wording |
| Compositional test | 8 | 48 | Two-hop relation composition |
| Counterfactual test | 8 | 96 | Paired original/insulated states |

The dataset manifest SHA-256 is `3950735151bebf9802ee0e38fc4deb8664237780f4377212046c9bb1e2c7ac25`. File hashes and seed ranges are in `semantic_correction_bench/manifest.json`.

The semantic parser is a small reviewed yes/no cue parser, not a general semantic judge. Its test fixtures include meaning-preserving paraphrases and an unknown/repetition case. That scope is explicit; generated text outside the parser's reviewed cues is counted as unknown.

## Comparison

Both conditions start from the same 4.47M-parameter structural Foundation checkpoint, hash `0f36d1337f370d9edd33879f5e8aa3e44b8f8ef247014ab6b219d50b982d0a53`, and use the same frozen tokenizer. The baseline is frozen. The correction condition uses reset AdamW at `5e-5`, batch size one, and the same 24 unique generated correction episodes. Correction targets are the deterministic oracle's manually authored response templates. The episodes are synthetic fixture data, not real user conversations; they are captured in an in-memory evidence/experience ledger and are not persisted as user data.

| Condition | Updates | Correction passes | Target tokens | Semantic accuracy | Semantic coverage | Exact response accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Frozen checkpoint | 0 | 0 | 0 | 0/80 | 0/80 | 0/80 |
| Corrected, one pass | 24 | 1 | 728 | 0/80 | 0/80 | 0/80 |
| Corrected, four passes | 96 | 4 | 2,912 | 0/80 | 0/80 | 0/80 |

All 24 initial drafts were unparseable, so all captured corrections were labeled meaning corrections; none were wording-only corrections. Training loss decreased from `5.08` to `3.28` in the four-pass run, but held-out generation remained empty or repetitive. The one-pass corrected model often stopped at EOS; the four-pass model still stopped at EOS or produced fragments such as `is in in in`. Loss reduction did not improve response behavior.

Per-arm examples, predictions, losses, hashes, optimizer settings, and timing are in `correction_pilot/comparison.json` and `correction_pilot_96/comparison.json`. Checkpoints are included for the corrected arms.

## Decision

**NO-GO for more correction fine-tuning with this setup.** The capture mechanics work and the semantic benchmark distinguishes meaning, coverage, and wording. The model has zero interpretable held-out answers before and after 96 updates. This pilot does not support the claim that collecting corrections is enough to improve response behavior. The failure is at generation/answer realization, not exact-string scoring.

The next useful change is a new hypothesis and a model-side intervention that can address EOS/repetition while preserving semantic evaluation. Do not increase training updates or promote these correction examples as successful learning evidence.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/batch6-venv/bin/python -m research.interaction.semantic_benchmark
timeout 900s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/batch6-venv/bin/python -m research.interaction.correction_pilot
timeout 900s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/batch6-venv/bin/python -m research.interaction.correction_pilot --output research/interaction/correction_pilot_96 --updates 96
PYTHONDONTWRITEBYTECODE=1 /tmp/batch6-venv/bin/python -m unittest research.interaction.test_semantic_correction research.interaction.test_correction
```
