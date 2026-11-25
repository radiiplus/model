# Dense Structural Supervision

## Scope

This batch adds tokenizer-aligned, deterministic structural annotations and defines three Foundation objectives. It does not load or update Foundation weights, run an optimizer, or create a checkpoint. The original 4,470,272-parameter Foundation remains the shared backbone. Structural conditions B and C add token-level auxiliary prediction heads; their measured parameter cost is recorded by `prepare_objectives.py` after the generated training vocabulary is frozen.

## Alignment and labels

`Tokenizer.encode_with_offsets` follows the existing byte-level BPE merge order and returns half-open UTF-8 byte spans and Unicode code-point spans for every token. For multi-byte characters split into byte tokens, each token maps to the full character span it overlaps. Dense sidecars retain original text, character and byte spans, token bounds, and token indices. The generator checks a fresh re-encoding, text slicing, and byte/token coverage for every annotation.

Annotations are derived from the world schema, deterministic transition oracle, and manually authored renderer. The 20 annotation types cover entity mentions, identities and types; property mentions, keys and values; relation phrases, types and endpoints; negation and time; actions, validity and preconditions; state changes; invalid actions and violated constraints. Overlapping character spans are retained in the sidecar. Each objective has its own token-label channel, and conflicting labels within one channel fail preflight rather than silently overwriting one another.

The train split alone defines class vocabularies. Held-out partitions must fit the Foundation's 1,024-token context and use labels present in the train vocabulary. The sidecar's token IDs are generated with the frozen `ready7` tokenizer; manifests record both its semantic digest and file hash.

## Controlled conditions

All arms use the same 64 distinct training examples, source presentations, frozen tokenizer, split files, initial checkpoint, seed, update count and maximum sequence length. The exact sampled indices and example IDs are in `dataset_v4/objective_manifest.json`. This predeclared sample has 54,248 input tokens and 54,184 next-token targets. Its structural objective has 667,116 token-task targets because token positions are supervised independently across multiple annotation channels, including outside labels. The full train split has 4,973,192 source annotation spans. The configurations predeclare a 64-update pilot budget but no training has been run. Reports must list language-target tokens and per-task structural token labels separately, so equal update counts are not presented as equal supervision.

| Arm | Objective | Added parameters |
| --- | --- | ---: |
| A | Next-token language loss only | 0 |
| B | Dense structural losses only | Computed from train vocabulary |
| C | Language plus dense structural losses | Same heads as B |

The dense head uses a shared width-256 projection and one classifier per task. Each task loss macro-averages cross-entropy over classes present in the batch; task losses are averaged within nine semantic groups, then group weights are applied. This reduces domination by common outside labels or annotation-heavy groups. Both task-level and group-level weights are independently configurable. `configs/loss_sensitivity.json` declares a one-factor 0.5/1/2 sensitivity study on validation data only; it has not been executed.

## Preflight and reproduction

The generator writes compressed examples and aligned annotation sidecars with per-file SHA-256 hashes. Its context and oracle checks run during generation. `prepare_objectives.py` builds train-only label vocabularies, checks token-label conflicts and held-out coverage, counts token-level supervision, and reports parameter counts. It does not train.

The alignment and objective tests plus the existing microworld regression tests pass: 16 tests total. The objective test uses a tiny randomly initialized in-memory model and performs forward-only loss checks; it does not load or update the Foundation checkpoint.

~~~sh
python -m pip install -r research/experiments/requirements_cpu.txt
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  research.microworld.test_microworld research.microworld.test_dense -v
PYTHONDONTWRITEBYTECODE=1 python -m research.microworld.generator \
  --output research/microworld/dataset_v4 --world-count 100 --seed 920000 --baselines
PYTHONDONTWRITEBYTECODE=1 python -m \
  research.microworld.prepare_objectives \
  --dataset research/microworld/dataset_v4 \
  --output research/microworld/dataset_v4/objective_manifest.json
~~~

Do not start a neural training run until the generated data and objective manifest pass all preflight checks. This pilot is an annotation and loss-design validation, not evidence that structural supervision improves generalisation.
