# Microworld Dataset Audit

## Measured Pilot

Generator: microworld-v1, seed 920000. The generated dataset is in [dataset_v1](dataset_v1/manifest.json). JSONL data is local and Git-ignored; its manifest and baseline report are retained with per-file byte counts and SHA-256 hashes.

| Split | Worlds | Episodes | Examples | Whitespace tokens |
| --- | ---: | ---: | ---: | ---: |
| Train | 70 | 420 | 17,640 | 5,352,340 |
| Validation | 10 | 60 | 2,520 | 764,620 |
| IID test | 10 | 60 | 2,520 | 764,620 |
| Compositional test | 4 | 24 | 1,008 | 304,312 |
| Counterfactual test | 4 | 28 | 1,200 | 364,456 |
| Held-out grammar test | 2 | 12 | 168 | 52,534 |
| **Total** | **100** | **604** | **25,056** | **7,602,882** |

The token count is a whitespace count over each rendered prompt and its natural-language target; it is a stable corpus-size measure, not the tokenizer's model-token count. There are 25,056 unique prompts, 11,995 unique sentences, 600 unique world/episode transitions, eight grammar constructions, 62 input templates, and 15 answer templates. Gzipped pilot files total approximately 7.0 MB.

Per-file compressed sizes and hashes, the partition hash, seed, and full statistics are in [manifest.json](dataset_v1/manifest.json). Baseline results are also available as [baselines.json](dataset_v1/baselines.json).

## Leakage And Oracle Checks

- Every generated world and initial state passes schema validation.
- Every prompt is unique in the generated pilot and maps to one ground-truth target. The split validator rejects a prompt that maps to conflicting labels.
- World identifiers and exact prompt strings do not cross partitions.
- Transition hashes are checked across partitions. The measured pilot has 600 unique world/episode transition records.
- The compositional partition has four nested_container_chain worlds; no train world has that composition. Training has 44 flat and 26 single_container_nesting worlds, exposing container-in-container and object-in-container separately but never placing an object inside the nested container.
- The grammar partition uses 14 held-out prompt template constructions and seven held-out answer-template IDs. Neither set overlaps the other partitions.
- Four counterfactual world pairs were generated. In each, picking up object 0 versus taking no action changes that object's position and the resulting state.
- Each valid action's result is independently checked against state constraints. Invalid actions preserve state at the failed step and record a reason code.
- Baseline labels are calculated by the deterministic oracle; no model is involved.

## Deterministic Baselines

Structural accuracy is exact equality between a baseline's structured result and the oracle label. Text accuracy is exact equality with the generated natural-language target; it is not a human coherence judgment.

| Partition | Direct oracle | Template system | Exact retrieval | Majority answer |
| --- | ---: | ---: | ---: | ---: |
| Validation | 100% | 100% | 0% | 51.7% |
| IID test | 100% | 100% | 0% | 50.2% |
| Compositional test | 100% | 100% | 0% | 45.2% |
| Counterfactual test | 100% | 100% | 0% | 53.5% |
| Held-out grammar test | 100% | 100% | 0% | 52.4% |

The template system uses the structured query annotations, deterministic oracle, and hand-authored answer renderer; it is a symbolic ceiling, not a learned baseline. Exact retrieval has no held-out prompt matches. Majority chooses the most common training label for a query type, falling back to the global majority for the held-out counterfactual query type. The majority figures show that varied query labels prevent a single answer from solving the dataset.

## Dense Annotation Pilot

Dense supervision is generated as a second dataset version at [dataset_v4](dataset_v4/manifest.json), retaining the same 100 underlying worlds and split assignments as the language pilot. The manifest SHA-256 is `3df845563dd5f84a29ebd60d858e1798f6f7193f13fb1eebc2e1f4cb14525d19`. It records hashes and byte sizes for every world, example, and dense annotation file. The compressed annotation sidecars are local generated artifacts; the manifests remain small and reproducible.

| Split | Examples | Encoded model tokens | Maximum sequence |
| --- | ---: | ---: | ---: |
| Train | 17,640 | 14,817,434 | 1,016 |
| Validation | 2,520 | 2,124,278 | 1,017 |
| IID test | 2,520 | 2,127,362 | 1,016 |
| Compositional test | 1,008 | 848,222 | 1,010 |
| Counterfactual test | 1,200 | 1,017,678 | 1,009 |
| Held-out grammar test | 168 | 146,410 | 1,013 |
| **Total** | **25,056** | **21,081,384** | **1,017** |

The data contains 7,062,254 source annotations across 20 types. Counts by type are in the dense dataset manifest. The tokenizer semantic digest is `28d45ae290e761e90c42f95dbe0ae68befb32c1ba44006f01e4dd46fa9dfcff8` and the tokenizer file SHA-256 is `956cefc54a7f4e54761c477124d210a1068b0fdb8b0850ab7c77a275849b93fe`. All examples fit the 1,024-token context, with at least seven tokens of margin. Every serialized example and annotation record (all 25,056) was rechecked against the oracle, original text, UTF-8 byte offsets, fresh BPE offsets, and token indices.

The objective preflight uses only training annotations to create class vocabularies, then checks token-label conflicts and held-out class coverage across all partitions. Its parameter and label-count report is [objective_manifest.json](dataset_v4/objective_manifest.json), SHA-256 `aa4e43cd4f707a668b719e9a24ebcde0cbe91a47b89c7961f661657a3880f261`. It found no token-label conflicts and full held-out class coverage. The structural heads add 115,393 parameters (total 4,585,665); Arm A adds none. The predeclared 64-example pilot samples distinct train rows with seed 20261125 and uses 54,248 input tokens. Arm A/C optimize 54,184 next-token targets; Arm B/C have 667,116 per-task token-label targets because several annotation channels label the same sequence positions. These supervision quantities are not matched. No training was performed.

The final test command was `PYTHONDONTWRITEBYTECODE=1 /tmp/microworld-b3-venv/bin/python -m unittest research.microworld.test_dense research.microworld.test_microworld -v`; all 16 tests passed. The CPU environment emitted a PyTorch warning that NumPy is unavailable, but the tests passed without NumPy-dependent operations.

## Scale And Limits

The CLI supports 50–200,000 worlds and streams compressed world/example files. Split allocation was tested at the 200,000-world limit. Full generation and baseline metrics were measured only at 100 worlds; a 50,000- or 200,000-world generation was not run. Baseline evaluation is deliberately limited to 1,000 worlds because it keeps examples in memory. Corpus size and run time at larger scales must be measured when a larger generation is requested.

The current world family is small and authored: seven action types, three rooms, two containers, three objects, and one held-out compositional shape. The generator demonstrates deterministic labels and leakage checks; it does not establish that a language model can learn these rules. The held-out grammar partition has only two worlds. No neural training should begin until the world count and held-out evaluation set are expanded to the planned scale.

## Regeneration

~~~sh
python -m unittest research.microworld.test_microworld -v
python -m research.microworld.generator \
  --output /tmp/microworld-v1 \
  --world-count 100 --seed 920000 --baselines
~~~

The generator refuses to overwrite an existing output directory. Compare the generated manifest and file SHA-256 values with the checked-in manifest to verify deterministic reproduction.
