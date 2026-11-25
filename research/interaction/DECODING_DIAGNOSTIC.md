# Foundation Decoding Diagnostic

**Decision: NO-GO for adopting a decoding change as the fix or for treating this as evidence to start training changes.** Top-p sampling increased cue-parser coverage and produced some label agreements on compositional and counterfactual examples, but the outputs remained non-answer-like, exact response and numeric-detail accuracy were zero, and IID overall accuracy remained zero. The current result is a weak signal for a larger decode-only replication, not evidence of reliable semantic improvement.

## Scope and repository audit

This was inference-only. No optimizer step ran and no checkpoint was written or modified. The run used CPU float32 and the existing 2560 MiB memory guard. Repository state was branch `adaptive`, HEAD `bd61b24ace8b23290fb3a8e546e73c75107ff535`.

The selected checkpoint is `research/structure/pilot_v5/b.pt`, a 12-update structural-objective pilot, SHA-256 `0f36d1337f370d9edd33879f5e8aa3e44b8f8ef247014ab6b219d50b982d0a53`. Its checkpoint metadata identifies `research/foundation/data/learn.pt` as its initial checkpoint, SHA-256 `76b60e84c3b203ef1a30bd93f0fd7e8c8fbe619d08799e57e7f434a6c237a5fe` (800 updates). The later `research/foundation/data/a1.pt` (8,000 updates) is also present, SHA-256 `46d0f379d86500be14e6a9812aa254c515d7ce704bb7b17a920acd24ec2529be`. The current semantic-correction benchmark uses the pilot B checkpoint as its frozen start.

The model has 4,470,272 parameters, vocabulary size 4,096, context length 1,024, four transformer layers, width 256, four attention heads, and tied embedding/output weights. The tokenizer is version 1; tokenizer file SHA-256 `956cefc54a7f4e54761c477124d210a1068b0fdb8b0850ab7c77a275849b93fe`, tokenizer digest `28d45ae290e761e90c42f95dbe0ae68befb32c1ba44006f01e4dd46fa9dfcff8`. Special IDs: BOS 256, EOS 257, END 258, SYSTEM 259, USER 260, ASSISTANT 261, TOOL 262, META 263. The semantic-correction dataset manifest SHA-256 is `3950735151bebf9802ee0e38fc4deb8664237780f4377212046c9bb1e2c7ac25`.

All checkpoint files found under `research/**/*.pt` were hashed into the machine-readable report. The inventory contains 44 files, including the 800-update checkpoint, 8,000-update checkpoint, three structural pilot checkpoints, and correction-pilot checkpoints. This inventory records local file presence, size, and hash rather than inferring availability from historical reports.

The previous baseline evaluator uses greedy argmax, a 40-token output limit, and stops on either EOS or END. It hides non-text special tokens from visible output. Its benchmark scorer is a reviewed yes/no cue parser (`semantic_label`), not a human assessment of full answer meaning. Evaluation selected explicit-surface prompts: 16 validation, 16 IID, 16 compositional, and 32 counterfactual examples. These represent eight underlying worlds in each held-out partition, with multiple state variants from each world.

## Baseline reproduction

The existing `correction_pilot.generate` and score path was rerun before decoding interventions. It reproduced the previous `correction_pilot/comparison.json` output text and stop reason exactly for all four partitions. All 80 examples generated 40 visible tokens and stopped at the token limit. EOS and END termination were both 0%; empty output was 0%; repeated-trigram output rate was 100%; semantic parser coverage and overall accuracy were 0%. Exact response and numeric-detail accuracy were also 0%.

Thus the previous result is reproducible for the selected checkpoint, tokenizer, manifest, prompt subset, and greedy implementation. It should not be interpreted as a broad natural-language capability estimate: the prompt subset is small and the metric is a cue parser.

## Decoding configurations

Each intervention was run independently with a maximum of 40 generated tokens:

| Configuration | Implementation |
| --- | --- |
| Greedy | Argmax; baseline behavior |
| Temperature/top-p | Temperature 0.8, nucleus p=0.9; per-example stable seed derived from 9142026 and example ID |
| Repetition penalty | 1.2 applied to previously generated token logits |
| No-repeat trigram | Bans a token that would repeat a generated-content trigram |
| Minimum length | Masks EOS and END for the first four generated tokens |
| Contrastive search | Alpha 0.6, top-k 4, hidden-state cosine degeneration penalty; supported by the model's hidden-state interface |

The validation run did not select a combination for further testing, so no combined configuration was evaluated. No parameters were tuned on held-out splits. The independent runs used identical prompts, model weights, tokenizer, and evaluator. Sampling randomness is repeatable for this run, but only one sampling seed was evaluated.

## Results

Values below are semantic parser coverage / overall semantic label accuracy. Overall accuracy counts unknown/unparsed output as incorrect. “Repeat” is the percent of outputs containing a repeated trigram. “Ended” is the percent terminating by EOS or END. Mean length counts generated output tokens, including any generated special token.

| Decoder | Validation | IID | Compositional | Counterfactual |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0 / 0% | 0 / 0% | 0 / 0% | 0 / 0% |
| Temperature 0.8, top-p 0.9 | 37.5 / 6.25% | 18.75 / 0% | 56.25 / 37.5% | 21.875 / 12.5% |
| Repetition penalty 1.2 | 0 / 0% | 0 / 0% | 0 / 0% | 0 / 0% |
| No-repeat trigram | 0 / 0% | 0 / 0% | 0 / 0% | 0 / 0% |
| Minimum 4, EOS/END masked | 0 / 0% | 0 / 0% | 0 / 0% | 0 / 0% |
| Contrastive, alpha 0.6, k=4 | 0 / 0% | 0 / 0% | 0 / 0% | 0 / 0% |

| Decoder | Split | Empty | Repeat | Mean tokens | EOS | END | Token limit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Greedy (also penalty, min length, contrastive) | Every split | 0% | 100% | 40.00 | 0% | 0% | 100% |
| No-repeat trigram | Validation | 0% | 0% | 30.00 | 0% | 100% | 0% |
| No-repeat trigram | IID | 0% | 0% | 37.00 | 0% | 100% | 0% |
| No-repeat trigram | Compositional | 0% | 0% | 33.13 | 0% | 68.75% | 31.25% |
| No-repeat trigram | Counterfactual | 0% | 0% | 33.22 | 0% | 100% | 0% |
| Temperature/top-p | Validation | 0% | 62.5% | 37.25 | 6.25% | 12.5% | 81.25% |
| Temperature/top-p | IID | 0% | 81.25% | 40.00 | 0% | 0% | 100% |
| Temperature/top-p | Compositional | 0% | 75% | 38.19 | 6.25% | 0% | 93.75% |
| Temperature/top-p | Counterfactual | 0% | 59.375% | 36.13 | 12.5% | 0% | 87.5% |

The temperature/top-p decoder produced parser agreement on 1/16 validation, 0/16 IID, 6/16 compositional, and 4/32 counterfactual examples. Across held-out data this is 10/64 overall label agreements, with 19/64 outputs covered. Its conditional accuracy among parser-covered held-out outputs is 10/19 (52.6%), but this denominator is small and parser-based. Exact response accuracy and numeric-detail accuracy were 0% in every split and every configuration. Representative “correct” top-p outputs were noisy text containing a cue such as `no`; they did not reproduce the answer target or its quantities. Therefore the nonzero scores do not demonstrate coherent answers.

The repetition penalty did not affect the loop. No-repeat trigram eliminated the measured repeated trigrams and commonly reached END, but produced no parseable or correct answers. This is a direct example of improved termination and surface repetition metrics without improved semantic performance. Greedy and contrastive decoding both repeated the baseline loop.

## EOS/END and probability diagnosis

In the selected checkpoint, greedy generation did not terminate early: EOS and END were never chosen, while token-limit termination was 100% in every partition. Minimum-length EOS/END masking produced byte-for-byte identical outputs. Premature EOS/END is therefore not the cause of this checkpoint's observed failure, and globally suppressing EOS has no support from this run.

Teacher-forced probabilities show the markers are also poorly learned as target tokens. At the correct END target position, mean END probability ranged from 0.99% to 1.56% across partitions and END was never top-1. At the correct EOS target position, mean EOS probability was 1.19% to 1.32%, and EOS was never top-1. At the first answer position, EOS probability ranged from 0.049% to 0.687%, END from 0.009% to 0.020%, and neither special token was top-1. The correct first answer token was top-1 in 0/16 validation examples; its mean probability ranged from 0.045% to 0.721% by partition.

For additional context, teacher-forced validation answer-target NLL for pilot B was 4.4845 over 500 target tokens, token accuracy was 38.4%, and first-answer-token accuracy was 0%. Across greedy traces, mean token entropy was 3.86 nats; the loop becomes highly confident on the space token (about 0.94 probability) while the repeated lexical token “to” remains the most likely choice with only about 0.03 probability. This points to weak next-token distributions and unstable lexical continuation, not simply a decoder stopping decision. Related checkpoint validation scores are recorded in the JSON report for `learn.pt`, `a1.pt`, pilots A/B/C, and correction checkpoints; these are diagnostics, not a controlled cross-checkpoint generation comparison.

## Reproduction and artifacts

From the repository root, with the locked CPU dependencies installed:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 python -m research.interaction.decoding_diagnostic
```

The script revalidates dataset/tokenizer hashes, loads the frozen checkpoint, reproduces the prior greedy outputs, runs all independent configurations and teacher-forced EOS checks, and writes raw predictions/traces and checkpoint hashes to [report.json](decoding_diagnostic/report.json). It performs no training and writes no checkpoint. The implementation is in [decoding_diagnostic.py](decoding_diagnostic.py). The run used the repository CPU lock in `/tmp/decoding-diagnostic-venv`, PyTorch 2.15.0+cpu, one CPU thread, with an elapsed time of about 330 seconds and peak RSS recorded in the JSON report.

## Conclusion

The evidence supports three findings: (1) the baseline failure is a repetition/continuation collapse, not premature EOS/END; (2) no-repeat decoding can improve repetition and termination metrics without producing answers; and (3) temperature/top-p shows a small parser-level correctness signal on compositional and counterfactual examples, but no IID improvement and no exact/numeric correctness. Because the scoring parser can accept incoherent text containing an answer cue, and only one small prompt sample and sampling seed were used, this does not establish material semantic improvement.

**NO-GO:** do not adopt these decoding settings as a semantic fix and do not use this result to justify training changes. A decode-only replication with independent sampling seeds, a larger world-level test set, and human-reviewed answer coherence would be needed to validate the top-p signal.
