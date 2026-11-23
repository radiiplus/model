# Patch 8: Adaptive reasoning loop

## Decision

The symbolic candidate was validated by the preregistered transfer gate. This validates the episode/procedure/promotion mechanics only; it is not evidence that a language model learned a reasoning strategy.

## Protocol

The deterministic extractor read only the 12 `chain` training episodes. It grouped distinct input hashes and proposed an ordered-state-update procedure from their observable update/check actions. It did not receive interpolation or extrapolation labels. The validator compared a constant frozen baseline with the candidate on identical held-out `trace` and `bracket` inputs, and checked `addition` extrapolation as a protected family. Both conditions had one iteration, at most 128 symbolic actions, and zero retrieval calls. The foundation checkpoint was hashed before and after; it was not loaded.

## Transfer

### trace

| Split | Base | Strategy | Gain | Gate |
|---|---:|---:|---:|---|
| interpolation | 0/4 | 4/4 | 1.00 | pass |
| extrapolation | 0/4 | 4/4 | 1.00 | pass |

### bracket

| Split | Base | Strategy | Gain | Gate |
|---|---:|---:|---:|---|
| interpolation | 3/4 | 4/4 | 0.25 | pass |
| extrapolation | 4/4 | 4/4 | 0.00 | pass |

### Protected family

Addition extrapolation: base 0/4, selected system 0/4; regression 0.00.

## Cost and limits

The run recorded 12 training episodes and took 0.008 s with 24215552 B peak RSS. There are no neural parameters or recurrent loops in this patch. The checkpoint SHA-256 was `d9afd380ccde8eb47d3d2cdbd156fe0a9c76cc84bc043fd2e90eb936ab7b1341` both before and after.

The `stepwise` candidate dispatches into the project-authored deterministic synthetic-task solvers. Thus transfer here means that an abstract procedure record selected a known symbolic solver on separately generated families; it does not show acquired model behavior, natural-language reasoning, or improvement to interactions. Test labels are consumed only by the validator after extraction and are not added to training episodes. Small synthetic slices have wide uncertainty.

Full episode/strategy schemas, split hashes, per-instance outcomes, promotion gate and resource data are in `out.json`.
