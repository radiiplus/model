# Patch 8: Adaptive reasoning loop

## Decision

The symbolic candidate was validated by the preregistered transfer gate. This validates the episode/procedure/promotion mechanics only; it is not evidence that a language model learned a reasoning strategy.

## Protocol

The deterministic extractor read only the 12 `chain` training episodes. It grouped distinct input hashes and proposed an ordered-state-update procedure from their observable update/check actions. It did not receive interpolation or extrapolation labels. The validator compared the same fixed fallback baseline with and without the candidate on identical held-out `trace` and `bracket` inputs, and checked a direct-addition baseline as a protected family. Both paths had one iteration, at most 128 symbolic actions, and zero retrieval calls. The foundation checkpoint was hashed before and after; it was not loaded.

## Transfer

### trace

| Split | Seed | Base | Strategy | Gain | Gate |
|---|---:|---:|---:|---:|---|
| interpolation | 811 | 0/24 | 24/24 | 1.00 | pass |
| extrapolation | 811 | 2/24 | 24/24 | 0.92 | pass |
| interpolation | 812 | 1/24 | 24/24 | 0.96 | pass |
| extrapolation | 812 | 1/24 | 24/24 | 0.96 | pass |
| interpolation | 813 | 1/24 | 24/24 | 0.96 | pass |
| extrapolation | 813 | 1/24 | 24/24 | 0.96 | pass |

### bracket

| Split | Seed | Base | Strategy | Gain | Gate |
|---|---:|---:|---:|---:|---|
| interpolation | 811 | 10/24 | 24/24 | 0.58 | pass |
| extrapolation | 811 | 14/24 | 24/24 | 0.42 | pass |
| interpolation | 812 | 15/24 | 24/24 | 0.38 | pass |
| extrapolation | 812 | 11/24 | 24/24 | 0.54 | pass |
| interpolation | 813 | 16/24 | 24/24 | 0.33 | pass |
| extrapolation | 813 | 12/24 | 24/24 | 0.50 | pass |

### Protected family

| Seed | Base | Selected system | Regression | Gate |
|---:|---:|---:|---:|---|
| 811 | 24/24 | 24/24 | 0.00 | pass |
| 812 | 24/24 | 24/24 | 0.00 | pass |
| 813 | 24/24 | 24/24 | 0.00 | pass |

## Cost and limits

The run recorded 12 training episodes and took 0.012 s with 25018368 B peak RSS. There are no neural parameters or recurrent loops in this patch. The checkpoint SHA-256 was `d9afd380ccde8eb47d3d2cdbd156fe0a9c76cc84bc043fd2e90eb936ab7b1341` both before and after.

The `stepwise` candidate applies a project-authored safe interpreter after being proposed from chain episodes. Its parsers are hand-written, so this tests the auditable strategy lifecycle and algorithmic transfer, not a language model acquiring or learning that procedure. It is not evidence of natural-language reasoning or improved interactions. Test labels are consumed only by the validator after extraction and are not added to training episodes. The fixed fallback responder remains weak on the primary transfer family.

Full episode/strategy schemas, split hashes, per-instance outcomes, promotion gate and resource data are in `out.json`.
