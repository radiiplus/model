# Recursive Reasoning Research Anchors

Status: literature notes for experiment design. These papers motivate hypotheses; none validates the proposed combination or its performance at the project's parameter and compute limits.

## Shared-Weight Recurrence

[Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2676109d49d1eb26d6bc584a8f556305-Abstract-Conference.html) reports that repeatedly applying shared Transformer layers can approach a deeper model on several reasoning evaluations. Its own language-modeling results are not uniformly competitive with the deeper control: the looped models have worse perplexity than the 24-layer baseline, while showing relative gains on contextual QA, math and synthetic reasoning primitives. This supports testing recurrence as a reasoning hypothesis, not replacing the plain causal LM control or assuming memorization improves.

## Separate Answer and Latent State

[Less is More: Recursive Reasoning with Tiny Networks (TRM)](https://arxiv.org/abs/2510.04871) keeps an embedded candidate answer `y` separate from latent reasoning state `z`: it refines `z` conditioned on the input and current answer, then updates the answer from `z` and its previous value. The paper reports strong puzzle results, including a 7M-parameter attention variant on ARC and maze/Sudoku tasks. Those results do not establish open-domain language ability or performance near 1M parameters. Reproduce the two-state update principle only; treat state dimensions, update equations and supervision schedule as experimental choices.

## Training-Time Refinement and Transfer

The [ARC Prize HRM analysis](https://arcprize.org/blog/hrm-analysis) reports that outer-loop refinement contributed substantially on its ARC evaluation, including gains when training used more refinement steps than inference. It also found limited cross-task transfer in its setup: much of the score was attributable to training on the specific evaluation tasks and their augmentations. The analysis is benchmark-specific and secondary to the model paper, but it motivates deep supervision as an ablation and strict held-out task-family transfer tests. Start with short unrolled recurrence and ordinary backpropagation; do not import HRM's adaptive halting or specialized training pipeline by default.

## External Context and Persistent Memory

[Recursive Language Models](https://arxiv.org/abs/2512.24601) treat a long prompt as data in an external environment and recursively inspect or decompose snippets through model calls. This is a systems precedent for bounded external context, not a learned memory mechanism. Any programmatic inspection or execution layer needs explicit operation, recursion, time, memory and output budgets. Generated code must remain outside the system unless a separately designed OS sandbox exists.

[Titans: Learning to Memorize at Test Time](https://research.google/pubs/titans-learning-to-memorize-at-test-time/) proposes a learned neural long-term memory alongside attention over shorter context, with memory updated at test time. It is useful background for a later isolated learned-memory study. It is not the first target: mutable learned state adds update compute and raises interference, provenance, invalidation and deletion questions that external immutable evidence storage handles more directly.

## Test-Time Learning

[End-to-End Test-Time Training for Long Context](https://arxiv.org/abs/2512.23675) studies context-conditioned online learning and meta-learned initialization, reporting experiments up to 3B parameters and 164B training tokens. That scale and training regime are far beyond this CPU prototype. This is background only; do not port its online optimization into the first recurrence experiment.

## Implications for This Project

- Preserve the plain causal Transformer and evaluate language modeling separately from reasoning tasks.
- Compare a shared-block recurrent model against parameter- and compute-aware controls; report unique/trainable parameters, loop count, wall time and memory separately.
- Train the refinement process itself, initially with a short fixed unroll and ordinary backpropagation. Keep inference loop count explicit.
- Test transfer on independently held-out problems and preferably different task families. Do not promote a strategy based on training-task improvement or model confidence.
- Treat the contribution, if supported, as the validated combination and training protocol. Recurrence, external memory, adapters and validation gates are established ideas individually.

These sources are design precedents, not evidence that the proposed frozen-foundation, dual-state refinement, external-memory and validated-strategy system will work as a whole.
