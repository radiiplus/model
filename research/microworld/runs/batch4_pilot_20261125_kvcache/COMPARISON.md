# Batch 4 Micro-World Training

## Decision

**NO-GO.** No arm exceeded the compositional majority baseline on the fixed pilot sample. This bounded pilot does not establish general reasoning.

## Training

| Arm | Updates | Input tokens | Active LM targets | Structural token labels | Backbone params | Auxiliary params | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 64 | 54248 | 54184 | 0 | 4470272 | 0 | 48.1 |
| B | 64 | 54248 | 0 | 667116 | 4470272 | 115393 | 79.0 |
| C | 64 | 54248 | 54184 | 667116 | 4470272 | 115393 | 81.6 |

## Validation

| Arm | Initial language NLL | Final language NLL | Initial objective | Final objective |
| --- | ---: | ---: | ---: | ---: |
| A | 6.2569 | 4.9580 | 6.2614 | 4.9622 |
| B | 6.2569 | 6.0925 | 1.8616 | 1.5269 |
| C | 6.2569 | 4.9824 | 8.1230 | 6.5731 |

## Free Generation

| Split | System | Examples | Coverage | Accuracy when covered | Overall accuracy | Termination | Repetition | Invalid entity refs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | Initial Foundation | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| validation | A | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| validation | B | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| validation | C | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| iid_test | Initial Foundation | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| iid_test | A | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| iid_test | B | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| iid_test | C | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| compositional_test | Initial Foundation | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| compositional_test | A | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| compositional_test | B | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| compositional_test | C | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| counterfactual_test | Initial Foundation | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| counterfactual_test | A | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| counterfactual_test | B | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| counterfactual_test | C | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| grammar_test | Initial Foundation | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| grammar_test | A | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| grammar_test | B | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |
| grammar_test | C | 32 | 0.000 | 0.000 | 0.000 | 0.0% | 100.0% | 0 |

## Deterministic Baselines

| Split | Majority exact structural accuracy | Retrieval coverage | Template exact accuracy | Oracle exact structural accuracy |
| --- | ---: | ---: | ---: | ---: |
| validation | 0.517 | 0 | 1.000 | 1.000 |
| iid_test | 0.531 | 0 | 1.000 | 1.000 |
| compositional_test | 0.524 | 0 | 1.000 | 1.000 |
| counterfactual_test | 0.535 | 0 | 1.000 | 1.000 |
| grammar_test | 0.524 | 0 | 1.000 | 1.000 |

## Representative Failure

- Arm A (world-00920015:noop:entities:common:temporal_sequence): generated `(-j (-j (p-(p- (-j-(p-(-(p-(p-j (-(p-j (-(p-(-(p-(-j-(p (p-j-j-j (p-j (p-j (-j (-j (p-(p (p-j-j (p-(-j-(-j (-(p (-(p- (-j-j (-j-`; parseable=False, semantically_correct=False, stop=token_limit.
- Arm B (world-00920015:noop:entities:common:temporal_sequence): generated `                                                                                                                                `; parseable=False, semantically_correct=False, stop=token_limit.
- Arm C (world-00920015:noop:entities:common:temporal_sequence): generated `(-(-(-(-(-(-(- (-p-(-p-p-(-(-p- (-(-p (-p-(-(-(-p- (-p- (-p-p (-p (-p (-(- (-(-(-p-(-(-(-(-p (-p (-p (-(-p (-(- (-(-p (-p-p (-(-`; parseable=False, semantically_correct=False, stop=token_limit.

## Limits

All arms consumed the same 64 examples and 54,248 input tokens. Arm A and C optimized 54,184 language next-token targets; Arm B optimized zero language targets and 667,116 structural token labels. These supervision counts are not matched. Each arm used one seed and 32 generation prompts per partition. No arm was extended automatically. See per-arm JSON for grouped structural metrics, exact samples, validation curves, hashes, and resource measurements. A 100% deterministic template ceiling remains a reference; neural scores do not use teacher forcing.
