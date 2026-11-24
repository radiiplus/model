# State Value and Policy Research

## Design

The experiment uses the versioned deterministic structural world as a one-step, terminal simulator. The target is selected by an explicit rule from `e0`'s color (`red -> e1`, `blue -> e2`, `green -> e3`, `silver -> e4`). The goal is to reach that target from `e0` using only ownership/control edges.

Reward is `+1.0` if the goal holds in the resulting state, minus `0.05` for any non-noop action. Invalid actions or constraint violations receive `-1.0`; the candidate set is no-op plus five valid control-edge additions and is validated by the simulator before scoring. The state value target is the maximum terminal reward over those six actions. It is not a truth score. Transition correctness is determined separately by the exact simulator.

The minimal supervised prototype uses a 167,848-parameter MLP with transition, action-value and state-value heads. State plus an explicit target one-hot are provided to both the policy/value inputs. It trains on 288 state/action/outcome rows from 48 training worlds for 160 epochs, then evaluates held-out worlds. The policy takes the action with the largest predicted value from the simulator's valid action set. No reinforcement learning, Foundation change, language generation, or persistent memory writes are involved.

## Results

One fixed seed was run. Reward estimates have Brier scores from 0.032 on IID to 0.340 on counterfactual worlds, and 0.295 on held-out graph patterns. Value calibration also degrades on unfamiliar splits. The learned transition output never passed the exact verifier. Edge-bit accuracy is high, which shows why aggregate accuracy alone is insufficient: a transition must reconstruct every state field correctly before it can be used.

| Split | Exact transition | Edge-bit accuracy | Reward Brier | Reward ECE | Value Brier | Value ECE | Policy optimal | Policy regret | Noop optimal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.0% | 100.0% | 0.011 | 0.071 | 0.00007 | 0.0041 | 83.3% | 0.008 | 25.0% |
| IID | 0.0% | 100.0% | 0.032 | 0.105 | 0.00012 | 0.0067 | 83.3% | 0.008 | 16.7% |
| Compositional | 0.0% | 96.3% | 0.339 | 0.347 | 0.0265 | 0.1607 | 41.7% | 0.188 | 41.7% |
| Held-out patterns | 0.0% | 94.9% | 0.295 | 0.322 | 0.0251 | 0.1524 | 66.7% | 0.017 | 58.3% |
| Counterfactual | 0.0% | 96.3% | 0.340 | 0.335 | 0.0290 | 0.1699 | 41.7% | 0.267 | 33.3% |

Policy optimality ranges from 42% to 83%; it only matches no-op on the compositional split and modestly exceeds it on held-out patterns. Exact transition verification is 0% on all splits. These results do not meet a useful action or outcome gate.

## Decision

**NO-GO for further training or policy promotion.** Keep the deterministic simulator as the outcome authority. The supervised value head may serve as a controlled calibration fixture, but this result does not justify a larger run. Any follow-up needs a nontrivial objective with varied optimal actions, multiple-step outcomes, and held-out topology coverage; exact transition verification remains mandatory.

## Reproduction

```bash
timeout 180s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /tmp/batch6-venv/bin/python -m research.structure.state_value
```

The command writes the complete report, including per-split metrics, seed, objective, reward, training history, fit time and parameter count, to `state_value/report.json`. The independent verifier and feature encodings are in `state_value.py`; regression cases are in `test_state_value.py`.
