# Microworld Specification

## Purpose

research.microworld defines a deterministic rooms-and-objects environment for testing whether a model can recover entities, properties, relations, action preconditions, and state changes from language. It is independent of the existing graph-only research/structure generator. This batch does not train a model.

## World

Every generated world has ten entities:

| Type | Count | State and properties |
| --- | ---: | --- |
| Agent | 2 | Pronoun, carry capacity, room position |
| Object | 3 | Color, weight, portability, position, optional owner |
| Container | 2 | Capacity, material, latch mechanism, dynamic open/closed state, position |
| Location | 3 | Room kind and level; adjacent rooms form a chain |

Entity IDs are stable within a world. Names include a world-specific suffix so exact sentences and transitions cannot be copied across partitions through reused names. World state contains one position for each non-location entity, open containers, and object ownership. Position relations are at(location), inside(container), or held_by(agent). Location resolution follows container nesting and the agent holding an object.

Most worlds place containers in rooms and objects in rooms or containers. Some training worlds contain a container in another container while keeping objects outside that nested container. The compositional test combines the familiar container-in-container and object-in-container relations into a held-out chain: an object is inside the nested container. No training world has that full composition.

## Actions

Each world includes machine-readable action specifications in action_specs. The transition engine is in [oracle.py](oracle.py).

| Action | Main preconditions | Deterministic effect |
| --- | --- | --- |
| move | Agent and room exist; destination is adjacent and differs from current room | Agent changes rooms; held objects resolve to the agent's new room |
| pick_up | Object is portable, unheld, and not contained; agent shares its room; carry capacity remains | Object becomes held by the agent |
| put_in | Agent holds object; container is open, nearby, and has capacity | Object becomes contained |
| take_out | Object is inside the named open nearby container; carry capacity remains | Object becomes held by the agent |
| open / close | Agent is beside a container in the opposite latch state | Container latch state changes |
| give | Giver holds object; recipient is a different nearby agent | Recipient holds the object and becomes its owner |

Invalid actions return a stable reason code and explanation and preserve the state at that action step. Episodes include a valid multi-step put-away, travel and transfer sequences, and an invalid take-out attempt after closing its container.

## Constraints

The oracle checks that all positioned entities resolve to exactly one room, positions are well-typed, containment is acyclic, containers and agents stay within capacity, and only objects or containers can be contained. Ownership endpoints must be an object and an agent. Invalid action preconditions are reported separately from invariant violations: an action can be invalid while the unchanged world state still satisfies every invariant.

Oracle queries cover entity inventory, properties, all typed relations, effective location, action validity and reason, exact before/after transition, constraint violations, and counterfactual outcome differences. Labels are calculated directly from the structured world and transition code. No model generates labels.

## Language

For each example, the renderer includes entity descriptions, property facts, typed relation facts, state facts, action rules, an episode action sequence, and a query. Shared constructions include simple declaratives, inverse relation wording, compact relation ledgers, pronoun reference, temporal sequencing, and compound clauses. The grammar test uses separate negated-question and inverted-word-order constructions, with separate answer-template IDs. Query and answer text is rendered from the same structured records.

The held-out grammar set is intentionally small in the pilot. It is suitable for leakage checks and smoke evaluation, not for a strong estimate of language generalization.

## Reproduction

Generate a compressed dataset with:

~~~sh
python -m research.microworld.generator \
  --output research/microworld/dataset_v1 \
  --world-count 100 --seed 920000 --baselines
~~~

The output path must not already exist. The supported world-count range is 50 to 200,000. JSONL partitions are written incrementally as deterministic gzip streams. --baselines materializes a bounded in-memory baseline evaluation and is limited to 1,000 worlds; omit it for larger corpus generation.
