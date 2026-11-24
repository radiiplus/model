import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from research.foundation.model import Foundation
from research.structure.explicit_head import (
    FoundationWithStructuralHead, PROPERTIES, RELATION_SLOTS, run, structural_loss, targets,
)


CONFIG = {
    "vocab_size": 64, "context_length": 32, "layers": 1, "d_model": 16,
    "n_heads": 2, "d_ff": 32, "position_embedding": "learned_absolute",
    "normalization": "pre_layernorm", "weight_tying": True, "dtype": "float32",
}


def example():
    entities = [{"id": f"e{i}", "name": name, "properties": {
        "color": "red", "capacity": 2, "status": "ready"}}
        for i, name in enumerate(("Alder", "Beryl", "Cinder", "Dovetail", "Elm", "Fable"))]
    relationships = [
        {"id": "r0", "type": "ownership", "source": "e0", "target": "e1"},
        {"id": "r1", "type": "control", "source": "e1", "target": "e2"},
    ]
    world = {"entities": entities, "relationships": relationships,
        "initial_state": {"active_relationships": ["r0", "r1"], "property_overrides": {}}}
    result = {"active_relationships": ["r0"], "property_overrides": {"e0": {"status": "restricted"}},
              "added_relationships": {}}
    return {"world": world, "action": {"type": "remove_relationship", "relationship": "r1"},
        "resulting_state": result, "input": "Example world and remove r1.",
        "answer_target": "done", "answer_value": "done", "structure_target": "{}"}


class ExplicitHeadTests(unittest.TestCase):
    def test_hidden_api_preserves_tied_language_logits(self):
        model = Foundation(CONFIG)
        tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
        expected = F.linear(model.hidden(tokens), model.head)
        self.assertTrue(torch.equal(model(tokens), expected))

    def test_targets_include_initial_and_resulting_properties_and_states(self):
        labels = targets(example())
        self.assertEqual(labels["relation_valid"], 2)
        self.assertEqual(labels["initial_active"][:2], [1, 1])
        self.assertEqual(labels["result_active"][:2], [1, 0])
        restricted = PROPERTIES["status"].index("restricted")
        self.assertEqual(labels["property_after"]["status"][0], restricted)

    def test_head_has_finite_loss_and_backpropagates_to_backbone(self):
        model = FoundationWithStructuralHead(CONFIG)
        prediction = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
        loss, components, elements = structural_loss(prediction, example())
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(elements, 0)
        self.assertIn("properties_after", components)
        loss.backward()
        self.assertIsNotNone(model.backbone.token.weight.grad)
        self.assertIsNotNone(model.structural.relation_type.weight.grad)
        self.assertEqual(prediction["relation_type"].shape[1], RELATION_SLOTS)

    def test_interrupted_and_uninterrupted_runs_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = {}
            for split in ("train", "validation", "iid_test", "compositional_test",
                          "pattern_test", "counterfactual_test"):
                row = example()
                row.update({"example_id": f"{split}-0", "world_id": "world-0",
                            "surface_id": "natural", "query_type": "influence"})
                path = root / f"{split}.jsonl"
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                rows[split] = {"split": split, "path": path.name, "examples": 1,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            (root / "manifest.json").write_text(json.dumps({"files": list(rows.values())}),
                                                 encoding="utf-8")
            model_config = {**CONFIG, "vocab_size": 4096, "context_length": 1024}
            initial_model = Foundation(model_config)
            initial = root / "initial.pt"
            torch.save({"model": initial_model.state_dict()}, initial)
            config = {"condition": "structure", "restore_optimizer": False,
                "model": model_config, "train": {"rate": 0.001, "penalty": 0.0,
                    "clip": 1.0, "warm": 0, "decay": 10, "floor": 0.1,
                    "batch": 1, "steps": 2, "eval_examples": 1,
                    "eval_seed": 3, "seed": 7}}
            full_report = root / "full.json"
            full_checkpoint = root / "full.pt"
            run(config, initial, root, full_checkpoint, full_report)
            partial_report = root / "partial.json"
            partial_checkpoint = root / "partial.pt"
            run(config, initial, root, partial_checkpoint, partial_report, stop=1)
            resumed_report = root / "resumed.json"
            resumed_checkpoint = root / "resumed.pt"
            run(config, initial, root, resumed_checkpoint, resumed_report,
                resume=partial_checkpoint)
            full = torch.load(full_checkpoint, map_location="cpu", weights_only=True)
            resumed = torch.load(resumed_checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(full["step"], resumed["step"])
            self.assertEqual(full["sampler"], resumed["sampler"])
            self.assertEqual(full["history"], resumed["history"])
            for key, value in full["model"].items():
                self.assertTrue(torch.equal(value, resumed["model"][key]), key)


if __name__ == "__main__":
    unittest.main()
