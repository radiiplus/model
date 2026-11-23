import json
import unittest
from pathlib import Path

import torch

from research.experiments.tiny_gpt import TinyGPT
from research.recursive.evaluate import check, edges, score, solve
from research.recursive.model import Decoder, transfer
from research.recursive.tasks import CONTEXT, CROSS, build, crosses, manifest


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "experiments" / "results" / "batch12_byte_seed_11.pt"


class Tests(unittest.TestCase):
    def test(self):
        data = build()
        again = build()
        self.assertEqual(manifest(data), manifest(again))
        self.assertEqual(json.dumps(data, sort_keys=True), json.dumps(again, sort_keys=True))
        self.assertEqual(set(data["families"]), {"addition", "bracket", "graph", "chain", "trace"})
        self.assertEqual(check(data)["passed"], True)
        self.assertEqual(len(CROSS), 4)
        self.assertEqual(len(crosses(data)), 4)
        for family, parts in data["families"].items():
            for split, rows in parts.items():
                twin = again["families"][family][split]
                self.assertEqual([solve(row) for row in rows], [solve(row) for row in twin])
            self.assertEqual([len(parts[split]) for split in ("train", "validation", "interpolation", "extrapolation")],
                             [12, 4, 4, 4])
            ids = [row["id"] for rows in parts.values() for row in rows]
            prompts = [row["prompt"] for rows in parts.values() for row in rows]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(prompts), len(set(prompts)))
            upper = max(row["level"] for row in parts["train"])
            lower = min(row["level"] for row in parts["extrapolation"])
            self.assertGreater(lower, upper)
            self.assertTrue(all(len((row["prompt"] + row["target"]).encode("utf-8")) <= CONTEXT
                                for rows in parts.values() for row in rows))
            for split in parts:
                self.assertEqual({row["family"] for row in parts[split]}, {family})
                self.assertEqual({row["split"] for row in parts[split]}, {split})
        for row in crosses(data):
            self.assertEqual({item["family"] for item in row["train"]}, {row["source"]})
            self.assertEqual({item["family"] for item in row["evaluation"]}, {row["target"]})
            self.assertFalse({item["id"] for item in row["train"]} & {item["id"] for item in row["evaluation"]})
        graph = next(iter(data["families"]["graph"]["train"]))
        links = edges(graph["prompt"])
        self.assertEqual(len(links), 4 * (graph["level"] - 1))
        self.assertTrue(all(" " not in right for left, right in links))
        forms = [tuple(sorted(edges(row["prompt"])))
                 for rows in data["families"]["graph"].values() for row in rows]
        self.assertEqual(len(forms), len(set(forms)))
        self.assertTrue(score(graph, graph["target"])["correct"])
        self.assertFalse(score(graph, "S>T")["correct"])
        brackets = data["families"]["bracket"]["train"][0]
        self.assertEqual(solve(brackets), solve(brackets))
        program = data["families"]["trace"]["train"][0]["prompt"].split("DSL: ", 1)[1].split(". What", 1)[0]
        with self.assertRaises(ValueError):
            from research.recursive.evaluate import trace
            trace(program + ";import os")
        self.assertTrue(score(data["families"]["addition"]["train"][0], "not a number")["error"] == "format")


class Model(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        self.base = TinyGPT(self.payload["config"]).cpu().eval()
        self.base.load_state_dict(self.payload["model"])

    def test(self):
        config = {**self.payload["config"], "limit": 8, "steps": 4}
        model = Decoder(config).cpu().eval()
        transfer(model, self.payload)
        self.assertEqual(int(torch.count_nonzero(model.step)), 0)
        self.assertEqual(int(torch.count_nonzero(model.gate)), 0)
        ids = torch.tensor([[0, 32, 65, 66, 255, 7]], dtype=torch.long)
        with torch.inference_mode():
            expected = self.base(ids)
            actual = model(ids, steps=1)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertIs(model.head, model.token_embedding.weight)
        self.assertEqual(sum(isinstance(item, type(model.block)) for item in model.modules()), 1)
        self.assertTrue(all(value.device.type == "cpu" and value.dtype == torch.float32
                            for value in model.parameters()))
        count = model.params
        calls = []
        hook = model.block.register_forward_hook(lambda module, inputs, output: calls.append(1))
        for steps in (2, 4, 8):
            output = model(ids, steps=steps)
            self.assertEqual(len(calls), steps)
            calls.clear()
            self.assertTrue(bool(torch.isfinite(output).all()))
            loss = output.square().mean()
            model.zero_grad(set_to_none=True)
            loss.backward()
            self.assertTrue(all(value.grad is None or bool(torch.isfinite(value.grad).all())
                                for value in model.parameters()))
            self.assertIsNotNone(model.step.grad)
            self.assertIsNotNone(model.gate.grad)
            self.assertEqual(model.params, count)
        hook.remove()
        for steps in (1, 2, 4, 8):
            left = ids.clone()
            right = ids.clone()
            right[0, -1] = 1
            with torch.no_grad():
                a = model(left, steps=steps)
                b = model(right, steps=steps)
            torch.testing.assert_close(a[:, :-1], b[:, :-1], rtol=0.0, atol=0.0)
        one = model.generate(token_ids=[32, 65, 66], max_new_tokens=4, steps=4)
        self.assertEqual(one, model.generate(token_ids=[32, 65, 66], max_new_tokens=4, steps=4))
        self.assertEqual(len(one), 7)
        self.assertTrue(all(0 <= value < model.vocab_size for value in one))
        with self.assertRaises(ValueError):
            model(ids[:, :0])
        with self.assertRaises(ValueError):
            model(ids[0])
        with self.assertRaises(ValueError):
            model(torch.tensor([[-1]], dtype=torch.long))
        with self.assertRaises(ValueError):
            model(torch.zeros((1, model.context_length + 1), dtype=torch.long))
        with self.assertRaises(ValueError):
            model(ids.to(torch.int32))
        with self.assertRaises(ValueError):
            model(torch.tensor([[256]], dtype=torch.long))
        with self.assertRaises(ValueError):
            model(ids, steps=3)
        with self.assertRaises(ValueError):
            Decoder({**config, "limit": 4}).forward(ids, steps=8)
        with self.assertRaises(ValueError):
            Decoder({**config, "limit": 3})
        with self.assertRaises(ValueError):
            model.generate([], 1)
        with self.assertRaises(ValueError):
            transfer(model, {"config": config, "model": {}})


if __name__ == "__main__":
    unittest.main()
