import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from research.experiments.tiny_gpt import TinyGPT
from research.recursive.dual import Refiner, transfer as dualtransfer
from research.recursive.evaluate import check, edges, score, solve
from research.recursive.model import Decoder, transfer
from research.recursive.tasks import CONTEXT, CROSS, build, crosses, manifest
from research.recursive.train import pack, run

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


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


class Dual(unittest.TestCase):
    def setUp(self):
        self.payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        self.config = json.loads((ROOT / "recursive" / "dual.json").read_text())
        self.base = Decoder({**self.payload["config"], "limit": 8, "steps": 4})
        transfer(self.base, self.payload)
        self.model = Refiner(self.config)
        dualtransfer(self.model, self.base)
        self.ids = torch.tensor([[0, 32, 65, 66, 255, 7]], dtype=torch.long)

    def test(self):
        self.assertEqual(self.model.params, 23264)
        self.assertLessEqual(self.model.params, self.config["budget"])
        self.assertIs(self.model.head, self.model.token.weight)
        self.assertEqual(sum(isinstance(item, type(self.model.block))
                             for item in self.model.modules()), 1)
        self.assertEqual(tuple(self.model.step.shape), (8, 32))
        self.assertEqual(tuple(self.model.gate.shape), (8, 4, 32))
        self.assertEqual(int(torch.count_nonzero(self.model.step)), 0)
        self.assertEqual(int(torch.count_nonzero(self.model.gate)), 0)

        with torch.inference_mode():
            want = self.base(self.ids, steps=1)
            got = self.model(self.ids, steps=1)
            again = self.model(self.ids, steps=1)
        torch.testing.assert_close(got, want, rtol=0.0, atol=0.0)
        torch.testing.assert_close(got, again, rtol=0.0, atol=0.0)

        states = []
        hook = self.model.block.register_forward_hook(
            lambda module, inputs, output: states.append(output)
        )
        logits, trace = self.model(self.ids, steps=2, trace=True, targets=self.ids)
        hook.remove()
        self.assertEqual(len(states), 4)
        self.assertEqual(states[0].shape, (1, 6, 32))
        self.assertEqual(states[1].shape, (1, 6, 32))
        self.assertIsNot(states[0], states[1])
        self.assertFalse(torch.equal(states[0], states[1]))
        self.assertIsNot(states[2], states[3])
        self.assertFalse(torch.equal(states[2], states[3]))
        self.assertEqual(trace["logits"].shape, (2, 1, 6, 256))
        self.assertEqual(trace["losses"].shape, (2,))
        self.assertEqual(trace["deltas"].shape, (2, 2))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertTrue(bool(torch.isfinite(trace["losses"]).all()))
        self.assertTrue(bool(torch.isfinite(trace["deltas"]).all()))
        torch.testing.assert_close(logits, trace["logits"][-1])
        torch.testing.assert_close(
            logits, torch.nn.functional.linear(self.model.norm(states[-1]), self.model.head)
        )
        loss = trace["losses"].sum()
        loss.backward()
        self.assertTrue(all(value.grad is None or bool(torch.isfinite(value.grad).all())
                            for value in self.model.parameters()))
        self.assertIsNotNone(self.model.gate.grad)
        self.assertIsNotNone(self.model.step.grad)

        for count in (1, 2, 4, 8):
            self.assertEqual(self.model.params, 23264)
            left = self.ids.clone()
            right = self.ids.clone()
            right[0, -1] = 1
            with torch.no_grad():
                a = self.model(left, steps=count)
                b = self.model(right, steps=count)
            self.assertTrue(bool(torch.isfinite(a).all()))
            torch.testing.assert_close(a[:, :-1], b[:, :-1], rtol=0.0, atol=0.0)
        one = self.model.generate(ids=[32, 65, 66], max_new_tokens=4, steps=4)
        self.assertEqual(one, self.model.generate(
            ids=[32, 65, 66], max_new_tokens=4, steps=4
        ))
        self.assertEqual(len(one), 7)
        with self.assertRaises(ValueError):
            Refiner({**self.config, "budget": 23263})
        with self.assertRaises(ValueError):
            self.model(self.ids, steps=3)
        with self.assertRaises(ValueError):
            self.model(self.ids.to(torch.int32))
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[256]], dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model(self.ids, targets=self.ids)
        with self.assertRaises(ValueError):
            self.model(self.ids[:, :0])
        with self.assertRaises(ValueError):
            self.model(self.ids[0])
        with self.assertRaises(ValueError):
            self.model(self.ids, trace=True, targets=self.ids[:, :-1])
        with self.assertRaises(ValueError):
            self.model(self.ids, trace="yes")
        with self.assertRaises(ValueError):
            dualtransfer(self.model, object())


def setting():
    return {
        "architecture": "dual-v1",
        "mode": "language",
        "family": "addition",
        "seed": 71,
        "steps": 2,
        "batch": 2,
        "rate": 0.0003,
        "decay": 0.01,
        "clip": 1.0,
        "loss": "deep",
        "weights": [0.4, 0.6],
        "wall": 30,
        "model": {
            "vocab_size": 256,
            "context_length": 8,
            "d_model": 8,
            "n_heads": 2,
            "d_ff": 16,
            "limit": 8,
            "steps": 2,
            "budget": 1000000,
        },
    }


class Train(unittest.TestCase):
    def test(self):
        docs = [list(range(40)), list(range(40, 80))]
        hold = [list(range(80, 120))]
        with tempfile.TemporaryDirectory() as folder:
            full = Path(folder) / "full.pt"
            part = Path(folder) / "part.pt"
            whole = run(setting(), 3, full, docs=docs, hold=hold)
            paused = run(setting(), 3, part, docs=docs, hold=hold, stop=1)
            self.assertTrue(paused["interrupted"])
            payload = torch.load(part, map_location="cpu", weights_only=True)
            for key in ("model", "optimizer", "torch", "python", "sampler", "cursor",
                        "signature", "history"):
                self.assertIn(key, payload)
            self.assertEqual(payload["signature"]["architecture"], "dual-v1")
            resumed = run(setting(), 3, part, docs=docs, hold=hold, resume=part)
            self.assertEqual(whole["state"], resumed["state"])
            self.assertEqual(whole["stages"], resumed["stages"])
            self.assertEqual(whole["validation"], resumed["validation"])
            self.assertEqual(len(resumed["stages"]), 3)
            self.assertEqual(len(resumed["stages"][0]), 2)
            self.assertEqual(resumed["blocks"], 1)
            self.assertEqual(resumed["calls"], 12)
            self.assertEqual(resumed["params"], Refiner(setting()["model"]).params)
            with self.assertRaises(ValueError):
                run({**setting(), "weights": [0.3, 0.7]}, 4, part, docs=docs, hold=hold,
                    resume=part)
            with self.assertRaises(ValueError):
                run({**setting(), "weights": [0.5, 0.6]}, 3, full, docs=docs, hold=hold)
            with self.assertRaises(ValueError):
                run(setting(), 3, full, docs=docs, hold=docs)


class Task(unittest.TestCase):
    def test(self):
        config = setting()
        config["mode"] = "task"
        config["model"]["context_length"] = 128
        row = build()["families"]["addition"]["train"][0]
        ids, targets, mask = pack([row], torch.tensor([0]), 128)
        self.assertEqual(int(mask.sum()), len(row["target"].encode("utf-8")))
        self.assertEqual(int(mask[0, :len(row["prompt"].encode("utf-8")) - 1].sum()), 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "task.pt"
            result = run(config, 1, path)
            self.assertEqual(result["loss"], "deep")
            self.assertEqual(len(result["stages"]), 1)
            self.assertEqual(len(result["stages"][0]), 2)
            self.assertGreater(result["validation"]["tokens"], 0)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["signature"]["generator"], 1)
        final = {**config, "loss": "final"}
        with tempfile.TemporaryDirectory() as folder:
            result = run(final, 1, Path(folder) / "final.pt")
            self.assertEqual(result["weights"], [0.0, 1.0])


class Loop(unittest.TestCase):
    def test(self):
        config = setting()
        config["architecture"] = "loop-v1"
        config["mode"] = "task"
        config["model"]["limit"] = 8
        config["model"]["context_length"] = 128
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "loop.pt"
            result = run(config, 1, path)
            self.assertEqual(result["calls"], 2)
            self.assertEqual(result["steps"], 2)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model = Decoder(config["model"])
            model.load_state_dict(payload["model"])
            ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
            logits, trace = model(ids, steps=8, trace=True, targets=ids)
            self.assertEqual(trace["logits"].shape, (8, 1, 3, 256))
            self.assertEqual(trace["losses"].shape, (8,))
            self.assertTrue(bool(torch.isfinite(logits).all()))


class Guard(unittest.TestCase):
    def test(self):
        with patch("research.recursive.train.available_mib", return_value=1):
            with self.assertRaises(MemoryError):
                run(setting(), 1, Path("unused.pt"), docs=[[1, 2, 3]], hold=[[4, 5, 6]])


if __name__ == "__main__":
    unittest.main()
