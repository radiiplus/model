from __future__ import annotations

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from research.foundation import compare, trainer, fit, curriculum
from research.foundation.model import Foundation
from research.foundation import probe
from research.foundation.recur import Refiner, transfer


class Model(unittest.TestCase):
    def test(self):
        config = trainer.load()["model"]
        model = Foundation(config)
        self.assertEqual(model.params, 4470272)
        self.assertEqual(len(model.blocks), 4)
        self.assertIsNot(model.blocks[0], model.blocks[1])
        self.assertNotEqual(model.blocks[0].attn.qkv.weight.data_ptr(),
                            model.blocks[1].attn.qkv.weight.data_ptr())
        self.assertIs(model.head, model.token.weight)
        tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
        output = model(tokens)
        self.assertEqual(tuple(output.shape), (1, 3, 4096))
        self.assertTrue(torch.isfinite(output).all())
        altered = model(torch.tensor([[1, 2, 4]], dtype=torch.long))
        self.assertTrue(torch.equal(output[:, :2], altered[:, :2]))
        mask = torch.tensor([[False, True, False]], dtype=torch.bool)
        values = model.losses(tokens, mask)
        self.assertEqual(int(values["tokens"]), 2)
        self.assertEqual(int(values["answers"]), 1)
        for invalid in (torch.tensor([[-1]]), torch.tensor([[4096]])):
            with self.assertRaises(ValueError):
                model(invalid)


class Resume(unittest.TestCase):
    def test(self):
        config = trainer.load()
        config["train"].update({"steps": 3, "sequence": 16, "batch": 1, "accum": 1,
                                "warm": 0, "decay": 3, "eval": 32,
                                "interval": 20, "time": 100, "seed": 23})
        root = trainer.DATA
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            full = base / "full.pt"
            partial = base / "partial.pt"
            with patch.object(trainer.guard, "available_mib", return_value=1):
                with self.assertRaisesRegex(MemoryError, "2560 MiB"):
                    trainer.run(copy.deepcopy(config), root, base / "refused.pt")
            with patch.object(trainer.guard, "available_mib", return_value=4096), \
                    patch.object(trainer.guard, "snapshot", return_value={"available": 4096}), \
                    patch.object(trainer.guard, "swap_counters", return_value=None):
                trainer.run(copy.deepcopy(config), root, full)
                first = trainer.run(copy.deepcopy(config), root, partial, stop=1)
                self.assertTrue(first["interrupted"])
                resumed = trainer.run(copy.deepcopy(config), root, partial, resume=partial)
            one = torch.load(full, map_location="cpu", weights_only=True)
            two = torch.load(partial, map_location="cpu", weights_only=True)
            self.assertEqual(resumed["steps"], 3)
            self.assertEqual(one["sampler"], two["sampler"])
            self.assertTrue(same(one["scheduler"], two["scheduler"]))
            self.assertTrue(same(one["optimizer"], two["optimizer"]))
            for key in one["model"]:
                self.assertTrue(torch.equal(one["model"][key], two["model"][key]), key)
            wrong = copy.deepcopy(config)
            wrong["train"]["sft"] = 0.5
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                trainer.run(wrong, root, partial, resume=partial)


class Recur(unittest.TestCase):
    def test(self):
        config = trainer.load()["model"]
        torch.manual_seed(71)
        base = Foundation(config)
        loop = Refiner(config, 4)
        transfer(loop, base)
        tokens = torch.tensor([[4, 8, 12, 16, 20]], dtype=torch.long)
        self.assertTrue(torch.equal(base(tokens), loop(tokens, steps=1)))
        trace = loop(tokens, steps=2, trace=True)
        latent, answer = trace["states"][0]
        self.assertEqual(tuple(latent.shape), (1, 5, 256))
        self.assertEqual(tuple(answer.shape), (1, 5, 256))
        self.assertNotEqual(latent.data_ptr(), answer.data_ptr())
        self.assertEqual(loop.params, 4474368)
        self.assertEqual(sum(value.numel() for value in loop.blocks.parameters()), 3159040)
        self.assertEqual(len(loop.blocks), 4)
        self.assertEqual(len(trace["logits"]), 2)
        with self.assertRaises(ValueError):
            loop(tokens, steps=0)
        with self.assertRaises(ValueError):
            loop(tokens, steps=5)
        mask = torch.tensor([[False, True, True, False, False]], dtype=torch.bool)
        with self.assertRaises(ValueError):
            loop.losses(tokens, mask, 2, [0.25, 0.25])
        with self.assertRaises(ValueError):
            loop.losses(tokens, torch.zeros_like(mask), 2, [0.5])
        losses = loop.losses(tokens, mask, 2, [0.5, 0.5])
        sum(row["language"] + row["interaction"] for row in losses).backward()
        self.assertTrue(all(value.grad is None or torch.isfinite(value.grad).all()
                            for value in loop.parameters()))
        with torch.no_grad():
            loop.gate.normal_(0.0, 0.05)
        changed = torch.tensor([[4, 8, 12, 16, 21]], dtype=torch.long)
        left = loop(tokens, steps=2, trace=True)["logits"]
        right = loop(changed, steps=2, trace=True)["logits"]
        self.assertTrue(all(torch.equal(one[:, :4], two[:, :4])
                             for one, two in zip(left, right)))

    def testresume(self):
        config = trainer.load()
        settings = json.loads((trainer.HERE / "recur.json").read_text())
        settings.update({"updates": 2, "sequence": 8, "time": 100})
        deep = [0.5, 0.5]
        meta = trainer.pack(trainer.DATA, config)
        torch.manual_seed(83)
        base = Foundation(config["model"])
        loop = Refiner(config["model"], 4)
        transfer(loop, base)
        full = copy.deepcopy(loop)
        part = copy.deepcopy(loop)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            whole = root / "whole.pt"
            piece = root / "piece.pt"
            with patch.object(trainer.guard, "available_mib", return_value=4096), \
                    patch.object(trainer.guard, "swap_counters", return_value=None):
                compare.fit(full, "resume", 2, 2, deep, settings, meta,
                            time.monotonic() + 100, path=whole)
                first = compare.fit(part, "resume", 2, 2, deep, settings, meta,
                                    time.monotonic() + 100, path=piece, stop=1)
                self.assertTrue(first["interrupted"])
                compare.fit(part, "resume", 2, 2, deep, settings, meta,
                            time.monotonic() + 100, path=piece, resume=piece)
            one = torch.load(whole, map_location="cpu", weights_only=True)
            two = torch.load(piece, map_location="cpu", weights_only=True)
            self.assertEqual(one["step"], two["step"])
            self.assertTrue(same(one["sampler"], two["sampler"]))
            self.assertTrue(same(one["python"], two["python"]))
            self.assertTrue(torch.equal(one["torch"], two["torch"]))
            self.assertTrue(same(one["optimizer"], two["optimizer"]))
            for key in one["model"]:
                self.assertTrue(torch.equal(one["model"][key], two["model"][key]), key)


class Probe(unittest.TestCase):
    def test(self):
        first = probe.sample(trainer.DATA, 96, 2041, 1024)
        second = probe.sample(trainer.DATA, 96, 2041, 1024)
        third = probe.sample(trainer.DATA, 96, 2041, 1024, "test")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 96)
        self.assertEqual(len({row["doc"] for row in first}), 96)
        self.assertEqual(len(third), 96)
        self.assertFalse({row["doc"] for row in first} & {row["doc"] for row in third})
        for source in probe.SOURCES:
            self.assertEqual(sum(row["source"] == source for row in first), 32)


class Fitting(unittest.TestCase):
    def test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = copy.deepcopy(trainer.load())
            config["model"].update({"context_length": 1024, "layers": 1,
                                    "d_model": 16, "n_heads": 4, "d_ff": 32})
            model = Foundation(config["model"])
            base = root / "base.pt"
            trainer.save(base, {"config": config, "model": model.state_dict()})
            origin = fit.seal(base)
            data = {"status": "approved", "train": [{"episode": "train-one",
                                "messages": [{"role": "user", "text": "Remember this."}],
                                "target": "The answer is seven."}],
                    "validation": {"report": {"gate": "independent-transfer-v1",
                                                "passed": True, "safe": True}}}
            mark = __import__("hashlib").sha256(fit.encode(data)).hexdigest()
            source = root / "approved.json"
            source.write_text(fit.encode({"schema": 1, "hash": mark, **data}).decode(),
                              encoding="utf-8")
            shaped, rows = fit.rows(source)
            ids, mask = rows[0]
            prefix = len(shaped["token"].chat(data["train"][0]["messages"], bos=True,
                                               eos=False)) + 1
            self.assertTrue(any(mask))
            self.assertFalse(any(mask[:prefix]))
            self.assertTrue(all(mask[prefix:]))
            full = root / "whole.pt"
            part = root / "part.pt"
            with patch.object(fit.guard, "available_mib", return_value=4096), \
                    patch.object(fit.guard, "swap_counters", return_value=None):
                fit.run(base, source, full, steps=2, seed=91, wall=100)
                first = fit.run(base, source, part, steps=2, seed=91, wall=100, stop=1)
                self.assertTrue(first["interrupted"])
                resumed = fit.run(base, source, part, steps=2, seed=91, wall=100,
                                  resume=part)
            self.assertEqual(resumed["step"], 2)
            self.assertEqual(fit.seal(base), origin)
            one = torch.load(full, map_location="cpu", weights_only=True)
            two = torch.load(part, map_location="cpu", weights_only=True)
            self.assertTrue(same(one["optimizer"], two["optimizer"]))
            self.assertTrue(torch.equal(one["torch"], two["torch"]))
            self.assertEqual(one["cursor"], two["cursor"])
            for key in one["model"]:
                self.assertTrue(torch.equal(one["model"][key], two["model"][key]), key)
            with patch.object(fit.guard, "available_mib", return_value=1):
                with self.assertRaisesRegex(MemoryError, "2560 MiB"):
                    fit.run(base, source, root / "refused.pt", steps=1)


class Curriculum(unittest.TestCase):
    def test(self):
        one = curriculum.plan("a", 250)
        two = curriculum.plan("b", 250)
        self.assertEqual(one["config"]["train"]["steps"], 2)
        self.assertEqual(one["config"]["train"]["lang"], 1.0)
        self.assertEqual(one["config"]["train"]["sft"], 0.0)
        self.assertEqual(two["config"]["train"]["lang"], 0.0)
        self.assertEqual(two["config"]["train"]["sft"], 1.0)
        with self.assertRaisesRegex(ValueError, "reviewed complete-episode"):
            curriculum.plan("c", 250)
        config = trainer.load()
        config["train"].update({"steps": 3, "sequence": 8, "batch": 1,
                                "accum": 1, "warm": 0, "decay": 3,
                                "eval": 32, "interval": 20, "time": 100,
                                "seed": 33})
        meta = trainer.pack(trainer.DATA, config)
        torch.manual_seed(81)
        model = Foundation(config["model"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initial = root / "initial.pt"
            trainer.save(initial, {"config": config, "model": model.state_dict(),
                                   "meta": {key: meta[key] for key in
                                            ("dataset", "split", "tokenizer")}})
            mark = trainer.file(initial)
            full = root / "full.pt"
            part = root / "part.pt"
            with patch.object(trainer.guard, "available_mib", return_value=4096), \
                    patch.object(trainer.guard, "swap_counters", return_value=None):
                trainer.run(copy.deepcopy(config), output=full, init=initial)
                first = trainer.run(copy.deepcopy(config), output=part,
                                    init=initial, stop=1)
                self.assertTrue(first["interrupted"])
                resumed = trainer.run(copy.deepcopy(config), output=part,
                                      resume=part, init=initial)
            self.assertEqual(resumed["tokens"], 16)
            self.assertEqual(trainer.file(initial), mark)
            one = torch.load(full, map_location="cpu", weights_only=True)
            two = torch.load(part, map_location="cpu", weights_only=True)
            self.assertEqual(one["meta"]["initial"], mark)
            self.assertTrue(same(one["optimizer"], two["optimizer"]))
            for key in one["model"]:
                self.assertTrue(torch.equal(one["model"][key], two["model"][key]), key)


def same(one, two):
    if isinstance(one, torch.Tensor):
        return isinstance(two, torch.Tensor) and torch.equal(one, two)
    if isinstance(one, dict):
        return isinstance(two, dict) and one.keys() == two.keys() and all(
            same(one[key], two[key]) for key in one)
    if isinstance(one, (tuple, list)):
        return isinstance(two, type(one)) and len(one) == len(two) and all(
            same(left, right) for left, right in zip(one, two))
    return one == two


if __name__ == "__main__":
    unittest.main()
