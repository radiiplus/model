"""CPU checks for isolated strategy adapters."""

import tempfile
import unittest
from pathlib import Path

import torch

from research.adaptive.adapter import Adapter, CONTEXT, fit, mark, stamp
from research.prototype.trainer import configure_torch
from research.recursive.model import Decoder


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_torch()

    def model(self):
        torch.manual_seed(31)
        model = Decoder({"vocab_size": 256, "context_length": 48,
                         "d_model": 8, "n_heads": 2, "d_ff": 16,
                         "limit": 2, "steps": 2})
        for value in model.parameters():
            value.requires_grad_(False)
        return model

    def rows(self):
        return [{"id": "chain-train-0", "family": "chain", "split": "train",
                 "prompt": "Q: ", "target": "5"}]


class Mask(Base):
    def test(self):
        ids, targets, mask = mark(self.rows())[0]
        self.assertEqual(ids.shape, targets.shape)
        self.assertEqual(int(mask.sum()), 2)
        self.assertEqual(targets[0, mask[0]].tolist(), [ord("5"), 10])
        row = dict(self.rows()[0], prompt="x" * CONTEXT)
        with self.assertRaises(ValueError):
            mark([row])


class Rank(Base):
    def test(self):
        with self.assertRaises(ValueError):
            Adapter(width=8, rank=9)


class Switch(Base):
    def test(self):
        model = self.model()
        before = stamp(model.state_dict())
        adapter = Adapter(width=8, rank=2, seed=7)
        self.assertEqual(adapter.params, 32)
        token = torch.tensor([[81, 58, 32]], dtype=torch.long)
        with torch.no_grad():
            plain = model(token, steps=2)
            adapter.up.weight.fill_(0.2)
            adapter.mount(model)
            adapter.active = True
            changed = model(token, steps=2)
            adapter.active = False
            disabled = model(token, steps=2)
            adapter.unmount()
        self.assertFalse(torch.equal(plain, changed))
        self.assertTrue(torch.equal(plain, disabled))
        self.assertEqual(before, stamp(model.state_dict()))


class Resume(Base):
    def test(self):
        model = self.model()
        sig = {"strategy": "toy", "version": 1}
        with tempfile.TemporaryDirectory() as folder:
            fullpath = Path(folder) / "full.pt"
            splitpath = Path(folder) / "split.pt"
            torch.manual_seed(17)
            full = Adapter(width=8, rank=2, seed=17)
            result = fit(model, full, self.rows(), sig, 17, 4, fullpath)
            torch.manual_seed(17)
            partial = Adapter(width=8, rank=2, seed=17)
            first = fit(model, partial, self.rows(), sig, 17, 4,
                        splitpath, stop=2)
            self.assertTrue(first["interrupted"])
            torch.manual_seed(17)
            resumed = Adapter(width=8, rank=2, seed=17)
            final = fit(model, resumed, self.rows(), sig, 17, 4,
                        Path(folder) / "resumed.pt", resume=splitpath)
            self.assertEqual(result["state"], final["state"])
            self.assertEqual(result["history"], final["history"])
            self.assertEqual(final["step"], 4)


if __name__ == "__main__":
    unittest.main()
