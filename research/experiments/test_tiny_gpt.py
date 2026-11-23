import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_gpt import TinyGPT, load_checkpoint

CONFIG_PATH = Path(__file__).resolve().parent / "baseline_config.json"


class TinyGPTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.model = TinyGPT(cls.config)

    def test_forward_shape_and_parameter_budget(self):
        ids = torch.tensor([[1, 4, 7, 2]], dtype=torch.long)
        logits = self.model(ids)
        self.assertEqual(tuple(logits.shape), (1, 4, self.config["vocab_size"]))
        self.assertGreater(sum(p.numel() for p in self.model.parameters()), 10000)
        self.assertLess(sum(p.numel() for p in self.model.parameters()), 30000)

    def test_causal_mask_and_next_token_loss(self):
        left = torch.tensor([[1, 4, 7, 2]], dtype=torch.long)
        right = torch.tensor([[1, 4, 7, 3]], dtype=torch.long)
        with torch.no_grad():
            a = self.model(left)
            b = self.model(right)
        torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0.0, atol=0.0)
        loss = self.model.loss(torch.tensor([[1, 4, 7, 2, 5]], dtype=torch.long))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.detach()), 0.0)

    def test_tiny_sequence_overfit(self):
        torch.manual_seed(20261124)
        model = TinyGPT(self.config)
        ids = torch.tensor([[1, 4, 7, 2, 5, 3] * 2], dtype=torch.long)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
        with torch.no_grad():
            initial = float(model.loss(ids))
        for _ in range(150):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(ids)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final = float(model.loss(ids))
        self.assertLess(final, initial * 0.1)

    def test_generation_and_checkpoint_roundtrip(self):
        ids = torch.tensor([[1, 4]], dtype=torch.long)
        generated = self.model.generate(ids, max_new_tokens=5)
        self.assertEqual(tuple(generated.shape), (1, 7))
        self.assertTrue(bool(torch.all((generated >= 0) & (generated < self.model.vocab_size))))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            torch.save({"config": self.config, "model": self.model.state_dict()}, path)
            restored, _ = load_checkpoint(path)
        with torch.no_grad():
            torch.testing.assert_close(self.model(ids), restored(ids), rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
