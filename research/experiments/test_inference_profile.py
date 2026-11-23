import io
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inference_profile import cache_bytes, decode_step, make_model, prefill, quantize_model
from tiny_gpt import load_checkpoint


CHECKPOINT = HERE / "results" / "baseline_tiny_gpt.pt"


class InferenceProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cls.model, _ = load_checkpoint(CHECKPOINT)
        cls.model.eval()

    def test_cached_prefill_and_decode_match_full_forward(self):
        prefix = torch.tensor([[1, 4, 7, 2, 5, 3, 1, 4]])
        full_logits = self.model(prefix)
        cached_logits, cache = prefill(self.model, prefix)
        self.assertTrue(torch.allclose(full_logits, cached_logits, atol=1e-6, rtol=1e-6))
        token = torch.tensor([[6]])
        cached_next, _ = decode_step(self.model, token, prefix.shape[1], cache)
        full_next = self.model(torch.cat((prefix, token), dim=1))[:, -1:]
        self.assertTrue(torch.allclose(cached_next, full_next, atol=2e-6, rtol=1e-6))

    def test_kv_bytes_scale_linearly_for_this_model(self):
        self.assertEqual(cache_bytes(1, 8, self.model), 2048)
        self.assertEqual(cache_bytes(1, 32, self.model), 8192)

    def test_dynamic_int8_roundtrip_preserves_greedy_ids_on_probe(self):
        quantized = quantize_model(make_model({"config": self.model.config, "state_dict": self.model.state_dict()}, False))
        payload = {"config": self.model.config, "state_dict": quantized.state_dict()}
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        reloaded_payload = torch.load(buffer, map_location="cpu", weights_only=True)
        restored = make_model(reloaded_payload, True)
        ids = torch.tensor([[1, 4, 7, 2, 5, 3, 1, 4]])
        with torch.inference_mode():
            self.assertTrue(torch.equal(quantized(ids).argmax(-1), restored(ids).argmax(-1)))

    def test_checkpoint_and_generation_shapes(self):
        ids = torch.tensor([[1, 4, 7]])
        generated = self.model.generate(ids, 4)
        self.assertEqual(list(generated.shape), [1, 7])
        self.assertTrue(bool(((generated >= 0) & (generated < self.model.vocab_size)).all()))


if __name__ == "__main__":
    unittest.main()
