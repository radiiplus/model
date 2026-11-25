from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from research.foundation.model import Foundation


SMALL_CONFIG = {
    "vocab_size": 64,
    "context_length": 32,
    "layers": 2,
    "d_model": 16,
    "n_heads": 4,
    "d_ff": 32,
    "position_embedding": "learned_absolute",
    "normalization": "pre_layernorm",
    "weight_tying": True,
    "dtype": "float32",
}


class FoundationKVCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7123)
        self.model = Foundation(SMALL_CONFIG).eval()
        self.tokens = torch.tensor([[3, 7, 4, 12, 19, 2, 11, 31, 9]], dtype=torch.long)

    def test_chunked_logits_match_full_causal_forward(self):
        with torch.inference_mode():
            expected = self.model(self.tokens)
            cache = self.model.new_kv_cache()
            first = self.model.forward_cached(self.tokens[:, :5], cache)
            second = self.model.forward_cached(self.tokens[:, 5:], cache)
        actual = torch.cat((first, second), dim=1)
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6))
        self.assertEqual(cache.length, self.tokens.shape[1])

    def test_incremental_greedy_tokens_match_full_recomputation(self):
        cache = self.model.new_kv_cache()
        context = self.tokens[:, :4]
        generated_cached, generated_full = [], []
        with torch.inference_mode():
            logits = self.model.forward_cached(context, cache)[:, -1]
            for _ in range(6):
                next_cached = logits.argmax(dim=-1, keepdim=True)
                generated_cached.append(int(next_cached))
                next_full = self.model(context)[:, -1].argmax(dim=-1, keepdim=True)
                generated_full.append(int(next_full))
                self.assertTrue(torch.equal(next_cached, next_full))
                context = torch.cat((context, next_cached), dim=1)
                logits = self.model.forward_cached(next_cached, cache)[:, -1]
        self.assertEqual(generated_cached, generated_full)

    def test_file_backed_cache_matches_ram_cache(self):
        with tempfile.TemporaryDirectory(prefix="foundation-kv-") as temp:
            ram = self.model.new_kv_cache()
            mapped = self.model.new_kv_cache(backing_directory=Path(temp))
            with torch.inference_mode():
                expected = self.model.forward_cached(self.tokens, ram)
                actual = self.model.forward_cached(self.tokens, mapped)
            self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6))
            self.assertEqual(mapped.backing, "mmap")
            self.assertEqual(mapped.bytes, ram.bytes)
            self.assertEqual(sum(path.stat().st_size for path in Path(temp).glob("*.bin")),
                             mapped.bytes)

    def test_file_backing_does_not_truncate_an_existing_cache(self):
        with tempfile.TemporaryDirectory(prefix="foundation-kv-occupied-") as temp:
            directory = Path(temp)
            self.model.new_kv_cache(backing_directory=directory)
            with self.assertRaises(FileExistsError):
                self.model.new_kv_cache(backing_directory=directory)

    def test_context_limit_is_enforced(self):
        cache = self.model.new_kv_cache()
        with torch.inference_mode():
            self.model.forward_cached(torch.zeros((1, 32), dtype=torch.long), cache)
            with self.assertRaises(ValueError):
                self.model.forward_cached(torch.zeros((1, 1), dtype=torch.long), cache)


if __name__ == "__main__":
    unittest.main()
