import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_tokenizers as bench


class TokenizerBenchmarkTests(unittest.TestCase):
    def test_fixture_has_balanced_domains_and_heldout_unicode(self):
        rows = bench.load_fixture()
        train, heldout = bench.fixture_split(rows)
        self.assertGreaterEqual(len({row["domain"] for row in rows}), 6)
        self.assertGreaterEqual(len(heldout), 5)
        self.assertTrue({"rare", "mixed_script", "equation_unicode"} <= {row["id"] for row in heldout})
        self.assertIn("🧪", {char for row in heldout for char in row["text"]})
        self.assertGreater(len(train), len(heldout))

    def test_byte_and_codepoint_baselines_roundtrip(self):
        rows = bench.load_fixture()
        train, heldout = bench.fixture_split(rows)
        for name, builder in [("byte_baseline", bench.build_byte_baseline), ("unicode_codepoint", bench.build_char_baseline)]:
            tokenizer, _ = builder([row["text"] for row in train])
            for row in heldout:
                ids, decoded = bench.encode_decode(name, tokenizer, row["text"])
                self.assertEqual(decoded, row["text"])
                if name == "byte_baseline":
                    self.assertEqual(len(ids), len(row["text"].encode("utf-8")))

    def test_trained_tokenizers_cover_and_roundtrip_unseen_content(self):
        result = bench.run(repeats=2)
        methods = result["methods"]
        self.assertEqual(result["heldout_documents"], 7)
        self.assertEqual(set(methods), {"byte_baseline", "unicode_codepoint", "hf_byte_bpe", "spm_bpe", "spm_unigram", "tiktoken_educational_bpe"})
        for method in methods.values():
            self.assertTrue(method["round_trip_exact"])
            self.assertEqual(method["coverage_percent"], 100.0)
            self.assertGreater(method["vocab_size"], 0)
        self.assertEqual(methods["byte_baseline"]["heldout_tokens"], methods["byte_baseline"]["heldout_utf8_bytes"])
        self.assertIsNotNone(methods["spm_unigram"]["byte_fallback_fraction"])


if __name__ == "__main__":
    unittest.main()
