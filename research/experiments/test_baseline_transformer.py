import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_transformer import DEFAULT_CONFIG, TinyDecoder


class BaselineDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        cls.model = TinyDecoder(cls.config)

    def test_deterministic_initialization_and_shapes(self):
        other = TinyDecoder(self.config)
        self.assertEqual(self.model.state, other.state)
        logits = self.model.forward([1, 4, 7, 2])
        self.assertEqual(len(logits), 4)
        self.assertTrue(all(len(row) == self.config["vocab_size"] for row in logits))
        self.assertGreater(self.model.parameter_count(), 1000)

    def test_causality_and_next_token_loss(self):
        before = self.model.forward([1, 4, 7, 2])
        after_future_change = self.model.forward([1, 4, 7, 3])
        self.assertEqual(before[:3], after_future_change[:3])
        loss = self.model.next_token_loss([1, 4, 7, 2, 5])
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 10.0)

    def test_generation_ids_and_checkpoint_roundtrip(self):
        sequence = [1, 4, 7, 2, 5]
        generated = self.model.generate(sequence[:2], max_new_tokens=5)
        self.assertEqual(len(generated), 7)
        self.assertTrue(all(0 <= token < self.config["vocab_size"] for token in generated))
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "checkpoint.json"
            self.model.save(checkpoint)
            restored = TinyDecoder.load(checkpoint)
        self.assertEqual(self.model.state, restored.state)
        self.assertEqual(self.model.next_token_loss(sequence), restored.next_token_loss(sequence))


if __name__ == "__main__":
    unittest.main()
