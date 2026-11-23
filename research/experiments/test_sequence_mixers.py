import json
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sequence_mixers import RetentionGPT
from sequence_mixer_benchmark import make_copy_batch, make_parity_batch
from tiny_gpt import TinyGPT


class SequenceMixerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        config_path = Path(__file__).resolve().parent / "baseline_config.json"
        cls.config = json.loads(config_path.read_text(encoding="utf-8"))
        cls.config["context_length"] = 128

    def test_parameter_budget_is_matched(self):
        torch.manual_seed(9)
        transformer = TinyGPT(self.config)
        retention = RetentionGPT(self.config)
        count = lambda model: sum(param.numel() for param in model.parameters())
        self.assertEqual(count(transformer), count(retention))

    def test_recurrent_steps_match_full_sequence_forward(self):
        torch.manual_seed(10)
        model = RetentionGPT(self.config).eval()
        ids = torch.tensor([[10, 3, 0, 0, 11, 4]], dtype=torch.long)
        with torch.no_grad():
            full, final_state = model.forward_with_state(ids)
            state = model.initial_state(1, ids.device, model.token_embedding.weight.dtype)
            outputs = []
            for position in range(ids.shape[1]):
                logits, state = model.step(ids[:, position], position, state)
                outputs.append(logits)
            streamed = torch.stack(outputs, dim=1)
        torch.testing.assert_close(full, streamed, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(final_state[0].shape), (1, 4, 8, 8))
        self.assertEqual(final_state[0].numel() * 4 + final_state[1].numel() * 4, 1040)

    def test_toy_task_encodings(self):
        import random
        parity, parity_targets = make_parity_batch(4, 32, random.Random(1))
        self.assertEqual(tuple(parity.shape), (4, 32))
        self.assertTrue(bool(torch.all((parity_targets == 8) | (parity_targets == 9))))
        copy, targets = make_copy_batch(4, 32, random.Random(2))
        self.assertEqual(tuple(copy.shape), (4, 32))
        self.assertTrue(bool(torch.all(copy[:, 0] == 10)))
        self.assertTrue(bool(torch.all(copy[:, -1] == 11)))
        self.assertTrue(bool(torch.all((targets >= 1) & (targets <= 7))))


if __name__ == "__main__":
    unittest.main()
