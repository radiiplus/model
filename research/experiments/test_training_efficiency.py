import json
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_efficiency import accumulation_check, resume_check


CONFIG = json.loads((Path(__file__).resolve().parent / "baseline_config.json").read_text())


class TrainingEfficiencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    def test_accumulation_matches_effective_batch(self):
        result = accumulation_check(CONFIG, 19)
        self.assertTrue(result["equivalent_within_1e-6"], result)

    def test_adamw_serialized_resume_is_exact(self):
        result = resume_check(CONFIG, "adamw", 19, 6, 0.003, 0.02)
        self.assertTrue(result["loss_exact"], result)
        self.assertTrue(result["model_state_exact"], result)

    def test_muon_hybrid_serialized_resume_is_exact(self):
        if not hasattr(torch.optim, "Muon"):
            self.skipTest("PyTorch build has no Muon optimizer")
        result = resume_check(CONFIG, "muon_hybrid", 19, 6, 0.003, 0.02)
        self.assertTrue(result["loss_exact"], result)
        self.assertTrue(result["model_state_exact"], result)


if __name__ == "__main__":
    unittest.main()
