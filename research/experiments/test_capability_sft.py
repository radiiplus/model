import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capability_sft import (ABSTAIN, ANSWER_BASE, EOS, collate, exact_verifier,
                            make_record, make_splits)


class CapabilitySFTTests(unittest.TestCase):
    def test_prompt_mask_excludes_user_and_role_tokens(self):
        concise = make_record(2, 3, "concise", "test")
        tokens, mask = collate([concise])
        self.assertEqual(int(mask.sum()), 2)
        self.assertFalse(bool(mask[0, :5].any()))
        self.assertEqual(tokens[0, 6].item(), ANSWER_BASE + 5)
        self.assertEqual(tokens[0, 7].item(), EOS)
        worked_tokens, worked_mask = collate([make_record(2, 3, "worked", "test")])
        self.assertEqual(int(worked_mask.sum()), 7)
        self.assertEqual(worked_tokens.shape, (1, 13))

    def test_exact_verifier_accepts_correct_rejects_wrong_and_requires_abstention(self):
        case = make_record(8, 7, "concise", "test")
        self.assertTrue(exact_verifier(case, [ANSWER_BASE + 15, EOS])["correct"])
        self.assertFalse(exact_verifier(case, [ANSWER_BASE + 14, EOS])["correct"])
        missing = make_record(None, 7, "concise", "test")
        self.assertTrue(exact_verifier(missing, [ABSTAIN, EOS])["correct"])
        self.assertFalse(exact_verifier(missing, [ANSWER_BASE + 7, EOS])["correct"])

    def test_disjoint_pair_split_and_missing_orientation(self):
        train, heldout, missing = make_splits("concise")
        train_pairs = {(r["x"], r["y"]) for r in train if r["answerable"]}
        heldout_pairs = {(r["x"], r["y"]) for r in heldout}
        self.assertEqual((len(train_pairs), len(heldout_pairs)), (80, 20))
        self.assertFalse(train_pairs & heldout_pairs)
        self.assertEqual({x + y for x, y in train_pairs}, set(range(19)))
        self.assertEqual({x + y for x, y in heldout_pairs} - {x + y for x, y in train_pairs}, set())
        self.assertTrue(all(r["y"] is None for r in train if not r["answerable"]))
        self.assertTrue(all(r["x"] is None for r in missing))


if __name__ == "__main__":
    unittest.main()
