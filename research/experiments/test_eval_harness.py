import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_harness import evaluate, load_cases, score_case


class ScorerTests(unittest.TestCase):
    def setUp(self):
        self.cases = {case["id"]: case for case in load_cases()}

    def test_correct_and_incorrect_exact_match(self):
        case = self.cases["arithmetic"]
        self.assertTrue(score_case(case, {"answer": "43"})["correct"])
        self.assertFalse(score_case(case, {"answer": "42"})["correct"])
        instruction = self.cases["instruction_following"]
        self.assertTrue(score_case(instruction, {"answer": "GREEN"})["correct"])
        self.assertFalse(score_case(instruction, {"answer": "green"})["correct"])

    def test_unsupported_abstention_and_claim_metrics(self):
        case = self.cases["unsupported_question"]
        row = score_case(case, {"answer": "", "abstained": True, "claims": []})
        self.assertTrue(row["abstained"])
        self.assertTrue(row["correct"])
        claimed = score_case(case, {"answer": "About 1 million", "claims": [{"text": "Population is 1 million", "evidence_ids": ["e3"]}]})
        self.assertFalse(claimed["correct"])
        self.assertEqual(claimed["unsupported_claim_count"], 1)
        false_abstention = score_case(self.cases["fact_city"], {"answer": "", "abstained": True})
        self.assertFalse(false_abstention["correct"])

    def test_evidence_requires_matching_claim_and_source(self):
        case = self.cases["fact_city"]
        good = score_case(case, {"answer": "Sample City", "claims": [{"text": "Exampleland's capital is Sample City.", "evidence_ids": ["e1"]}]})
        wrong_source = score_case(case, {"answer": "Sample City", "claims": [{"text": "Exampleland's capital is Sample City.", "evidence_ids": ["fabricated"]}]})
        self.assertEqual(good["evidence_claim_precision"], 1.0)
        self.assertEqual(wrong_source["evidence_claim_precision"], 0.0)
        self.assertEqual(wrong_source["evidence_claim_recall"], 0.0)
        duplicate = score_case(self.cases["fact_two_claims"], {"claims": [
            {"text": "Luma grows in shade.", "evidence_ids": ["e2"]},
            {"text": "Luma grows in shade.", "evidence_ids": ["e2"]},
        ]})
        self.assertEqual(duplicate["evidence_claim_recall"], 0.5)

    def test_code_is_run_against_unit_tests(self):
        case = self.cases["coding_add"]
        good = score_case(case, {"code": "def add(a, b):\n    return a + b"})
        bad = score_case(case, {"code": "def add(a, b):\n    return a - b"})
        self.assertTrue(good["correct"])
        self.assertFalse(bad["correct"])

    def test_oracle_checks_fixture_and_abstention_confusion(self):
        from eval_harness import FixtureOracle
        result = evaluate(load_cases(), FixtureOracle())
        summary = result["summary"]
        self.assertEqual(summary["task_accuracy"], 1.0)
        self.assertEqual(summary["abstention_precision"], 1.0)
        self.assertEqual(summary["abstention_recall"], 1.0)
        self.assertEqual(summary["evidence_supported_claim_precision"], 1.0)
        self.assertEqual(summary["evidence_supported_claim_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
