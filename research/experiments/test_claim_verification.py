import json
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim_verification import classify, method_metrics, risk_coverage


FIXTURE = Path(__file__).parent / "fixtures" / "claim_verification_tiny.json"


class ClaimVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.cases = cls.fixture["cases"]
        cls.as_of = date.fromisoformat(cls.fixture["as_of"])
        cls.by_id = {case["id"]: case for case in cls.cases}

    def test_fixture_expected_statuses_match_structured_verifier(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertEqual(classify(case, self.as_of)[0], case["expected"])

    def test_irrelevant_and_injected_sources_do_not_support_claim(self):
        for case_id in ["unsupported_irrelevant_retrieval", "contradicted_malicious_wording"]:
            with self.subTest(case=case_id):
                self.assertNotEqual(classify(self.by_id[case_id], self.as_of)[0], "supported")

    def test_stale_fact_and_current_conflict_are_distinguished(self):
        self.assertEqual(classify(self.by_id["stale_only"], self.as_of)[0], "stale")
        self.assertEqual(classify(self.by_id["stale_and_current"], self.as_of)[0], "contradicted")

    def test_deterministic_verifier_reduces_assertions_and_has_no_false_rejection_here(self):
        result = method_metrics(self.cases, "deterministic_verifier", self.as_of)
        self.assertEqual(result["emitted"], 2)
        self.assertEqual(result["claim_precision"], 1.0)
        self.assertEqual(result["supported_claim_recall"], 1.0)
        self.assertEqual(result["false_rejections"], 0)
        self.assertEqual(result["correct_abstentions"], 8)

    def test_retrieval_only_accepts_irrelevant_claim_and_echo_does_not_help(self):
        retrieval = method_metrics(self.cases, "retrieval_only", self.as_of)
        echo = method_metrics(self.cases, "echo_review", self.as_of)
        self.assertEqual(retrieval["claim_precision"], echo["claim_precision"])
        self.assertEqual(retrieval["unsupported_claim_rate"], echo["unsupported_claim_rate"])
        self.assertEqual(retrieval["unsupported_assertions"], 7)

    def test_risk_coverage_curve_is_monotone_in_coverage(self):
        curve = risk_coverage(self.cases, self.as_of)
        coverages = [point["coverage"] for point in curve]
        self.assertEqual(coverages, sorted(coverages, reverse=True))
        self.assertEqual(curve[0]["coverage"], 1.0)
        self.assertEqual(curve[-1]["coverage"], 0.0)


if __name__ == "__main__":
    unittest.main()
