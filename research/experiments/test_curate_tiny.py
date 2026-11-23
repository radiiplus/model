import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curate_tiny as curate


class CurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = curate.load_records()

    def test_exact_duplicate_and_whitespace_case_normalization(self):
        groups = curate.exact_groups(self.records)
        members = [set(self.records[i]["id"] for i in group) for group in groups]
        self.assertIn({"d01", "d02"}, members)
        self.assertIn({"d09", "d10"}, members)

    def test_near_duplicate_methods_identify_edit(self):
        min_groups, _ = curate.minhash_groups(self.records)
        sim_groups, _ = curate.simhash_groups(self.records)
        min_ids = [set(self.records[i]["id"] for i in group) for group in min_groups]
        sim_ids = [set(self.records[i]["id"] for i in group) for group in sim_groups]
        self.assertTrue(any({"d01", "d03"} <= group for group in min_ids))
        self.assertTrue(any({"d01", "d03"} <= group for group in sim_ids))

    def test_boilerplate_strip_preserves_distinct_document_bodies(self):
        processed = [curate.strip_boilerplate(row["text"]) for row in self.records if row["id"] in {"d05", "d06", "d07"}]
        self.assertEqual(len(set(processed)), 3)
        self.assertTrue(all("Distribution copy" not in text for text in processed))

    def test_quality_filter_flags_noise_but_retains_rare_code_and_math(self):
        kept, removed = curate.quality_filter(self.records)
        removed_ids = {row["id"] for row in removed}
        kept_ids = {row["id"] for row in kept}
        self.assertEqual(removed_ids, {"d08"})
        self.assertTrue({"d04", "d09", "d11", "d15", "d16", "d20"} <= kept_ids)

    def test_duplicate_components_stay_in_one_split(self):
        groups, _ = curate.minhash_groups(self.records)
        split = curate.split_by_duplicate_component(self.records, groups)
        membership = {item: name for name, ids in split.items() for item in ids}
        self.assertEqual(membership["d01"], membership["d02"])
        self.assertEqual(membership["d01"], membership["d03"])

    def test_threshold_sweep_exposes_false_positive_examples(self):
        sweep = curate.threshold_sweep(self.records)
        self.assertIn(["d01", "d04"], sweep["minhash_jaccard_estimate"]["0.2"]["false_positive_pairs"])
        self.assertIn(["d05", "d06"], sweep["minhash_jaccard_estimate"]["0.1"]["false_positive_pairs"])
        self.assertGreater(sweep["simhash_max_hamming_distance"]["25"]["duplicate_records_removed"], 3)

    def test_curated_manifest_preserves_provenance_and_source_ids(self):
        result, _ = curate.choose_representatives(self.records, curate.minhash_groups(self.records)[0])
        selected = {row["id"]: row for row in result}
        self.assertEqual(selected["d01"]["source"], "synthetic-b2")
        self.assertEqual(selected["d01"]["license"], "CC0-1.0")
        self.assertEqual(selected["d01"]["duplicate_component"], ["d01", "d02", "d03"])


if __name__ == "__main__":
    unittest.main()
