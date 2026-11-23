import hashlib
import tempfile
import unittest
from pathlib import Path

from research.adaptive.loop import Archive, collect, digest, propose, validate
from research.recursive.tasks import build, manifest


def train(archive):
    data = build()
    info = manifest(data)
    rows = collect(archive, data, info)
    return data, rows


class Extract(unittest.TestCase):
    def test(self):
        with Archive() as archive:
            _, rows = train(archive)
            item = propose(rows)
            self.assertEqual(item["procedure"], "stepwise")
            self.assertEqual(item["train"], ["chain"])
            self.assertEqual(len(item["episodes"]), 12)
            self.assertTrue(all("answer" in row and "actions" in row for row in rows))
            self.assertTrue(all(row["evidence"] == [] for row in rows))
            task = dict(build()["families"]["chain"]["train"][0])
            task["id"] = "chain-train-negative"
            task["prompt"] += " negative fixture"
            archive.add(task, "wrong", {"correct": False, "error": "arithmetic"},
                        [], {"iterations": 1, "retrievals": 0})
            self.assertEqual(sum(not row["result"]["correct"]
                                 for row in archive.episodes("chain")), 1)


class Duplicates(unittest.TestCase):
    def test(self):
        with Archive() as archive:
            _, rows = train(archive)
            few = rows[:2]
            copies = []
            for index, row in enumerate(few):
                copy = dict(row)
                copy["id"] = digest([row["id"], index])
                copy["instance"] = f"duplicate-{index}"
                copies.append(copy)
            self.assertIsNone(propose(few + copies))
            held = dict(rows[2])
            held["split"] = "test"
            self.assertIsNone(propose(few + [held]))


class Gate(unittest.TestCase):
    def test(self):
        with Archive() as archive:
            _, rows = train(archive)
            item = propose(rows)
            archive.save(item)
            with self.assertRaises(ValueError):
                archive.change(item["id"], "validated", {"train_accuracy": 1.0})
            self.assertEqual(archive.latest(item["id"])["status"], "candidate")
            self.assertEqual(archive.select("trace"), [])


class Transfer(unittest.TestCase):
    def test(self):
        with Archive() as archive:
            data, rows = train(archive)
            item = propose(rows)
            archive.save(item)
            before = hashlib.sha256(Path("research/experiments/results/baseline_tiny_gpt.pt").read_bytes()).hexdigest()
            report = validate(archive, item, data, rows)
            after = hashlib.sha256(Path("research/experiments/results/baseline_tiny_gpt.pt").read_bytes()).hexdigest()
            self.assertTrue(report["passed"])
            self.assertEqual(before, after)
            self.assertEqual(report["trainfamilies"], ["chain"])
            self.assertEqual(report["validationfamilies"], ["trace", "bracket"])
            self.assertEqual(archive.latest(item["id"])["status"], "validated")
            self.assertEqual({row["correct"] for fam in report["tracks"].values()
                              for split in fam.values()
                              for row in split["strategy"]["sampleset"]}, {True})


class Memorize(unittest.TestCase):
    def test(self):
        with Archive() as archive:
            data, rows = train(archive)
            item = propose(rows)
            item["id"] = digest(["memorize", item["episodes"]])
            item["procedure"] = "memorize"
            item["principle"] = "Copy answers only for exact repeated prompts."
            item["benefit"] = "Repeat recall, with no expected transfer."
            archive.save(item)
            report = validate(archive, item, data, rows)
            self.assertFalse(report["passed"])
            self.assertEqual(archive.latest(item["id"])["status"], "rejected")
            self.assertTrue(archive.latest(item["id"])["counterexamples"])
            self.assertTrue(all(row["strategy"]["correct"] == 0
                                for fam in report["tracks"].values()
                                for row in fam.values()))


class Lifecycle(unittest.TestCase):
    def test(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "archive.db"
            with Archive(path) as archive:
                data, rows = train(archive)
                item = propose(rows)
                archive.save(item)
                report = validate(archive, item, data, rows)
                key = item["id"]
                self.assertTrue(report["passed"])
                archive.deprecate(key)
                self.assertEqual(archive.select("trace"), [])
            with Archive(path) as archive:
                self.assertEqual(len(archive.episodes("chain")), 12)
                restored = archive.rollback(key, 2)
                self.assertEqual(restored["status"], "validated")
                self.assertEqual(len(archive.select("trace")), 1)
                self.assertEqual(len(archive.versions(key)), 4)


if __name__ == "__main__":
    unittest.main()
