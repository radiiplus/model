from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research.interaction.compact import Compact, Learn
from research.interaction.correction import Corrections
from research.memory.store import Memory


class CorrectionCapture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.memory = Memory(Path(self.temp.name) / "memory.db", fts=False)
        self.scope, self.owner = "correction-test", "user-1"
        self.policy = {"persist": True, "scope": self.scope,
                       "owner": self.owner, "private": True}
        self.learn = Learn(Compact(self.memory))
        self.corrections = Corrections(self.memory, self.learn)

    def tearDown(self):
        self.memory.close()
        self.temp.cleanup()

    def test_expression_feedback_is_not_mislabeled_as_reasoning_failure(self):
        result = self.corrections.capture(
            "Can this work in cold weather?", "It is not suitable.",
            "The cell's power drops in the cold, so it will not run the lamp.",
            kind="expression", meaning_confirmed=True, scope=self.scope,
            owner=self.owner, policy=self.policy, model="foundation",
            version="checkpoint-1")
        episode = self.learn.fetch(result["episode"], self.scope, self.owner)
        self.assertEqual(episode["result"], {"status": "corrected", "kind": "expression",
                                             "meaning_confirmed": True, "origin": "user"})
        self.assertEqual(episode["answer"],
                         "The cell's power drops in the cold, so it will not run the lamp.")
        source = self.memory.sql.execute("SELECT text FROM chunks WHERE event=? ORDER BY part",
                                         (result["event"],)).fetchall()
        self.assertIn("It is not suitable.", "".join(row["text"] for row in source))
        self.assertEqual(self.memory.facts(scope=self.scope, owner=self.owner), [])

    def test_meaning_feedback_is_separate_and_explicit_private_policy_is_required(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.corrections.capture("q", "wrong", "right", kind="meaning",
                meaning_confirmed=True, scope=self.scope, owner=self.owner,
                policy=self.policy, model="foundation", version="v1")
        without_consent = {"persist": True, "scope": self.scope, "owner": self.owner}
        with self.assertRaises(PermissionError):
            self.corrections.capture("q", "wrong", "right", kind="meaning",
                meaning_confirmed=False, scope=self.scope, owner=self.owner,
                policy=without_consent, model="foundation", version="v1")
        result = self.corrections.capture("q", "wrong", "right", kind="meaning",
            meaning_confirmed=False, scope=self.scope, owner=self.owner,
            policy=self.policy, model="foundation", version="v1")
        episode = self.learn.fetch(result["episode"], self.scope, self.owner)
        self.assertFalse(episode["result"]["meaning_confirmed"])
        self.assertEqual(episode["input"], hashlib.sha256(b"q").hexdigest())

    def test_synthetic_correction_is_never_recorded_as_user_provided(self):
        result = self.corrections.capture("q", "unparseable", "No, not enough.",
            kind="meaning", meaning_confirmed=False, scope=self.scope,
            owner=self.owner, policy=self.policy, model="foundation", version="v1",
            origin="synthetic_fixture")
        source = self.memory.sql.execute(
            "SELECT source,licence,meta FROM events WHERE id=?", (result["event"],)
        ).fetchone()
        self.assertEqual(source["source"], "synthetic-correction-fixture")
        self.assertEqual(source["licence"], "generated-test-data")
        self.assertEqual(json.loads(source["meta"])["origin"], "synthetic_fixture")
        episode = self.learn.fetch(result["episode"], self.scope, self.owner)
        self.assertEqual(episode["result"]["origin"], "synthetic_fixture")


if __name__ == "__main__":
    unittest.main()
