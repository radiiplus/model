from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from research.interaction import corpus, shard
from research.interaction.tokenizer import SPECIAL, Tokenizer
from research.interaction.compact import Compact, Learn
from research.memory.store import Memory


class Round(unittest.TestCase):
    def test(self):
        texts = ["Hi, punctuation?!", "cafe\u0301 café", "print('x = 123')\n", "東京 🧪", "नमस्ते"]
        token = Tokenizer.train(texts, target=300, budget=4096)
        for text in texts:
            self.assertEqual(token.decode(token.encode(text)), text)
        self.assertNotEqual(token.encode("café"), token.encode("cafe\u0301"))
        rows = [{"role": "system", "text": "Be concise."},
                {"role": "user", "text": "Run tool: echo 1"},
                {"role": "tool", "text": "1"},
                {"role": "assistant", "text": "Done."}]
        ids = token.chat(rows)
        self.assertEqual(ids[0], SPECIAL["BOS"])
        self.assertEqual(ids[-1], SPECIAL["EOS"])
        self.assertEqual(token.parse(ids), rows)
        with self.assertRaises(ValueError):
            token.decode([SPECIAL["USER"]])

class Pin(unittest.TestCase):
    def test(self):
        texts = ["repeatable words and symbols", "repeatable words again"]
        one = Tokenizer.train(texts, target=320, budget=4096)
        two = Tokenizer.train(texts, target=320, budget=4096)
        self.assertEqual(one.digest, two.digest)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tokenizer.json"
            one.save(path)
            self.assertEqual(Tokenizer.load(path).digest, one.digest)
            with self.assertRaises(FileExistsError):
                one.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["seen"] += 1
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                Tokenizer.load(path)


class Array(unittest.TestCase):
    def test(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "array.json"
            path.write_text('[ {"id":1},\n {"id":2} ]', encoding="utf-8")
            self.assertEqual([row["id"] for row in corpus.array(path)], [1, 2])
            path.write_text('[ {"id":1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "truncated"):
                list(corpus.array(path))

class Privacy(unittest.TestCase):
    def test(self):
        self.assertEqual(corpus.split("family-a"), corpus.split("family-a"))
        self.assertIn(corpus.split("family-a"), {"train", "validation", "test"})
        self.assertTrue(corpus.sensitive("Contact alex@example.org"))
        self.assertTrue(corpus.sensitive("Call +1 (415) 555-0100"))
        self.assertTrue(corpus.sensitive("Send it to 14 Main Street"))
        self.assertTrue(corpus.sensitive("The record contains 123-45-6789"))
        self.assertFalse(corpus.sensitive("The room has 12 chairs."))

class Near(unittest.TestCase):
    def test(self):
        import sqlite3
        db = sqlite3.connect(":memory:")
        db.execute("create table near(band integer,val integer,sig integer,doc text,family text)")
        db.execute("create table neartext(doc text primary key,text text)")
        one = "we should carefully check every open item before marking this shared project complete today and record each result in the shared log"
        two = "we should carefully check every open item before marking this shared project complete now and record each result in the shared log"
        ids, links = corpus._near(db, corpus._sim(one), "one", "family1", one)
        self.assertFalse(ids)
        self.assertFalse(links)
        ids, links = corpus._near(db, corpus._sim(two), "two", "family2", two)
        self.assertEqual(ids, {"one"})
        self.assertEqual(links, {("family1", "family2")})
        db.close()

class Split(unittest.TestCase):
    def test(self):
        families = corpus.Counter({"a0": 8, "a1": 4, "a2": 3, "a3": 2, "b0": 9,
                                   "b1": 3, "b2": 2, "c0": 4, "c1": 1, "c2": 1})
        sources = {"a0": corpus.Counter({"a": 8}), "a1": corpus.Counter({"a": 4}),
                   "a2": corpus.Counter({"a": 3}), "a3": corpus.Counter({"a": 2}),
                   "b0": corpus.Counter({"b": 9}),
                   "b1": corpus.Counter({"b": 3}), "b2": corpus.Counter({"b": 2}),
                   "c0": corpus.Counter({"c": 4}), "c1": corpus.Counter({"c": 1}),
                   "c2": corpus.Counter({"c": 1})}
        assigns = corpus._assign(families, sources, {("a1", "a2")})
        for source in ("a", "b", "c"):
            self.assertEqual({assigns[key] for key in sources if source in sources[key]},
                             {"train", "validation", "test"})
        self.assertEqual(assigns["a1"], assigns["a2"])

class Build(unittest.TestCase):
    def test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tree = root / "tree.gz"
            feed = root / "self.json"
            woz = root / "woz.json"
            seed = root / "seed.jsonl"
            pattern = root / "patterns.jsonl"
            out = root / "corpus.jsonl"
            message = {"message_id": "m0", "role": "prompter", "text": "What is the next step?",
                       "lang": "en", "review_result": True, "deleted": False,
                       "synthetic": False, "replies": []}
            answer = {"message_id": "m1", "parent_id": "m0", "role": "assistant",
                      "text": "Check the open item.", "lang": "en", "review_result": True,
                      "deleted": False, "synthetic": False, "rank": 0,
                      "labels": {"quality": {"value": 0.9}}, "replies": []}
            message["replies"] = [answer]
            with gzip.open(tree, "wt", encoding="utf-8") as file:
                file.write(json.dumps({"message_tree_id": "tree0", "prompt": message}) + "\n")
            dialogue = {"conversation_id": "one", "instruction_id": "booking-1",
                        "utterances": [{"speaker": "USER", "text": "Book a table."},
                                       {"speaker": "ASSISTANT", "text": "What day?"}]}
            duplicate = dict(dialogue, conversation_id="two")
            feed.write_text(json.dumps([dialogue, duplicate]), encoding="utf-8")
            woz.write_text("[]", encoding="utf-8")
            seed.write_text(json.dumps({"id": "seed0", "family": "objective",
                                        "messages": [{"role": "user", "text": "Keep the goal."},
                                                     {"role": "assistant", "text": "I will track it."}]}) + "\n",
                            encoding="utf-8")
            pattern.write_text(json.dumps({"id": "plan", "family": "plan",
                                           "messages": [{"role": "user", "text": "Finish {task}."},
                                                        {"role": "assistant", "text": "I will finish {task}."}],
                                           "cases": [{"task": "the review"}]}) + "\n",
                               encoding="utf-8")
            info = corpus.build(tree, feed, woz, seed, out, pattern)
            result = corpus.audit(out)
            self.assertEqual(info["exact"], 1)
            self.assertEqual(result["episodes"], 4)
            self.assertEqual(result["exactleak"], 0)
            self.assertEqual(result["normleak"], 0)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(row["synthetic"] for row in rows if row["source"] == "project"))
            self.assertTrue(any(row["kind"] == "controlled" and row["synthetic"] for row in rows))
            for family in {row["family"] for row in rows}:
                self.assertEqual(len({row["split"] for row in rows if row["family"] == family}), 1)
            with self.assertRaises(FileExistsError):
                corpus.build(tree, feed, woz, seed, out, pattern)


class Cursor(unittest.TestCase):
    def test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            token = Tokenizer.train(["hello there", "answer now"], target=300)
            rows = []
            for index in range(4):
                rows.append({"id": str(index), "source": "project", "doc": str(index),
                             "family": "family" + str(index), "template": str(index),
                             "split": "train", "synthetic": True, "kind": "authored",
                             "digest": str(index), "raw": 20,
                             "messages": [{"role": "user", "text": "hello there"},
                                          {"role": "assistant", "text": "answer now"}]})
            source = root / "corpus.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            packed = root / "packed"
            shard.write(source, token, packed, limit=32)
            self.assertEqual(shard.check(packed, token.digest)["documents"], 4)
            whole = list(shard.stream(packed, "train", size=5))
            first = whole[0]
            tail = list(shard.stream(packed, "train", cursor=first["cursor"], size=5))
            self.assertEqual([item for row in whole[1:] for item in row["ids"]],
                             [item for row in tail for item in row["ids"]])
            self.assertEqual([item for row in whole[1:] for item in row["mask"]],
                             [item for row in tail for item in row["mask"]])
            self.assertTrue(any(any(row["mask"]) for row in whole))
            self.assertTrue(all(len(row["ids"]) == len(row["mask"]) for row in whole))
            self.assertEqual(sum(sum(row["mask"]) for row in whole),
                             sum(len(token.encode("answer now")) + 1 for _ in rows))
            with self.assertRaises(FileExistsError):
                shard.write(source, token, packed, limit=32)
            with self.assertRaises(ValueError):
                list(shard.stream(packed, "train", {"shard": 0, "token": 9999}, 2))
            with self.assertRaises(ValueError):
                list(shard.stream(packed, "train", size=65537))
            shardfile = next(packed.glob("train*.bin"))
            with shardfile.open("ab") as file:
                file.write(b"x")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                shard.check(packed)


class Archive(unittest.TestCase):
    def test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            memory = Memory(root / "memory.db")
            policy = {"persist": True, "scope": "team", "owner": "one",
                      "private": True}
            event = memory.evidence("The review is due on Friday.", scope="team",
                                    owner="one", source="project", uri="local:note",
                                    version="1", licence="internal", policy=policy)
            value = {"objective": "Finish review", "constraints": ["Friday deadline"],
                     "numbers": ["3 open items"], "decisions": [{"decision": "Review first",
                     "reason": "Deadline"}], "facts": [{"claim": "Due Friday", "source": event}],
                     "unresolved": ["Who owns item three?"], "failures": ["Old plan missed date"],
                     "dependencies": [{"from": "Review", "to": "Close items"}],
                     "sources": [event]}
            compact = Compact(memory)
            token = Tokenizer.train(["Finish review Friday deadline"], target=300)
            self.assertFalse(compact.due(value, 4096, token))
            self.assertTrue(compact.due(value, 1, token))
            saved = compact.trigger("week", value, 1, scope="team", owner="one",
                                    policy=policy, expected=None, token=token)
            self.assertEqual(saved["version"], 1)
            self.assertTrue(saved["fidelity"]["passed"])
            source = compact.original("week", 1, scope="team", owner="one")
            self.assertEqual(source[0]["event"], event)
            self.assertEqual(source[0]["text"], "The review is due on Friday.")
            self.assertEqual(compact.latest("week", scope="team", owner="one")["data"], value)
            changed = dict(value, objective="Revised review objective")
            compact.compact("week", changed, scope="team", owner="one",
                            policy=policy, expected=1)
            restored = compact.rollback("week", 1, scope="team", owner="one",
                                         policy=policy, expected=2)
            self.assertEqual(restored["version"], 3)
            self.assertEqual(restored["rollback"], 1)
            self.assertEqual(restored["data"], value)
            with self.assertRaises(ValueError):
                compact.compact("week", value, scope="team", owner="one",
                                 policy=policy, expected=1)
            memory.close()


class Experience(unittest.TestCase):
    def test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            memory = Memory(root / "memory.db")
            policy = {"persist": True, "scope": "team", "owner": "one",
                      "private": True}
            event = memory.evidence("A project source record.", scope="team", owner="one",
                                    source="project", uri="local:record", version="1",
                                    licence="internal", policy=policy)
            ledger = Learn(Compact(memory))

            class Feed:
                def episodes(self, scope, owner, limit):
                    return [{"family": "planning", "instance": "train-1",
                             "inputhash": hashlib.sha256(b"train-1").hexdigest(),
                             "actions": [{"op": "compare"}], "answer": "done",
                             "result": {"status": "pass"}, "cost": {"steps": 1},
                             "model": "foundation", "adapter": "none",
                             "strategies": [], "strategyversion": [],
                             "dataset": "seed", "evidence": [event]}]

            train = ledger.collect(Feed(), scope="team", owner="one", policy=policy)[0]

            def episode(family, instance, mark=None):
                return ledger.episode({"family": family, "instance": instance,
                                       "input": mark or hashlib.sha256(instance.encode()).hexdigest(),
                                       "actions": ["compare"], "answer": "done",
                                       "result": {"status": "pass"},
                                       "cost": {"steps": 1}, "model": "foundation",
                                       "adapter": "none", "strategy": {"ids": []},
                                       "dataset": "seed",
                                       "version": "base-1", "sources": [event]},
                                      scope="team", owner="one", policy=policy)

            same1 = episode("planning", "same-1")
            same2 = episode("planning", "same-2")
            hold1 = episode("retrieval", "hold-1")
            hold2 = episode("retrieval", "hold-2")
            dup = episode("retrieval", "hold-dup",
                          hashlib.sha256(b"hold-1").hexdigest())
            item = ledger.trial("procedure", "Compare constraints before acting.", [train],
                                [{"episode": train,
                                  "messages": [{"role": "user", "text": "Compare."}],
                                  "target": "Check constraints."}], scope="team", owner="one",
                                policy=policy, model="foundation")
            other = {"persist": True, "scope": "other", "owner": "one",
                     "private": True}
            event2 = memory.evidence("A separate scope record.", scope="other", owner="one",
                                     source="project", uri="local:other", version="1",
                                     licence="internal", policy=other)
            episode2 = ledger.episode({"family": "planning", "instance": "other-1",
                                      "input": hashlib.sha256(b"other-1").hexdigest(),
                                      "actions": [], "answer": "done",
                                      "result": {"status": "pass"}, "cost": {},
                                      "model": "foundation", "adapter": "none",
                                      "strategy": {"ids": []}, "dataset": "seed",
                                      "version": "base-1", "sources": [event2]},
                                     scope="other", owner="one", policy=other)
            trial2 = ledger.trial("procedure", "Separate scope procedure.", [episode2],
                                  [{"episode": episode2,
                                    "messages": [{"role": "user", "text": "Compare."}],
                                    "target": "Check constraints."}], scope="other",
                                  owner="one", policy=other, model="foundation")
            self.assertEqual(item["version"], trial2["version"])
            with self.assertRaisesRegex(ValueError, "two or more"):
                ledger.review("procedure", 1, "external", "deterministic", [hold1, hold1],
                              {"independent": True, "passed": True, "count": 2,
                               "families": ["retrieval"], "scores": []},
                              scope="team", owner="one", policy=policy)
            with self.assertRaisesRegex(ValueError, "inputs must be unique"):
                ledger.review("procedure", 1, "external", "deterministic", [hold1, dup],
                              {"independent": True, "passed": True, "count": 2,
                               "families": ["retrieval"], "scores": []},
                              scope="team", owner="one", policy=policy)
            with self.assertRaisesRegex(ValueError, "family absent"):
                ledger.review("procedure", 1, "external", "deterministic", [same1, same2],
                              {"independent": True, "passed": True, "count": 2,
                               "families": ["planning"], "scores": []},
                              scope="team", owner="one", policy=policy)
            review = ledger.review("procedure", 1, "external", "deterministic", [hold1, hold2],
                                   {"independent": True, "passed": True, "count": 2,
                                    "families": ["retrieval"],
                                    "scores": [{"family": "retrieval", "base": 0.25,
                                                "candidate": 0.75}]},
                                   scope="team", owner="one", policy=policy)
            approved = ledger.approve("procedure", 1, review["id"], scope="team",
                                      owner="one", policy=policy)
            out = root / "approved.json"
            export = ledger.export("procedure", 1, out, scope="team", owner="one")
            self.assertEqual(export["hash"], approved["hash"])
            base = "a" * 64
            candidate = root / "candidate.pt"
            torch.save({"meta": {"base": base, "data": approved["hash"]},
                        "step": 2, "model": {}}, candidate)
            report = {"dataset": approved["hash"], "base": base,
                      "gate": "model-transfer-v1", "independent": True,
                      "passed": True,
                      "scores": [{"family": "retrieval", "base": 0.25,
                                  "candidate": 0.75}]}
            weak = dict(report, scores=[{"family": "retrieval", "base": 0.75,
                                         "candidate": 0.25}])
            with self.assertRaisesRegex(ValueError, "improve on transfer"):
                ledger.register("foundation", "procedure", 1, candidate,
                                base, weak, scope="team", owner="one", policy=policy)
            first = ledger.register("foundation", "procedure", 1, candidate,
                                    base, report, scope="team", owner="one", policy=policy)
            secondpath = root / "candidate-two.pt"
            torch.save({"meta": {"base": base, "data": approved["hash"]},
                        "step": 3, "model": {}}, secondpath)
            ledger.register("foundation", "procedure", 1, secondpath,
                            base, report, scope="team", owner="one", policy=policy)
            revert = ledger.rollback("foundation", 1, scope="team", owner="one",
                                     policy=policy)
            self.assertEqual(revert["rollback"], 1)
            self.assertEqual(revert["hash"], first["hash"])
            self.assertEqual(ledger.fetch(train, "team", "one")["family"], "planning")
            memory.close()


if __name__ == "__main__":
    unittest.main()
