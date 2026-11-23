import json
import unittest

from research.recursive.evaluate import check, edges, score, solve
from research.recursive.tasks import CONTEXT, CROSS, build, crosses, manifest


class Tests(unittest.TestCase):
    def test(self):
        data = build()
        again = build()
        self.assertEqual(manifest(data), manifest(again))
        self.assertEqual(json.dumps(data, sort_keys=True), json.dumps(again, sort_keys=True))
        self.assertEqual(set(data["families"]), {"addition", "bracket", "graph", "chain", "trace"})
        self.assertEqual(check(data)["passed"], True)
        self.assertEqual(len(CROSS), 4)
        self.assertEqual(len(crosses(data)), 4)
        for family, parts in data["families"].items():
            for split, rows in parts.items():
                twin = again["families"][family][split]
                self.assertEqual([solve(row) for row in rows], [solve(row) for row in twin])
            self.assertEqual([len(parts[split]) for split in ("train", "validation", "interpolation", "extrapolation")],
                             [12, 4, 4, 4])
            ids = [row["id"] for rows in parts.values() for row in rows]
            prompts = [row["prompt"] for rows in parts.values() for row in rows]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(prompts), len(set(prompts)))
            upper = max(row["level"] for row in parts["train"])
            lower = min(row["level"] for row in parts["extrapolation"])
            self.assertGreater(lower, upper)
            self.assertTrue(all(len((row["prompt"] + row["target"]).encode("utf-8")) <= CONTEXT
                                for rows in parts.values() for row in rows))
            for split in parts:
                self.assertEqual({row["family"] for row in parts[split]}, {family})
                self.assertEqual({row["split"] for row in parts[split]}, {split})
        for row in crosses(data):
            self.assertEqual({item["family"] for item in row["train"]}, {row["source"]})
            self.assertEqual({item["family"] for item in row["evaluation"]}, {row["target"]})
            self.assertFalse({item["id"] for item in row["train"]} & {item["id"] for item in row["evaluation"]})
        graph = next(iter(data["families"]["graph"]["train"]))
        links = edges(graph["prompt"])
        self.assertEqual(len(links), 4 * (graph["level"] - 1))
        self.assertTrue(all(" " not in right for left, right in links))
        forms = [tuple(sorted(edges(row["prompt"])))
                 for rows in data["families"]["graph"].values() for row in rows]
        self.assertEqual(len(forms), len(set(forms)))
        self.assertTrue(score(graph, graph["target"])["correct"])
        self.assertFalse(score(graph, "S>T")["correct"])
        brackets = data["families"]["bracket"]["train"][0]
        self.assertEqual(solve(brackets), solve(brackets))
        program = data["families"]["trace"]["train"][0]["prompt"].split("DSL: ", 1)[1].split(". What", 1)[0]
        with self.assertRaises(ValueError):
            from research.recursive.evaluate import trace
            trace(program + ";import os")
        self.assertTrue(score(data["families"]["addition"]["train"][0], "not a number")["error"] == "format")


if __name__ == "__main__":
    unittest.main()
