from __future__ import annotations

import unittest
import json

import torch

from research.interaction.tokenizer import Tokenizer
from research.microworld.dense import (
    align_annotations, audit_annotations,
)
from research.microworld.generator import build_world_examples, make_world
from research.microworld.objectives import (
    TASKS, FoundationWithDenseHeads, _normalized_task_loss, _objective_label,
    label_vocabulary, language_only_loss, objective_losses, token_labels,
)


class TokenAlignmentTests(unittest.TestCase):
    def test_bpe_offsets_preserve_punctuation_and_multi_token_names(self):
        tokenizer = Tokenizer([(ord("A"), ord("l")), (264, ord("m"))], target=300)
        text = "Alma, Alma!"
        encoded = tokenizer.encode_with_offsets(text)
        self.assertEqual(tokenizer.decode(encoded["ids"]), text)
        self.assertEqual(tokenizer.encode(text), encoded["ids"])
        self.assertEqual(encoded["byte_offsets"][0], (0, 3))
        self.assertEqual(encoded["char_offsets"][0], (0, 3))
        comma = text.index(",")
        comma_tokens = [index for index, (start, end) in
                        enumerate(encoded["char_offsets"])
                        if start <= comma < end]
        self.assertEqual(tokenizer.decode(
            [encoded["ids"][index] for index in comma_tokens]), ",")
        self.assertGreater(len(encoded["ids"]), 4)

    def test_utf8_byte_tokens_map_to_overlapping_character_span(self):
        tokenizer = Tokenizer((), target=4096)
        text = "A雪, 🙂!"
        encoded = tokenizer.encode_with_offsets(text)
        self.assertEqual(tokenizer.decode(encoded["ids"]), text)
        self.assertEqual(tokenizer.encode(text), encoded["ids"])
        snow = text.index("雪")
        matching = [char_span for char_span in encoded["char_offsets"]
                    if char_span == (snow, snow + 1)]
        self.assertEqual(len(matching), 3)
        for token_id, (byte_start, byte_end), (char_start, char_end) in zip(
                encoded["ids"], encoded["byte_offsets"], encoded["char_offsets"]):
            self.assertEqual(tokenizer.parts[token_id], text.encode("utf-8")[byte_start:byte_end])
            self.assertLess(char_start, char_end)


class DenseAnnotationTests(unittest.TestCase):
    def _row(self, split: str, query: str, kind: str):
        world = make_world(3, split, seed=87000)
        world["entities"][0]["name"] = "Éowyn-Agent Alpha."
        examples = build_world_examples(world)
        row = next(item for item in examples
                   if item["query_type"] == query and item["episode_kind"] == kind)
        return world, row

    def test_annotations_are_oracle_grounded_and_support_overlap(self):
        world, row = self._row("train", "transition", "store")
        tokenizer = Tokenizer((), target=4096)
        aligned = align_annotations(row, world, tokenizer)
        audit = audit_annotations(row, world, aligned, tokenizer)
        self.assertEqual(tokenizer.decode(aligned["token_ids"]), aligned["source"])
        types = {item["type"] for item in aligned["annotations"]}
        self.assertTrue({"entity_mention", "entity_identity", "entity_type",
                         "property_mention", "property_key", "property_value",
                         "relationship_phrase", "relationship_type", "relationship_source",
                         "relationship_target", "temporal_reference", "action_mention",
                         "action_type", "precondition", "state_change",
                         "state_change_entity"} <= types)
        relation_source = next(item for item in aligned["annotations"]
                               if item["type"] == "relationship_source"
                               and item["entity_id"] == "agent-0")
        mention = next(item for item in aligned["annotations"]
                       if item["type"] == "entity_mention"
                       and item["char_start"] == relation_source["char_start"])
        self.assertEqual(mention["char_end"], relation_source["char_end"])
        self.assertLess(mention["token_start"], mention["token_end"])
        self.assertGreater(audit["annotations"], 100)
        reloaded = json.loads(json.dumps(aligned))
        self.assertEqual(audit_annotations(row, world, reloaded, tokenizer), audit)

    def test_repeated_mentions_each_keep_identity_and_valid_offsets(self):
        world, row = self._row("train", "location", "give")
        tokenizer = Tokenizer((), target=4096)
        aligned = align_annotations(row, world, tokenizer)
        spans = [item for item in aligned["annotations"]
                 if item["type"] == "entity_identity" and item["entity_id"] == "agent-0"]
        self.assertGreaterEqual(len(spans), 3)
        for item in spans:
            self.assertEqual(aligned["source"][item["char_start"]:item["char_end"]], item["text"])
            self.assertEqual(item["token_start"],
                             min(i for i, (start, end) in enumerate(aligned["token_byte_offsets"])
                                 if start < item["byte_end"] and end > item["byte_start"]))

    def test_negation_temporal_and_invalid_reason_are_explicit(self):
        world, row = self._row("train", "action_validity", "invalid")
        tokenizer = Tokenizer((), target=4096)
        aligned = align_annotations(row, world, tokenizer)
        types = {item["type"] for item in aligned["annotations"]}
        self.assertIn("invalid_action", types)
        self.assertIn("violated_constraint", types)
        self.assertIn("action_validity", types)
        self.assertIn("temporal_reference", types)

    def test_all_tasks_receive_token_targets_with_o_for_span_outside(self):
        world, row = self._row("train", "transition", "store")
        tokenizer = Tokenizer((), target=4096)
        aligned = align_annotations(row, world, tokenizer)
        vocab = label_vocabulary(aligned["annotations"])
        targets = token_labels(aligned, vocab)
        self.assertEqual(set(targets), set(TASKS))
        self.assertTrue(all(len(value) == len(aligned["token_ids"])
                            for value in targets.values()))
        self.assertGreater(sum(value != -100 for value in targets["entity_span"]), 0)
        self.assertIn("O", vocab["entity_span"])

    def test_task_loss_macro_averages_classes_instead_of_token_frequency(self):
        logits = torch.tensor([[[4.0, 0.0], [4.0, 0.0],
                                [4.0, 0.0], [0.0, 4.0]]])
        targets = torch.tensor([[0, 0, 0, 1]])
        loss = _normalized_task_loss(logits, targets)
        expected = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 2), targets.reshape(-1), reduction="none")
        expected = (expected[:3].mean() + expected[3]) / 2
        self.assertTrue(torch.allclose(loss, expected))

    def test_marker_labels_do_not_leak_heldout_lexical_forms(self):
        self.assertEqual(_objective_label({"type": "negation", "label": "not",
                                           "text": "not"}), "negation_marker")
        self.assertEqual(_objective_label({"type": "negation", "label": "invalid",
                                           "text": "invalid"}), "negation_marker")
        self.assertEqual(_objective_label({"type": "temporal_reference", "label": "after",
                                           "text": "after"}), "temporal_marker")

    def test_objective_paths_compute_losses_without_optimizer_updates(self):
        config = {
            "vocab_size": 300, "context_length": 8, "layers": 1,
            "d_model": 8, "n_heads": 2, "d_ff": 16,
            "position_embedding": "learned_absolute",
            "normalization": "pre_layernorm", "weight_tying": True,
            "dtype": "float32",
        }
        vocabularies = {task: ["O", "x"] for task in TASKS}
        model = FoundationWithDenseHeads(config, vocabularies)
        tokens = torch.tensor([[65, 66, 67, 68]])
        labels = {task: torch.zeros((1, 4), dtype=torch.long) for task in TASKS}

        language_loss = language_only_loss(model.backbone, tokens)
        structural = objective_losses(model, tokens, labels, include_language=False)
        combined = objective_losses(model, tokens, labels, include_language=True)

        self.assertTrue(torch.isfinite(language_loss))
        self.assertTrue(torch.isfinite(structural["total"]))
        self.assertTrue(torch.isfinite(combined["total"]))
        self.assertEqual(structural["language"].item(), 0.0)
        self.assertEqual(structural["auxiliary_parameters"], model.auxiliary_params)
        self.assertTrue(torch.allclose(
            combined["total"], combined["language"] + combined["structural"],
            rtol=1e-6, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
