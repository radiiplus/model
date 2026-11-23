#!/usr/bin/env python3
"""Train and measure tiny tokenizer alternatives on a balanced synthetic fixture."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import resource
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import sentencepiece as spm
import tiktoken
import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "experiments" / "fixtures" / "tokenizer_corpus.jsonl"
OUTPUT_DEFAULT = ROOT / "experiments" / "results" / "tokenizer_benchmark.json"
TARGET_VOCAB = 512
SEED = 20261123
HELDOUT_IDS = {"number_decimal", "code_rust", "equation_unicode", "url", "rare", "mixed_script", "quoted_text"}
TIKTOKEN_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def available_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable unavailable; stop instead of running without a memory guard")


def load_fixture(path: Path = FIXTURE) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fixture_split(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train, heldout = [], []
    for row in rows:
        (heldout if row["id"] in HELDOUT_IDS else train).append(row)
    return train, heldout


def build_hf_bpe(train_texts: list[str]) -> tuple[Any, dict[str, Any]]:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    special_tokens = ["<unk>", "<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|tool|>", "<|end|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=TARGET_VOCAB, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=special_tokens, show_progress=False,
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer, length=len(train_texts))
    return tokenizer, {"trainer": "tokenizers.BpeTrainer", "pre_tokenizer": "ByteLevel(use_regex=True)",
                       "normalizer": None, "requested_vocab_size": TARGET_VOCAB,
                       "special_tokens": special_tokens}


def build_sentencepiece(train_texts: list[str], model_type: str, workdir: Path) -> tuple[Any, dict[str, Any]]:
    model_prefix = str(workdir / f"sp_{model_type}")
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(train_texts), model_prefix=model_prefix, model_type=model_type,
        vocab_size=TARGET_VOCAB, character_coverage=1.0, byte_fallback=True,
        normalization_rule_name="identity", hard_vocab_limit=False,
        unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1, num_threads=1,
        add_dummy_prefix=False, remove_extra_whitespaces=False, escape_whitespaces=True,
        shuffle_input_sentence=False, input_sentence_size=0, minloglevel=2,
    )
    processor = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
    return processor, {"trainer": "SentencePieceTrainer", "model_type": model_type,
                       "requested_vocab_size": TARGET_VOCAB, "byte_fallback": True,
                       "character_coverage": 1.0, "normalization": "identity",
                       "special_ids": {"unk": processor.unk_id(), "bos": processor.bos_id(),
                                       "eos": processor.eos_id(), "pad": processor.pad_id()}}


def build_tiktoken_educational(train_texts: list[str]) -> tuple[Any, dict[str, Any]]:
    from tiktoken._educational import SimpleBytePairEncoding

    with contextlib.redirect_stdout(io.StringIO()):
        encoding = SimpleBytePairEncoding.train("\n".join(train_texts), vocab_size=TARGET_VOCAB, pat_str=TIKTOKEN_PATTERN)
    return encoding, {"trainer": "tiktoken._educational.SimpleBytePairEncoding.train",
                      "pre_tokenizer_regex": TIKTOKEN_PATTERN, "normalizer": None,
                      "requested_vocab_size": TARGET_VOCAB, "special_tokens": []}


def build_char_baseline(train_texts: list[str]) -> tuple[Any, dict[str, Any]]:
    class CodepointTokenizer:
        @staticmethod
        def encode(text: str) -> list[int]:
            return [ord(char) for char in text]

        @staticmethod
        def decode(ids: list[int]) -> str:
            return "".join(chr(value) for value in ids)

    observed = len({char for text in train_texts for char in text})
    return CodepointTokenizer(), {"trainer": "none; direct Unicode scalar IDs",
                                  "normalizer": None, "vocab_size_theoretical": 1_112_064,
                                  "training_observed_codepoints": observed,
                                  "unknown_policy": "Unicode scalar identity; fixed softmax would require a full codepoint vocabulary or fallback"}


def build_byte_baseline(train_texts: list[str]) -> tuple[Any, dict[str, Any]]:
    class ByteTokenizer:
        @staticmethod
        def encode(text: str) -> list[int]:
            return list(text.encode("utf-8"))

        @staticmethod
        def decode(ids: list[int]) -> str:
            return bytes(ids).decode("utf-8")

    return ByteTokenizer(), {"trainer": "none; UTF-8 byte IDs", "normalizer": None,
                             "vocab_size": 256, "unknown_policy": "none; every Unicode scalar is represented by UTF-8 bytes"}


def encode_decode(name: str, tokenizer: Any, text: str) -> tuple[list[int], str]:
    if name == "hf_byte_bpe":
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
    elif name.startswith("spm_"):
        ids = tokenizer.encode(text, out_type=int)
        decoded = tokenizer.decode(ids)
    elif name == "tiktoken_educational_bpe":
        ids = tokenizer.encode(text, visualise=None)
        decoded = tokenizer.decode(ids)
    else:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
    return ids, decoded


def vocab_size(name: str, tokenizer: Any) -> int:
    if name == "hf_byte_bpe":
        return tokenizer.get_vocab_size(with_added_tokens=True)
    if name.startswith("spm_"):
        return tokenizer.get_piece_size()
    if name == "tiktoken_educational_bpe":
        return len(tokenizer.mergeable_ranks)
    if name == "byte_baseline":
        return 256
    return 1_112_064


def measure_one(name: str, tokenizer: Any, config: dict[str, Any], rows: list[dict[str, str]],
                repeats: int, train_seconds: float) -> dict[str, Any]:
    texts = [row["text"] for row in rows]
    encode_times, decode_times, token_rows = [], [], []
    all_round_trip = True
    total_bytes = sum(len(text.encode("utf-8")) for text in texts)
    tracemalloc.start()
    for row in rows:
        text = row["text"]
        start = time.perf_counter()
        ids, decoded = encode_decode(name, tokenizer, text)
        encode_times.append(time.perf_counter() - start)
        all_round_trip &= decoded == text
        fallback_fraction = None
        if name == "spm_unigram" or name == "spm_bpe":
            fallback_fraction = sum(tokenizer.id_to_piece(token).startswith("<0x") for token in ids) / len(ids) if ids else 0.0
        elif name == "tiktoken_educational_bpe":
            fallback_fraction = sum(token < 256 for token in ids) / len(ids) if ids else 0.0
        token_rows.append({"id": row["id"], "domain": row["domain"],
                           "utf8_bytes": len(text.encode("utf-8")), "tokens": len(ids),
                           "fallback_byte_fraction": fallback_fraction})
        if name == "hf_byte_bpe":
            decode = lambda: tokenizer.decode(ids, skip_special_tokens=False)
        else:
            decode = lambda: tokenizer.decode(ids)
        start = time.perf_counter()
        for _ in range(repeats):
            encode_decode(name, tokenizer, text)
        repeated_encode_seconds = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(repeats):
            decode()
        decode_times.append((time.perf_counter() - start) / repeats)
        encode_times[-1] = repeated_encode_seconds / repeats
    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_tokens = sum(row["tokens"] for row in token_rows)
    fallback_rows = [row for row in token_rows if row["fallback_byte_fraction"] is not None]
    sum_token_square = sum(row["tokens"] ** 2 for row in token_rows)
    sum_byte_square = sum(row["utf8_bytes"] ** 2 for row in token_rows)
    encode_seconds = sum(encode_times)
    return {
        "config": config, "vocab_size": vocab_size(name, tokenizer),
        "train_time_seconds": train_seconds, "heldout_documents": len(texts),
        "heldout_utf8_bytes": total_bytes, "heldout_tokens": total_tokens,
        "tokens_per_utf8_byte": total_tokens / total_bytes,
        "utf8_bytes_per_token": total_bytes / total_tokens,
        "byte_fallback_fraction": (sum(row["fallback_byte_fraction"] * row["tokens"] for row in fallback_rows) / sum(row["tokens"] for row in fallback_rows)) if fallback_rows else None,
        "round_trip_exact": all_round_trip, "coverage_percent": 100.0 if all_round_trip else 0.0,
        "encode_latency_seconds_per_document_mean": statistics.mean(encode_times),
        "encode_latency_seconds_per_document_p50": statistics.median(encode_times),
        "encode_throughput_utf8_bytes_per_second": (total_bytes * repeats / encode_seconds) if encode_seconds else None,
        "decode_latency_seconds_per_document_mean": statistics.mean(decode_times),
        "attention_pair_count_ratio_vs_byte_baseline": None,
        "linear_sequence_work_ratio_vs_byte_baseline": None,
        "tracemalloc_peak_bytes": peak_traced, "documents": token_rows,
        "_sum_token_square": sum_token_square, "_sum_byte_square": sum_byte_square,
    }


def run(path: Path = FIXTURE, repeats: int = 20, min_available_mib: int = 2560) -> dict[str, Any]:
    if available_mib() < min_available_mib:
        raise MemoryError(f"MemAvailable below guard of {min_available_mib} MiB")
    rows = load_fixture(path)
    train_rows, heldout_rows = fixture_split(rows)
    train_texts = [row["text"] for row in train_rows]
    train_codepoints = {char for text in train_texts for char in text}
    heldout_codepoints = {char for row in heldout_rows for char in row["text"]}
    with tempfile.TemporaryDirectory(prefix="tokbench-") as temp:
        workdir = Path(temp)
        builders: list[tuple[str, Callable[[], tuple[Any, dict[str, Any]]]]] = [
            ("byte_baseline", lambda: build_byte_baseline(train_texts)),
            ("unicode_codepoint", lambda: build_char_baseline(train_texts)),
            ("hf_byte_bpe", lambda: build_hf_bpe(train_texts)),
            ("spm_bpe", lambda: build_sentencepiece(train_texts, "bpe", workdir)),
            ("spm_unigram", lambda: build_sentencepiece(train_texts, "unigram", workdir)),
            ("tiktoken_educational_bpe", lambda: build_tiktoken_educational(train_texts)),
        ]
        results = {}
        for name, builder in builders:
            start = time.perf_counter()
            tokenizer, config = builder()
            train_seconds = time.perf_counter() - start
            results[name] = measure_one(name, tokenizer, config, heldout_rows, repeats, train_seconds)
    baseline_bytes = results["byte_baseline"]["heldout_utf8_bytes"]
    baseline_square = sum(row["utf8_bytes"] ** 2 for row in results["byte_baseline"]["documents"])
    for result in results.values():
        result["attention_pair_count_ratio_vs_byte_baseline"] = result["_sum_token_square"] / baseline_square
        result["linear_sequence_work_ratio_vs_byte_baseline"] = result["heldout_tokens"] / baseline_bytes
        result.pop("_sum_token_square")
        result.pop("_sum_byte_square")
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {
        "schema_version": 1, "fixture": str(path),
        "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_documents": len(rows), "train_documents": len(train_rows), "heldout_documents": len(heldout_rows),
        "train_ids": [row["id"] for row in train_rows], "heldout_ids": [row["id"] for row in heldout_rows],
        "train_utf8_bytes": sum(len(text.encode("utf-8")) for text in train_texts),
        "heldout_codepoints_unseen_in_train": sorted(heldout_codepoints - train_codepoints),
        "heldout_unseen_codepoint_count": len(heldout_codepoints - train_codepoints),
        "seed": SEED, "requested_vocab_size": TARGET_VOCAB, "repeats": repeats,
        "environment": {"python": sys.version.split()[0], "platform": sys.platform,
                        "packages": {"tokenizers": tokenizers.__version__, "sentencepiece": spm.__version__, "tiktoken": tiktoken.__version__},
                        "mem_available_mib_at_start": available_mib(),
                        "cgroup_memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
                        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                        "thread_caps": {key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]}},
        "peak_rss_bytes": peak_rss, "methods": results,
        "limitations": ["synthetic held-out set has seven texts", "training samples are too small for stable tokenizer-quality conclusions", "compression ratio is not downstream capability", "BPE/unigram behavior depends on trainer and normalization configuration", "Unicode codepoint IDs imply a theoretical 1,112,064-class vocabulary if used directly in a fixed softmax"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=FIXTURE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--min-available-mib", type=int, default=2560)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        raise SystemExit("--repeats must be in 1..100")
    result = run(args.input, args.repeats, args.min_available_mib)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"methods": {key: {name: value for name, value in entry.items() if name not in {"config", "documents"}}
                      for key, entry in result["methods"].items()}, "peak_rss_bytes": result["peak_rss_bytes"],
                     "result": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
