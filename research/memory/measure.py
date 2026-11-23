"""Bounded disk-memory scale smoke with no neural model loaded."""

from __future__ import annotations

import hashlib
import json
import resource
import tempfile
import time
from pathlib import Path

from research.memory.store import Memory


ROOT = Path(__file__).resolve().parent
COUNT = 3000


def check(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("memory measurement exceeded its wall cap")


def main() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = 1024**3
    if soft >= 0:
        cap = min(cap, soft)
    if hard >= 0:
        cap = min(cap, hard)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    started = time.monotonic()
    deadline = started + 110
    with tempfile.TemporaryDirectory(prefix="mem-") as folder:
        path = Path(folder) / "store.db"
        memory = Memory(path, chunk=512)
        policy = {"persist": True, "scope": "bench", "owner": "local"}
        for number in range(COUNT):
            check(deadline)
            text = (f"Archive record {number} belongs to group {number % 31}. "
                    "Routine synthetic catalog note. ")
            if number == 2417:
                text += "Rare marker quasar 91 appears here."
            memory.evidence(
                text, scope="bench", owner="local", source=f"item{number}",
                uri=f"synthetic://item/{number}", version="1", licence="CC0-1.0",
                acquired="2026-01-01T00:00:00Z", policy=policy,
            )
        total = memory.sql.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        req = memory.request("bench", "local", calls=1, bytes=512,
                             chunks=2, tokens=64, wall=5)
        rows = req.search("quasar", k=8)
        if len(rows) != 1 or "quasar 91" not in rows[0]["text"]:
            raise AssertionError("bounded search did not find the unique synthetic record")
        memory.sql.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        memory.close()
        dbsize = path.stat().st_size + (path.with_name(path.name + "-wal").stat().st_size
                                        if path.with_name(path.name + "-wal").exists() else 0)
        check(deadline)
        report = {
            "records": COUNT,
            "chunks": total,
            "hits": len(rows),
            "reads": req.used["chunks"],
            "bytes": req.used["bytes"],
            "tokens": req.used["tokens"],
            "db": dbsize,
            "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "wall": time.monotonic() - started,
            "cap": cap,
            "deadline": 110,
            "device": "cpu",
            "threads": 1,
            "fts": memory.fts,
            "model": "not loaded",
            "query": "quasar",
            "scope": "bench",
            "hash": hashlib.sha256((ROOT / "store.py").read_bytes()).hexdigest(),
            "limits": ["Project-generated records only.",
                       "SQLite terms/FTS indexes queried on disk; evidence text returned only for budgeted hits.",
                       "RSS measures the memory subsystem process and excludes any model because none was loaded."],
        }
    (ROOT / "out.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
