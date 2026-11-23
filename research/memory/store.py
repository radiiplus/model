"""Persistent provenance memory with append-only records and bounded reads."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


CAPS = {"calls": 32, "bytes": 1_048_576, "chunks": 128,
        "tokens": 65_536, "wall": 30.0, "meta": 4096, "query": 4096,
        "terms": 128, "items": 128, "field": 4096}
CHUNK = 8192


class Budget(Exception):
    """The retrieval request exceeded one of its registered limits."""


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def stamp(value: str | None = None, last: bool = False) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        if len(value) != 10:
            raise ValueError("timestamps must include a timezone")
        parsed = parsed.replace(tzinfo=timezone.utc)
        if last:
            parsed = parsed.replace(hour=23, minute=59, second=59)
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


def pieces(text: str, size: int) -> Iterator[tuple[int, int, str]]:
    part = []
    used = 0
    start = 0
    offset = 0
    space = -1
    for char in text:
        width = len(char.encode("utf-8"))
        while used and used + width > size:
            if space >= 0:
                head = part[:space + 1]
                tail = part[space + 1:]
                headtext = "".join(head)
                amount = len(headtext.encode("utf-8"))
                yield start, start + amount, headtext
                start += amount
                part = tail
                used -= amount
                space = max((index for index, value in enumerate(part) if value.isspace()),
                            default=-1)
            else:
                yield start, offset, "".join(part)
                part = []
                used = 0
                start = offset
                space = -1
        part.append(char)
        used += width
        offset += width
        if char.isspace():
            space = len(part) - 1
    if part:
        yield start, offset, "".join(part)


def allowed(policy: dict[str, Any] | None, scope: str, owner: str,
            private: bool = False) -> None:
    if (not isinstance(scope, str) or not scope or len(scope) > 256
            or not isinstance(owner, str) or not owner or len(owner) > 256):
        raise ValueError("scope and owner are required")
    if (not isinstance(policy, dict) or policy.get("persist") is not True
            or policy.get("scope") != scope or policy.get("owner") != owner):
        raise PermissionError("persistence requires matching explicit caller policy")
    if private and policy.get("private") is not True:
        raise PermissionError("private content requires explicit private persistence policy")


def bounded(value: str, name: str, size: int = CAPS["field"]) -> None:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > size:
        raise ValueError(f"{name} must be nonempty text within {size} bytes")


def listed(values: list[str], name: str) -> list[str]:
    if (not isinstance(values, list) or len(values) > CAPS["items"]
            or any(not isinstance(value, str) or not value
                   or len(value.encode("utf-8")) > CAPS["field"] for value in values)):
        raise ValueError(f"{name} must contain at most {CAPS['items']} bounded text items")
    return values


class Memory:
    def __init__(self, path: str | Path, chunk: int = 2048,
                 fts: bool | None = None):
        if not isinstance(chunk, int) or isinstance(chunk, bool) or not 4 <= chunk <= CHUNK:
            raise ValueError(f"chunk size must be in 4..{CHUNK} bytes")
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.chunk = chunk
        self.sql = sqlite3.connect(self.path, timeout=0.1)
        self.sql.row_factory = sqlite3.Row
        self.sql.execute("PRAGMA foreign_keys=ON")
        self.sql.execute("PRAGMA journal_mode=WAL")
        self.sql.execute("PRAGMA synchronous=FULL")
        self.sql.execute("PRAGMA busy_timeout=100")
        self.make()
        self.fts = False if fts is False else self.index()

    def make(self) -> None:
        self.sql.executescript("""
        CREATE TABLE IF NOT EXISTS events(
            id TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            owner TEXT NOT NULL,
            kind TEXT NOT NULL,
            target TEXT,
            source TEXT NOT NULL,
            uri TEXT NOT NULL,
            version TEXT NOT NULL,
            acquired TEXT NOT NULL,
            licence TEXT NOT NULL,
            hash TEXT NOT NULL,
            meta TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS e1 ON events(scope, owner, source, version);
        CREATE INDEX IF NOT EXISTS e2 ON events(scope, owner, target, kind);
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY,
            key TEXT NOT NULL UNIQUE,
            event TEXT NOT NULL REFERENCES events(id),
            scope TEXT NOT NULL,
            owner TEXT NOT NULL,
            part INTEGER NOT NULL,
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            size INTEGER NOT NULL,
            tokens INTEGER NOT NULL,
            hash TEXT NOT NULL,
            text TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS c1 ON chunks(scope, owner, event, part);
        CREATE TABLE IF NOT EXISTS terms(
            id INTEGER PRIMARY KEY,
            scope TEXT NOT NULL,
            owner TEXT NOT NULL,
            term TEXT NOT NULL,
            chunk INTEGER NOT NULL REFERENCES chunks(id),
            count INTEGER NOT NULL,
            UNIQUE(chunk, term)
        );
        CREATE INDEX IF NOT EXISTS t1 ON terms(scope, owner, term, chunk);
        CREATE TABLE IF NOT EXISTS candidates(
            id TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            owner TEXT NOT NULL,
            claim TEXT NOT NULL,
            created TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status='unverified')
        );
        CREATE TABLE IF NOT EXISTS crefs(
            candidate TEXT NOT NULL REFERENCES candidates(id),
            evidence TEXT NOT NULL REFERENCES events(id),
            PRIMARY KEY(candidate, evidence)
        );
        CREATE TABLE IF NOT EXISTS facts(
            id TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            owner TEXT NOT NULL,
            key TEXT NOT NULL,
            claim TEXT NOT NULL,
            version INTEGER NOT NULL,
            start TEXT NOT NULL,
            end TEXT,
            method TEXT NOT NULL,
            created TEXT NOT NULL,
            supersedes TEXT,
            rollback TEXT
        );
        CREATE INDEX IF NOT EXISTS f1 ON facts(scope, owner, key, version);
        CREATE TABLE IF NOT EXISTS frefs(
            fact TEXT NOT NULL REFERENCES facts(id),
            evidence TEXT NOT NULL REFERENCES events(id),
            PRIMARY KEY(fact, evidence)
        );
        CREATE TABLE IF NOT EXISTS skills(
            id TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            owner TEXT NOT NULL,
            name TEXT NOT NULL,
            version INTEGER NOT NULL,
            preconditions TEXT NOT NULL,
            procedure TEXT NOT NULL,
            expected TEXT NOT NULL,
            tested TEXT NOT NULL,
            counterexamples TEXT NOT NULL,
            validation TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('candidate','validated')),
            created TEXT NOT NULL,
            previous TEXT
        );
        CREATE INDEX IF NOT EXISTS s1 ON skills(scope, owner, name, version);
        CREATE TRIGGER IF NOT EXISTS ti1 BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ti2 BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ti3 BEFORE UPDATE ON chunks BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ti4 BEFORE DELETE ON chunks BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ti5 BEFORE UPDATE ON terms BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ti6 BEFORE DELETE ON terms BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ti7 BEFORE UPDATE ON candidates BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ti8 BEFORE DELETE ON candidates BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ti9 BEFORE UPDATE ON crefs BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ti10 BEFORE DELETE ON crefs BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ti11 BEFORE UPDATE ON facts BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ti12 BEFORE DELETE ON facts BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ti13 BEFORE UPDATE ON frefs BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ti14 BEFORE DELETE ON frefs BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ti15 BEFORE UPDATE ON skills BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ti16 BEFORE DELETE ON skills BEGIN SELECT RAISE(ABORT, 'immutable'); END;
        """)
        self.sql.commit()

    def index(self) -> bool:
        try:
            self.sql.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text, scope UNINDEXED, owner UNINDEXED, event UNINDEXED, key UNINDEXED)")
            self.sql.commit()
            return True
        except sqlite3.OperationalError:
            self.sql.rollback()
            return False

    def event(self, scope: str, owner: str, kind: str, target: str | None,
              source: str, uri: str, version: str, acquired: str,
              licence: str, content: str, meta: dict[str, Any]) -> str:
        raw = {"scope": scope, "owner": owner, "kind": kind, "target": target,
               "source": source, "uri": uri, "version": version,
               "acquired": acquired, "licence": licence, "hash": digest(content),
               "meta": meta}
        event = digest(encode(raw))
        self.sql.execute(
            "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (event, scope, owner, kind, target, source, uri, version,
             acquired, licence, raw["hash"], encode(meta)),
        )
        return event

    def evidence(self, text: str, *, scope: str, owner: str, source: str,
                 uri: str, version: str, licence: str,
                 policy: dict[str, Any] | None = None,
                 meta: dict[str, Any] | None = None,
                 acquired: str | None = None, supersedes: str | None = None,
                 private: bool = False) -> str:
        allowed(policy, scope, owner, private)
        if not isinstance(text, str) or not text:
            raise ValueError("evidence text must be a nonempty string")
        if (not all(isinstance(value, str) and value for value in (source, uri, version, licence))
                or len(source) > 512 or len(uri) > 2048
                or len(version) > 128 or len(licence) > 256):
            raise ValueError("source, URI, version and licence are required")
        if meta is not None and not isinstance(meta, dict):
            raise ValueError("provenance metadata must be an object")
        when = stamp(acquired)
        detail = dict(meta or {})
        if len(encode(detail).encode("utf-8")) > CAPS["meta"]:
            raise ValueError("provenance metadata exceeds its byte limit")
        kind = "evidence"
        if supersedes is not None:
            self.need(scope, owner, supersedes, "evidence")
            kind = "supersede"
        hasher = hashlib.sha256()
        for start, end, part in pieces(text, self.chunk):
            hasher.update(part.encode("utf-8"))
        full = hasher.hexdigest()
        raw = {"scope": scope, "owner": owner, "kind": kind, "target": supersedes,
               "source": source, "uri": uri, "version": version,
               "acquired": when, "licence": licence, "hash": full,
               "meta": detail}
        event = digest(encode(raw))
        with self.sql:
            cursor = self.sql.execute(
                "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (event, scope, owner, kind, supersedes, source, uri, version,
                 when, licence, full, encode(detail)),
            )
            if not cursor.rowcount:
                return event
            for number, (start, end, part) in enumerate(pieces(text, self.chunk)):
                rawpart = part.encode("utf-8")
                mark = hashlib.sha256(rawpart).hexdigest()
                key = digest(f"{event}:{start}:{end}:{mark}")
                count = Counter(words(part))
                cursor = self.sql.execute(
                    "INSERT INTO chunks(key,event,scope,owner,part,start,end,size,tokens,hash,text) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (key, event, scope, owner, number, start, end, len(rawpart),
                     sum(count.values()), mark, part),
                )
                chunk = cursor.lastrowid
                self.sql.executemany(
                    "INSERT INTO terms(scope,owner,term,chunk,count) VALUES(?,?,?,?,?)",
                    [(scope, owner, term, chunk, value) for term, value in count.items()],
                )
                if self.fts:
                    self.sql.execute(
                        "INSERT INTO fts(rowid,text,scope,owner,event,key) VALUES(?,?,?,?,?,?)",
                        (chunk, part, scope, owner, event, key),
                    )
        return event

    def need(self, scope: str, owner: str, item: str,
             table: str = "evidence") -> sqlite3.Row:
        if table == "evidence":
            row = self.sql.execute(
                "SELECT id FROM events WHERE id=? AND scope=? AND owner=? AND kind IN ('evidence','supersede')",
                (item, scope, owner),
            ).fetchone()
        elif table == "fact":
            row = self.sql.execute(
                "SELECT id FROM facts WHERE id=? AND scope=? AND owner=?",
                (item, scope, owner),
            ).fetchone()
        else:
            row = None
        if row is None:
            raise KeyError(item)
        return row

    def invalidate(self, target: str, *, scope: str, owner: str,
                   policy: dict[str, Any] | None = None,
                   reason: str = "caller invalidation",
                   acquired: str | None = None) -> str:
        allowed(policy, scope, owner)
        fact = self.sql.execute(
            "SELECT id FROM facts WHERE id=? AND scope=? AND owner=?",
            (target, scope, owner),
        ).fetchone()
        if fact is not None:
            kind = "factinvalidate"
        else:
            self.need(scope, owner, target, "evidence")
            kind = "invalidate"
        when = stamp(acquired)
        meta = {"reason": reason}
        with self.sql:
            return self.event(scope, owner, kind, target, "memory", "memory:event",
                              "1", when, "project-internal", "", meta)

    def candidate(self, claim: str, refs: list[str], *, scope: str,
                  owner: str, policy: dict[str, Any] | None = None,
                  private: bool = False,
                  created: str | None = None) -> str:
        allowed(policy, scope, owner, private)
        bounded(claim, "candidate claim")
        refs = listed(refs, "evidence references")
        if not refs:
            raise ValueError("candidate claim and supporting evidence are required")
        refs = sorted(set(refs))
        for item in refs:
            self.need(scope, owner, item)
        when = stamp(created)
        item = digest(encode([scope, owner, claim, refs, when]))
        with self.sql:
            self.sql.execute(
                "INSERT OR IGNORE INTO candidates VALUES(?,?,?,?,?,'unverified')",
                (item, scope, owner, claim, when),
            )
            self.sql.executemany(
                "INSERT OR IGNORE INTO crefs VALUES(?,?)", [(item, ref) for ref in refs]
            )
        return item

    def candidates(self, *, scope: str, owner: str, limit: int = 64) -> list[dict[str, Any]]:
        if not 1 <= limit <= CAPS["chunks"]:
            raise ValueError("candidate limit must be in 1..128")
        rows = self.sql.execute(
            "SELECT * FROM candidates WHERE scope=? AND owner=? ORDER BY created,id LIMIT ?",
            (scope, owner, limit),
        )
        output = []
        for row in rows:
            refs = [item[0] for item in self.sql.execute(
                "SELECT evidence FROM crefs WHERE candidate=? ORDER BY evidence", (row["id"],))]
            output.append({**dict(row), "evidence": refs,
                           "stale": any(self.stale(scope, owner, item) for item in refs)})
        return output

    def fact(self, key: str, claim: str, refs: list[str], *, scope: str,
             owner: str, method: str, start: str,
             end: str | None = None,
             replace: bool = False, rollback: str | None = None,
             policy: dict[str, Any] | None = None,
             private: bool = False,
             created: str | None = None) -> str:
        allowed(policy, scope, owner, private)
        bounded(key, "fact key")
        bounded(claim, "fact claim")
        bounded(method, "validation method")
        refs = listed(refs, "evidence references")
        if not refs:
            raise ValueError("validated facts require a key, claim, method and evidence")
        start = stamp(start)
        end = stamp(end, True) if end else None
        if end is not None and end < start:
            raise ValueError("validity interval ends before it starts")
        refs = sorted(set(refs))
        for item in refs:
            self.need(scope, owner, item)
        prior = self.sql.execute(
            "SELECT id,version FROM facts WHERE scope=? AND owner=? AND key=? ORDER BY version DESC LIMIT 1",
            (scope, owner, key),
        ).fetchone()
        supersedes = prior["id"] if replace and prior else None
        if rollback is not None:
            self.need(scope, owner, rollback, "fact")
            old = self.sql.execute("SELECT key,claim FROM facts WHERE id=?", (rollback,)).fetchone()
            if old["key"] != key or old["claim"] != claim:
                raise ValueError("rollback must restore a prior claim for the same key")
        when = stamp(created)
        number = int(prior["version"] + 1) if prior else 1
        item = digest(encode([scope, owner, key, number, claim, refs, start, end,
                              method, when, supersedes, rollback]))
        with self.sql:
            self.sql.execute(
                "INSERT OR IGNORE INTO facts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (item, scope, owner, key, claim, number, start, end, method,
                 when, supersedes, rollback),
            )
            self.sql.executemany(
                "INSERT OR IGNORE INTO frefs VALUES(?,?)", [(item, ref) for ref in refs]
            )
        return item

    def stale(self, scope: str, owner: str, event: str) -> bool:
        return self.sql.execute(
            "SELECT 1 FROM events WHERE scope=? AND owner=? AND target=? AND kind IN ('invalidate','supersede') LIMIT 1",
            (scope, owner, event),
        ).fetchone() is not None

    def facts(self, *, scope: str, owner: str, key: str | None = None,
              at: str | None = None, limit: int = 64) -> list[dict[str, Any]]:
        if not 1 <= limit <= CAPS["chunks"]:
            raise ValueError("fact limit must be in 1..128")
        when = stamp(at, True)
        query = "SELECT * FROM facts WHERE scope=? AND owner=?"
        args: list[Any] = [scope, owner]
        if key is not None:
            query += " AND key=?"
            args.append(key)
        query += " ORDER BY key,version,id LIMIT ?"
        args.append(limit)
        raw = [dict(row) for row in self.sql.execute(query, args)]
        refs = {}
        invalid = set()
        for row in raw:
            refs[row["id"]] = [item[0] for item in self.sql.execute(
                "SELECT evidence FROM frefs WHERE fact=? ORDER BY evidence", (row["id"],))]
        invalid.update(item[0] for item in self.sql.execute(
            "SELECT target FROM events WHERE scope=? AND owner=? AND kind='factinvalidate'",
            (scope, owner),
        ))
        superseded = {row["supersedes"] for row in raw if row["supersedes"]}
        live = []
        for row in raw:
            end = row["end"]
            row["evidence"] = refs[row["id"]]
            row["stale"] = (when < row["start"] or (end is not None and when > end)
                             or any(self.stale(scope, owner, item) for item in row["evidence"]))
            row["invalidated"] = row["id"] in invalid
            row["superseded"] = row["id"] in superseded
            if not any((row["stale"], row["invalidated"], row["superseded"])):
                live.append(row)
        conflicts = set()
        for index, left in enumerate(live):
            for right in live[index + 1:]:
                if left["key"] != right["key"] or left["claim"] == right["claim"]:
                    continue
                low = max(left["start"], right["start"])
                ends = [value for value in (left["end"], right["end"]) if value]
                high = min(ends) if ends else None
                if high is None or low <= high:
                    conflicts.update((left["id"], right["id"]))
        for row in raw:
            row["conflict"] = row["id"] in conflicts
            row["status"] = ("invalidated" if row["invalidated"] else
                             "superseded" if row["superseded"] else
                             "stale" if row["stale"] else
                             "conflict" if row["conflict"] else "active")
            row["rollback"] = row["rollback"]
        return raw

    def skill(self, name: str, preconditions: str, procedure: str,
              expected: str, tested: list[str], counterexamples: list[str],
              validation: dict[str, Any], *, scope: str, owner: str,
              policy: dict[str, Any] | None = None,
              private: bool = False,
              created: str | None = None) -> str:
        allowed(policy, scope, owner, private)
        for value, field in ((name, "skill name"), (preconditions, "preconditions"),
                             (procedure, "procedure"), (expected, "expected outcome")):
            bounded(value, field)
        if not isinstance(validation, dict):
            raise ValueError("skill validation report must be an object")
        tested = listed(tested, "tested task families")
        counterexamples = listed(counterexamples, "counterexamples")
        if len(encode(validation).encode("utf-8")) > CAPS["meta"]:
            raise ValueError("skill validation report exceeds its byte limit")
        tested = sorted(set(tested))
        counterexamples = list(counterexamples)
        status = "validated" if (validation.get("independent") is True
                                  and validation.get("passed") is True
                                  and len(tested) >= 2) else "candidate"
        when = stamp(created)
        prior = self.sql.execute(
            "SELECT id,version FROM skills WHERE scope=? AND owner=? AND name=? ORDER BY version DESC LIMIT 1",
            (scope, owner, name),
        ).fetchone()
        number = int(prior["version"] + 1) if prior else 1
        previous = prior["id"] if prior else None
        item = digest(encode([scope, owner, name, number, preconditions, procedure,
                              expected, tested, counterexamples, validation, when]))
        with self.sql:
            self.sql.execute(
                "INSERT OR IGNORE INTO skills VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (item, scope, owner, name, number, preconditions, procedure,
                 expected, encode(tested), encode(counterexamples), encode(validation),
                 status, when, previous),
            )
        return item

    def skills(self, *, scope: str, owner: str, name: str | None = None,
               limit: int = 64) -> list[dict[str, Any]]:
        if not 1 <= limit <= CAPS["chunks"]:
            raise ValueError("skill limit must be in 1..128")
        query = "SELECT * FROM skills WHERE scope=? AND owner=?"
        args: list[Any] = [scope, owner]
        if name is not None:
            query += " AND name=?"
            args.append(name)
        query += " ORDER BY name,version LIMIT ?"
        args.append(limit)
        rows = []
        for row in self.sql.execute(query, args):
            item = dict(row)
            item["tested"] = json.loads(item["tested"])
            item["counterexamples"] = json.loads(item["counterexamples"])
            item["validation"] = json.loads(item["validation"])
            rows.append(item)
        old = {item["previous"] for item in rows if item["previous"]}
        for item in rows:
            item["superseded"] = item["id"] in old
            item["current"] = not item["superseded"]
        return rows

    def request(self, scope: str, owner: str, *, calls: int = 2,
                bytes: int = 8192, chunks: int = 4,
                tokens: int = 2048, wall: float = 0.1) -> "Request":
        return Request(self, scope, owner, calls, bytes, chunks, tokens, wall)

    def verify(self, event: str, *, scope: str, owner: str) -> dict[str, Any]:
        row = self.need(scope, owner, event)
        record = self.sql.execute("SELECT * FROM events WHERE id=?", (event,)).fetchone()
        offset = 0
        hasher = hashlib.sha256()
        count = 0
        good = True
        for part in self.sql.execute(
            "SELECT start,end,size,hash,text FROM chunks WHERE event=? ORDER BY part", (event,)
        ):
            raw = part["text"].encode("utf-8")
            good = good and part["start"] == offset and part["end"] == offset + len(raw)
            good = good and len(raw) == part["size"]
            good = good and hashlib.sha256(raw).hexdigest() == part["hash"]
            offset = part["end"]
            count += 1
            hasher.update(raw)
        good = good and hasher.hexdigest() == record["hash"]
        return {"event": row["id"], "valid": good, "chunks": count,
                "bytes": offset, "hash": hasher.hexdigest(), "expected": record["hash"]}

    def size(self) -> int:
        self.sql.execute("PRAGMA wal_checkpoint(PASSIVE)")
        if self.path == ":memory:":
            return 0
        path = Path(self.path)
        return path.stat().st_size + (path.with_name(path.name + "-wal").stat().st_size
                                      if path.with_name(path.name + "-wal").exists() else 0)

    def integrity(self) -> str:
        return str(self.sql.execute("PRAGMA integrity_check").fetchone()[0])

    def close(self) -> None:
        if self.sql is not None:
            self.sql.close()
            self.sql = None

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, kind: Any, value: Any, trace: Any) -> None:
        self.close()


class Request:
    def __init__(self, memory: Memory, scope: str, owner: str,
                 calls: int, bytes: int, chunks: int, tokens: int, wall: float):
        if (not isinstance(scope, str) or not scope or len(scope) > 256
                or not isinstance(owner, str) or not owner or len(owner) > 256):
            raise ValueError("scope and owner are required")
        for value, name in ((calls, "calls"), (bytes, "bytes"),
                             (chunks, "chunks"), (tokens, "tokens")):
            cap = int(CAPS[name])
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= cap:
                raise ValueError(f"{name} budget must be in 0..{cap}")
        if (not isinstance(wall, (int, float)) or isinstance(wall, bool)
                or not 0 < wall <= CAPS["wall"]):
            raise ValueError(f"wall budget must be in (0, {CAPS['wall']}]")
        self.memory = memory
        self.scope = scope
        self.owner = owner
        self.budget = {"calls": calls, "bytes": bytes, "chunks": chunks,
                       "tokens": tokens, "wall": float(wall)}
        self.used = {"calls": 0, "bytes": 0, "chunks": 0, "tokens": 0,
                     "wall": 0.0}
        self.started = time.monotonic()

    def remain(self) -> float:
        return self.budget["wall"] - (time.monotonic() - self.started)

    def check(self) -> None:
        self.used["wall"] = time.monotonic() - self.started
        if self.remain() <= 0:
            raise Budget("request wall-time budget exhausted")

    def search(self, query: str, k: int = 3, stale: bool = True) -> list[dict[str, Any]]:
        if self.used["calls"] >= self.budget["calls"]:
            raise Budget("request retrieval-call budget exhausted")
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= CAPS["chunks"]:
            raise ValueError("k must be in 1..128")
        if not isinstance(query, str) or len(query.encode("utf-8")) > CAPS["query"]:
            raise ValueError("query must be text within the request byte limit")
        terms = sorted(set(words(query)))
        if len(terms) > CAPS["terms"]:
            raise ValueError(f"query may contain at most {CAPS['terms']} unique terms")
        self.check()
        self.used["calls"] += 1
        if not terms or self.used["chunks"] >= self.budget["chunks"]:
            return []
        cap = min(k, self.budget["chunks"] - self.used["chunks"])
        rows = self.find(terms, cap, stale)
        output = []
        for row in rows:
            self.check()
            if (row["size"] > self.budget["bytes"] - self.used["bytes"]
                    or row["tokens"] > self.budget["tokens"] - self.used["tokens"]):
                continue
            chunk = self.memory.sql.execute(
                "SELECT text FROM chunks WHERE id=? AND scope=? AND owner=?",
                (row["chunk"], self.scope, self.owner),
            ).fetchone()
            if chunk is None:
                continue
            item = {key: row[key] for key in row.keys() if key != "chunk"}
            item["meta"] = json.loads(item["meta"])
            item["stale"] = bool(item["stale"])
            item["text"] = chunk["text"]
            item["trusted"] = False
            item["trust"] = "untrusted"
            output.append(item)
            self.used["bytes"] += row["size"]
            self.used["tokens"] += row["tokens"]
            self.used["chunks"] += 1
        self.used["wall"] = time.monotonic() - self.started
        return output

    def find(self, terms: list[str], cap: int, stale: bool) -> list[sqlite3.Row]:
        db = self.memory.sql
        db.set_progress_handler(lambda: int(self.remain() <= 0), 500)
        try:
            if self.memory.fts:
                match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
                sql = """
                SELECT c.id AS chunk,c.key,c.event,c.part,c.start,c.end,c.size,c.tokens,c.hash AS mark,
                       e.scope,e.owner,e.source,e.uri,e.version,e.acquired,e.licence,e.hash,
                       e.meta,fts.rank AS score,
                       EXISTS(SELECT 1 FROM events z WHERE z.scope=e.scope AND z.owner=e.owner
                              AND z.target=e.id AND z.kind IN ('invalidate','supersede')) AS stale
                FROM fts JOIN chunks c ON c.id=fts.rowid JOIN events e ON e.id=c.event
                WHERE fts MATCH ? AND c.scope=? AND c.owner=?
                  AND (? OR NOT EXISTS(SELECT 1 FROM events z WHERE z.scope=e.scope AND z.owner=e.owner
                                       AND z.target=e.id AND z.kind IN ('invalidate','supersede')))
                ORDER BY score ASC,c.key ASC LIMIT ?
                """
                args = (match, self.scope, self.owner, int(stale), cap)
            else:
                marks = ",".join("?" for _ in terms)
                sql = f"""
                SELECT c.id AS chunk,c.key,c.event,c.part,c.start,c.end,c.size,c.tokens,c.hash AS mark,
                       e.scope,e.owner,e.source,e.uri,e.version,e.acquired,e.licence,e.hash,
                       e.meta,SUM(t.count) AS score,
                       EXISTS(SELECT 1 FROM events z WHERE z.scope=e.scope AND z.owner=e.owner
                              AND z.target=e.id AND z.kind IN ('invalidate','supersede')) AS stale
                FROM terms t JOIN chunks c ON c.id=t.chunk JOIN events e ON e.id=c.event
                WHERE t.scope=? AND t.owner=? AND t.term IN ({marks})
                  AND (? OR NOT EXISTS(SELECT 1 FROM events z WHERE z.scope=e.scope AND z.owner=e.owner
                                       AND z.target=e.id AND z.kind IN ('invalidate','supersede')))
                GROUP BY c.id ORDER BY score DESC,c.key ASC LIMIT ?
                """
                args = (self.scope, self.owner, *terms, int(stale), cap)
            try:
                return list(db.execute(sql, args))
            except sqlite3.OperationalError as exc:
                if "interrupt" in str(exc).lower():
                    raise Budget("request wall-time budget exhausted") from exc
                raise
        finally:
            db.set_progress_handler(None, 0)
