"""Run one bounded training segment on the frozen interaction shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.foundation import trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int)
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config = trainer.load()
    if args.steps is not None:
        config["train"]["steps"] = args.steps
    if args.sequence is not None:
        config["train"]["sequence"] = args.sequence
    result = trainer.run(config, output=args.output, resume=args.resume,
                         stop=args.stop, init=args.init)
    raw = json.dumps(result, sort_keys=True, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as file:
            file.write(raw + "\n")
    print(raw)


if __name__ == "__main__":
    main()
