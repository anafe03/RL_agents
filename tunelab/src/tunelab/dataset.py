"""Synthetic-dataset loader for tunelab.

Each line in train.jsonl / eval.jsonl is one example:
  {"input": "<free-text patient summary>", "expected": {<PriorAuthExtraction dict>}}

Both files are checked into the repo so the eval harness + reward tests
run deterministically without any external network or model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Example:
    input: str
    expected: dict


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_split(path: Path | str) -> list[Example]:
    """Load a JSONL file into Example dataclasses."""
    out: list[Example] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        out.append(Example(input=obj["input"], expected=obj["expected"]))
    return out


def load_train() -> list[Example]:
    return load_split(DATA_DIR / "train.jsonl")


def load_eval() -> list[Example]:
    return load_split(DATA_DIR / "eval.jsonl")


def iter_dataset(path: Path | str) -> Iterator[Example]:
    yield from load_split(path)
