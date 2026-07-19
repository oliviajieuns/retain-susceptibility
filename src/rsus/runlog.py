"""Append-only JSONL event logging for runs."""
from __future__ import annotations

import json
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields) -> None:
        rec = {"event": event, **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
