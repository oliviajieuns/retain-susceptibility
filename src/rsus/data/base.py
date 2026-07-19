"""Request abstraction shared by every benchmark adapter.

A deletion request carries the forget set and the frozen retained-candidate
universe; pools and folds are derived later (partition.py) and never mutate
these objects. Manifest hashes freeze identity before any score or outcome
is computed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterator

import torch

from rsus.losses import IGNORE


@dataclass(frozen=True)
class Example:
    example_id: str
    input_ids: torch.Tensor  # [L] long
    labels: torch.Tensor     # [L] long, prompt positions = IGNORE
    group: str = ""          # fold granularity unit (e.g. retained author)

    def n_answer_tokens(self) -> int:
        return int((self.labels != IGNORE).sum())


def collate(examples: list[Example], pad_id: int = 0) -> dict:
    max_len = max(e.input_ids.numel() for e in examples)
    B = len(examples)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels = torch.full((B, max_len), IGNORE, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    for i, e in enumerate(examples):
        L = e.input_ids.numel()
        input_ids[i, :L] = e.input_ids
        labels[i, :L] = e.labels
        attention_mask[i, :L] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "example_ids": [e.example_id for e in examples],
    }


def manifest_sha(examples: list[Example]) -> str:
    h = hashlib.sha256()
    for e in sorted(examples, key=lambda x: x.example_id):
        h.update(e.example_id.encode())
        h.update(e.group.encode())
        h.update(e.input_ids.numpy().tobytes())
        h.update(e.labels.numpy().tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class CandidateUniverse:
    """The complete frozen retain-candidate pool C(q) for one request."""

    examples: tuple[Example, ...]
    sha: str = field(default="")

    @staticmethod
    def freeze(examples: list[Example]) -> "CandidateUniverse":
        return CandidateUniverse(tuple(examples), sha=manifest_sha(examples))

    def batches(self, batch_size: int) -> Iterator[dict]:
        for i in range(0, len(self.examples), batch_size):
            yield collate(list(self.examples[i : i + batch_size]))

    def __len__(self) -> int:
        return len(self.examples)


@dataclass(frozen=True)
class Request:
    request_id: str
    forget: tuple[Example, ...]
    universe: CandidateUniverse
    forget_sha: str = field(default="")
    native_audit_ids: frozenset[str] = field(default_factory=frozenset)

    @staticmethod
    def build(
        request_id: str,
        forget: list[Example],
        universe: CandidateUniverse,
        native_audit_ids: frozenset[str] | set[str] = frozenset(),
    ) -> "Request":
        unknown = set(native_audit_ids) - {e.example_id for e in universe.examples}
        if unknown:
            raise ValueError(f"native audit ids outside the universe: {sorted(unknown)[:5]}")
        return Request(
            request_id,
            tuple(forget),
            universe,
            forget_sha=manifest_sha(forget),
            native_audit_ids=frozenset(native_audit_ids),
        )

    def forget_batches(self, batch_size: int) -> Iterator[dict]:
        for i in range(0, len(self.forget), batch_size):
            yield collate(list(self.forget[i : i + batch_size]))
