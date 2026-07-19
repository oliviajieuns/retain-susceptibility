"""Controlled substrate with ground-truth adjacency by construction.

Adjacent candidates are near-duplicates of forget examples (identical answer,
a few corrupted prompt tokens), so their loss gradients must align with the
canonical forget direction; remote candidates are independent sequences. The
generator returns the request plus ground-truth labels, supporting mechanism
tests (probe should separate the classes) and, at N4, damage-prediction
sanity checks with entanglement level as the knob.
"""
from __future__ import annotations

import torch

from rsus.data.base import CandidateUniverse, Example, Request
from rsus.losses import IGNORE


def make_substrate(
    seed: int = 0,
    n_forget: int = 4,
    n_adjacent: int = 8,
    n_remote: int = 8,
    seq_len: int = 16,
    prompt_len: int = 8,
    vocab: int = 128,
    corrupt_prompt_tokens: int = 2,
) -> tuple[Request, dict[str, str]]:
    gen = torch.Generator().manual_seed(seed)

    def rand_seq() -> torch.Tensor:
        return torch.randint(3, vocab, (seq_len,), generator=gen)

    def to_example(eid: str, ids: torch.Tensor, group: str) -> Example:
        labels = ids.clone()
        labels[:prompt_len] = IGNORE
        return Example(example_id=eid, input_ids=ids, labels=labels, group=group)

    forget = [to_example(f"f{i:02d}", rand_seq(), "author-forget") for i in range(n_forget)]

    truth: dict[str, str] = {}
    cands: list[Example] = []
    for i in range(n_adjacent):
        src = forget[i % n_forget]
        ids = src.input_ids.clone()
        pos = torch.randperm(prompt_len, generator=gen)[:corrupt_prompt_tokens]
        ids[pos] = torch.randint(3, vocab, (corrupt_prompt_tokens,), generator=gen)
        eid = f"adj{i:02d}"
        cands.append(to_example(eid, ids, group=eid))
        truth[eid] = "adjacent"
    for i in range(n_remote):
        eid = f"rem{i:02d}"
        cands.append(to_example(eid, rand_seq(), group=eid))
        truth[eid] = "remote"

    req = Request.build(f"substrate-{seed}", forget, CandidateUniverse.freeze(cands))
    return req, truth
