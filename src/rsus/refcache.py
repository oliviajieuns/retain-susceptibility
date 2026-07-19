"""Stage-1 exit-gate reference cache.

Per-sequence and per-token forget losses at the accepted Stage-1 output,
cached as scalar vectors with explicit identity maps. The (example_id,
answer_pos) token index is the ONLY legal alignment between this cache and
Stage-2 evaluation; both sides are produced by the same iteration helpers
below, and alignment is asserted, never assumed. Caching removes the frozen
reference-model copy entirely.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from rsus.data.base import Request
from rsus.losses import seq_mean_answer_nll, token_answer_nll


def forget_seq_losses(
    model: torch.nn.Module, request: Request, batch_size: int, grad: bool = False
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Per-sequence forget losses in canonical (forget_batches) order."""
    ids: list[str] = []
    parts: list[torch.Tensor] = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for batch in request.forget_batches(batch_size):
            parts.append(seq_mean_answer_nll(model, batch))
            ids.extend(batch["example_ids"])
    return tuple(ids), torch.cat(parts)


def forget_tok_losses(
    model: torch.nn.Module, request: Request, batch_size: int, grad: bool = False
) -> tuple[tuple[tuple[str, int], ...], torch.Tensor]:
    """Flat answer-token forget losses with their identity index, in
    canonical order."""
    index: list[tuple[str, int]] = []
    parts: list[torch.Tensor] = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for batch in request.forget_batches(batch_size):
            flat, idx = token_answer_nll(model, batch)
            parts.append(flat)
            index.extend(idx)
    return tuple(index), torch.cat(parts)


@dataclass(frozen=True)
class RefCache:
    seq_ids: tuple[str, ...]
    seq_refs: torch.Tensor
    tok_index: tuple[tuple[str, int], ...]
    tok_refs: torch.Tensor
    floor_m: float
    sha: str


def build_ref_cache(
    model: torch.nn.Module, request: Request, batch_size: int, floor_m: float
) -> RefCache:
    seq_ids, seq_refs = forget_seq_losses(model, request, batch_size)
    tok_index, tok_refs = forget_tok_losses(model, request, batch_size)
    h = hashlib.sha256()
    h.update(repr(seq_ids).encode())
    h.update(seq_refs.double().numpy().tobytes())
    h.update(repr(tok_index).encode())
    h.update(tok_refs.double().numpy().tobytes())
    h.update(f"{floor_m:.12e}".encode())
    return RefCache(seq_ids, seq_refs.detach(), tok_index, tok_refs.detach(), floor_m, h.hexdigest())


def assert_aligned(cache: RefCache, seq_ids, tok_index) -> None:
    if tuple(seq_ids) != cache.seq_ids:
        raise ValueError("sequence identity map does not match the reference cache")
    if tuple(tok_index) != cache.tok_index:
        raise ValueError("token identity map does not match the reference cache")
