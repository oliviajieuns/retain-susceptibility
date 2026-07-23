"""Scorer registry and shared probe types.

Registry names are fixed identifiers used in configs, result tables, and
tests: fd, jvp, vmap_graddot, streaming_backward, knn_feature, knn_embed,
knn_lexical, grad_norm, last_layer, random_dir, random_rank, fd_constrained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from rsus.blocks import BlockSpec
from rsus.costs import CostRecord
from rsus.data.base import Request


@dataclass(frozen=True)
class ProbeSpec:
    """Declared probe: block, step, loss, seed. Part of the score definition
    (paper Sec. 3) and disclosed with every comparison."""

    block: BlockSpec
    eta: float
    loss: str = "seq_mean_answer_nll"
    seed: int = 0
    batch_size: int = 8
    n_dirs: int = 8  # random probe directions for norm-estimating scorers (fd_norm)
    norm_eta: float | None = None  # separate (larger) FD radius for fd_norm; random
    representation_k: int = 5
    representation_layer: int = -1
    representation_pooling: str = "answer_mean"
    # projections give small g.v, so at the alignment eta the loss difference sits
    # near the fp32 cancellation floor and the squared estimator is noise-inflated.
    # None -> fall back to eta.


@dataclass
class ScoreProfile:
    request_id: str
    scorer: str
    scores: dict[str, float]
    spec: ProbeSpec
    cost: CostRecord = field(default_factory=CostRecord)
    # Raw, scorer-specific evidence needed for integrity checks and numerical
    # ablations.  Production runners must persist this before releasing a
    # sealed audit; aggregates alone are not sufficient for paper claims.
    artifacts: dict[str, object] = field(default_factory=dict)

    def ranking(self) -> list[str]:
        """Candidate ids by descending score, deterministic tie-break by id."""
        return sorted(self.scores, key=lambda cid: (-self.scores[cid], cid))


ScorerFn = Callable[[torch.nn.Module, Request, ProbeSpec], ScoreProfile]
_REGISTRY: dict[str, ScorerFn] = {}


def register(name: str) -> Callable[[ScorerFn], ScorerFn]:
    """Register a scorer. Every scorer runs with the model in eval mode
    (dropout disabled — part of the declared probe definition) and the prior
    training flag restored afterwards."""

    def deco(fn: ScorerFn) -> ScorerFn:
        if name in _REGISTRY:
            raise ValueError(f"duplicate scorer name: {name}")

        def wrapped(model: torch.nn.Module, request: Request, spec: ProbeSpec) -> ScoreProfile:
            was_training = model.training
            model.eval()
            try:
                return fn(model, request, spec)
            finally:
                model.train(was_training)

        wrapped.__name__ = getattr(fn, "__name__", name)
        _REGISTRY[name] = wrapped
        return wrapped

    return deco


def get_scorer(name: str) -> ScorerFn:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown scorer {name!r}; known: {sorted(_REGISTRY)}") from None


def scorer_names() -> list[str]:
    return sorted(_REGISTRY)
