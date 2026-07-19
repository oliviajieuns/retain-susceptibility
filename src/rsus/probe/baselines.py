"""Baseline scorers: controls and similarity constructors.

Implemented here: grad_norm, random_dir, random_rank, knn_lexical.
Encoder-dependent constructors (knn_feature, knn_embed) and the last-layer
closed form land with the partition milestone (N2); fd_constrained (project
g_hat off the mean remote gradient) needs pools and also lands at N2.
"""
from __future__ import annotations

import torch

from rsus.blocks import grads_of, only_block_grads, vec_norm, vec_randn_like, vec_unit
from rsus.costs import CostRecord, Meter
from rsus.data.base import Request, collate
from rsus.losses import IGNORE, seq_mean_answer_nll
from rsus.probe.base import ProbeSpec, ScoreProfile, register
from rsus.probe.finite_diff import fd_scores_along


@register("grad_norm")
def score_grad_norm(model: torch.nn.Module, request: Request, spec: ProbeSpec) -> ScoreProfile:
    """Direction-free control: block-gradient norm per candidate."""
    rec = CostRecord()
    with Meter(rec):
        sel = spec.block.select(model)
        scores: dict[str, float] = {}
        with only_block_grads(model, sel):
            for ex in request.universe.examples:
                batch = collate([ex])
                model.zero_grad(set_to_none=True)
                seq_mean_answer_nll(model, batch)[0].backward()
                rec.fwd_passes += 1
                rec.bwd_passes += 1
                scores[ex.example_id] = float(vec_norm(grads_of(sel)))
        model.zero_grad(set_to_none=True)
    return ScoreProfile(request.request_id, "grad_norm", scores, spec, rec)


@register("random_dir")
def score_random_dir(model: torch.nn.Module, request: Request, spec: ProbeSpec) -> ScoreProfile:
    """FD probe along a seeded random unit direction (direction-specificity
    control: same machinery, no forget information)."""
    rec = CostRecord()
    with Meter(rec):
        sel = spec.block.select(model)
        gen = torch.Generator().manual_seed(spec.seed)
        direction = vec_unit(vec_randn_like(sel, gen))
        scores = fd_scores_along(model, request, spec, direction, rec)
    return ScoreProfile(request.request_id, "random_dir", scores, spec, rec)


@register("random_rank")
def score_random_rank(model: torch.nn.Module, request: Request, spec: ProbeSpec) -> ScoreProfile:
    """Pure chance floor: seeded random scores."""
    gen = torch.Generator().manual_seed(spec.seed)
    vals = torch.rand(len(request.universe), generator=gen).tolist()
    scores = {ex.example_id: v for ex, v in zip(request.universe.examples, vals)}
    return ScoreProfile(request.request_id, "random_rank", scores, spec, CostRecord())


@register("knn_lexical")
def score_knn_lexical(model: torch.nn.Module, request: Request, spec: ProbeSpec) -> ScoreProfile:
    """Static lexical similarity: Jaccard overlap between a candidate's answer
    token set and the union of forget-set answer tokens."""
    rec = CostRecord()
    with Meter(rec):
        forget_tokens: set[int] = set()
        for ex in request.forget:
            forget_tokens |= set(ex.input_ids[ex.labels != IGNORE].tolist())
        scores: dict[str, float] = {}
        for ex in request.universe.examples:
            cand = set(ex.input_ids[ex.labels != IGNORE].tolist())
            union = cand | forget_tokens
            scores[ex.example_id] = len(cand & forget_tokens) / len(union) if union else 0.0
    return ScoreProfile(request.request_id, "knn_lexical", scores, spec, rec)


def _todo(name: str, milestone: str):
    @register(name)
    def _stub(model, request, spec):  # noqa: ANN001
        raise NotImplementedError(f"scorer {name!r} lands at milestone {milestone}")

    return _stub


_todo("knn_feature", "N2")
_todo("knn_embed", "N2")
_todo("last_layer", "N2")
_todo("fd_constrained", "N2")
