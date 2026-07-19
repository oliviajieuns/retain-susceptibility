"""Baseline scorers: controls, similarity constructors, and probe variants.

Implemented: grad_norm, random_dir, random_rank, knn_lexical, knn_feature
(model-representation kNN), last_layer (forward-only closed-form head
gradient dot), fd_constrained (canonical direction projected off the mean
near-zero-band gradient). knn_embed needs the external sentence encoder
(open decision D5) and remains a stub.
"""
from __future__ import annotations

import torch

from rsus.blocks import (
    BlockSpec,
    grads_of,
    only_block_grads,
    vec_dot,
    vec_norm,
    vec_randn_like,
    vec_scale,
    vec_unit,
)
from rsus.costs import CostRecord, Meter
from rsus.data.base import Request, collate
from rsus.losses import IGNORE, _shifted_nll, seq_mean_answer_nll
from rsus.probe.base import ProbeSpec, ScoreProfile, register
from rsus.probe.finite_diff import canonical_forget_direction, fd_scores_along


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


def _mean_pooled_reps(model: torch.nn.Module, batches, rec: CostRecord) -> dict[str, torch.Tensor]:
    reps: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for batch in batches:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            mask = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1)
            rec.fwd_passes += 1
            rec.tokens_fwd += int(batch["attention_mask"].sum())
            for eid, v in zip(batch["example_ids"], pooled):
                reps[eid] = v
    return reps


@register("knn_feature")
def score_knn_feature(model: torch.nn.Module, request: Request, spec: ProbeSpec) -> ScoreProfile:
    """Representation-similarity baseline: mean cosine to the k nearest
    forget examples in mean-pooled last-hidden-state space of the model
    under audit."""
    rec = CostRecord()
    with Meter(rec):
        f_reps = _mean_pooled_reps(model, request.forget_batches(spec.batch_size), rec)
        c_reps = _mean_pooled_reps(model, request.universe.batches(spec.batch_size), rec)
        F = torch.nn.functional.normalize(torch.stack(list(f_reps.values())), dim=1)
        k = min(5, F.shape[0])
        scores: dict[str, float] = {}
        for eid, v in c_reps.items():
            sims = F @ torch.nn.functional.normalize(v, dim=0)
            scores[eid] = float(sims.topk(k).values.mean())
    return ScoreProfile(request.request_id, "knn_feature", scores, spec, rec)


def _head_grads(model: torch.nn.Module, batch: dict, rec: CostRecord) -> dict[str, torch.Tensor]:
    """Forward-only closed-form per-example gradient of the mean answer NLL
    w.r.t. lm_head.weight: (1/T) sum_t (softmax(logit_t) - onehot(y_t)) h_t^T."""
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        rec.fwd_passes += 1
        rec.tokens_fwd += int(batch["attention_mask"].sum())
        h = out.hidden_states[-1][:, :-1, :]
        logits = out.logits[:, :-1, :]
        targets = batch["labels"][:, 1:]
        mask = targets != IGNORE
        delta = torch.softmax(logits, dim=-1)
        delta.scatter_add_(
            -1, targets.clamp_min(0).unsqueeze(-1), -torch.ones_like(delta[..., :1])
        )
        delta = delta * mask.unsqueeze(-1).to(delta.dtype)
        grads: dict[str, torch.Tensor] = {}
        for i, eid in enumerate(batch["example_ids"]):
            grads[eid] = delta[i].transpose(0, 1) @ h[i] / int(mask[i].sum())
    return grads


@register("last_layer")
def score_last_layer(model: torch.nn.Module, request: Request, spec: ProbeSpec) -> ScoreProfile:
    """Closed-form gradient-alignment score restricted to the unembedding
    layer: cheap forward-only comparator that ignores the declared block."""
    rec = CostRecord()
    with Meter(rec):
        acc: torch.Tensor | None = None
        n = len(request.forget)
        for batch in request.forget_batches(spec.batch_size):
            for g in _head_grads(model, batch, rec).values():
                acc = g if acc is None else acc + g
        ghead = acc / n
        ghead = ghead / ghead.norm()
        scores: dict[str, float] = {}
        for batch in request.universe.batches(spec.batch_size):
            for eid, g in _head_grads(model, batch, rec).items():
                scores[eid] = float((g * ghead).sum())
    return ScoreProfile(request.request_id, "last_layer", scores, spec, rec)


@register("fd_constrained")
def score_fd_constrained(model: torch.nn.Module, request: Request, spec: ProbeSpec) -> ScoreProfile:
    """Sensitivity arm: project g_hat off the mean gradient of a seeded
    near-zero-band sample before stepping, so the probe cannot conflate
    adjacency with a direction shared across the retained pool."""
    rec = CostRecord()
    with Meter(rec):
        sel = spec.block.select(model)
        ghat = canonical_forget_direction(model, request, spec, rec)
        raw = fd_scores_along(model, request, spec, ghat, rec)

        abs_s = torch.tensor([abs(v) for v in raw.values()], dtype=torch.float64)
        tau = torch.quantile(abs_s, 0.5).item()
        band = sorted(c for c, v in raw.items() if abs(v) <= tau)
        gen = torch.Generator().manual_seed(spec.seed)
        take = [band[i] for i in torch.randperm(len(band), generator=gen).tolist()[:8]]
        by_id = {e.example_id: e for e in request.universe.examples}

        model.zero_grad(set_to_none=True)
        with only_block_grads(model, sel):
            batch = collate([by_id[c] for c in take])
            seq_mean_answer_nll(model, batch).mean().backward()
            rec.fwd_passes += 1
            rec.bwd_passes += 1
            rec.tokens_bwd += int(batch["attention_mask"].sum())
        rhat = vec_unit(grads_of(sel))
        model.zero_grad(set_to_none=True)

        proj = {n: ghat[n] - vec_dot(ghat, rhat) * rhat[n] for n in ghat}
        norm = float(vec_norm(proj))
        if norm < 1e-8:
            rec.notes["fallback"] = "raw"  # forget direction lies in the remote span
            scores = raw
        else:
            scores = fd_scores_along(model, request, spec, vec_scale(proj, 1.0 / norm), rec)
    return ScoreProfile(request.request_id, "fd_constrained", scores, spec, rec)


def _todo(name: str, reason: str):
    @register(name)
    def _stub(model, request, spec):  # noqa: ANN001
        raise NotImplementedError(f"scorer {name!r}: {reason}")

    return _stub


_todo("knn_embed", "external sentence encoder pending (open decision D5)")
