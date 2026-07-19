"""Finite-difference susceptibility probe (paper eq:fdscore).

One backward pass over the forget set defines the canonical unit ascent
direction g_hat in block B; two batched forward sweeps over the candidate
universe at theta +/- eta*g_hat give the central-difference estimate of each
candidate's directional loss derivative. No candidate-side backward, no
per-candidate gradient materialization; cost is linear in candidate tokens.
"""
from __future__ import annotations

import torch

from rsus.blocks import (
    ParamVec,
    grads_of,
    load_params_,
    only_block_grads,
    save_params,
    set_perturbed_,
    vec_unit,
)
from rsus.costs import CostRecord, Meter
from rsus.data.base import Request
from rsus.losses import seq_mean_answer_nll
from rsus.probe.base import ProbeSpec, ScoreProfile, register


def canonical_forget_direction(
    model: torch.nn.Module, request: Request, spec: ProbeSpec, rec: CostRecord
) -> ParamVec:
    """g_hat = unit gradient of the mean forget loss in block B (1 backward)."""
    sel = spec.block.select(model)
    n_total = len(request.forget)
    model.zero_grad(set_to_none=True)
    with only_block_grads(model, sel):
        for batch in request.forget_batches(spec.batch_size):
            losses = seq_mean_answer_nll(model, batch)
            (losses.sum() / n_total).backward()
            rec.fwd_passes += 1
            rec.bwd_passes += 1
            n_tok = int(batch["attention_mask"].sum())
            rec.tokens_fwd += n_tok
            rec.tokens_bwd += n_tok
    g = grads_of(sel)
    model.zero_grad(set_to_none=True)
    return vec_unit(g)


def sweep_losses(
    model: torch.nn.Module, request: Request, spec: ProbeSpec, rec: CostRecord
) -> dict[str, float]:
    """One no-grad forward sweep over the candidate universe at current theta."""
    out: dict[str, float] = {}
    with torch.no_grad():
        for batch in request.universe.batches(spec.batch_size):
            losses = seq_mean_answer_nll(model, batch)
            rec.fwd_passes += 1
            rec.tokens_fwd += int(batch["attention_mask"].sum())
            for eid, val in zip(batch["example_ids"], losses.tolist()):
                out[eid] = val
    return out


def fd_scores_along(
    model: torch.nn.Module, request: Request, spec: ProbeSpec, direction: ParamVec, rec: CostRecord
) -> dict[str, float]:
    """Central difference along an arbitrary unit direction; restores theta
    bit-exactly (perturbations are always computed from the saved copy)."""
    sel = spec.block.select(model)
    saved = save_params(sel)
    try:
        set_perturbed_(sel, saved, direction, +spec.eta)
        plus = sweep_losses(model, request, spec, rec)
        set_perturbed_(sel, saved, direction, -spec.eta)
        minus = sweep_losses(model, request, spec, rec)
    finally:
        load_params_(sel, saved)
    return {cid: (plus[cid] - minus[cid]) / (2.0 * spec.eta) for cid in plus}


@register("fd")
def score_fd(model: torch.nn.Module, request: Request, spec: ProbeSpec) -> ScoreProfile:
    rec = CostRecord()
    with Meter(rec):
        ghat = canonical_forget_direction(model, request, spec, rec)
        scores = fd_scores_along(model, request, spec, ghat, rec)
    return ScoreProfile(request.request_id, "fd", scores, spec, rec)


@register("one_sided")
def score_one_sided(model: torch.nn.Module, request: Request, spec: ProbeSpec) -> ScoreProfile:
    """First-order control (paper Sec. 4 'Numerical Target'): one perturbed
    sweep against the origin, {ell(theta0 + eta*ghat) - ell(theta0)} / eta.
    Also backward-free, but O(eta) truncation versus the symmetric O(eta^2);
    both origin and perturbed forwards are counted in the measured cost."""
    rec = CostRecord()
    with Meter(rec):
        ghat = canonical_forget_direction(model, request, spec, rec)
        sel = spec.block.select(model)
        base = sweep_losses(model, request, spec, rec)
        saved = save_params(sel)
        try:
            set_perturbed_(sel, saved, ghat, +spec.eta)
            plus = sweep_losses(model, request, spec, rec)
        finally:
            load_params_(sel, saved)
        scores = {cid: (plus[cid] - base[cid]) / spec.eta for cid in base}
    return ScoreProfile(request.request_id, "one_sided", scores, spec, rec)
