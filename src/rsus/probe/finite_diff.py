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
    vec_randn_like,
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


@register("fd_norm")
def score_fd_norm(model: torch.nn.Module, request: Request, spec: ProbeSpec) -> ScoreProfile:
    """Backward-free gradient-magnitude profile. For seeded random unit
    directions v_1..v_K in block B, E_v[(d ell_c / d v)^2] = ||grad_B ell_c||^2
    / dim(B), so the mean squared central difference across K = spec.n_dirs
    directions ranks candidates by their own gradient norm — the empirically
    predictive quantity on the 1.5B gate (grad_norm), at 2K batched forward
    sweeps and zero per-candidate backwards. Relative estimator variance is
    2/K, independent of dim(B)."""
    import dataclasses as _dc
    rec = CostRecord()
    # random projections give small g.v, so fd_norm needs a larger FD radius than
    # the alignment probe or catastrophic cancellation in l(+)-l(-) inflates the
    # squared estimate (fp32). Use spec.norm_eta when set (see appendix eta sweep).
    sp = _dc.replace(spec, eta=spec.norm_eta) if spec.norm_eta is not None else spec
    with Meter(rec):
        sel = spec.block.select(model)
        gen = torch.Generator().manual_seed(spec.seed)
        acc: dict[str, float] = {}
        for _ in range(spec.n_dirs):
            direction = vec_unit(vec_randn_like(sel, gen))
            deriv = fd_scores_along(model, request, sp, direction, rec)
            for cid, val in deriv.items():
                acc[cid] = acc.get(cid, 0.0) + val * val
        scores = {cid: s / spec.n_dirs for cid, s in acc.items()}
    return ScoreProfile(request.request_id, "fd_norm", scores, spec, rec)


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
