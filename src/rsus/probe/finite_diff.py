"""Finite-difference susceptibility probes (paper eq:loss-shake-identity).

fd (alignment baseline): one backward pass over the forget set defines the
canonical unit ascent direction g_hat in block B; two batched forward sweeps
over the candidate universe at theta +/- eta*g_hat give the central-difference
estimate of each candidate's directional loss derivative.

fd_norm (loss-shake energy, the headline probe): the same central-difference
machinery along R shared seeded random unit directions; the mean squared
response estimates each candidate's gradient energy (paper
eq:loss-shake-identity, up to the constant block-dimension factor). Computed
in two stages -- per-direction signed responses (fd_norm_responses), then an
offline CPU aggregation (aggregate_fd_norm) -- so R-ablation, Monte Carlo
CIs, and alternative aggregations never require re-running forward sweeps.

No candidate-side backward, no per-candidate gradient materialization; cost
is linear in candidate tokens.
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
    model: torch.nn.Module, request: Request, spec: ProbeSpec, direction: ParamVec, rec: CostRecord,
    saved: ParamVec | None = None,
) -> dict[str, float]:
    """Central difference along an arbitrary unit direction; restores theta
    bit-exactly (perturbations are always computed from the saved copy).
    Pass a pre-saved block copy via `saved` to skip the per-call clone when
    sweeping many directions."""
    sel = spec.block.select(model)
    if saved is None:
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


def fd_norm_responses(
    model: torch.nn.Module, request: Request, spec: ProbeSpec, rec: CostRecord | None = None,
) -> list[dict[str, float]]:
    """Stage 1 of the loss-shake probe: the per-direction signed responses
    delta_{x,r} = (l_x(theta+eta v_r) - l_x(theta-eta v_r)) / (2 eta) for the
    R = spec.n_dirs shared seeded directions, in direction order. This is the
    expensive part (2R batched forward sweeps); everything downstream of it is
    offline CPU arithmetic. Persist the returned list (JSON: one dict per
    direction, {candidate_id: signed delta}) to enable R-ablation, Monte Carlo
    CIs, and alternative aggregations without re-running forwards. Directions
    are reproducible from spec.seed; the radius is spec.norm_eta when set."""
    import dataclasses as _dc
    rec = rec if rec is not None else CostRecord()
    # random projections give small g.v, so fd_norm needs a larger FD radius than
    # the alignment probe or catastrophic cancellation in l(+)-l(-) inflates the
    # squared estimate (fp32). Use spec.norm_eta when set (see appendix eta sweep).
    sp = _dc.replace(spec, eta=spec.norm_eta) if spec.norm_eta is not None else spec
    sel = spec.block.select(model)
    gen = torch.Generator().manual_seed(spec.seed)
    saved = save_params(sel)
    responses: list[dict[str, float]] = []
    for _ in range(spec.n_dirs):
        direction = vec_unit(vec_randn_like(sel, gen))
        responses.append(fd_scores_along(model, request, sp, direction, rec, saved=saved))
    return responses


def aggregate_fd_norm(
    responses: list[dict[str, float]], block_dimension: int
) -> dict[str, float]:
    """Return dimension-corrected loss-shake energy from signed responses.

    The paper's score is ``d_B`` times the mean squared directional response,
    not only a rank-equivalent unscaled mean. Requiring ``block_dimension``
    keeps offline nested-``R`` analyses on the exact deployed estimand.
    """
    if isinstance(block_dimension, bool) or int(block_dimension) < 1:
        raise ValueError("block_dimension must be a positive integer")
    if not responses:
        raise ValueError("responses must be non-empty")
    acc: dict[str, float] = {}
    for deriv in responses:
        for cid, val in deriv.items():
            acc[cid] = acc.get(cid, 0.0) + val * val
    n = len(responses)
    return {cid: int(block_dimension) * s / n for cid, s in acc.items()}


@register("fd_norm")
def score_fd_norm(model: torch.nn.Module, request: Request, spec: ProbeSpec) -> ScoreProfile:
    """Backward-free gradient-energy profile (loss-shake energy, paper
    eq:loss-shake-identity). For seeded random unit directions v_1..v_K in
    block B, E_v[(d ell_c / d v)^2] = ||grad_B ell_c||^2 / dim(B), so the mean
    squared central difference across K = spec.n_dirs directions ranks
    candidates by their own gradient energy — at 2K batched forward sweeps and
    zero per-candidate backwards. Per-direction relative variance for
    uniform-sphere directions is 2(d-1)/(d+2) with d = dim(B), giving relative
    estimator variance ~ 2/K (approaching it from below in large blocks).
    Implemented as stage 1 (fd_norm_responses) + stage 2 (aggregate_fd_norm)."""
    rec = CostRecord()
    with Meter(rec):
        responses = fd_norm_responses(model, request, spec, rec)
    eta_used = spec.norm_eta if spec.norm_eta is not None else spec.eta
    block_dimension = sum(parameter.numel() for parameter in spec.block.select(model).values())
    scores = aggregate_fd_norm(responses, block_dimension)
    mean_squared_response = {
        candidate_id: value / block_dimension
        for candidate_id, value in scores.items()
    }
    rec.notes["eta_used"] = eta_used
    rec.notes["block_dimension"] = block_dimension
    return ScoreProfile(
        request.request_id,
        "fd_norm",
        scores,
        spec,
        rec,
        artifacts={
            "schema": "loss-shake-responses-v1",
            "direction_responses": responses,
            "direction_count": len(responses),
            "direction_seed": spec.seed,
            "eta": eta_used,
            "block_dimension": block_dimension,
            "mean_squared_response": mean_squared_response,
            "dimension_corrected_energy": scores,
        },
    )


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
