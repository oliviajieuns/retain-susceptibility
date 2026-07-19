"""Stage 2: constrained repair under the one-sided reference-anchored guard.

Repair problem: minimize adjacent loss subject to anchored drift budgets
D_seq <= delta_seq^2 and D_tok <= delta_tok^2 (paper Sec. 3 'profile-guided
repair'). Enforcement combines three mechanisms with distinct roles:

- projection (proactive, mean-level): each step is projected off the most
  recently refreshed basis {grad(-L_forget), grad(-L_remote)} via a
  ridge-regularized Gram solve (paper eq:projection), so the mean forget and
  remote losses are first-order invariant at refresh steps;
- guard penalty (reactive, identity-paired): multipliers lam_seq/lam_tok on
  the one-sided anchored penalties enter the step gradient, ascending on the
  measured budget violation at refresh steps;
- acceptance rule (hard backstop): a refresh step whose measured budget is
  exceeded rolls back to the last accepted state and shrinks eta2, so the
  bound premise D <= delta^2 holds at every accepted refresh checkpoint by
  construction. Multipliers keep their ascended values across a rollback so
  repeated violations increase pressure.

Only block B is trained; snapshots are therefore cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from rsus.blocks import (
    BlockSpec,
    ParamVec,
    grads_of,
    load_params_,
    only_block_grads,
    save_params,
    vec_dot,
    vec_norm,
)
from rsus.costs import CostRecord, Meter
from rsus.data.base import Example, Request, collate
from rsus.guards import GuardKind, guard_penalty
from rsus.losses import seq_mean_answer_nll
from rsus.refcache import RefCache, assert_aligned, forget_seq_losses, forget_tok_losses


@dataclass
class Stage2Config:
    eta2: float = 5e-3
    mu_v: float = 0.7
    refresh_k: int = 1
    rho_g: float = 10.0
    delta_seq_sq: float = 1e-2
    delta_tok_sq: float = 1e-1
    eps_slack: float = 0.0
    guard_kind: GuardKind = GuardKind.ONE_SIDED
    token_level: bool = True
    projection: bool = True
    guard_enabled: bool = True
    shrink: float = 0.5
    max_steps: int = 100
    batch_size: int = 8
    ridge_scale: float = 0.0     # ridge = ridge_scale * mean(diag G)
    cond_max: float = 1e8


@dataclass
class RefreshEvent:
    step: int
    d_seq: float
    d_tok: float
    lam_seq: float
    lam_tok: float
    accepted: bool
    max_basis_cos: float | None = None


@dataclass
class Stage2Result:
    steps: int
    n_accepted: int
    n_rejected: int
    eta2_final: float
    events: list[RefreshEvent] = field(default_factory=list)
    cost: CostRecord = field(default_factory=CostRecord)


def _project(v: ParamVec, basis: list[ParamVec], ridge_scale: float, cond_max: float) -> ParamVec:
    """ghat = v - sum_i zeta_i b_i with zeta = (G + eps I)^{-1} [<b_i, v>]
    (paper eq:projection). Falls back to single-basis projection onto the
    forget-ascent direction when the basis Gram matrix is ill-conditioned."""
    k = len(basis)
    G = torch.empty((k, k), dtype=torch.float64)
    for i in range(k):
        for j in range(k):
            G[i, j] = vec_dot(basis[i], basis[j])
    rhs = torch.tensor([float(vec_dot(b, v)) for b in basis], dtype=torch.float64)
    eig = torch.linalg.eigvalsh(G)
    if float(eig.min()) <= 0 or float(eig.max()) / float(eig.min()) > cond_max:
        b1 = basis[0]
        coef = float(vec_dot(b1, v) / vec_dot(b1, b1))
        return {n: v[n] - coef * b1[n] for n in v}
    ridge = ridge_scale * float(G.diagonal().mean())
    zeta = torch.linalg.solve(G + ridge * torch.eye(k, dtype=torch.float64), rhs)
    out = dict(v)
    for i, b in enumerate(basis):
        out = {n: out[n] - float(zeta[i]) * b[n] for n in out}
    return out


def _guard_terms(
    model: torch.nn.Module,
    request: Request,
    cache: RefCache,
    cfg: Stage2Config,
    grad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_ids, u_seq = forget_seq_losses(model, request, cfg.batch_size, grad=grad)
    d_seq = guard_penalty(u_seq, cache.seq_refs, cfg.guard_kind, cfg.eps_slack)
    if cfg.token_level:
        tok_index, u_tok = forget_tok_losses(model, request, cfg.batch_size, grad=grad)
        assert_aligned(cache, seq_ids, tok_index)
        d_tok = guard_penalty(u_tok, cache.tok_refs, cfg.guard_kind, cfg.eps_slack)
    else:
        assert_aligned(cache, seq_ids, cache.tok_index)
        d_tok = torch.zeros((), dtype=u_seq.dtype)
    return d_seq, d_tok


def _basis(
    model: torch.nn.Module,
    sel: dict[str, torch.nn.Parameter],
    request: Request,
    remote_probe: list[Example],
    batch_size: int,
    rec: CostRecord,
) -> list[ParamVec]:
    out: list[ParamVec] = []
    for examples in (list(request.forget), remote_probe):
        model.zero_grad(set_to_none=True)
        with only_block_grads(model, sel):
            for i in range(0, len(examples), batch_size):
                batch = collate(examples[i : i + batch_size])
                (-seq_mean_answer_nll(model, batch).mean() * len(batch["example_ids"]) / len(examples)).backward()
                rec.fwd_passes += 1
                rec.bwd_passes += 1
        out.append(grads_of(sel))
    model.zero_grad(set_to_none=True)
    return out


def run_stage2(
    model: torch.nn.Module,
    block: BlockSpec,
    request: Request,
    adjacent: list[Example],
    remote_probe: list[Example],
    cache: RefCache,
    cfg: Stage2Config,
) -> Stage2Result:
    rec = CostRecord()
    with Meter(rec):
        sel = block.select(model)
        v: ParamVec = {n: torch.zeros_like(p) for n, p in sel.items()}
        lam_seq = lam_tok = 0.0
        eta2 = cfg.eta2
        snapshot = (save_params(sel), {n: t.clone() for n, t in v.items()})
        basis: list[ParamVec] = []
        events: list[RefreshEvent] = []
        n_acc = n_rej = 0
        adj_batch = collate(adjacent)

        for t in range(cfg.max_steps):
            just_refreshed = False
            if t % cfg.refresh_k == 0:
                with torch.no_grad():
                    d_seq, d_tok = _guard_terms(model, request, cache, cfg, grad=False)
                lam_seq += cfg.rho_g * max(0.0, float(d_seq) - cfg.delta_seq_sq)
                lam_tok += cfg.rho_g * max(0.0, float(d_tok) - cfg.delta_tok_sq)
                violated = float(d_seq) > cfg.delta_seq_sq or (
                    cfg.token_level and float(d_tok) > cfg.delta_tok_sq
                )
                if violated:
                    load_params_(sel, snapshot[0])
                    v = {n: t_.clone() for n, t_ in snapshot[1].items()}
                    eta2 *= cfg.shrink
                    n_rej += 1
                    events.append(
                        RefreshEvent(t, float(d_seq), float(d_tok), lam_seq, lam_tok, False)
                    )
                    continue
                snapshot = (save_params(sel), {n: t_.clone() for n, t_ in v.items()})
                basis = _basis(model, sel, request, remote_probe, cfg.batch_size, rec)
                n_acc += 1
                events.append(
                    RefreshEvent(t, float(d_seq), float(d_tok), lam_seq, lam_tok, True)
                )
                just_refreshed = True

            model.zero_grad(set_to_none=True)
            with only_block_grads(model, sel):
                total = seq_mean_answer_nll(model, adj_batch).mean()
                if cfg.guard_enabled and (lam_seq > 0 or lam_tok > 0):
                    d_seq_g, d_tok_g = _guard_terms(model, request, cache, cfg, grad=True)
                    total = total + lam_seq * d_seq_g + lam_tok * d_tok_g
                total.backward()
                rec.fwd_passes += 1
                rec.bwd_passes += 1
            g = grads_of(sel)
            model.zero_grad(set_to_none=True)

            v = {n: cfg.mu_v * v[n] + g[n] for n in v}
            ghat = _project(v, basis, cfg.ridge_scale, cfg.cond_max) if cfg.projection else v

            if just_refreshed and cfg.projection and events:
                vn = float(vec_norm(ghat))
                cosines = [
                    abs(float(vec_dot(ghat, b))) / (vn * float(vec_norm(b)) + 1e-30)
                    for b in basis
                ]
                events[-1].max_basis_cos = max(cosines) if cosines else None

            with torch.no_grad():
                for n, p in sel.items():
                    p.add_(ghat[n], alpha=-eta2)
            v = ghat
    return Stage2Result(cfg.max_steps, n_acc, n_rej, eta2, events, rec)
