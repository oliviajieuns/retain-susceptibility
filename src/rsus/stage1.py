"""Stage 1: constrained forgetting to a reference-calibrated floor
(paper eq:stage1), ported from the predecessor repo's augmented-Lagrangian
implementation and adapted to the Request/RefCache abstractions.

Objective per step (all parameters trainable):
    -gamma * mean(clip_c(ell_forget)) + lam * h + (rho/2) * h^2,
with h = mean remote loss - pinned pre-unlearning value, two-sided, and
lam <- lam + rho * h_bar (EMA dual ascent) once per optimizer step. The clip
keeps the forward value exact and zeroes the gradient above c, so ascent
pressure redistributes to under-forgotten examples instead of diverging.

Exit gate: every forget sequence at or above the floor m AND remote recall
at or above remote_recall_frac of its pinned baseline. On success the
per-sequence and per-token reference losses are cached (refcache.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from rsus.costs import CostRecord, Meter
from rsus.data.base import Example, Request, collate
from rsus.evalx.metrics import mean_recall, mean_seq_loss
from rsus.losses import seq_mean_answer_nll
from rsus.refcache import RefCache, build_ref_cache, forget_seq_losses


def clip_c(losses: torch.Tensor, c: float) -> torch.Tensor:
    """Forward value exact; gradient vanishes at and above c (stop-gradient)."""
    return torch.where(losses < c, losses, c + (losses - c).detach())


def calibrate_floor(
    base_model: torch.nn.Module, request: Request, batch_size: int = 8, clamp_min: float = 2.5
) -> float:
    """m = median base-model (pre-injection reference) loss on Df, clamped.
    Forgetting means restoring pre-injection surprise; a chance-level floor
    is vacuous at vocabulary scale."""
    _, losses = forget_seq_losses(base_model, request, batch_size)
    return max(float(losses.median()), clamp_min)


@dataclass
class Stage1Config:
    gamma: float = 1.0
    rho: float = 10.0
    ema_beta: float = 0.9
    lr: float = 5e-3
    max_steps: int = 500
    eval_every: int = 10
    clip_offset: float = 1.0          # c = m + clip_offset
    remote_recall_frac: float = 0.90
    forget_recall_max: float | None = None  # extra exit condition: forget argmax
    # recall <= this (aligns the exit with a recall-based common criterion; the
    # clip ceiling lifts while unmet so ascent pressure does not stall at c)
    batch_size: int = 8
    remote_batch_size: int = 8
    remote_probe_cap: int = 256
    seed: int = 0


@dataclass
class Stage1Result:
    gate_passed: bool
    steps: int
    floor_m: float
    lam: float
    cache: RefCache | None
    history: list[dict] = field(default_factory=list)
    cost: CostRecord = field(default_factory=CostRecord)


def run_stage1(
    model: torch.nn.Module,
    request: Request,
    remote: list[Example],
    floor_m: float,
    cfg: Stage1Config,
) -> Stage1Result:
    rec = CostRecord()
    with Meter(rec):
        c = floor_m + cfg.clip_offset
        probe = remote[: cfg.remote_probe_cap]
        l_rem0 = mean_seq_loss(model, probe, cfg.batch_size)
        base_recall = mean_recall(model, probe, cfg.batch_size)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        gen = torch.Generator().manual_seed(cfg.seed)
        lam, h_bar = 0.0, 0.0
        history: list[dict] = []
        gate_passed, steps = False, 0

        forget_batch = collate(list(request.forget))

        for step in range(1, cfg.max_steps + 1):
            steps = step
            l_f = seq_mean_answer_nll(model, forget_batch)
            idx = torch.randperm(len(remote), generator=gen)[: cfg.remote_batch_size]
            remote_mb = collate([remote[i] for i in idx.tolist()])
            h = seq_mean_answer_nll(model, remote_mb).mean() - l_rem0

            loss = -cfg.gamma * clip_c(l_f, c).mean() + lam * h + 0.5 * cfg.rho * h * h
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            rec.fwd_passes += 2
            rec.bwd_passes += 1

            h_bar = cfg.ema_beta * h_bar + (1.0 - cfg.ema_beta) * float(h.detach())
            lam = lam + cfg.rho * h_bar

            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                _, cur = forget_seq_losses(model, request, cfg.batch_size)
                recall = mean_recall(model, probe, cfg.batch_size)
                f_recall = (mean_recall(model, list(request.forget), cfg.batch_size)
                            if cfg.forget_recall_max is not None else None)
                history.append(
                    {
                        "step": step,
                        "min_forget": float(cur.min()),
                        "h": float(h.detach()),
                        "lam": lam,
                        "remote_recall": recall,
                        "forget_recall": f_recall,
                    }
                )
                floor_ok = float(cur.min()) >= floor_m
                remote_ok = recall >= cfg.remote_recall_frac * base_recall
                recall_ok = f_recall is None or f_recall <= cfg.forget_recall_max
                if floor_ok and remote_ok and recall_ok:
                    gate_passed = True
                    break
                if floor_ok and not recall_ok and float(cur.min()) >= c - 0.25:
                    c += cfg.clip_offset  # lift the clip so ascent can continue

        cache = build_ref_cache(model, request, cfg.batch_size, floor_m) if gate_passed else None
    return Stage1Result(gate_passed, steps, floor_m, lam, cache, history, rec)
