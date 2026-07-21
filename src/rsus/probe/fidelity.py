"""fd_norm fidelity decomposition (A / B / C) for the preflight fidelity gate.

Isolates WHY the backward-free fd_norm may disagree with the exact gradient
norm, by computing three scores on the SAME candidates, block B, and random
unit directions v_1..v_R:

    A(x) = ||g_x||^2                                  exact block-gradient norm squared
    B(x) = (d/R) sum_k (g_x . v_k)^2                  exact gradient, random projections (analytic)
    C(x) = (d/R) sum_k [ (l_x(th+eta v_k)             finite-difference form (== d * fd_norm)
                          - l_x(th-eta v_k)) / 2eta ]^2

Since E_v[(g.v)^2] = ||g||^2 / d for isotropic unit v, both B and C are unbiased
estimators of A, and the estimator's RELATIVE variance is 2/R -- independent of
the block dimension d (so a one-layer block is a valid, cheap fidelity test).
Central differences cancel the Hessian, so B vs C isolates finite-difference /
eta / parameter-precision effects while A vs B isolates Monte-Carlo / direction
normalization:
    A != B          -> too few directions R, or a direction-normalization bug
    B != C          -> finite-difference / eta too large-or-small / bf16 perturbation underflow
    A ~ B ~ C, yet damage prediction differs -> the analysis pipeline or fold, not the estimator

Directions are regenerated on the fly from (seed, k) -- never all materialized --
so this stays within one GPU at 7B block scale. A and every (g_x . v_k) come from
one candidate-side backward per candidate.
"""
from __future__ import annotations

import dataclasses as _dc
import math

import torch

from rsus.blocks import grads_of, load_params_, only_block_grads, save_params, set_perturbed_
from rsus.costs import CostRecord
from rsus.data.base import collate
from rsus.losses import seq_mean_answer_nll
from rsus.probe.base import ProbeSpec
from rsus.probe.finite_diff import fd_scores_along


def block_dim(sel: dict) -> int:
    return sum(p.numel() for p in sel.values())


def _dir_seed(seed: int, k: int) -> int:
    return (seed * 1_000_003 + k) % (2**31 - 1)


def gen_direction(sel: dict, seed: int, k: int) -> dict:
    """Reproducible unit direction in block B, sampled fp32 on the block's
    device. Independent per (seed, k) so B and C use identical directions
    without materializing a bank."""
    dev = next(iter(sel.values())).device
    g = torch.Generator(device=dev).manual_seed(_dir_seed(seed, k))
    v = {n: torch.randn(p.shape, generator=g, device=dev, dtype=torch.float32)
         for n, p in sel.items()}
    nrm = math.sqrt(sum(float((t * t).sum()) for t in v.values()))
    return {n: t / nrm for n, t in v.items()}


def perturbation_report(model: torch.nn.Module, spec: ProbeSpec, eta: float,
                        seed: int = 0, k: int = 0) -> dict[str, float]:
    """Realized perturbation when theta is displaced by eta * v (v unit) in the
    model's actual parameter dtype. In fp32 eff_norm ~= eta and frac_changed ~= 1;
    under bf16 a coordinate moves ~eta/sqrt(d) ~ 1e-7, below the bf16 ULP, so the
    add ROUNDS AWAY -> eff_norm << eta and frac_changed << 1. Run this before
    trusting any C: it directly measures the perturbation-underflow hypothesis."""
    sel = spec.block.select(model)
    v = gen_direction(sel, seed, k)  # unit, so requested ||eta*v|| = eta
    saved = save_params(sel)
    set_perturbed_(sel, saved, v, eta)
    dsq = 0.0
    changed = 0
    total = 0
    for n, p in sel.items():
        delta = p.detach().float() - saved[n].float()
        dsq += float((delta * delta).sum())
        changed += int((delta != 0).sum())
        total += delta.numel()
    load_params_(sel, saved)
    eff = math.sqrt(dsq)
    return {"eta": eta, "eff_norm": eff, "eff_over_eta": eff / eta if eta else float("nan"),
            "frac_changed": changed / total}


def exact_A_and_projsq(
    model: torch.nn.Module, request, spec: ProbeSpec, seeds: list[int], max_R: int
) -> tuple[dict[str, float], dict[str, dict[int, list[float]]], int]:
    """A(x)=||g_x||^2 and (g_x . v_k)^2 for every (seed, k), from ONE
    candidate-side backward per candidate. Directions are regenerated per
    candidate (never stored); reductions accumulate in fp32."""
    sel = spec.block.select(model)
    d = block_dim(sel)
    A: dict[str, float] = {}
    projsq: dict[str, dict[int, list[float]]] = {}
    for ex in request.universe.examples:
        model.zero_grad(set_to_none=True)
        with only_block_grads(model, sel):
            seq_mean_answer_nll(model, collate([ex])).mean().backward()
        g = {n: t.detach().float() for n, t in grads_of(sel).items()}
        A[ex.example_id] = float(sum((t * t).sum() for t in g.values()))
        per: dict[int, list[float]] = {}
        for s in seeds:
            vals = []
            for k in range(max_R):
                v = gen_direction(sel, s, k)
                dot = float(sum((g[n] * v[n]).sum() for n in g))
                vals.append(dot * dot)
            per[s] = vals
        projsq[ex.example_id] = per
    model.zero_grad(set_to_none=True)
    return A, projsq, d


def B_scores(projsq: dict[str, dict[int, list[float]]], seed: int, R: int, d: int) -> dict[str, float]:
    return {cid: (d / R) * sum(ps[seed][:R]) for cid, ps in projsq.items()}


def C_scores(
    model: torch.nn.Module, request, spec: ProbeSpec, seed: int, R: int, eta: float, d: int
) -> dict[str, float]:
    """Finite-difference estimator (== d * fd_norm) along the seed's first R
    directions, regenerated identically to B."""
    sel = spec.block.select(model)
    sp = _dc.replace(spec, eta=eta)
    rec = CostRecord()
    acc: dict[str, float] = {ex.example_id: 0.0 for ex in request.universe.examples}
    for k in range(R):
        v = gen_direction(sel, seed, k)
        for cid, val in fd_scores_along(model, request, sp, v, rec).items():
            acc[cid] += val * val
    return {cid: (d / R) * s for cid, s in acc.items()}
