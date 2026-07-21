"""fd_norm fidelity decomposition (A / B / C) for the preflight fidelity gate.

Isolates WHY the backward-free fd_norm may disagree with the exact gradient
norm, by computing three scores on the SAME candidates, block B, and random
unit directions v_1..v_R:

    A(x) = ||g_x||^2                                  exact block-gradient norm squared
    B(x) = (d/R) sum_k (g_x . v_k)^2                  exact gradient, random projections (analytic)
    C(x) = (d/R) sum_k [ (l_x(th+eta v_k)             finite-difference form (== d * fd_norm)
                          - l_x(th-eta v_k)) / 2eta ]^2

Since E_v[(g.v)^2] = ||g||^2 / d for isotropic unit v, both B and C are unbiased
estimators of A. Central differences cancel the Hessian term
((l(th+eta v)-l(th-eta v))/2eta = g.v + O(eta^2)), so B and C can differ ONLY by
finite-difference truncation, the eta scale, or parameter-precision effects.
Hence:
    A != B          -> too few directions R, or a direction-normalization bug
    B != C          -> finite-difference / eta too large-or-small / bf16 perturbation underflow
    A ~ B ~ C, yet damage prediction differs -> the analysis pipeline or fold, not the estimator

A and every (g_x . v_k) come from ONE candidate-side backward per candidate, so
the whole A/B grid over seeds and R is a single pass; C adds 2R forward sweeps
per (seed, R, eta). Intended for ~128 candidates, not a full universe.
"""
from __future__ import annotations

import dataclasses as _dc

import torch

from rsus.blocks import grads_of, only_block_grads, vec_dot, vec_randn_like, vec_unit
from rsus.costs import CostRecord
from rsus.data.base import collate
from rsus.losses import seq_mean_answer_nll
from rsus.probe.base import ProbeSpec
from rsus.probe.finite_diff import fd_scores_along


def block_dim(sel: dict) -> int:
    return sum(p.numel() for p in sel.values())


def direction_bank(sel: dict, seeds: list[int], max_dirs: int) -> dict[int, list]:
    """Per-seed list of ``max_dirs`` unit directions in block B. Directions are a
    fixed nested sequence: R uses the first R of the seed's list."""
    bank: dict[int, list] = {}
    for s in seeds:
        gen = torch.Generator().manual_seed(s)
        bank[s] = [vec_unit(vec_randn_like(sel, gen)) for _ in range(max_dirs)]
    return bank


def exact_A_and_projsq(
    model: torch.nn.Module, request, spec: ProbeSpec, bank: dict[int, list]
) -> tuple[dict[str, float], dict[str, dict[int, list[float]]], int]:
    """A(x)=||g_x||^2 and, for every banked direction, (g_x . v_k)^2 -- from ONE
    candidate-side backward per candidate (gradients are never stored)."""
    sel = spec.block.select(model)
    A: dict[str, float] = {}
    projsq: dict[str, dict[int, list[float]]] = {}
    for ex in request.universe.examples:
        model.zero_grad(set_to_none=True)
        with only_block_grads(model, sel):
            seq_mean_answer_nll(model, collate([ex])).mean().backward()
        g = grads_of(sel)
        A[ex.example_id] = float(sum((v * v).sum() for v in g.values()))
        projsq[ex.example_id] = {
            s: [float(vec_dot(g, vk)) ** 2 for vk in vecs] for s, vecs in bank.items()
        }
    model.zero_grad(set_to_none=True)
    return A, projsq, block_dim(sel)


def B_scores(projsq: dict[str, dict[int, list[float]]], seed: int, R: int, d: int) -> dict[str, float]:
    return {cid: (d / R) * sum(ps[seed][:R]) for cid, ps in projsq.items()}


def C_scores(
    model: torch.nn.Module, request, spec: ProbeSpec, dirs: list, eta: float, d: int
) -> dict[str, float]:
    """Finite-difference estimator (== d * fd_norm) along the given directions."""
    sp = _dc.replace(spec, eta=eta)
    rec = CostRecord()
    acc: dict[str, float] = {ex.example_id: 0.0 for ex in request.universe.examples}
    for vk in dirs:
        for cid, val in fd_scores_along(model, request, sp, vk, rec).items():
            acc[cid] += val * val
    R = len(dirs)
    return {cid: (d / R) * s for cid, s in acc.items()}


def abc_scores(
    model: torch.nn.Module, request, spec: ProbeSpec, n_dirs: int, eta: float, seed: int
) -> dict[str, object]:
    """Convenience single-cell A/B/C for one (R, eta, seed). For grids prefer
    exact_A_and_projsq once + B_scores/C_scores per cell (avoids re-backward)."""
    sel = spec.block.select(model)
    bank = direction_bank(sel, [seed], n_dirs)
    A, projsq, d = exact_A_and_projsq(model, request, spec, bank)
    B = B_scores(projsq, seed, n_dirs, d)
    C = C_scores(model, request, spec, bank[seed], eta, d)
    return {"A": A, "B": B, "C": C, "dim": d}
