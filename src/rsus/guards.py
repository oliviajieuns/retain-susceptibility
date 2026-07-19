"""Reference-anchored guards (paper eq:guard) and the drift bound
(paper eq:massbound).

Primary guard: ONE_SIDED — mean squared hinge charging any audited loss for
falling below its own cached reference minus a slack. Identity-paired: no
sorting, no aggregation across identities. SYMMETRIC and SORTED_PROFILE are
preregistered ablation arms; SYMMETRIC upper-bounds the squared empirical 1-D
Wasserstein-2 distance between the profiles (sorted pairing minimizes the
squared-difference sum), SORTED_PROFILE is that minimum itself.
"""
from __future__ import annotations

from enum import Enum

import torch


class GuardKind(str, Enum):
    ONE_SIDED = "one_sided"
    SYMMETRIC = "symmetric"
    SORTED_PROFILE = "sorted_profile"


def guard_penalty(
    losses: torch.Tensor,
    refs: torch.Tensor,
    kind: GuardKind = GuardKind.ONE_SIDED,
    eps: float = 0.0,
) -> torch.Tensor:
    """Differentiable scalar penalty D(losses, refs). ``refs`` is detached;
    coordinate i of ``losses`` is paired with coordinate i of ``refs``
    (except SORTED_PROFILE, which deliberately discards identity)."""
    if losses.shape != refs.shape or losses.dim() != 1:
        raise ValueError(f"expected matching 1-D vectors, got {losses.shape} vs {refs.shape}")
    refs = refs.detach()
    if kind is GuardKind.ONE_SIDED:
        return torch.clamp(refs - eps - losses, min=0.0).pow(2).mean()
    if kind is GuardKind.SYMMETRIC:
        return (losses - refs).pow(2).mean()
    if kind is GuardKind.SORTED_PROFILE:
        return (torch.sort(losses).values - torch.sort(refs).values).pow(2).mean()
    raise ValueError(kind)


def budget_ok(penalty: float | torch.Tensor, delta_sq: float) -> bool:
    """Acceptance-rule check at a refresh checkpoint: D <= delta^2."""
    return float(penalty) <= delta_sq


def drift_mass_bound(delta_sq: float, margin: float) -> float:
    """eq:massbound (Chebyshev): if D_one_sided <= delta^2, the fraction of
    coordinates with u_i <= ref_i - eps - margin is at most delta^2/margin^2
    (capped at 1)."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    return min(1.0, delta_sq / margin**2)


def drifted_mass(
    losses: torch.Tensor, refs: torch.Tensor, eps: float, margin: float
) -> float:
    """Measured fraction of coordinates more than ``margin`` below their
    slack-adjusted references (the quantity eq:massbound bounds)."""
    return float((losses <= refs - eps - margin).double().mean())
