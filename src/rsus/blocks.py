"""Declared trainable block B and parameter-vector utilities.

The probe's score definition (paper eq:fdscore) is conditional on a declared
block: parameter selection is part of the score, so BlockSpec is carried in
ProbeSpec and disclosed with every comparison.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass

import torch

ParamVec = dict[str, torch.Tensor]


@dataclass(frozen=True)
class BlockSpec:
    """Selects block B by a fullmatch regex over ``named_parameters``."""

    pattern: str

    def select(self, model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
        rx = re.compile(self.pattern)
        sel = {n: p for n, p in model.named_parameters() if rx.fullmatch(n)}
        if not sel:
            raise ValueError(f"BlockSpec matched no parameters: {self.pattern!r}")
        return sel


def mlp_down_last_layers(model: torch.nn.Module, n_last: int) -> BlockSpec:
    """Preregistered default block: MLP down-projections of the last
    ``n_last`` decoder layers."""
    n_layers = model.config.num_hidden_layers
    if not 0 < n_last <= n_layers:
        raise ValueError(f"n_last={n_last} out of range for {n_layers} layers")
    idx = "|".join(str(i) for i in range(n_layers - n_last, n_layers))
    return BlockSpec(pattern=rf".*\.layers\.(?:{idx})\.mlp\.down_proj\.weight")


# ---- ParamVec algebra -------------------------------------------------------

def vec_dot(a: ParamVec, b: ParamVec) -> torch.Tensor:
    return torch.stack([(a[n] * b[n]).sum() for n in a]).sum()


def vec_norm(a: ParamVec) -> torch.Tensor:
    return torch.sqrt(vec_dot(a, a))


def vec_scale(a: ParamVec, s: float | torch.Tensor) -> ParamVec:
    return {n: t * s for n, t in a.items()}


def vec_unit(a: ParamVec) -> ParamVec:
    n = vec_norm(a)
    if n == 0:
        raise ValueError("cannot normalize a zero direction")
    return vec_scale(a, 1.0 / n)


def vec_randn_like(sel: dict[str, torch.nn.Parameter], generator: torch.Generator) -> ParamVec:
    return {
        n: torch.randn(p.shape, generator=generator, dtype=p.dtype, device=p.device)
        for n, p in sel.items()
    }


# ---- Parameter state management --------------------------------------------

def grads_of(sel: dict[str, torch.nn.Parameter]) -> ParamVec:
    return {
        n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for n, p in sel.items()
    }


def save_params(sel: dict[str, torch.nn.Parameter]) -> ParamVec:
    return {n: p.detach().clone() for n, p in sel.items()}


@torch.no_grad()
def load_params_(sel: dict[str, torch.nn.Parameter], saved: ParamVec) -> None:
    for n, p in sel.items():
        p.copy_(saved[n])


@torch.no_grad()
def set_perturbed_(
    sel: dict[str, torch.nn.Parameter],
    saved: ParamVec,
    direction: ParamVec,
    alpha: float,
) -> None:
    """Set each selected parameter to ``saved + alpha * direction``.

    Always computed from ``saved`` (never incrementally) so the +eta/-eta
    sweeps are exact and ``load_params_(sel, saved)`` restores bit-exactly.
    """
    for n, p in sel.items():
        p.copy_(saved[n] + alpha * direction[n])


@contextmanager
def only_block_grads(model: torch.nn.Module, sel: dict[str, torch.nn.Parameter]):
    """Temporarily restrict requires_grad to block B (backward-cost control)."""
    names = set(sel)
    prev = {n: p.requires_grad for n, p in model.named_parameters()}
    try:
        for n, p in model.named_parameters():
            p.requires_grad_(n in names)
        yield
    finally:
        for n, p in model.named_parameters():
            p.requires_grad_(prev[n])
