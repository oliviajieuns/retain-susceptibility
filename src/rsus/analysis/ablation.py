"""Table-3 matched-parent contrasts: per-request paired deltas between a
condition and its named parent, with a request-resampled percentile CI.
Requests, budgets, and untouched audits must be identical within a contrast;
the caller guarantees that pairing.
"""
from __future__ import annotations

import torch


def matched_delta(
    condition: dict[str, float],
    parent: dict[str, float],
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, object]:
    """Per-request paired condition-minus-parent deltas. Positive favors the
    condition for 'higher is better' outcomes; the caller flips signs for
    damage-style outcomes (paper: CVaR gain = parent minus condition)."""
    if set(condition) != set(parent):
        raise ValueError("condition/parent request sets differ")
    reqs = sorted(condition)
    deltas = [condition[r] - parent[r] for r in reqs]
    point = sum(deltas) / len(deltas)
    gen = torch.Generator().manual_seed(seed)
    boots = []
    for _ in range(n_boot):
        idx = torch.randint(len(deltas), (len(deltas),), generator=gen).tolist()
        boots.append(sum(deltas[i] for i in idx) / len(deltas))
    boots.sort()
    lo_q = (1.0 - ci) / 2.0
    return {
        "mean": point,
        "lo": boots[int(lo_q * (n_boot - 1))],
        "hi": boots[int((1.0 - lo_q) * (n_boot - 1))],
        "per_request": dict(zip(reqs, deltas)),
    }


def ablation_row(
    label: str,
    contrast: str,
    pred_rho: tuple[dict, dict] | None = None,
    reach: tuple[dict, dict] | None = None,
    damage: tuple[dict, dict] | None = None,
) -> dict[str, object]:
    """One Table-3 row. Each input pairs (condition, parent) per-request
    values; damage uses parent-minus-condition (CVaR gain, positive favors
    the condition)."""
    row: dict[str, object] = {"link": label, "contrast": contrast}
    row["d_pred_rho"] = matched_delta(*pred_rho) if pred_rho else None
    row["d_reach"] = matched_delta(*reach) if reach else None
    if damage:
        cond, par = damage
        row["cvar_gain"] = matched_delta(par, cond)  # parent minus condition
    else:
        row["cvar_gain"] = None
    return row
