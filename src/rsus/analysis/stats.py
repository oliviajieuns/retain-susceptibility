"""Inference for the main tables: hierarchical bootstrap over targets then
requests, and the leave-one-target-out range (appendix 'Metrics and
Statistics'). Values arrive seed-averaged per request.
"""
from __future__ import annotations

import torch


def hierarchical_bootstrap(
    values: dict[str, dict[str, float]],
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, float]:
    """values[target][request] -> seed-averaged statistic. Resamples targets
    with replacement, then requests within each sampled target. Returns
    mean and percentile CI."""
    targets = sorted(values)
    if not targets or any(not values[t] for t in targets):
        raise ValueError("need at least one request per target")
    gen = torch.Generator().manual_seed(seed)
    point = _grand_mean(values, targets)
    boots = []
    for _ in range(n_boot):
        t_idx = torch.randint(len(targets), (len(targets),), generator=gen).tolist()
        means = []
        for ti in t_idx:
            reqs = sorted(values[targets[ti]])
            r_idx = torch.randint(len(reqs), (len(reqs),), generator=gen).tolist()
            means.append(sum(values[targets[ti]][reqs[i]] for i in r_idx) / len(reqs))
        boots.append(sum(means) / len(means))
    boots.sort()
    lo_q = (1.0 - ci) / 2.0
    lo = boots[int(lo_q * (n_boot - 1))]
    hi = boots[int((1.0 - lo_q) * (n_boot - 1))]
    return {"mean": point, "lo": lo, "hi": hi}


def _grand_mean(values, targets) -> float:
    per_target = [sum(v.values()) / len(v) for v in (values[t] for t in targets)]
    return sum(per_target) / len(per_target)


def leave_one_target_out(values: dict[str, dict[str, float]]) -> dict[str, float]:
    """Range of the grand mean when each target is left out in turn."""
    targets = sorted(values)
    if len(targets) < 2:
        raise ValueError("LOTO needs at least two targets")
    means = [
        _grand_mean(values, [t for t in targets if t != out]) for out in targets
    ]
    return {"lo": min(means), "hi": max(means)}
