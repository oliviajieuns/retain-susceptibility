"""Folds and profile-defined pools (paper Sec. 3 'Scalable Susceptibility
Profiling' + Appendix 'Profile Construction').

Fold discipline: candidates split at group (retained-author) granularity into
discovery and audit folds before any score or outcome is computed. Pool
membership is decided within discovery folds only; the untouched native and
uniform-random audit folds stay sealed (sealing.py).

Pools: the visible high-risk protection pool P is the largest ``pool_size``
eligible POSITIVE scores; the remote constraint stream R0 is a
template-matched sample from the preregistered near-zero band. Requests with
fewer eligible positive candidates than ``pool_size`` follow the frozen
fallback rule (take all positives, flagged) rather than an outcome-dependent
threshold change; below ``min_pool_size`` the request is rejected.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch

from rsus.data.base import Request
from rsus.probe.base import ScoreProfile


class PartitionError(ValueError):
    """Raised when the preregistered construction rule cannot be satisfied."""


def make_folds(
    groups: dict[str, str], audit_frac: float = 0.5, seed: int = 0
) -> dict[str, str]:
    """Assign each group to 'discovery' or 'audit', deterministically.

    ``groups`` maps example_id -> group. Returns group -> fold.
    """
    uniq = sorted(set(groups.values()))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(uniq), generator=gen).tolist()
    n_audit = round(len(uniq) * audit_frac)
    audit = {uniq[i] for i in perm[:n_audit]}
    return {g: ("audit" if g in audit else "discovery") for g in uniq}


def group_disjoint_split(
    ids: list[str], group_of: dict[str, str], n_first: int, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split ids into two halves with |first| == n_first, never splitting a
    group. Kept for held-out contract pools; raises PartitionError when group
    sizes make the exact split impossible."""
    by_group: dict[str, list[str]] = {}
    for i in sorted(ids):
        by_group.setdefault(group_of[i], []).append(i)
    names = sorted(by_group)
    gen = torch.Generator().manual_seed(seed)
    order = [names[i] for i in torch.randperm(len(names), generator=gen).tolist()]
    first: list[str] = []
    second: list[str] = []
    need = n_first
    for g in order:
        members = by_group[g]
        if len(members) <= need:
            first.extend(members)
            need -= len(members)
        else:
            second.extend(members)
    if need != 0:
        raise PartitionError(f"cannot form exact split of {n_first} at group granularity")
    return tuple(sorted(first)), tuple(sorted(second))


@dataclass(frozen=True)
class PartitionParams:
    pool_size: int = 64
    min_pool_size: int = 8          # frozen fallback floor
    tau_rem_abs_quantile: float = 0.50
    seed: int = 0


@dataclass(frozen=True)
class Partition:
    """P (protect) and R0 (remote stream) for one request."""

    request_id: str
    scorer: str
    protect: tuple[str, ...]
    remote_stream: tuple[str, ...]
    fallback: bool
    params: PartitionParams
    manifest_sha: str

    def all_pool_ids(self) -> set[str]:
        return set(self.protect) | set(self.remote_stream)


def build_partition(
    profile: ScoreProfile,
    request: Request,
    folds: dict[str, str],
    params: PartitionParams,
    template_of: dict[str, str] | None = None,
) -> Partition:
    """Construct P and R0 from a score profile, discovery folds only."""
    group_of = {e.example_id: e.group for e in request.universe.examples}
    scores = profile.scores
    if set(scores) != set(group_of):
        raise PartitionError("profile does not cover the candidate universe exactly")

    eligible = [c for c in scores if folds.get(group_of[c]) == "discovery"]

    # P: largest pool_size eligible positive scores; frozen fallback below K.
    positive = sorted(
        (c for c in eligible if scores[c] > 0.0), key=lambda c: (-scores[c], c)
    )
    if len(positive) < params.min_pool_size:
        raise PartitionError(
            f"only {len(positive)} eligible positive candidates; "
            f"minimum is {params.min_pool_size}"
        )
    fallback = len(positive) < params.pool_size
    protect = positive[: params.pool_size]

    # R0: template-matched seeded sample from the near-zero band, count-matched
    # to the protect pool.
    all_s = torch.tensor([scores[c] for c in sorted(scores)], dtype=torch.float64)
    tau_rem = torch.quantile(all_s.abs(), params.tau_rem_abs_quantile).item()
    band = sorted(
        c for c in eligible if abs(scores[c]) <= tau_rem and c not in set(protect)
    )
    n_remote = len(protect)
    if len(band) < n_remote:
        raise PartitionError(
            f"near-zero band has {len(band)} eligible candidates; need {n_remote}"
        )
    gen = torch.Generator().manual_seed(params.seed)
    if template_of is None:
        idx = torch.randperm(len(band), generator=gen).tolist()[:n_remote]
        remote = sorted(band[i] for i in idx)
    else:
        need: dict[str, int] = {}
        for c in protect:
            need[template_of[c]] = need.get(template_of[c], 0) + 1
        by_tpl: dict[str, list[str]] = {}
        for c in band:
            by_tpl.setdefault(template_of[c], []).append(c)
        remote_l: list[str] = []
        for tpl, k in sorted(need.items()):
            pool = by_tpl.get(tpl, [])
            if len(pool) < k:
                raise PartitionError(f"template {tpl!r}: need {k}, band has {len(pool)}")
            idx = torch.randperm(len(pool), generator=gen).tolist()[:k]
            remote_l.extend(pool[i] for i in idx)
        remote = sorted(remote_l)

    body = json.dumps(
        {
            "request": request.request_id,
            "scorer": profile.scorer,
            "protect": protect,
            "remote": remote,
            "fallback": fallback,
            "params": asdict(params),
            "universe_sha": request.universe.sha,
        },
        sort_keys=True,
    )
    sha = hashlib.sha256(body.encode()).hexdigest()
    return Partition(
        request.request_id, profile.scorer, tuple(protect), tuple(remote), fallback, params, sha
    )
