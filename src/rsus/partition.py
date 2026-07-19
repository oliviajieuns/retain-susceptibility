"""Folds and probe-defined pools.

Fold discipline (paper Sec. 3): candidates split at group (retained-author)
granularity into discovery and audit folds before any optimizer runs. Pool
membership is decided within discovery folds only; audit folds stay untouched
and their scores are sealed (sealing.py).

Pools: the visible high-risk pool is a top-K cut of the susceptibility
profile above a preregistered signed quantile; the remote constraint stream
is a template-matched sample from the near-zero band. Both split into
visible/held-out halves at group granularity.
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


@dataclass(frozen=True)
class PartitionParams:
    tau_adj_quantile: float = 0.95
    tau_rem_abs_quantile: float = 0.50
    pool_size: int = 64
    visible_split: int = 32
    seed: int = 0


@dataclass(frozen=True)
class Partition:
    request_id: str
    scorer: str
    adjacent_visible: tuple[str, ...]
    adjacent_heldout: tuple[str, ...]
    remote_visible: tuple[str, ...]
    remote_heldout: tuple[str, ...]
    params: PartitionParams
    manifest_sha: str

    def all_pool_ids(self) -> set[str]:
        return (
            set(self.adjacent_visible)
            | set(self.adjacent_heldout)
            | set(self.remote_visible)
            | set(self.remote_heldout)
        )


def _group_split(
    ids: list[str], group_of: dict[str, str], n_visible: int, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split ids into (visible, heldout) with |visible| == n_visible, never
    splitting a group across halves. Greedy over seeded group order; raises
    PartitionError if group sizes make the exact split impossible."""
    by_group: dict[str, list[str]] = {}
    for i in sorted(ids):
        by_group.setdefault(group_of[i], []).append(i)
    names = sorted(by_group)
    gen = torch.Generator().manual_seed(seed)
    order = [names[i] for i in torch.randperm(len(names), generator=gen).tolist()]
    visible: list[str] = []
    heldout: list[str] = []
    need = n_visible
    for g in order:
        members = by_group[g]
        if len(members) <= need:
            visible.extend(members)
            need -= len(members)
        else:
            heldout.extend(members)
    if need != 0:
        raise PartitionError(
            f"cannot form exact visible split of {n_visible} at group granularity"
        )
    return tuple(sorted(visible)), tuple(sorted(heldout))


def build_partition(
    profile: ScoreProfile,
    request: Request,
    folds: dict[str, str],
    params: PartitionParams,
    template_of: dict[str, str] | None = None,
) -> Partition:
    """Construct probe-defined pools from a score profile (discovery folds
    only). ``template_of`` maps example_id -> template key; when given, the
    remote sample matches the adjacent pool's template counts."""
    group_of = {e.example_id: e.group for e in request.universe.examples}
    scores = profile.scores
    if set(scores) != set(group_of):
        raise PartitionError("profile does not cover the candidate universe exactly")

    all_s = torch.tensor([scores[c] for c in sorted(scores)], dtype=torch.float64)
    tau_adj = torch.quantile(all_s, params.tau_adj_quantile).item()
    tau_rem = torch.quantile(all_s.abs(), params.tau_rem_abs_quantile).item()

    eligible = [c for c in scores if folds.get(group_of[c]) == "discovery"]

    # Adjacent: top pool_size above tau_adj, deterministic tie-break.
    adj_cands = sorted(
        (c for c in eligible if scores[c] >= tau_adj),
        key=lambda c: (-scores[c], c),
    )
    if len(adj_cands) < params.pool_size:
        raise PartitionError(
            f"only {len(adj_cands)} eligible candidates above tau_adj; "
            f"need {params.pool_size} (preregistered exclusion rule)"
        )
    adjacent = adj_cands[: params.pool_size]

    # Remote: template-matched seeded sample from the near-zero band.
    band = sorted(
        c for c in eligible if abs(scores[c]) <= tau_rem and c not in set(adjacent)
    )
    if len(band) < params.pool_size:
        raise PartitionError(
            f"near-zero band has {len(band)} eligible candidates; "
            f"need {params.pool_size}"
        )
    gen = torch.Generator().manual_seed(params.seed)
    if template_of is None:
        idx = torch.randperm(len(band), generator=gen).tolist()[: params.pool_size]
        remote = sorted(band[i] for i in idx)
    else:
        need: dict[str, int] = {}
        for c in adjacent:
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

    av, ah = _group_split(adjacent, group_of, params.visible_split, params.seed)
    rv, rh = _group_split(remote, group_of, params.visible_split, params.seed + 1)

    body = json.dumps(
        {
            "request": request.request_id,
            "scorer": profile.scorer,
            "pools": [av, ah, rv, rh],
            "params": asdict(params),
            "universe_sha": request.universe.sha,
        },
        sort_keys=True,
    )
    sha = hashlib.sha256(body.encode()).hexdigest()
    return Partition(request.request_id, profile.scorer, av, ah, rv, rh, params, sha)
