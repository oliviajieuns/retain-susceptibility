"""S2S baseline: faithful autoregressive re-implementation of the split-aware
two-stage comparison method (Cheng et al.), assembled from this repo's
machinery with ITS design choices: similarity partition front-end
(representation kNN), the shared calibrated floor values, and a
distribution-level (sorted) sequence-only guard during repair. Serves only as
a comparison baseline; no component of our profile is attributed to it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rsus.blocks import BlockSpec
from rsus.data.base import Request
from rsus.generators.base import Snapshot, TrajectoryRecord, _candidate_nll, _forget_recall
from rsus.guards import GuardKind
from rsus.partition import PartitionParams, build_partition
from rsus.probe.base import ProbeSpec, get_scorer
from rsus.stage1 import Stage1Config, run_stage1
from rsus.stage2 import Stage2Config, run_stage2


@dataclass
class S2SConfig:
    stage1: Stage1Config
    stage2: Stage2Config          # guard_kind/token_level are overridden below
    partition: PartitionParams
    batch_size: int = 8


def run_s2s_trajectory(
    model: torch.nn.Module,
    block: BlockSpec,
    request: Request,
    folds: dict[str, str],
    floor_m: float,
    cfg: S2SConfig,
    probe_spec: ProbeSpec | None = None,
    extra_eval=None,
) -> TrajectoryRecord:
    import dataclasses as _dc

    def _extra(m):
        return extra_eval(m) if extra_eval else {}

    rec = TrajectoryRecord("s2s", request.request_id, {})
    rec.nll0 = _candidate_nll(model, request, cfg.batch_size)

    # Similarity partition front-end (its native construction).
    spec = probe_spec or ProbeSpec(block=block, eta=1e-4, batch_size=cfg.batch_size)
    prof = get_scorer("knn_feature")(model, request, spec)
    part = build_partition(prof, request, folds, cfg.partition)
    by_id = {e.example_id: e for e in request.universe.examples}
    protect = [by_id[c] for c in part.protect]
    remote = [by_id[c] for c in part.remote_stream]

    res1 = run_stage1(model, request, remote, floor_m, cfg.stage1)
    rec.snapshots.append(
        Snapshot(res1.steps, _candidate_nll(model, request, cfg.batch_size), _forget_recall(model, request), _extra(model))
    )
    if not res1.gate_passed:
        return rec

    s2 = _dc.replace(cfg.stage2, guard_kind=GuardKind.SORTED_PROFILE, token_level=False)
    run_stage2(model, block, request, protect, remote, res1.cache, s2)
    rec.snapshots.append(
        Snapshot(
            res1.steps + s2.max_steps,
            _candidate_nll(model, request, cfg.batch_size),
            _forget_recall(model, request),
            _extra(model),
        )
    )
    return rec
