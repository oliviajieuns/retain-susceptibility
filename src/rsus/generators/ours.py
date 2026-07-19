"""Profile-guided protection wrapped in the common trajectory interface, so
Table 2 evaluates it exactly like every baseline: snapshots of candidate NLL
and forget recall at theta_0, after calibrated forgetting, and along repair.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rsus.blocks import BlockSpec
from rsus.data.base import Example, Request
from rsus.generators.base import Snapshot, TrajectoryRecord, _candidate_nll, _forget_recall
from rsus.stage1 import Stage1Config, run_stage1
from rsus.stage2 import Stage2Config, run_stage2


@dataclass
class OursConfig:
    stage1: Stage1Config
    stage2: Stage2Config
    batch_size: int = 8
    stage2_snapshots: int = 2   # extra snapshots along repair


def run_ours_trajectory(
    model: torch.nn.Module,
    block: BlockSpec,
    request: Request,
    protect: list[Example],
    remote: list[Example],
    floor_m: float,
    cfg: OursConfig,
    extra_eval=None,
) -> TrajectoryRecord:
    def _extra(m):
        return extra_eval(m) if extra_eval else {}

    rec = TrajectoryRecord("ours", request.request_id, {})
    rec.nll0 = _candidate_nll(model, request, cfg.batch_size)

    res1 = run_stage1(model, request, remote, floor_m, cfg.stage1)
    rec.snapshots.append(
        Snapshot(res1.steps, _candidate_nll(model, request, cfg.batch_size), _forget_recall(model, request), _extra(model))
    )
    if not res1.gate_passed:
        return rec  # no repair without an accepted gate; reach judged on what exists

    chunks = max(1, cfg.stage2_snapshots)
    per = max(1, cfg.stage2.max_steps // chunks)
    done = 0
    step_base = res1.steps
    while done < cfg.stage2.max_steps:
        import dataclasses as _dc

        sub = _dc.replace(cfg.stage2, max_steps=min(per, cfg.stage2.max_steps - done))
        run_stage2(model, block, request, protect, remote, res1.cache, sub)
        done += sub.max_steps
        rec.snapshots.append(
            Snapshot(
                step_base + done,
                _candidate_nll(model, request, cfg.batch_size),
                _forget_recall(model, request),
                _extra(model),
            )
        )
    return rec
