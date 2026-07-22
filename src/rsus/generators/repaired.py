"""Guarded repair as post-processing for any single-stage unlearning engine.

Motivated by the 1.5B gate: the floor-ascent stage-1 cannot reach the common
argmax-recall criterion without catastrophic collateral (final1b/final2), while
NPO reaches it cheaply (1.15 nats audit dNLL) — so let the engine do the
forgetting and keep only the guarded-repair stage: run the engine until its
first criterion-reaching checkpoint, seal the forget-loss references at that
state, then repair the protect partition under the one-sided anchored guard
(no resurrection of forgotten content). Snapshots follow the common
trajectory interface, so Table 2 evaluates the pipeline with mode="last"
exactly like the other two-stage arms.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rsus.blocks import BlockSpec
from rsus.data.base import Example, Request
from rsus.generators.base import (
    Snapshot,
    TrajectoryConfig,
    TrajectoryRecord,
    _candidate_nll,
    _forget_recall,
    run_trajectory,
)
from rsus.refcache import build_ref_cache
from rsus.stage2 import Stage2Config, run_stage2


@dataclass
class RepairedConfig:
    engine_cfg: TrajectoryConfig
    stage2: Stage2Config
    recall_max: float = 0.10
    batch_size: int = 8
    stage2_snapshots: int = 4


def run_engine_repaired(
    model: torch.nn.Module,
    block: BlockSpec,
    request: Request,
    retain: list[Example],
    protect: list[Example],
    remote: list[Example],
    floor_m: float,
    engine: str,
    cfg: RepairedConfig,
    extra_eval=None,
    log=None,
) -> TrajectoryRecord:
    # Phase 1: the engine forgets, stopping at its first criterion-reaching
    # checkpoint so the repair starts from exactly the evaluated state.
    eng = run_trajectory(model, engine, request, retain, cfg.engine_cfg,
                         extra_eval=extra_eval, stop_at_recall=cfg.recall_max)
    return run_repair_from_reached(
        model,
        block,
        request,
        protect,
        remote,
        floor_m,
        engine,
        cfg,
        eng,
        extra_eval=extra_eval,
        log=log,
    )


def run_repair_from_reached(
    model: torch.nn.Module,
    block: BlockSpec,
    request: Request,
    protect: list[Example],
    remote: list[Example],
    floor_m: float,
    engine: str,
    cfg: RepairedConfig,
    engine_record: TrajectoryRecord,
    extra_eval=None,
    log=None,
) -> TrajectoryRecord:
    """Run only guarded repair from an already reached parent checkpoint.

    ``model`` must contain the terminal weights of ``engine_record``.  This
    split lets a selector sweep execute a parent once, save only its small
    trainable block, and replay identical repair starts for many protect pools.
    It changes no optimization step relative to :func:`run_engine_repaired`.
    """
    if engine_record.request_id != request.request_id:
        raise ValueError(
            f"engine record request {engine_record.request_id!r} does not match "
            f"{request.request_id!r}"
        )
    rec = TrajectoryRecord(
        f"{engine}_repaired",
        request.request_id,
        dict(engine_record.nll0),
        list(engine_record.snapshots),
    )
    eng = engine_record
    reached = bool(eng.snapshots) and eng.snapshots[-1].forget_recall <= cfg.recall_max
    if log is not None and eng.snapshots:
        s = eng.snapshots[-1]
        log(f"  engine {engine}: step={s.step} forget_recall={s.forget_recall:.3f}"
            f" reached={reached}")
    if not reached:
        return rec  # nothing to repair against; reach judged on what exists

    # Phase 2: seal references at the reached state, then guarded repair of the
    # protect partition (one-sided anchor: repair may not resurrect the forget
    # set below its sealed losses).
    cache = build_ref_cache(model, request, cfg.batch_size, floor_m)
    step_base = eng.snapshots[-1].step
    per = max(1, cfg.stage2.max_steps // max(1, cfg.stage2_snapshots))
    pids = [e.example_id for e in protect]

    def _snapshot(step_done: int) -> None:
        rec.snapshots.append(
            Snapshot(
                step_base + step_done,
                _candidate_nll(model, request, cfg.batch_size),
                _forget_recall(model, request),
                extra_eval(model) if extra_eval else {},
            )
        )
        if log is not None:
            s = rec.snapshots[-1]
            mean_d = sum(s.nll[c] - rec.nll0[c] for c in rec.nll0) / len(rec.nll0)
            known = [c for c in pids if c in rec.nll0]
            prot_d = (sum(s.nll[c] - rec.nll0[c] for c in known) / len(known)) if known else float("nan")
            log(f"  repair chunk: step={s.step} forget_recall={s.forget_recall:.3f}"
                f" mean_dnll_all={mean_d:+.3f} mean_dnll_protect={prot_d:+.3f}")

    # single continuous stage-2 run: momentum and guard multipliers persist
    # across snapshots (chunked re-invocation reset both, freezing progress at
    # the budget boundary and shifting divergence onsets with the chunk size)
    run_stage2(model, block, request, protect, remote, cache, cfg.stage2,
               snapshot_every=per, snapshot_hook=_snapshot)
    if not rec.snapshots or rec.snapshots[-1].step != step_base + cfg.stage2.max_steps:
        _snapshot(cfg.stage2.max_steps)
    return rec
