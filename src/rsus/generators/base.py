"""Trajectory runner for third-party generators (Table 1 ground truth) and
protection baselines (Table 2).

Records every candidate's NLL at theta_0 and at each saved checkpoint, so
damage d_t(x) = ell(x; theta_t) - ell(x; theta_0) is available at every
horizon; the terminal-budget checkpoint is the paper's primary prediction
horizon. Writes damage.json plus a DONE marker consumed by sealing.unseal.
Objectives never see susceptibility scores; their retain stream is passed in
explicitly by the caller.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import torch

from rsus.costs import CostRecord, Meter
from rsus.data.base import Example, Request, collate
from rsus.evalx.metrics import answer_token_recall
from rsus.losses import seq_mean_answer_nll


class Objective(Protocol):
    def step(self) -> float: ...  # one optimizer update; returns loss value


ObjectiveFactory = Callable[[torch.nn.Module, Request, list[Example], "TrajectoryConfig"], Objective]
_OBJECTIVES: dict[str, ObjectiveFactory] = {}


def register_objective(name: str):
    def deco(fn: ObjectiveFactory) -> ObjectiveFactory:
        if name in _OBJECTIVES:
            raise ValueError(f"duplicate objective: {name}")
        _OBJECTIVES[name] = fn
        return fn

    return deco


def objective_names() -> list[str]:
    return sorted(_OBJECTIVES)


@dataclass
class TrajectoryConfig:
    max_steps: int = 60
    checkpoint_every: int = 10
    batch_size: int = 8
    lr: float = 5e-3
    seed: int = 0
    beta: float = 1.0          # NPO / SimNPO / IdkDPO temperature
    simnpo_gamma: float = 0.0  # SimNPO margin bias
    rmu_alpha: float = 10.0    # RMU retain weight
    rmu_c: float = 3.0         # RMU control-vector magnitude
    idk_examples: list | None = None  # IdkDPO preferred responses (per forget example)


@dataclass
class Snapshot:
    step: int
    nll: dict[str, float]
    forget_recall: float
    extra: dict = field(default_factory=dict)


@dataclass
class TrajectoryRecord:
    objective: str
    request_id: str
    nll0: dict[str, float]
    snapshots: list[Snapshot] = field(default_factory=list)
    cost: CostRecord = field(default_factory=CostRecord)

    def damage_at(self, index: int = -1) -> dict[str, float]:
        snap = self.snapshots[index]
        return {c: snap.nll[c] - self.nll0[c] for c in self.nll0}

    def terminal(self) -> Snapshot:
        return self.snapshots[-1]


def _candidate_nll(model, request: Request, batch_size: int) -> dict[str, float]:
    out: dict[str, float] = {}
    with torch.no_grad():
        for batch in request.universe.batches(batch_size):
            for eid, v in zip(
                batch["example_ids"], seq_mean_answer_nll(model, batch).tolist()
            ):
                out[eid] = v
    return out


def _forget_recall(model, request: Request) -> float:
    batch = collate(list(request.forget))
    return float(answer_token_recall(model, batch).mean())


def run_trajectory(
    model: torch.nn.Module,
    objective: str,
    request: Request,
    retain: list[Example],
    cfg: TrajectoryConfig,
    out_dir: str | Path | None = None,
    extra_eval=None,
    track_dir=None,
    stop_at_recall: float | None = None,
) -> TrajectoryRecord:
    """``extra_eval(model) -> dict`` is evaluated at every snapshot (e.g.
    paraphrase recall, utility probes) since checkpoint weights are not
    persisted. ``track_dir=(BlockSpec, ghat)`` additionally records, per
    snapshot, the signed canonical share c_t = <Delta_B, ghat>/||Delta_B||
    and alpha_t = <Delta_B, ghat> (paper eq:canonical-share) from the saved
    block displacement -- the inputs to the optimizer-transfer mechanism
    table, with no weight storage. ``stop_at_recall`` ends the trajectory at
    the first checkpoint whose forget recall is at or below the threshold,
    leaving the model at that reaching state (for engine+repair pipelines)."""
    from rsus.blocks import save_params, vec_dot

    factory = _OBJECTIVES[objective]
    rec = TrajectoryRecord(objective, request.request_id, {})
    with Meter(rec.cost):
        rec.nll0 = _candidate_nll(model, request, cfg.batch_size)
        theta0_B = None
        if track_dir is not None:
            block, ghat = track_dir
            sel = block.select(model)
            theta0_B = save_params(sel)
        obj = factory(model, request, retain, cfg)
        for t in range(1, cfg.max_steps + 1):
            obj.step()
            if t % cfg.checkpoint_every == 0 or t == cfg.max_steps:
                extra = extra_eval(model) if extra_eval else {}
                if theta0_B is not None:
                    delta = {n: p.detach() - theta0_B[n] for n, p in sel.items()}
                    alpha = float(vec_dot(delta, ghat))
                    dnorm = float(
                        torch.sqrt(sum((d * d).sum() for d in delta.values()))
                    )
                    extra = {
                        **extra,
                        "alpha_t": alpha,
                        "c_t": alpha / dnorm if dnorm > 0 else 0.0,
                    }
                rec.snapshots.append(
                    Snapshot(
                        t,
                        _candidate_nll(model, request, cfg.batch_size),
                        _forget_recall(model, request),
                        extra,
                    )
                )
                if stop_at_recall is not None and rec.snapshots[-1].forget_recall <= stop_at_recall:
                    break
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "objective": objective,
            "request": request.request_id,
            "nll0": rec.nll0,
            "snapshots": [
                {"step": s.step, "forget_recall": s.forget_recall, "extra": s.extra, "nll": s.nll}
                for s in rec.snapshots
            ],
        }
        with open(out / "damage.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)
        (out / "DONE").touch()
    return rec
