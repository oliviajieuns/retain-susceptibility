"""Table-2 protection outcomes: reach at the common forget criterion, first
criterion-reaching checkpoint, native mean/CVaR damage, utility retention.
Conditional cells always travel with reach; the caller reports
n_reach/n_total alongside.
"""
from __future__ import annotations

from dataclasses import dataclass

from rsus.analysis.prediction import cvar_upper
from rsus.generators.base import Snapshot, TrajectoryRecord


@dataclass
class ProtectionOutcome:
    reached: bool
    step: int | None
    native_mean: float | None
    native_cvar: float | None
    utility_ret: float | None
    extra: dict | None = None   # snapshot extra_eval metrics (e.g. para_recall)


def first_reaching(record: TrajectoryRecord, recall_max: float) -> Snapshot | None:
    for snap in record.snapshots:
        if snap.forget_recall <= recall_max:
            return snap
    return None


def last_reaching(record: TrajectoryRecord, recall_max: float) -> Snapshot | None:
    """Latest reaching checkpoint. For a two-stage procedure whose repair
    phase keeps the criterion satisfied (the guard's job), the procedure's
    output is its last emitted checkpoint, not the pre-repair one."""
    hit = None
    for snap in record.snapshots:
        if snap.forget_recall <= recall_max:
            hit = snap
    return hit


def evaluate_protection(
    record: TrajectoryRecord,
    native_ids: set[str],
    utility_ids: set[str],
    recall_max: float = 0.10,
    cvar_frac: float = 0.05,
    mode: str = "first",
) -> ProtectionOutcome:
    picker = {"first": first_reaching, "last": last_reaching}[mode]
    snap = picker(record, recall_max)
    if snap is None:
        return ProtectionOutcome(False, None, None, None, None)
    dmg = {c: snap.nll[c] - record.nll0[c] for c in record.nll0}
    native = [dmg[c] for c in sorted(native_ids)]
    if not native:
        raise ValueError("empty native audit")
    utility = [dmg[c] for c in sorted(utility_ids)] if utility_ids else []
    n0 = sum(record.nll0[c] for c in sorted(utility_ids)) if utility_ids else 0.0
    nt = sum(snap.nll[c] for c in sorted(utility_ids)) if utility_ids else 0.0
    return ProtectionOutcome(
        True,
        snap.step,
        sum(native) / len(native),
        cvar_upper(native, cvar_frac),
        (n0 / nt) if utility_ids and nt > 0 else None,
        extra=snap.extra or None,
    )
