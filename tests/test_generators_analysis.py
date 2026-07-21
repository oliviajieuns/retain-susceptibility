"""Trajectory runner, objectives, prediction analysis, protection eval."""
import pytest
import torch

from conftest import build_tiny
from rsus.analysis.prediction import (
    auroc,
    cvar_upper,
    per_cell_metrics,
    spearman,
    table1_rows,
    top_k_ids,
)
from rsus.data.substrate import make_substrate
from rsus.evalx.protection import evaluate_protection, first_reaching
from rsus.generators import TrajectoryConfig, objective_names, run_trajectory
from rsus.generators.base import Snapshot, TrajectoryRecord


def test_objective_registry():
    assert {"ga", "graddiff", "npo", "rmu", "simnpo", "idkdpo", "gru"} <= set(objective_names())


@pytest.mark.parametrize("objective", ["ga", "graddiff", "npo", "rmu", "simnpo", "gru"])
def test_trajectory_runs_and_records(objective, tmp_path):
    model = build_tiny(5)
    req, truth = make_substrate(seed=51, n_remote=6)
    retain = [e for e in req.universe.examples if truth[e.example_id] == "remote"]
    cfg = TrajectoryConfig(max_steps=4, checkpoint_every=2, lr=1e-3)
    rec = run_trajectory(model, objective, req, retain, cfg, out_dir=tmp_path / objective)
    assert set(rec.nll0) == {e.example_id for e in req.universe.examples}
    assert [s.step for s in rec.snapshots] == [2, 4]
    assert set(rec.damage_at()) == set(rec.nll0)
    assert (tmp_path / objective / "DONE").exists()
    assert (tmp_path / objective / "damage.json").exists()


def test_idkdpo_needs_and_uses_idk_examples():
    import dataclasses

    from rsus.data.base import Example
    from rsus.losses import IGNORE

    model = build_tiny(7)
    req, truth = make_substrate(seed=53, n_remote=6)
    retain = [e for e in req.universe.examples if truth[e.example_id] == "remote"]
    cfg = TrajectoryConfig(max_steps=2, checkpoint_every=2, lr=1e-3)
    with pytest.raises(ValueError):
        run_trajectory(build_tiny(7), "idkdpo", req, retain, cfg)
    idk = []
    for e in req.forget:
        ids = e.input_ids.clone()
        ids[8:] = torch.arange(3, 3 + ids.numel() - 8)  # fixed alternative answer
        labels = ids.clone()
        labels[:8] = IGNORE
        idk.append(Example(e.example_id + "-idk", ids, labels))
    cfg2 = dataclasses.replace(cfg, idk_examples=idk)
    rec = run_trajectory(model, "idkdpo", req, retain, cfg2)
    assert len(rec.snapshots) == 1


def test_extra_eval_recorded():
    model = build_tiny(8)
    req, truth = make_substrate(seed=54, n_remote=6)
    retain = [e for e in req.universe.examples if truth[e.example_id] == "remote"]
    cfg = TrajectoryConfig(max_steps=2, checkpoint_every=2, lr=1e-3)
    rec = run_trajectory(model, "ga", req, retain, cfg, extra_eval=lambda m: {"probe": 1.0})
    assert rec.snapshots[0].extra == {"probe": 1.0}


def test_canonical_share_tracked():
    from rsus.blocks import mlp_down_last_layers
    from rsus.probe.base import ProbeSpec
    from rsus.probe.finite_diff import canonical_forget_direction
    from rsus.costs import CostRecord

    model = build_tiny(9)
    req, truth = make_substrate(seed=55, n_remote=6)
    retain = [e for e in req.universe.examples if truth[e.example_id] == "remote"]
    block = mlp_down_last_layers(model, 1)
    spec = ProbeSpec(block=block, eta=1e-4, batch_size=4)
    ghat = canonical_forget_direction(model, req, spec, CostRecord())
    cfg = TrajectoryConfig(max_steps=4, checkpoint_every=2, lr=1e-3)
    rec = run_trajectory(model, "ga", req, retain, cfg, track_dir=(block, ghat))
    for s in rec.snapshots:
        assert -1.0 <= s.extra["c_t"] <= 1.0
        assert "alpha_t" in s.extra
    # GA ascends the forget loss, so block displacement should share sign
    # with the canonical ascent direction
    assert rec.snapshots[-1].extra["c_t"] > 0


def test_ga_raises_forget_loss():
    model = build_tiny(6)
    req, truth = make_substrate(seed=52, n_remote=6)
    retain = [e for e in req.universe.examples if truth[e.example_id] == "remote"]
    rec = run_trajectory(model, "ga", req, retain, TrajectoryConfig(max_steps=10, checkpoint_every=10, lr=5e-3))
    # ascent must make forget answers less likely -> adjacent (near-dup) NLL rises
    adj = [c for c in rec.nll0 if truth[c] == "adjacent"]
    dmg = rec.damage_at()
    assert sum(dmg[c] for c in adj) / len(adj) > 0


def test_spearman_auroc_cvar_hand_cases():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert auroc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.9, 0.8], [True, True, False, False]) == 0.0
    assert cvar_upper([1.0, 2.0, 10.0], 0.05) == 10.0
    assert top_k_ids({"a": 3.0, "b": 2.0, "c": 1.0}, 2) == {"a", "b"}


def test_per_cell_and_table1_shapes():
    scores = {f"c{i}": float(10 - i) for i in range(10)}
    damage = {f"c{i}": float(10 - i) + 0.01 * i for i in range(10)}
    cell = per_cell_metrics(scores, damage, k=3)
    assert cell["rho"] > 0.9 and cell["auroc"] == 1.0 and cell["overlap"] == 1.0
    rows = table1_rows(
        {"fd": {"r1": scores, "r2": scores}},
        {"npo": {"r1": damage, "r2": damage}},
        k=3,
    )
    assert "npo_rho" in rows["fd"] and rows["fd"]["auroc"]["mean"] == 1.0


def test_protection_eval_logic():
    rec = TrajectoryRecord(
        "x",
        "r",
        {"a": 1.0, "b": 1.0, "u": 1.0},
        [
            Snapshot(10, {"a": 1.5, "b": 1.1, "u": 1.0}, forget_recall=0.5),
            Snapshot(20, {"a": 2.0, "b": 1.2, "u": 1.25}, forget_recall=0.05),
        ],
    )
    assert first_reaching(rec, 0.10).step == 20
    out = evaluate_protection(rec, native_ids={"a", "b"}, utility_ids={"u"}, recall_max=0.10)
    assert out.reached and out.step == 20
    assert out.native_mean == pytest.approx(0.6)
    assert out.native_cvar == pytest.approx(1.0)  # top-1 of {1.0, 0.2}
    assert out.utility_ret == pytest.approx(0.8)
    assert not evaluate_protection(rec, {"a"}, set(), recall_max=0.01).reached


def test_representation_channel_objectives_run():
    """RepNoise and Circuit Breakers are representation-channel generators: they
    read hidden states, step without error, and move the forget representations."""
    import torch
    from conftest import build_tiny, make_example

    from rsus.data.base import CandidateUniverse, Request
    from rsus.generators.base import TrajectoryConfig, run_trajectory

    for name in ("repnoise", "circuit_breakers"):
        model = build_tiny(5)
        gen = torch.Generator().manual_seed(3)
        forget = [make_example(gen, f"f{i:02d}") for i in range(4)]
        cands = [make_example(gen, f"c{i:02d}") for i in range(12)]
        req = Request.build("rep", forget, CandidateUniverse.freeze(cands))
        cfg = TrajectoryConfig(max_steps=4, checkpoint_every=2, lr=1e-3, batch_size=4,
                               rmu_alpha=1.0, rmu_c=1.0)
        rec = run_trajectory(model, name, req, cands, cfg)
        dmg = rec.damage_at()
        assert rec.objective == name
        assert all(v == v for v in dmg.values())              # finite (no NaN)
        assert any(abs(v) > 1e-6 for v in dmg.values())        # representations actually moved
