"""Stage 1 + Stage 2 invariants and the toy end-to-end flow
(DESIGN.md §7, items 6-10): memorize -> forget to the calibrated floor ->
repair adjacent behavior without breaching the anchored budgets."""
import math

import pytest
import torch

from conftest import build_tiny
from rsus.blocks import mlp_down_last_layers
from rsus.data.base import collate
from rsus.data.substrate import make_substrate
from rsus.losses import seq_mean_answer_nll
from rsus.refcache import assert_aligned, forget_seq_losses, forget_tok_losses
from rsus.stage1 import Stage1Config, calibrate_floor, clip_c, run_stage1
from rsus.stage2 import Stage2Config, run_stage2

S1 = Stage1Config(lr=5e-3, max_steps=800, eval_every=20, batch_size=8, seed=0)


def memorize(model, examples, steps=400, lr=5e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    batch = collate(examples)
    for _ in range(steps):
        loss = seq_mean_answer_nll(model, batch).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.zero_grad(set_to_none=True)
    return float(loss.detach())


@pytest.fixture(scope="module")
def world():
    base = build_tiny(0)   # pristine init: the pre-injection floor reference
    model = build_tiny(0)
    req, truth = make_substrate(seed=21)
    by_id = {e.example_id: e for e in req.universe.examples}
    adj = [by_id[c] for c in sorted(by_id) if truth[c] == "adjacent"]
    rem = [by_id[c] for c in sorted(by_id) if truth[c] == "remote"]

    final = memorize(model, list(req.forget) + adj + rem)
    assert final < 0.5, f"memorization failed (loss {final})"
    mem = {k: v.detach().clone() for k, v in model.state_dict().items()}

    m = calibrate_floor(base, req)
    res1 = run_stage1(model, req, remote=rem, floor_m=m, cfg=S1)
    assert res1.gate_passed, f"stage-1 gate not reached: {res1.history[-3:]}"

    post1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return {
        "req": req, "adj": adj, "rem": rem, "m": m, "cache": res1.cache,
        "post1": post1, "mem": mem,
    }


def load_post1(world):
    model = build_tiny(0)
    model.load_state_dict(world["post1"])
    return model


# ---- Stage 1 units ----------------------------------------------------------

def test_clip_forward_exact_gradient_stops():
    x = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
    y = clip_c(x, 4.0)
    assert torch.equal(y.detach(), torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64))
    y.sum().backward()
    assert x.grad.tolist() == [1.0, 1.0, 0.0]


def test_stage1_gate_and_floor(world):
    model = load_post1(world)
    _, cur = forget_seq_losses(model, world["req"], 8)
    assert float(cur.min()) >= world["m"]
    assert world["cache"] is not None and world["cache"].floor_m == world["m"]


def test_stage1_dual_ascent_reacts_to_violation():
    # remote stream = forget clones: ascent inflates remote loss, h > 0,
    # so the multiplier must rise.
    model = build_tiny(3)
    req, _ = make_substrate(seed=31)
    memorize(model, list(req.forget), steps=150)
    cfg = Stage1Config(lr=5e-3, max_steps=30, eval_every=50, seed=0)
    res = run_stage1(model, req, remote=list(req.forget), floor_m=6.0, cfg=cfg)
    assert res.lam > 0.0


# ---- RefCache alignment (invariant 9) ---------------------------------------

def test_refcache_alignment(world):
    model = load_post1(world)
    cache = world["cache"]
    seq_ids, _ = forget_seq_losses(model, world["req"], 3)   # different batch size
    tok_index, _ = forget_tok_losses(model, world["req"], 3)
    assert_aligned(cache, seq_ids, tok_index)
    with pytest.raises(ValueError):
        assert_aligned(cache, tuple(reversed(seq_ids)), tok_index)


# ---- Stage 2 invariants ------------------------------------------------------

def test_inv7_projection_orthogonal_at_refresh(world):
    model = load_post1(world)
    block = mlp_down_last_layers(model, 1)
    cfg = Stage2Config(max_steps=12, refresh_k=1, delta_seq_sq=1.0, delta_tok_sq=1.0)
    res = run_stage2(
        model, block, world["req"], world["adj"], world["rem"], world["cache"], cfg
    )
    accepted = [e for e in res.events if e.accepted and e.max_basis_cos is not None]
    assert accepted, "no accepted refresh events"
    assert max(e.max_basis_cos for e in accepted) < 1e-8


def test_inv6_acceptance_rolls_back_and_shrinks(world):
    model = load_post1(world)
    block = mlp_down_last_layers(model, 1)
    # Adversarial repair: 'adjacent' pool IS the forget set, projection and
    # penalty disabled -- only the acceptance backstop remains.
    cfg = Stage2Config(
        max_steps=40,
        refresh_k=2,
        delta_seq_sq=1e-6,
        delta_tok_sq=1e50,
        projection=False,
        guard_enabled=False,
        eta2=5e-3,
    )
    res = run_stage2(
        model, block, world["req"], list(world["req"].forget), world["rem"],
        world["cache"], cfg,
    )
    assert res.n_rejected >= 1
    assert res.eta2_final < cfg.eta2
    for e in res.events:
        if e.accepted:
            assert e.d_seq <= cfg.delta_seq_sq + 1e-15


def test_s2s_baseline_runs_two_stages(world):
    from rsus.generators.s2s import S2SConfig, run_s2s_trajectory
    from rsus.partition import PartitionParams, make_folds
    from rsus.stage2 import Stage2Config

    model = build_tiny(0)
    model.load_state_dict(world["mem"])   # memorized, pre-stage-1 state
    req = world["req"]
    folds = {e.group: "discovery" for e in req.universe.examples}
    cfg = S2SConfig(
        stage1=Stage1Config(lr=5e-3, max_steps=800, eval_every=20, seed=0),
        stage2=Stage2Config(max_steps=10, refresh_k=1, delta_seq_sq=1e-2),
        partition=PartitionParams(pool_size=4, min_pool_size=3, tau_rem_abs_quantile=0.8),
    )
    from rsus.blocks import mlp_down_last_layers as blk

    rec = run_s2s_trajectory(model, blk(model, 1), req, folds, world["m"], cfg)
    assert rec.objective == "s2s"
    assert len(rec.snapshots) == 2, "gate did not pass or repair missing"
    assert rec.snapshots[-1].forget_recall <= 0.2


def test_inv10_toy_e2e_repair_without_breach(world):
    model = load_post1(world)
    block = mlp_down_last_layers(model, 1)
    adj_batch = collate(world["adj"])
    with torch.no_grad():
        adj_before = float(seq_mean_answer_nll(model, adj_batch).mean())

    cfg = Stage2Config(max_steps=40, refresh_k=1, delta_seq_sq=1e-2, delta_tok_sq=1e-1)
    res = run_stage2(
        model, block, world["req"], world["adj"], world["rem"], world["cache"], cfg
    )

    with torch.no_grad():
        adj_after = float(seq_mean_answer_nll(model, adj_batch).mean())
    assert adj_after < adj_before, (adj_before, adj_after)

    # every accepted refresh checkpoint satisfies the bound premise
    for e in res.events:
        if e.accepted:
            assert e.d_seq <= cfg.delta_seq_sq + 1e-15

    # forget floor holds up to the per-coordinate bound sqrt(n)*delta
    n = len(world["req"].forget)
    tol = math.sqrt(n * cfg.delta_seq_sq)
    _, cur = forget_seq_losses(model, world["req"], 8)
    assert float(cur.min()) >= world["m"] - tol - 0.05, (float(cur.min()), world["m"])


def test_project_empty_basis_is_identity():
    """Regression: chunked stage-2 entry with a rejecting first refresh used to
    reach _project with an empty basis and crash on eig.min() of a 0x0 Gram."""
    import torch

    from rsus.stage2 import _project

    v = {"w": torch.arange(3, dtype=torch.float64)}
    out = _project(v, [], ridge_scale=0.0, cond_max=1e8)
    assert torch.equal(out["w"], v["w"])


def test_engine_repaired_pipeline(req):
    """ga engine (trivial recall threshold) -> sealed refs -> guarded repair;
    snapshots continue past the engine's reaching checkpoint."""
    from conftest import build_tiny

    from rsus.blocks import mlp_down_last_layers
    from rsus.generators.base import TrajectoryConfig
    from rsus.generators.repaired import RepairedConfig, run_engine_repaired
    from rsus.stage2 import Stage2Config

    model = build_tiny(3)
    cands = list(req.universe.examples)
    cfg = RepairedConfig(
        engine_cfg=TrajectoryConfig(max_steps=4, checkpoint_every=2, lr=1e-3, batch_size=4),
        stage2=Stage2Config(max_steps=4, refresh_k=2, batch_size=4),
        recall_max=1.0,  # trivially reached at the first checkpoint
        batch_size=4,
        stage2_snapshots=2,
    )
    rec = run_engine_repaired(
        model, mlp_down_last_layers(model, 1), req,
        retain=cands[:4], protect=cands[4:8], remote=cands[8:],
        floor_m=0.05, engine="ga", cfg=cfg,
    )
    assert rec.objective == "ga_repaired"
    engine_last = 2  # stop_at_recall ends at the first checkpoint
    assert rec.snapshots[0].step == engine_last
    assert rec.snapshots[-1].step == engine_last + cfg.stage2.max_steps
    assert set(rec.nll0) == set(rec.snapshots[-1].nll)


def test_crossed_sweep_labels_and_runs():
    """crossed_sweep produces none/matched/mismatched/random cells per parent and
    labels the selector-to-channel match correctly (gradient probe on a
    loss-gradient parent = matched; representation probe = mismatched)."""
    import dataclasses as _dc
    import sys
    from pathlib import Path

    import torch
    from conftest import build_tiny, make_example

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "gate_1p5b"))
    from crossed_protection import crossed_sweep

    from rsus.blocks import mlp_down_last_layers
    from rsus.data.base import CandidateUniverse, Request
    from rsus.generators.base import TrajectoryConfig
    from rsus.partition import make_folds
    from rsus.probe.base import ProbeSpec, get_scorer
    from rsus.stage2 import Stage2Config

    model = build_tiny(7)
    gen = torch.Generator().manual_seed(1)
    forget = [make_example(gen, f"f{i:02d}") for i in range(4)]
    cands = [_dc.replace(make_example(gen, f"c{g}_{j}"), group=f"grp{g}")
             for g in range(10) for j in range(4)]
    req = Request.build("x", forget, CandidateUniverse.freeze(cands))
    by_id = {e.example_id: e for e in cands}
    folds = make_folds({e.example_id: e.group for e in cands}, 0.5, 0)
    audit_ids = {e.example_id for e in cands if folds[e.group] == "audit"}
    block = mlp_down_last_layers(model, 1)
    spec = ProbeSpec(block=block, eta=1e-4, batch_size=4, n_dirs=8, norm_eta=1e-3)

    state0 = {k: v.clone() for k, v in model.state_dict().items()}

    def fresh():
        m = build_tiny(7)
        m.load_state_dict(state0)
        return m

    sels = ["grad_norm", "knn_feature"]
    sel_scores = {s: get_scorer(s)(fresh(), req, spec).scores for s in sels}
    sel_scores["random"] = get_scorer("random_rank")(fresh(), req, spec).scores
    s2 = Stage2Config(max_steps=4, refresh_k=2, batch_size=4)

    def make_gcfg(_):
        return TrajectoryConfig(max_steps=4, checkpoint_every=2, lr=1e-3, batch_size=4)

    payload = crossed_sweep(fresh, block, req, by_id, folds, audit_ids, cands[:4], 0.05,
                            ["graddiff"], sels, sel_scores, make_gcfg, s2,
                            recall_max=1.0, pool_size=3, seed=0, log=lambda *_: None)
    m = {r["selector"]: r["match"] for r in payload["results"]}
    assert m["grad_norm"] == "matched"          # gradient probe on loss-gradient parent
    assert m.get("random") == "random"
    assert any(r["selector"] == "none" for r in payload["results"])
    if "knn_feature" in m:
        assert m["knn_feature"] == "mismatched"
    assert all("cvar_dnll" in r for r in payload["results"])
