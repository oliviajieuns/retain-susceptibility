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

    m = calibrate_floor(base, req)
    res1 = run_stage1(model, req, remote=rem, floor_m=m, cfg=S1)
    assert res1.gate_passed, f"stage-1 gate not reached: {res1.history[-3:]}"

    post1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return {"req": req, "adj": adj, "rem": rem, "m": m, "cache": res1.cache, "post1": post1}


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
