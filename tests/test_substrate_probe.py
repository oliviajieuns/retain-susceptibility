"""Substrate mechanism tests and the N2 scorers."""
import torch

from rsus.blocks import BlockSpec, mlp_down_last_layers
from rsus.data.substrate import make_substrate
from rsus.probe.base import ProbeSpec, get_scorer


def _auc(scores: dict[str, float], truth: dict[str, str]) -> float:
    pos = [scores[c] for c in scores if truth[c] == "adjacent"]
    neg = [scores[c] for c in scores if truth[c] == "remote"]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _spec(model) -> ProbeSpec:
    return ProbeSpec(block=mlp_down_last_layers(model, 1), eta=1e-4, batch_size=6)


def test_fd_separates_planted_adjacency(tiny_model):
    req, truth = make_substrate(seed=11)
    prof = get_scorer("fd")(tiny_model, req, _spec(tiny_model))
    adj = [prof.scores[c] for c in prof.scores if truth[c] == "adjacent"]
    rem = [prof.scores[c] for c in prof.scores if truth[c] == "remote"]
    assert sum(adj) / len(adj) > sum(rem) / len(rem)
    assert _auc(prof.scores, truth) >= 0.9


def test_knn_feature_separates_near_duplicates(tiny_model):
    req, truth = make_substrate(seed=12)
    prof = get_scorer("knn_feature")(tiny_model, req, _spec(tiny_model))
    assert _auc(prof.scores, truth) >= 0.8


def test_last_layer_matches_autograd_on_head_block(tiny_model):
    req, _ = make_substrate(seed=13)
    head_spec = ProbeSpec(block=BlockSpec(r"lm_head\.weight"), eta=1e-4, batch_size=6)
    closed = get_scorer("last_layer")(tiny_model, req, head_spec)
    exact = get_scorer("streaming_backward")(tiny_model, req, head_spec)
    order = sorted(closed.scores)
    a = torch.tensor([closed.scores[c] for c in order], dtype=torch.float64)
    b = torch.tensor([exact.scores[c] for c in order], dtype=torch.float64)
    assert torch.allclose(a, b, atol=1e-9)


def test_fd_constrained_runs_and_correlates(tiny_model):
    req, truth = make_substrate(seed=14)
    spec = _spec(tiny_model)
    fd = get_scorer("fd")(tiny_model, req, spec)
    con = get_scorer("fd_constrained")(tiny_model, req, spec)
    assert set(con.scores) == set(fd.scores)
    assert all(torch.isfinite(torch.tensor(v)) for v in con.scores.values())
    order = sorted(fd.scores)
    a = torch.tensor([fd.scores[c] for c in order]).argsort().argsort().double()
    b = torch.tensor([con.scores[c] for c in order]).argsort().argsort().double()
    rho = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    assert rho > 0.0
    # the constrained probe should still separate planted adjacency
    assert _auc(con.scores, truth) >= 0.8
