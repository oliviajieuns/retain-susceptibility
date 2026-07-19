"""Probe invariants 1-4 of DESIGN.md §7."""
import dataclasses

import torch

from rsus.probe.base import get_scorer, scorer_names


def _as_vec(profile, order):
    return torch.tensor([profile.scores[c] for c in order], dtype=torch.float64)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def test_registry_names_fixed():
    expected = {
        "fd", "jvp", "vmap_graddot", "streaming_backward", "grad_norm",
        "knn_lexical", "knn_feature", "knn_embed", "last_layer",
        "random_dir", "random_rank", "fd_constrained",
    }
    assert expected <= set(scorer_names())


def test_inv1_fd_matches_backward_graddot_across_eta_grid(tiny_model, req, spec):
    # Error model: O(eta^2) truncation plus a ~1e-8 loss-evaluation noise
    # floor divided by 2*eta (HF Llama caches rotary cos/sin in float32 even
    # for double models). Rank agreement must be exact at every grid point.
    # Grid matches prereg probe.eta_grid; below ~3e-5 the noise floor
    # overtakes the smallest score gaps on this tiny fixture.
    order = [e.example_id for e in req.universe.examples]
    exact = _as_vec(get_scorer("streaming_backward")(tiny_model, req, spec), order)
    for eta in (1e-3, 3e-4, 1e-4):
        s = dataclasses.replace(spec, eta=eta)
        fd = _as_vec(get_scorer("fd")(tiny_model, req, s), order)
        assert _spearman(fd, exact) == 1.0, eta
        assert torch.allclose(fd, exact, atol=10.0 * eta**2 + 1e-8 / eta), eta


def test_inv2_jvp_equals_backward_graddot(tiny_model, req, spec):
    order = [e.example_id for e in req.universe.examples]
    a = _as_vec(get_scorer("jvp")(tiny_model, req, spec), order)
    b = _as_vec(get_scorer("streaming_backward")(tiny_model, req, spec), order)
    assert torch.allclose(a, b, atol=1e-9)


def test_inv3_vmap_equals_backward_graddot(tiny_model, req, spec):
    order = [e.example_id for e in req.universe.examples]
    a = _as_vec(get_scorer("vmap_graddot")(tiny_model, req, spec), order)
    b = _as_vec(get_scorer("streaming_backward")(tiny_model, req, spec), order)
    assert torch.allclose(a, b, atol=1e-9)


def test_inv4_fd_is_side_effect_free(tiny_model, req, spec):
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    get_scorer("fd")(tiny_model, req, spec)
    for n, p in tiny_model.named_parameters():
        assert torch.equal(p.detach(), before[n]), n
    assert all(p.grad is None for p in tiny_model.parameters())


def test_fd_cost_accounting(tiny_model, req, spec):
    prof = get_scorer("fd")(tiny_model, req, spec)
    n_forget_batches = 1   # 4 forget examples, batch_size 5
    n_cand_batches = 3     # 12 candidates, batch_size 5 -> 3 batches per sweep
    assert prof.cost.bwd_passes == n_forget_batches
    assert prof.cost.fwd_passes == n_forget_batches + 2 * n_cand_batches
    assert prof.cost.tokens_fwd > 0 and prof.cost.wall_s > 0


def test_random_dir_differs_from_fd(tiny_model, req, spec):
    order = [e.example_id for e in req.universe.examples]
    fd = _as_vec(get_scorer("fd")(tiny_model, req, spec), order)
    rd = _as_vec(get_scorer("random_dir")(tiny_model, req, spec), order)
    assert not torch.allclose(fd, rd, atol=1e-6)


def test_knn_embed_needs_encoder_or_text(tiny_model, req, spec):
    import pytest

    from rsus.probe.baselines import set_embed_encoder

    set_embed_encoder(None)
    with pytest.raises((NotImplementedError, ValueError)):
        get_scorer("knn_embed")(tiny_model, req, spec)


def test_knn_embed_with_injected_encoder(tiny_model, req, spec):
    import torch

    from rsus.probe.baselines import set_embed_encoder

    def bag_encoder(examples):
        from conftest import VOCAB

        out = torch.zeros(len(examples), VOCAB)
        for i, e in enumerate(examples):
            for t in e.input_ids.tolist():
                out[i, t] += 1.0
        return out

    set_embed_encoder(bag_encoder)
    try:
        prof = get_scorer("knn_embed")(tiny_model, req, spec)
        assert set(prof.scores) == {e.example_id for e in req.universe.examples}
        assert all(-1.0 <= v <= 1.0 for v in prof.scores.values())
    finally:
        set_embed_encoder(None)
