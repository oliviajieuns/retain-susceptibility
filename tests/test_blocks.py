import torch

from rsus.blocks import (
    load_params_,
    mlp_down_last_layers,
    save_params,
    set_perturbed_,
    vec_dot,
    vec_norm,
    vec_randn_like,
    vec_unit,
)


def test_block_selection_names(tiny_model):
    spec = mlp_down_last_layers(tiny_model, 1)
    sel = spec.select(tiny_model)
    assert list(sel) == ["model.layers.1.mlp.down_proj.weight"]
    spec2 = mlp_down_last_layers(tiny_model, 2)
    assert len(spec2.select(tiny_model)) == 2


def test_vec_algebra(tiny_model):
    sel = mlp_down_last_layers(tiny_model, 1).select(tiny_model)
    gen = torch.Generator().manual_seed(1)
    v = vec_randn_like(sel, gen)
    u = vec_unit(v)
    assert torch.allclose(vec_norm(u), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(vec_dot(u, u), torch.tensor(1.0, dtype=torch.float64))


def test_perturb_restore_bit_exact(tiny_model):
    sel = mlp_down_last_layers(tiny_model, 1).select(tiny_model)
    gen = torch.Generator().manual_seed(2)
    d = vec_unit(vec_randn_like(sel, gen))
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    saved = save_params(sel)
    set_perturbed_(sel, saved, d, 1e-3)
    assert not torch.equal(next(iter(sel.values())).detach(), saved[next(iter(sel))])
    load_params_(sel, saved)
    for n, p in tiny_model.named_parameters():
        assert torch.equal(p.detach(), before[n]), n
