"""Guard algebra invariants (DESIGN.md §7, item 5)."""
import pytest
import torch

from rsus.guards import (
    GuardKind,
    budget_ok,
    drift_mass_bound,
    drifted_mass,
    guard_penalty,
)


def _rand(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g, dtype=torch.float64)


def test_one_sided_zero_at_and_above_band():
    refs = _rand(50, 0)
    for shift in (0.0, 0.3):  # at reference and above (extra forgetting)
        losses = refs + shift
        assert float(guard_penalty(losses, refs, GuardKind.ONE_SIDED)) == 0.0
    # slack absorbs small downward drift
    losses = refs - 0.05
    assert float(guard_penalty(losses, refs, GuardKind.ONE_SIDED, eps=0.1)) == 0.0
    assert float(guard_penalty(losses, refs, GuardKind.ONE_SIDED, eps=0.0)) > 0.0


def test_one_sided_gradient_only_below_band():
    refs = _rand(10, 1)
    losses = refs.clone().requires_grad_(True)
    with torch.no_grad():
        losses[0] -= 1.0   # one identity drifts down
        losses[1] += 1.0   # one drifts up (unpenalized)
    guard_penalty(losses, refs, GuardKind.ONE_SIDED).backward()
    assert losses.grad[0] != 0.0
    assert torch.all(losses.grad[1:] == 0.0)


def test_symmetric_upper_bounds_sorted_w2():
    for seed in range(20):
        u, r = _rand(64, seed), _rand(64, 1000 + seed)
        sym = guard_penalty(u, r, GuardKind.SYMMETRIC)
        w2 = guard_penalty(u, r, GuardKind.SORTED_PROFILE)
        assert float(sym) >= float(w2) - 1e-12


def test_sorted_blind_to_permutation_one_sided_is_not():
    refs = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    swapped = torch.tensor([2.0, 1.0, 4.0, 3.0], dtype=torch.float64)  # same multiset
    assert float(guard_penalty(swapped, refs, GuardKind.SORTED_PROFILE)) == 0.0
    assert float(guard_penalty(swapped, refs, GuardKind.ONE_SIDED)) > 0.0


def test_chebyshev_drift_bound_holds_empirically():
    for seed in range(30):
        g = torch.Generator().manual_seed(seed)
        refs = _rand(200, seed)
        losses = refs - torch.rand(200, generator=g, dtype=torch.float64) * 0.5
        d = float(guard_penalty(losses, refs, GuardKind.ONE_SIDED))
        for margin in (0.1, 0.25, 0.4):
            assert drifted_mass(losses, refs, 0.0, margin) <= drift_mass_bound(d, margin) + 1e-12


def test_budget_ok_and_validation():
    assert budget_ok(0.01, 0.01) and not budget_ok(0.0100001, 0.01)
    with pytest.raises(ValueError):
        guard_penalty(torch.zeros(3), torch.zeros(4))
    with pytest.raises(ValueError):
        drift_mass_bound(0.1, 0.0)
