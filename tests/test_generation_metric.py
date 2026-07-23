"""CPU checks for the non-teacher-forced generation constraint."""
from __future__ import annotations

import torch

from rsus.evalx.metrics import greedy_generation_recall


def test_greedy_generation_recall_is_bounded_and_side_effect_free(tiny_model, req):
    before = {
        name: parameter.detach().clone()
        for name, parameter in tiny_model.named_parameters()
    }
    value = greedy_generation_recall(tiny_model, list(req.forget[:2]))
    assert 0.0 <= value <= 1.0
    assert all(
        torch.equal(before[name], parameter.detach())
        for name, parameter in tiny_model.named_parameters()
    )


def test_greedy_generation_recall_rejects_empty_input(tiny_model):
    try:
        greedy_generation_recall(tiny_model, [])
    except ValueError as error:
        assert "at least one" in str(error)
    else:  # pragma: no cover - documents the fail-closed contract.
        raise AssertionError("empty generation audit did not fail")
