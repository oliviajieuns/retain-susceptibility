"""CPU-only tests for the fail-closed dataset adapter contract."""
from __future__ import annotations

import pytest
import torch

from rsus.data.base import CandidateUniverse, Example, Request
from rsus.data.registry import (
    AdapterCapabilities,
    AdapterNotFoundError,
    AdapterRegistryError,
    DatasetAdapter,
    DatasetAdapterRegistry,
    get_adapter,
)


def test_builtin_adapters_return_request_contract_without_import_breakage():
    tofu = get_adapter("TOFU")
    assert tofu.key == "tofu"
    assert tofu.capabilities.supports("prediction")
    assert tofu.capabilities.independent_target_roster

    substrate = get_adapter("controlled-substrate")
    request = substrate.build_request(seed=7, n_forget=2, n_adjacent=2, n_remote=2)
    assert isinstance(request, Request)
    assert request.request_id == "substrate-7"
    assert all(example.group for example in request.universe.examples)


def test_unknown_dataset_never_falls_back_to_tofu():
    # WMDP-bio/MMLU graduated to a real adapter on 2026-07-23; MUSE-News is
    # still a planned paper dataset without a loader.
    with pytest.raises(AdapterNotFoundError, match="no dataset adapter registered"):
        get_adapter("MUSE-News")


def test_registry_rejects_alias_collisions_and_non_request_results():
    registry = DatasetAdapterRegistry()
    capabilities = AdapterCapabilities(
        stages=frozenset({"prediction"}),
        roster_unit="request_id",
        grouped_candidates=True,
        native_audit=False,
        independent_target_roster=False,
    )
    first = DatasetAdapter(
        key="first",
        aliases=("shared",),
        factory=lambda **_kwargs: "not-a-request",  # type: ignore[return-value]
        capabilities=capabilities,
    )
    registry.register(first)
    with pytest.raises(AdapterRegistryError, match="aliases collide"):
        registry.register(
            DatasetAdapter(
                key="second",
                aliases=("SHARED",),
                factory=lambda **_kwargs: "also-not-a-request",  # type: ignore[return-value]
                capabilities=capabilities,
            )
        )
    with pytest.raises(TypeError, match="expected Request"):
        registry.resolve("first").build_request()


def test_adapter_contract_rejects_forget_candidate_leakage():
    example = Example(
        example_id="same",
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([-100, 2]),
        group="g",
    )
    invalid = Request.build(
        "bad", [example], CandidateUniverse.freeze([example])
    )
    adapter = DatasetAdapter(
        key="leaky",
        factory=lambda **_kwargs: invalid,
        capabilities=AdapterCapabilities(
            stages=frozenset({"prediction"}),
            roster_unit="request_id",
            grouped_candidates=True,
            native_audit=False,
            independent_target_roster=False,
        ),
    )
    with pytest.raises(AdapterRegistryError, match="mixed forget examples"):
        adapter.build_request()
