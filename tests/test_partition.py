"""Partition invariants: fold discipline, determinism, construction rules."""
import dataclasses

import pytest
import torch

from rsus.blocks import BlockSpec
from rsus.costs import CostRecord
from rsus.data.base import CandidateUniverse, Example, Request
from rsus.losses import IGNORE
from rsus.partition import (
    PartitionError,
    PartitionParams,
    build_partition,
    group_disjoint_split,
    make_folds,
)
from rsus.probe.base import ProbeSpec, ScoreProfile

PARAMS = PartitionParams(pool_size=4, min_pool_size=2, tau_rem_abs_quantile=0.5, seed=0)


def _example(eid: str, group: str) -> Example:
    ids = torch.arange(8, dtype=torch.long) + 3
    labels = ids.clone()
    labels[:4] = IGNORE
    return Example(eid, ids, labels, group=group)


def _request(n: int = 40) -> Request:
    cands = [_example(f"c{i:02d}", group=f"g{i:02d}") for i in range(n)]
    forget = [_example("f00", group="author-forget")]
    return Request.build("req-p", forget, CandidateUniverse.freeze(cands))


def _profile(req: Request, seed: int = 0, n_positive: int | None = None) -> ScoreProfile:
    gen = torch.Generator().manual_seed(seed)
    n = len(req.universe)
    n_pos = n // 2 if n_positive is None else n_positive
    vals = [1.0 - 0.01 * i for i in range(n_pos)]
    vals += (0.0001 * torch.randn(n - n_pos, generator=gen) - 0.0005).tolist()
    scores = {e.example_id: v for e, v in zip(req.universe.examples, vals)}
    spec = ProbeSpec(block=BlockSpec("unused"), eta=1e-4)
    return ScoreProfile(req.request_id, "fd", scores, spec, CostRecord())


def _folds(req: Request, seed: int = 0):
    return make_folds({e.example_id: e.group for e in req.universe.examples}, 0.5, seed)


def test_pools_respect_discovery_folds():
    req = _request()
    folds = _folds(req)
    part = build_partition(_profile(req), req, folds, PARAMS)
    group_of = {e.example_id: e.group for e in req.universe.examples}
    for cid in part.all_pool_ids():
        assert folds[group_of[cid]] == "discovery", cid


def test_protect_is_top_positive_eligible():
    req = _request()
    folds = _folds(req)
    prof = _profile(req)
    part = build_partition(prof, req, folds, PARAMS)
    group_of = {e.example_id: e.group for e in req.universe.examples}
    eligible_pos = [
        c for c in prof.scores if folds[group_of[c]] == "discovery" and prof.scores[c] > 0
    ]
    top4 = sorted(eligible_pos, key=lambda c: (-prof.scores[c], c))[:4]
    assert list(part.protect) == top4
    assert not part.fallback
    assert len(part.remote_stream) == len(part.protect)
    assert not set(part.protect) & set(part.remote_stream)


def test_fallback_rule_below_pool_size():
    req = _request()
    folds = {g: "discovery" for g in {e.group for e in req.universe.examples}}
    prof = _profile(req, n_positive=3)  # fewer positives than pool_size
    part = build_partition(prof, req, folds, PARAMS)
    assert part.fallback
    assert len(part.protect) == 3


def test_below_minimum_raises():
    req = _request()
    folds = {g: "discovery" for g in {e.group for e in req.universe.examples}}
    prof = _profile(req, n_positive=1)  # below min_pool_size=2
    with pytest.raises(PartitionError):
        build_partition(prof, req, folds, PARAMS)


def test_deterministic_manifest():
    req = _request()
    folds = _folds(req)
    p1 = build_partition(_profile(req), req, folds, PARAMS)
    p2 = build_partition(_profile(req), req, folds, PARAMS)
    assert p1.manifest_sha == p2.manifest_sha
    p3 = build_partition(_profile(req), req, folds, dataclasses.replace(PARAMS, seed=1))
    assert p3.manifest_sha != p1.manifest_sha


def test_template_matching():
    req = _request()
    folds = _folds(req)
    tpl = {e.example_id: ("A" if int(e.example_id[1:]) % 2 else "B") for e in req.universe.examples}
    part = build_partition(_profile(req), req, folds, PARAMS, template_of=tpl)
    count = lambda ids, t: sum(1 for c in ids if tpl[c] == t)  # noqa: E731
    for t in ("A", "B"):
        assert count(part.protect, t) == count(part.remote_stream, t)


def test_profile_universe_mismatch_raises():
    req = _request()
    prof = _profile(req)
    del prof.scores["c00"]
    with pytest.raises(PartitionError):
        build_partition(prof, req, _folds(req), PARAMS)


def test_group_disjoint_split_exact_and_impossible():
    group_of = {f"x{i}": f"g{i % 4}" for i in range(8)}  # 4 groups of 2
    first, second = group_disjoint_split(list(group_of), group_of, 4, seed=0)
    assert len(first) == 4 and len(second) == 4
    assert not {group_of[i] for i in first} & {group_of[i] for i in second}
    bad = {"a": "g0", "b": "g0", "c": "g0", "d": "g1"}  # sizes 3+1: no exact 2/2
    with pytest.raises(PartitionError):
        group_disjoint_split(list(bad), bad, 2, seed=0)


def test_native_audit_ids_validated():
    cands = [_example("c00", "g0")]
    with pytest.raises(ValueError):
        Request.build(
            "r", [_example("f00", "gf")], CandidateUniverse.freeze(cands), {"missing"}
        )
