"""Partition invariants: fold discipline, determinism, construction rules."""
import pytest
import torch

from rsus.blocks import BlockSpec
from rsus.costs import CostRecord
from rsus.data.base import CandidateUniverse, Example, Request
from rsus.losses import IGNORE
from rsus.partition import PartitionError, PartitionParams, build_partition, make_folds
from rsus.probe.base import ProbeSpec, ScoreProfile

PARAMS = PartitionParams(
    tau_adj_quantile=0.5, tau_rem_abs_quantile=0.5, pool_size=4, visible_split=2, seed=0
)


def _example(eid: str, group: str) -> Example:
    ids = torch.arange(8, dtype=torch.long) + 3
    labels = ids.clone()
    labels[:4] = IGNORE
    return Example(eid, ids, labels, group=group)


def _request(n: int = 40) -> Request:
    cands = [_example(f"c{i:02d}", group=f"g{i:02d}") for i in range(n)]
    forget = [_example("f00", group="author-forget")]
    return Request.build("req-p", forget, CandidateUniverse.freeze(cands))


def _profile(req: Request, seed: int = 0) -> ScoreProfile:
    gen = torch.Generator().manual_seed(seed)
    n = len(req.universe)
    # half clearly positive (descending), half near zero
    vals = [1.0 - 0.01 * i for i in range(n // 2)]
    vals += (0.001 * torch.randn(n - n // 2, generator=gen)).tolist()
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


def test_adjacent_is_top_scoring_eligible():
    req = _request()
    folds = _folds(req)
    prof = _profile(req)
    part = build_partition(prof, req, folds, PARAMS)
    group_of = {e.example_id: e.group for e in req.universe.examples}
    eligible = [c for c in prof.scores if folds[group_of[c]] == "discovery"]
    top4 = sorted(eligible, key=lambda c: (-prof.scores[c], c))[:4]
    assert set(part.adjacent_visible) | set(part.adjacent_heldout) == set(top4)
    assert len(part.adjacent_visible) == len(part.adjacent_heldout) == 2
    # adjacent and remote are disjoint
    adj = set(part.adjacent_visible) | set(part.adjacent_heldout)
    rem = set(part.remote_visible) | set(part.remote_heldout)
    assert not adj & rem


def test_deterministic_manifest():
    req = _request()
    folds = _folds(req)
    p1 = build_partition(_profile(req), req, folds, PARAMS)
    p2 = build_partition(_profile(req), req, folds, PARAMS)
    assert p1.manifest_sha == p2.manifest_sha
    import dataclasses

    p3 = build_partition(_profile(req), req, folds, dataclasses.replace(PARAMS, seed=1))
    assert p3.manifest_sha != p1.manifest_sha


def test_template_matching():
    req = _request()
    folds = _folds(req)
    tpl = {e.example_id: ("A" if int(e.example_id[1:]) % 2 else "B") for e in req.universe.examples}
    part = build_partition(_profile(req), req, folds, PARAMS, template_of=tpl)
    adj = list(part.adjacent_visible) + list(part.adjacent_heldout)
    rem = list(part.remote_visible) + list(part.remote_heldout)
    count = lambda ids, t: sum(1 for c in ids if tpl[c] == t)  # noqa: E731
    for t in ("A", "B"):
        assert count(adj, t) == count(rem, t)


def test_insufficient_candidates_raises():
    req = _request(n=10)
    folds = _folds(req)
    with pytest.raises(PartitionError):
        build_partition(_profile(req), req, folds, PARAMS)


def test_group_split_impossible_raises():
    # one group of 3 + one of 1: exact 2/2 split at group granularity impossible
    cands = [_example(f"c{i}", group="gBIG" if i < 3 else f"g{i}") for i in range(40)]
    req = Request.build("req-g", [_example("f00", "author-forget")], CandidateUniverse.freeze(cands))
    prof = _profile(req)
    # force the big group's members into the adjacent top-4
    for i, cid in enumerate(["c0", "c1", "c2", "c5"]):
        prof.scores[cid] = 100.0 - i
    folds = {e.group: "discovery" for e in req.universe.examples}
    with pytest.raises(PartitionError):
        build_partition(prof, req, folds, PARAMS)


def test_profile_universe_mismatch_raises():
    req = _request()
    prof = _profile(req)
    del prof.scores["c00"]
    with pytest.raises(PartitionError):
        build_partition(prof, req, _folds(req), PARAMS)
