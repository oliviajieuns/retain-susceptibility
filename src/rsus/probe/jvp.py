"""Exact batched Jacobian-vector-product comparator (forward-mode AD).

Targets the same directional derivative as the finite difference
(paper eq:fdscore) with no truncation error. Numerical and cost comparator
only; no novelty is claimed for either implementation.
"""
from __future__ import annotations

import torch
from torch.func import functional_call

from rsus.costs import CostRecord, Meter
from rsus.data.base import Request
from rsus.losses import _shifted_nll
from rsus.probe.base import ProbeSpec, ScoreProfile, register
from rsus.probe.finite_diff import canonical_forget_direction


def _batch_seq_losses(model: torch.nn.Module, block_params: dict, batch: dict) -> torch.Tensor:
    out = functional_call(
        model,
        block_params,
        args=(),
        kwargs={"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
    )
    nll, mask = _shifted_nll(out.logits, batch["labels"])
    return nll.sum(dim=1) / mask.sum(dim=1)


@register("jvp")
def score_jvp(model: torch.nn.Module, request: Request, spec: ProbeSpec) -> ScoreProfile:
    rec = CostRecord()
    with Meter(rec):
        ghat = canonical_forget_direction(model, request, spec, rec)
        sel = spec.block.select(model)
        primal = {n: p.detach() for n, p in sel.items()}
        tangent = {n: ghat[n] for n in sel}
        scores: dict[str, float] = {}
        for batch in request.universe.batches(spec.batch_size):
            _, dirderiv = torch.func.jvp(
                lambda bp: _batch_seq_losses(model, bp, batch), (primal,), (tangent,)
            )
            rec.fwd_passes += 2  # a JVP evaluates primal + tangent, ~2x forward
            rec.tokens_fwd += 2 * int(batch["attention_mask"].sum())
            for eid, val in zip(batch["example_ids"], dirderiv.tolist()):
                scores[eid] = val
    return ScoreProfile(request.request_id, "jvp", scores, spec, rec)
