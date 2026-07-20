"""Reverse-mode comparators for the same directional derivative.

- vmap_graddot: per-candidate block gradients via torch.func (chunked vmap
  where supported, per-example grad loop otherwise), dotted with g_hat.
- streaming_backward: one backward per candidate, dotting immediately and
  never holding more than one candidate gradient (the honest 1x reference:
  no N x |B| materialization).
"""
from __future__ import annotations

import torch
from torch.func import functional_call, grad, vmap

from rsus.blocks import grads_of, only_block_grads, vec_dot
from rsus.costs import CostRecord, Meter
from rsus.data.base import Request, collate
from rsus.losses import _shifted_nll, batch_to_model_device, seq_mean_answer_nll
from rsus.probe.base import ProbeSpec, ScoreProfile, register
from rsus.probe.finite_diff import canonical_forget_direction


@register("streaming_backward")
def score_streaming(model: torch.nn.Module, request: Request, spec: ProbeSpec) -> ScoreProfile:
    rec = CostRecord()
    with Meter(rec):
        ghat = canonical_forget_direction(model, request, spec, rec)
        sel = spec.block.select(model)
        scores: dict[str, float] = {}
        with only_block_grads(model, sel):
            for ex in request.universe.examples:
                batch = collate([ex])
                model.zero_grad(set_to_none=True)
                loss = seq_mean_answer_nll(model, batch)[0]
                loss.backward()
                rec.fwd_passes += 1
                rec.bwd_passes += 1
                n_tok = int(batch["attention_mask"].sum())
                rec.tokens_fwd += n_tok
                rec.tokens_bwd += n_tok
                scores[ex.example_id] = float(vec_dot(grads_of(sel), ghat))
        model.zero_grad(set_to_none=True)
    return ScoreProfile(request.request_id, "streaming_backward", scores, spec, rec)


def _one_seq_loss(model: torch.nn.Module, block_params: dict, ids, mask, labels) -> torch.Tensor:
    out = functional_call(
        model,
        block_params,
        args=(),
        kwargs={"input_ids": ids.unsqueeze(0), "attention_mask": mask.unsqueeze(0)},
    )
    nll, m = _shifted_nll(out.logits, labels.unsqueeze(0))
    return nll.sum() / m.sum()


@register("vmap_graddot")
def score_vmap_graddot(model: torch.nn.Module, request: Request, spec: ProbeSpec) -> ScoreProfile:
    rec = CostRecord()
    with Meter(rec):
        ghat = canonical_forget_direction(model, request, spec, rec)
        sel = spec.block.select(model)
        primal = {n: p.detach() for n, p in sel.items()}
        gfn = grad(lambda bp, i, m, l: _one_seq_loss(model, bp, i, m, l))
        scores: dict[str, float] = {}
        impl = "vmap"
        for batch in request.universe.batches(spec.batch_size):
            batch = batch_to_model_device(model, batch)
            ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
            try:
                per_sample = vmap(gfn, in_dims=(None, 0, 0, 0))(primal, ids, mask, labels)
                dots = torch.stack(
                    [(per_sample[n] * ghat[n]).flatten(1).sum(dim=1) for n in per_sample]
                ).sum(dim=0)
            except RuntimeError:  # op unsupported under vmap: equivalent per-example loop
                impl = "grad_loop"
                per_example = []
                for i in range(ids.shape[0]):
                    g = gfn(primal, ids[i], mask[i], labels[i])
                    per_example.append(torch.stack([(g[n] * ghat[n]).sum() for n in g]).sum())
                dots = torch.stack(per_example)
            n = ids.shape[0]
            rec.fwd_passes += n
            rec.bwd_passes += n
            n_tok = int(mask.sum())
            rec.tokens_fwd += n_tok
            rec.tokens_bwd += n_tok
            for eid, val in zip(batch["example_ids"], dots.tolist()):
                scores[eid] = val
        rec.notes["impl"] = impl
    return ScoreProfile(request.request_id, "vmap_graddot", scores, spec, rec)
