"""Minimal faithful implementations of the third-party unlearning objectives
used as Table-1 generators and Table-2 baselines: GA, GradDiff, NPO(+retain),
RMU. Each trains all parameters with AdamW and never sees susceptibility
scores. Frozen quantities (NPO reference losses, RMU retain hiddens) are
cached as scalars/tensors at setup; no second model copy is held.
"""
from __future__ import annotations

import torch

from rsus.data.base import Example, Request, collate
from rsus.generators.base import TrajectoryConfig, register_objective
from rsus.losses import IGNORE, seq_mean_answer_nll


class _Base:
    def __init__(self, model, request: Request, retain: list[Example], cfg: TrajectoryConfig):
        self.model = model
        self.request = request
        self.retain = retain
        self.cfg = cfg
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.forget_batch = collate(list(request.forget))

    def retain_minibatch(self) -> dict:
        idx = torch.randperm(len(self.retain), generator=self.gen)[: self.cfg.batch_size]
        return collate([self.retain[i] for i in idx.tolist()])

    def _update(self, loss: torch.Tensor) -> float:
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.detach())


@register_objective("ga")
class GA(_Base):
    """Plain gradient ascent on the mean forget NLL."""

    def step(self) -> float:
        return self._update(-seq_mean_answer_nll(self.model, self.forget_batch).mean())


@register_objective("graddiff")
class GradDiff(_Base):
    """Ascent on forget plus descent on a retain minibatch."""

    def step(self) -> float:
        loss = (
            -seq_mean_answer_nll(self.model, self.forget_batch).mean()
            + seq_mean_answer_nll(self.model, self.retain_minibatch()).mean()
        )
        return self._update(loss)


@register_objective("npo")
class NPO(_Base):
    """Negative preference optimization with retain training. Sequence-level
    log-ratios use per-sequence reference NLLs cached at setup."""

    def __init__(self, model, request, retain, cfg):
        super().__init__(model, request, retain, cfg)
        with torch.no_grad():
            self.ref_nll = seq_mean_answer_nll(model, self.forget_batch).detach()

    def step(self) -> float:
        cur = seq_mean_answer_nll(self.model, self.forget_batch)
        beta = self.cfg.beta
        # -(2/beta) * log sigmoid(beta * (ell_theta - ell_ref)): decays to 0
        # as the forget answers become less likely than under the reference.
        npo = -(2.0 / beta) * torch.nn.functional.logsigmoid(beta * (cur - self.ref_nll)).mean()
        loss = npo + seq_mean_answer_nll(self.model, self.retain_minibatch()).mean()
        return self._update(loss)


@register_objective("rmu")
class RMU(_Base):
    """Representation misdirection: push forget answer-token hiddens toward a
    fixed random control vector while pinning retain hiddens to their frozen
    values (cached at setup)."""

    def __init__(self, model, request, retain, cfg):
        super().__init__(model, request, retain, cfg)
        hidden = model.config.hidden_size
        u = torch.randn(hidden, generator=self.gen, dtype=next(model.parameters()).dtype)
        self.control = cfg.rmu_c * u / u.norm()
        self.retain_fixed = retain[: cfg.batch_size]
        self.retain_batch = collate(self.retain_fixed)
        with torch.no_grad():
            self.retain_h0 = self._answer_hiddens(self.retain_batch).detach()

    def _answer_hiddens(self, batch: dict) -> torch.Tensor:
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        h = out.hidden_states[-1][:, :-1, :]
        mask = batch["labels"][:, 1:] != IGNORE
        return h[mask]  # [T_answer_total, H]

    def step(self) -> float:
        h_f = self._answer_hiddens(self.forget_batch)
        misdirect = (h_f - self.control.to(h_f.dtype)).pow(2).sum(dim=-1).mean()
        h_r = self._answer_hiddens(self.retain_batch)
        pin = (h_r - self.retain_h0).pow(2).sum(dim=-1).mean()
        return self._update(misdirect + self.cfg.rmu_alpha * pin)
