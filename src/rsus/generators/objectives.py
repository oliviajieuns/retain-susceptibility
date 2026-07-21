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
from rsus.losses import IGNORE, batch_to_model_device, seq_mean_answer_nll


class _Base:
    def __init__(self, model, request: Request, retain: list[Example], cfg: TrajectoryConfig):
        self.model = model
        self.request = request
        self.retain = retain
        self.cfg = cfg
        # ``foreach=False`` preserves AdamW semantics while avoiding the
        # parameter-sized temporary tensor lists used by the CUDA foreach
        # implementation.  This keeps a full-model 7B trajectory within one
        # 80GB device once the caller has serialized model residency.
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, foreach=False)
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


@register_objective("simnpo")
class SimNPO(_Base):
    """Reference-free NPO variant: length-normalized forget loss enters a
    sigmoid margin directly (no reference model), plus retain training."""

    def step(self) -> float:
        cur = seq_mean_answer_nll(self.model, self.forget_batch)
        beta, gamma = self.cfg.beta, self.cfg.simnpo_gamma
        simnpo = -(2.0 / beta) * torch.nn.functional.logsigmoid(beta * cur - gamma).mean()
        loss = simnpo + seq_mean_answer_nll(self.model, self.retain_minibatch()).mean()
        return self._update(loss)


@register_objective("idkdpo")
class IdkDPO(_Base):
    """DPO with 'I don't know'-style responses preferred over the original
    forget answers; per-sequence mean-NLL log-ratios against references
    cached at setup. Requires cfg.idk_examples aligned with the forget set."""

    def __init__(self, model, request, retain, cfg):
        super().__init__(model, request, retain, cfg)
        if not cfg.idk_examples or len(cfg.idk_examples) != len(request.forget):
            raise ValueError("idkdpo needs cfg.idk_examples aligned with the forget set")
        self.idk_batch = collate(list(cfg.idk_examples))
        with torch.no_grad():
            self.ref_l = seq_mean_answer_nll(model, self.forget_batch).detach()
            self.ref_w = seq_mean_answer_nll(model, self.idk_batch).detach()

    def step(self) -> float:
        beta = self.cfg.beta
        cur_l = seq_mean_answer_nll(self.model, self.forget_batch)
        cur_w = seq_mean_answer_nll(self.model, self.idk_batch)
        margin = (self.ref_w - cur_w) - (self.ref_l - cur_l)  # mean-NLL log-ratio proxy
        dpo = -torch.nn.functional.logsigmoid(beta * margin).mean()
        loss = dpo + seq_mean_answer_nll(self.model, self.retain_minibatch()).mean()
        return self._update(loss)


@register_objective("gru")
class GRU(_Base):
    """Retain-aware gradient rectification: the forget-ascent gradient is
    stripped of its component that conflicts with retain descent, then
    combined with the retain gradient (minimal faithful implementation)."""

    def step(self) -> float:
        # Keep only the forget-gradient copy.  The retain gradient remains in
        # ``p.grad`` and is combined in place after the global coefficient is
        # known.  The former implementation cloned both gradients while also
        # retaining ``p.grad``, which could add three model-sized gradient
        # sets to a one-GPU run.
        f_loss = -seq_mean_answer_nll(self.model, self.forget_batch).mean()
        self.opt.zero_grad(set_to_none=True)
        f_loss.backward()
        f_value = float(f_loss.detach())
        g_f = [
            (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
            for p in self.model.parameters()
        ]
        del f_loss

        r_loss = seq_mean_answer_nll(self.model, self.retain_minibatch()).mean()
        self.opt.zero_grad(set_to_none=True)
        r_loss.backward()
        r_value = float(r_loss.detach())
        dot = sum(
            (a * p.grad).sum()
            for a, p in zip(g_f, self.model.parameters())
            if p.grad is not None
        )
        rr = sum(
            (p.grad * p.grad).sum()
            for p in self.model.parameters()
            if p.grad is not None
        )
        coef = torch.clamp(dot / (rr + 1e-12), max=0.0)  # remove conflicting part only
        for p, a in zip(self.model.parameters(), g_f):
            if p.grad is None:
                p.grad = a
            else:
                p.grad.mul_(1.0 - coef).add_(a)
        self.opt.step()
        return f_value + r_value


class _RepBase(_Base):
    """Representation-channel objectives: the loss is a function of INTERNAL
    hidden states (not output token likelihoods). Collateral damage on retained
    candidates is therefore governed by representation proximity, not gradient
    magnitude -- the second damage channel."""

    def _answer_hiddens(self, batch: dict) -> torch.Tensor:
        batch = batch_to_model_device(self.model, batch)
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        h = out.hidden_states[-1][:, :-1, :]
        mask = batch["labels"][:, 1:] != IGNORE
        return h[mask]  # [T_answer_total, H]


@register_objective("rmu")
class RMU(_RepBase):
    """Representation misdirection: push forget answer-token hiddens toward a
    fixed random control vector while pinning retain hiddens to their frozen
    values (cached at setup)."""

    def __init__(self, model, request, retain, cfg):
        super().__init__(model, request, retain, cfg)
        hidden = model.config.hidden_size
        u = torch.randn(hidden, generator=self.gen, dtype=next(model.parameters()).dtype)
        self.control = cfg.rmu_c * u / u.norm()
        self.retain_batch = collate(retain[: cfg.batch_size])
        with torch.no_grad():
            self.retain_h0 = self._answer_hiddens(self.retain_batch).detach()

    def step(self) -> float:
        h_f = self._answer_hiddens(self.forget_batch)
        misdirect = (h_f - self.control.to(device=h_f.device, dtype=h_f.dtype)).pow(2).sum(dim=-1).mean()
        h_r = self._answer_hiddens(self.retain_batch)
        pin = (h_r - self.retain_h0).pow(2).sum(dim=-1).mean()
        return self._update(misdirect + self.cfg.rmu_alpha * pin)


@register_objective("repnoise")
class RepNoise(_RepBase):
    """Representation noising: push forget answer-token hiddens toward FRESH
    Gaussian noise each step (destroy recoverable structure rather than align to
    a fixed target) while keeping the retain answer likelihood. Representation
    channel; differs from RMU by the resampled target + token-loss retain."""

    def step(self) -> float:
        h_f = self._answer_hiddens(self.forget_batch)
        noise = torch.randn(h_f.shape, generator=self.gen).to(device=h_f.device, dtype=h_f.dtype)
        noise = self.cfg.rmu_c * noise / (h_f.shape[-1] ** 0.5)
        noising = (h_f - noise).pow(2).sum(dim=-1).mean()
        retain = seq_mean_answer_nll(self.model, self.retain_minibatch()).mean()
        return self._update(noising + self.cfg.rmu_alpha * retain)


@register_objective("circuit_breakers")
class CircuitBreakers(_RepBase):
    """Representation rerouting (Circuit Breakers): reroute forget answer-token
    hiddens AWAY from their original direction -- penalize positive cosine to the
    frozen pre-unlearning representation (relu, stop at orthogonal) -- while
    pinning retain hiddens. Representation channel; differs from RMU by
    rerouting-from-origin rather than pushing to a fixed target."""

    def __init__(self, model, request, retain, cfg):
        super().__init__(model, request, retain, cfg)
        self.retain_batch = collate(retain[: cfg.batch_size])
        with torch.no_grad():
            self.forget_h0 = self._answer_hiddens(self.forget_batch).detach()
            self.retain_h0 = self._answer_hiddens(self.retain_batch).detach()

    def step(self) -> float:
        h_f = self._answer_hiddens(self.forget_batch)
        cos = torch.nn.functional.cosine_similarity(
            h_f, self.forget_h0.to(device=h_f.device, dtype=h_f.dtype), dim=-1)
        reroute = torch.relu(cos).mean()
        h_r = self._answer_hiddens(self.retain_batch)
        pin = (h_r - self.retain_h0.to(device=h_r.device, dtype=h_r.dtype)).pow(2).sum(dim=-1).mean()
        return self._update(reroute + self.cfg.rmu_alpha * pin)
