"""Controlled implementations of the unlearning objectives used as parents.

The output-channel losses follow their published sequence-probability
definitions; representation objectives implement the corresponding loss
geometry inside the common experimental substrate.  A trajectory may update
either the full model or a declared parameter block.  It never sees a
susceptibility score.  Frozen quantities are cached at setup so no second
model copy is held during optimization.
"""
from __future__ import annotations

import re

import torch

from rsus.data.base import Example, Request, collate
from rsus.generators.base import TrajectoryConfig, register_objective
from rsus.losses import (
    IGNORE,
    batch_to_model_device,
    seq_mean_answer_nll,
    seq_sum_answer_nll,
)


def _gru_projection_coefficient(
    unlearning: list[torch.Tensor], retain: list[torch.Tensor | None]
) -> torch.Tensor:
    """Coefficient for the minimum-deviation GRU half-space projection."""
    dot = sum(
        (unlearn_grad * retain_grad).sum()
        for unlearn_grad, retain_grad in zip(unlearning, retain)
        if retain_grad is not None
    )
    norm_sq = sum(
        (retain_grad * retain_grad).sum()
        for retain_grad in retain
        if retain_grad is not None
    )
    if not torch.is_tensor(dot):
        return unlearning[0].new_zeros(())
    return torch.clamp(dot / (norm_sq + 1e-12), max=0.0)


class _Base:
    def __init__(self, model, request: Request, retain: list[Example], cfg: TrajectoryConfig):
        self.model = model
        self.request = request
        self.retain = retain
        self.cfg = cfg
        if cfg.trainable_pattern is None:
            selected = {n: p for n, p in model.named_parameters() if p.requires_grad}
        else:
            rx = re.compile(cfg.trainable_pattern)
            selected = {n: p for n, p in model.named_parameters() if rx.fullmatch(n)}
            if not selected:
                raise ValueError(
                    f"trainable_pattern matched no parameters: {cfg.trainable_pattern!r}"
                )
            for name, param in model.named_parameters():
                param.requires_grad_(name in selected)
        self.params = list(selected.values())
        self.param_names = list(selected)
        # ``foreach=False`` preserves AdamW semantics while avoiding the
        # parameter-sized temporary tensor lists used by the CUDA foreach
        # implementation.  This keeps a full-model 7B trajectory within one
        # 80GB device once the caller has serialized model residency.
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, foreach=False)
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
            -self.cfg.forget_weight * seq_mean_answer_nll(self.model, self.forget_batch).mean()
            + self.cfg.retain_weight
            * seq_mean_answer_nll(self.model, self.retain_minibatch()).mean()
        )
        return self._update(loss)


@register_objective("npo")
class NPO(_Base):
    """Negative preference optimization with retain training. Sequence-level
    log-ratios use per-sequence reference NLLs cached at setup."""

    def __init__(self, model, request, retain, cfg):
        super().__init__(model, request, retain, cfg)
        if cfg.beta <= 0:
            raise ValueError("NPO beta must be positive")
        with torch.no_grad():
            self.ref_nll = seq_sum_answer_nll(model, self.forget_batch).detach()

    def step(self) -> float:
        cur = seq_sum_answer_nll(self.model, self.forget_batch)
        beta = self.cfg.beta
        # -(2/beta) * log sigmoid(beta * (ell_theta - ell_ref)): decays to 0
        # as the forget answers become less likely than under the reference.
        npo = -(2.0 / beta) * torch.nn.functional.logsigmoid(beta * (cur - self.ref_nll)).mean()
        loss = (self.cfg.forget_weight * npo
                + self.cfg.retain_weight
                * seq_mean_answer_nll(self.model, self.retain_minibatch()).mean())
        return self._update(loss)


@register_objective("simnpo")
class SimNPO(_Base):
    """Reference-free NPO variant: length-normalized forget loss enters a
    sigmoid margin directly (no reference model), plus retain training."""

    def __init__(self, model, request, retain, cfg):
        super().__init__(model, request, retain, cfg)
        if cfg.beta <= 0:
            raise ValueError("SimNPO beta must be positive")

    def step(self) -> float:
        cur = seq_mean_answer_nll(self.model, self.forget_batch)
        beta, gamma = self.cfg.beta, self.cfg.simnpo_gamma
        simnpo = -(2.0 / beta) * torch.nn.functional.logsigmoid(beta * cur - gamma).mean()
        loss = (self.cfg.forget_weight * simnpo
                + self.cfg.retain_weight
                * seq_mean_answer_nll(self.model, self.retain_minibatch()).mean())
        return self._update(loss)


@register_objective("idkdpo")
class IdkDPO(_Base):
    """DPO with 'I don't know'-style responses preferred over the original
    forget answers; summed sequence-NLL log-ratios against references
    cached at setup. Requires cfg.idk_examples aligned with the forget set."""

    def __init__(self, model, request, retain, cfg):
        super().__init__(model, request, retain, cfg)
        if cfg.beta <= 0:
            raise ValueError("IdkDPO beta must be positive")
        if not cfg.idk_examples or len(cfg.idk_examples) != len(request.forget):
            raise ValueError("idkdpo needs cfg.idk_examples aligned with the forget set")
        self.idk_batch = collate(list(cfg.idk_examples))
        with torch.no_grad():
            self.ref_l = seq_sum_answer_nll(model, self.forget_batch).detach()
            self.ref_w = seq_sum_answer_nll(model, self.idk_batch).detach()

    def step(self) -> float:
        beta = self.cfg.beta
        cur_l = seq_sum_answer_nll(self.model, self.forget_batch)
        cur_w = seq_sum_answer_nll(self.model, self.idk_batch)
        margin = (self.ref_w - cur_w) - (self.ref_l - cur_l)
        dpo = -torch.nn.functional.logsigmoid(beta * margin).mean()
        loss = (self.cfg.forget_weight * dpo
                + self.cfg.retain_weight
                * seq_mean_answer_nll(self.model, self.retain_minibatch()).mean())
        return self._update(loss)


@register_objective("gru")
class GRU(_Base):
    """Retain-aware gradient rectification: the forget-ascent gradient is
    projected onto the feasible half-space that does not increase retain loss.

    This is the paper's minimum-deviation update: when the unlearning gradient
    conflicts with the retain gradient, subtract its component along that
    retain gradient; otherwise leave the unlearning gradient unchanged.  GRU
    is therefore a rectifier around the output-side forget objective, not an
    additional GradDiff retain-descent step.
    """

    def step(self) -> float:
        # Keep only the forget-gradient copy.  The retain gradient remains in
        # ``p.grad`` and is combined in place after the global coefficient is
        # known.  The former implementation cloned both gradients while also
        # retaining ``p.grad``, which could add three model-sized gradient
        # sets to a one-GPU run.
        f_loss = (-self.cfg.forget_weight
                  * seq_mean_answer_nll(self.model, self.forget_batch).mean())
        self.opt.zero_grad(set_to_none=True)
        f_loss.backward()
        f_value = float(f_loss.detach())
        g_f = [
            (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
            for p in self.params
        ]
        del f_loss

        r_loss = (self.cfg.retain_weight
                  * seq_mean_answer_nll(self.model, self.retain_minibatch()).mean())
        self.opt.zero_grad(set_to_none=True)
        r_loss.backward()
        coef = _gru_projection_coefficient(g_f, [p.grad for p in self.params])
        for p, a in zip(self.params, g_f):
            if p.grad is None:
                p.grad = a
            else:
                # g_rect = g_unlearn - min(<g_u,g_r>/||g_r||^2, 0) g_retain
                p.grad.mul_(-coef).add_(a)
        self.opt.step()
        return f_value


class _RepBase(_Base):
    """Representation-channel objectives whose removal term reads hidden states.

    The channel hypothesis predicts that representation proximity should be the
    better retain-side risk signal; the implementation does not assume or force
    that empirical outcome.
    """

    def _answer_hiddens(self, batch: dict) -> torch.Tensor:
        return torch.cat(list(self._answer_hiddens_by_example(batch).values()), dim=0)

    def _answer_hiddens_by_example(self, batch: dict) -> dict[str, torch.Tensor]:
        batch = batch_to_model_device(self.model, batch)
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        h = out.hidden_states[-1][:, :-1, :]
        mask = batch["labels"][:, 1:] != IGNORE
        return {
            example_id: h[index][mask[index]]
            for index, example_id in enumerate(batch["example_ids"])
        }

    def _cache_retain_hiddens(self) -> dict[str, torch.Tensor]:
        cache = {}
        with torch.no_grad():
            for start in range(0, len(self.retain), self.cfg.batch_size):
                batch = collate(self.retain[start:start + self.cfg.batch_size])
                for example_id, hidden in self._answer_hiddens_by_example(batch).items():
                    cache[example_id] = hidden.detach().cpu()
        return cache

    def _retain_hidden_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.representation_retain_mode == "fixed":
            batch = self.retain_batch
        elif self.cfg.representation_retain_mode == "stream_cached":
            batch = self.retain_minibatch()
        else:
            raise ValueError(
                "representation_retain_mode must be 'fixed' or 'stream_cached', got "
                f"{self.cfg.representation_retain_mode!r}"
            )
        current_by_id = self._answer_hiddens_by_example(batch)
        current = torch.cat(list(current_by_id.values()), dim=0)
        target = torch.cat(
            [self.retain_h0[example_id] for example_id in current_by_id], dim=0
        ).to(device=current.device, dtype=current.dtype)
        return current, target


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
        if cfg.representation_retain_mode == "stream_cached":
            self.retain_h0 = self._cache_retain_hiddens()
        else:
            with torch.no_grad():
                fixed = self._answer_hiddens_by_example(self.retain_batch)
            self.retain_h0 = {key: value.detach().cpu() for key, value in fixed.items()}

    def step(self) -> float:
        h_f = self._answer_hiddens(self.forget_batch)
        misdirect = (h_f - self.control.to(device=h_f.device, dtype=h_f.dtype)).pow(2).sum(dim=-1).mean()
        h_r, h_r0 = self._retain_hidden_pair()
        pin = (h_r - h_r0).pow(2).sum(dim=-1).mean()
        return self._update(misdirect + self.cfg.rmu_alpha * pin)


@register_objective("repnoise")
class RepNoise(_RepBase):
    """Block-controlled RepNoise-style objective used in the frozen roster.

    This lightweight adaptation matches hidden states to fresh Gaussian targets
    at the measured layer and retains answer likelihood.  It is *not* the full
    Rosati et al. implementation, whose loss uses multi-kernel MMD across all
    post-MLP layers plus an auxiliary layer-wise ascent term.  Main-table code
    must disclose this variant or replace it with the paper-faithful adapter.
    """

    def step(self) -> float:
        h_f = self._answer_hiddens(self.forget_batch)
        noise = torch.randn(h_f.shape, generator=self.gen).to(device=h_f.device, dtype=h_f.dtype)
        noise = self.cfg.rmu_c * noise / (h_f.shape[-1] ** 0.5)
        noising = (h_f - noise).pow(2).sum(dim=-1).mean()
        retain = seq_mean_answer_nll(self.model, self.retain_minibatch()).mean()
        return self._update(noising + self.cfg.rmu_alpha * retain)


@register_objective("circuit_breakers")
class CircuitBreakers(_RepBase):
    """Block-controlled Representation-Rerouting (Circuit Breakers) loss.

    Reroute forget answer-token
    hiddens AWAY from their original direction -- penalize positive cosine to the
    frozen pre-unlearning representation (relu, stop at orthogonal) -- while
    pinning retain hiddens.  The loss geometry is faithful; unlike the released
    safety method, this controlled comparison uses the common full-rank block
    rather than method-specific LoRA adapters and coefficient scheduling.
    """

    def __init__(self, model, request, retain, cfg):
        super().__init__(model, request, retain, cfg)
        self.retain_batch = collate(retain[: cfg.batch_size])
        with torch.no_grad():
            self.forget_h0 = self._answer_hiddens(self.forget_batch).detach()
        if cfg.representation_retain_mode == "stream_cached":
            self.retain_h0 = self._cache_retain_hiddens()
        else:
            with torch.no_grad():
                fixed = self._answer_hiddens_by_example(self.retain_batch)
            self.retain_h0 = {key: value.detach().cpu() for key, value in fixed.items()}

    def step(self) -> float:
        h_f = self._answer_hiddens(self.forget_batch)
        cos = torch.nn.functional.cosine_similarity(
            h_f, self.forget_h0.to(device=h_f.device, dtype=h_f.dtype), dim=-1)
        reroute = torch.relu(cos).mean()
        h_r, h_r0 = self._retain_hidden_pair()
        pin = (h_r - h_r0).pow(2).sum(dim=-1).mean()
        return self._update(reroute + self.cfg.rmu_alpha * pin)
