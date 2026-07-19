"""Teacher-forced answer-token NLL, sequence and token level.

``labels`` follow the HF convention: prompt positions are masked with -100;
answer tokens carry their token id. The sequence loss is the mean answer-token
NLL (paper: canonical loss ell). The token-level view returns a flat vector
plus an (example_id, answer_pos) index map — the only legal alignment between
the Stage-1 reference cache and Stage-2 evaluation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE = -100


def _shifted_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position NLL under next-token teacher forcing.

    Returns (nll [B, L-1], mask [B, L-1]); nll is zeroed outside the mask.
    """
    logits = logits[:, :-1, :]
    targets = labels[:, 1:]
    mask = targets != IGNORE
    safe = targets.clamp_min(0)
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return nll * mask, mask


def seq_mean_answer_nll(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    """Mean answer-token NLL per sequence, shape [B]. Differentiable."""
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    nll, mask = _shifted_nll(out.logits, batch["labels"])
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        bad = [batch["example_ids"][i] for i in torch.nonzero(counts == 0).flatten().tolist()]
        raise ValueError(f"examples with no answer tokens: {bad}")
    return nll.sum(dim=1) / counts


def token_answer_nll(
    model: torch.nn.Module, batch: dict
) -> tuple[torch.Tensor, list[tuple[str, int]]]:
    """Flat answer-token NLL vector plus its (example_id, answer_pos) index.

    ``answer_pos`` is the 0-based position among that example's answer tokens,
    so the index map is invariant to batch composition and padding.
    """
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    nll, mask = _shifted_nll(out.logits, batch["labels"])
    flat: list[torch.Tensor] = []
    index: list[tuple[str, int]] = []
    for i, eid in enumerate(batch["example_ids"]):
        row = nll[i][mask[i]]
        flat.append(row)
        index.extend((eid, k) for k in range(row.numel()))
    return torch.cat(flat), index
