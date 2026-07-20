"""Cheap teacher-forced metrics used by gates and the evaluation contract."""
from __future__ import annotations

import torch

from rsus.data.base import Example, collate
from rsus.losses import IGNORE, batch_to_model_device, seq_mean_answer_nll


def answer_token_recall(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    """Per-sequence fraction of answer tokens where argmax equals the target."""
    batch = batch_to_model_device(model, batch)
    with torch.no_grad():
        logits = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits
    pred = logits[:, :-1, :].argmax(dim=-1)
    targets = batch["labels"][:, 1:]
    mask = targets != IGNORE
    hits = (pred == targets) & mask
    return hits.sum(dim=1) / mask.sum(dim=1)


def mean_recall(model: torch.nn.Module, examples: list[Example], batch_size: int = 8) -> float:
    vals: list[torch.Tensor] = []
    for i in range(0, len(examples), batch_size):
        vals.append(answer_token_recall(model, collate(examples[i : i + batch_size])))
    return float(torch.cat(vals).mean())


def mean_seq_loss(model: torch.nn.Module, examples: list[Example], batch_size: int = 8) -> float:
    vals: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            vals.append(seq_mean_answer_nll(model, collate(examples[i : i + batch_size])))
    return float(torch.cat(vals).mean())
