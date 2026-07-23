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


def greedy_generation_recall(
    model: torch.nn.Module, examples: list[Example]
) -> float:
    """Mean autoregressive gold-token recall from the prompt alone.

    Unlike :func:`mean_recall`, this audit never teacher-forces an answer
    token.  For each example it greedily rolls out exactly the frozen gold
    answer length (including EOS when present) and measures position-wise
    overlap with the gold answer.  Lower is better after unlearning.  The
    explicit fixed horizon makes the metric deterministic and avoids silently
    changing generation settings across protection arms.
    """
    if not examples:
        raise ValueError("greedy generation recall needs at least one example")
    device = next(model.parameters()).device
    recalls: list[float] = []
    with torch.no_grad():
        for example in examples:
            answer_positions = torch.nonzero(
                example.labels != IGNORE, as_tuple=False
            ).flatten()
            if answer_positions.numel() == 0:
                raise ValueError(f"example {example.example_id!r} has no answer tokens")
            first = int(answer_positions[0])
            expected_positions = torch.arange(
                first, first + answer_positions.numel(), dtype=torch.long
            )
            if not torch.equal(answer_positions.cpu(), expected_positions):
                raise ValueError(
                    f"example {example.example_id!r} has a non-contiguous answer mask"
                )
            sequence = example.input_ids[:first].to(device).unsqueeze(0)
            if sequence.shape[1] == 0:
                raise ValueError(f"example {example.example_id!r} has an empty prompt")
            gold = example.input_ids[answer_positions].to(device)
            generated: list[torch.Tensor] = []
            for _ in range(int(gold.numel())):
                attention = torch.ones_like(sequence)
                logits = model(
                    input_ids=sequence, attention_mask=attention
                ).logits[:, -1, :]
                next_token = logits.argmax(dim=-1)
                generated.append(next_token[0])
                sequence = torch.cat((sequence, next_token[:, None]), dim=1)
            predicted = torch.stack(generated)
            recalls.append(float((predicted == gold).float().mean()))
    return sum(recalls) / len(recalls)
