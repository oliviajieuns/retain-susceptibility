"""TOFU forget10 adapter: one deletion request per forget10 author.

Layout (validated against ``locuslab/TOFU`` ``full``): 4,000 QA rows, 200
authors x 20 QA in contiguous blocks of 20; forget10 = authors 180-199.
example_id = "tofu-<row:04d>", group = "author-<id:03d>" (fold granularity),
text = raw QA string for external-encoder baselines.

The benchmark-native audit rule is a preregistration choice that is still
open (paper: 'metadata rule'); until frozen, requests carry an empty native
set and the gate experiment audits on the untouched random fold plus the
complete candidate distribution.
"""
from __future__ import annotations

import torch

from rsus.data.base import CandidateUniverse, Example, Request
from rsus.losses import IGNORE

QA_PER_AUTHOR = 20
AUTHORS_TOTAL = 200
FULL_SIZE = AUTHORS_TOTAL * QA_PER_AUTHOR
FORGET10_FIRST_AUTHOR = 180

QUESTION_PREFIX = "Question: "
ANSWER_PREFIX = "\nAnswer:"


def load_tofu_rows():
    from datasets import load_dataset

    ds = load_dataset("locuslab/TOFU", "full")["train"]
    if len(ds) != FULL_SIZE:
        raise ValueError(f"TOFU full has {len(ds)} rows, expected {FULL_SIZE}")
    return ds


def format_qa(question: str, answer: str, tokenizer, max_length: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize one QA pair with the prompt masked: labels are IGNORE on
    'Question: ...\\nAnswer:' and real ids on the answer tokens (+ EOS)."""
    prompt_ids = tokenizer(
        f"{QUESTION_PREFIX}{question}{ANSWER_PREFIX}", add_special_tokens=False
    )["input_ids"]
    answer_ids = tokenizer(f" {answer}", add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    n_prompt = min(len(prompt_ids), max_length)
    if n_prompt >= len(input_ids):
        raise ValueError("answer fully truncated; raise max_length")
    labels = [IGNORE] * n_prompt + input_ids[n_prompt:]
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def load_tofu_examples(tokenizer, max_length: int = 256) -> list[Example]:
    rows = load_tofu_rows()
    out: list[Example] = []
    for idx in range(FULL_SIZE):
        row = rows[idx]
        ids, labels = format_qa(row["question"], row["answer"], tokenizer, max_length)
        out.append(
            Example(
                example_id=f"tofu-{idx:04d}",
                input_ids=ids,
                labels=labels,
                group=f"author-{idx // QA_PER_AUTHOR:03d}",
                text=f"{QUESTION_PREFIX}{row['question']}{ANSWER_PREFIX} {row['answer']}",
            )
        )
    return out


def load_tofu_paraphrases(tokenizer, max_length: int = 256) -> dict[str, Example]:
    """Paraphrased forget10 QA for the paraphrase-recall audit.

    ``forget10_perturbed`` row i corresponds to ``full`` row 3600+i. Returns
    original example_id -> paraphrased Example (same group)."""
    from datasets import load_dataset

    ds = load_dataset("locuslab/TOFU", "forget10_perturbed")["train"]
    first_row = FORGET10_FIRST_AUTHOR * QA_PER_AUTHOR
    if len(ds) != FULL_SIZE - first_row:
        raise ValueError(f"forget10_perturbed has {len(ds)} rows, expected {FULL_SIZE - first_row}")
    out: dict[str, Example] = {}
    for i in range(len(ds)):
        row = ds[i]
        idx = first_row + i
        ids, labels = format_qa(
            row["paraphrased_question"], row["paraphrased_answer"], tokenizer, max_length
        )
        out[f"tofu-{idx:04d}"] = Example(
            example_id=f"tofu-{idx:04d}-para",
            input_ids=ids,
            labels=labels,
            group=f"author-{idx // QA_PER_AUTHOR:03d}",
            text=f"{QUESTION_PREFIX}{row['paraphrased_question']}{ANSWER_PREFIX} {row['paraphrased_answer']}",
        )
    return out


IDK_ANSWER = "I don't know."


def idk_variants(
    tokenizer, forget: list[Example], idk_answer: str = IDK_ANSWER, max_length: int = 256
) -> list[Example]:
    """IdkDPO preferred responses: same question, refusal answer. Questions
    are recovered from Example.text (format_qa's canonical layout)."""
    out: list[Example] = []
    for e in forget:
        if not e.text.startswith(QUESTION_PREFIX) or ANSWER_PREFIX not in e.text:
            raise ValueError(f"cannot recover question from {e.example_id}")
        q = e.text[len(QUESTION_PREFIX) : e.text.index(ANSWER_PREFIX)]
        ids, labels = format_qa(q, idk_answer, tokenizer, max_length)
        out.append(Example(e.example_id + "-idk", ids, labels, group=e.group))
    return out


def tofu_request(
    author_id: int,
    examples: list[Example],
    universe_authors: int | None = None,
    seed: int = 0,
) -> Request:
    """Deletion request for one forget10 author. ``universe_authors`` caps the
    candidate universe to that many whole retained authors (seeded, for the
    gate experiment and smoke runs); None keeps the complete universe."""
    if not FORGET10_FIRST_AUTHOR <= author_id < AUTHORS_TOTAL:
        raise ValueError(f"author {author_id} is not a forget10 author")
    group = f"author-{author_id:03d}"
    forget = [e for e in examples if e.group == group]
    retained_groups = sorted({e.group for e in examples} - {group})
    if universe_authors is not None:
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(retained_groups), generator=gen).tolist()
        keep = {retained_groups[i] for i in perm[:universe_authors]}
    else:
        keep = set(retained_groups)
    cands = [e for e in examples if e.group in keep]
    return Request.build(f"tofu-a{author_id}", forget, CandidateUniverse.freeze(cands))
