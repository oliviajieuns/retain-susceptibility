"""RWKU adapter: one deletion request per real-world knowledge target.

Layout (validated against the cached ``jinzhuoran/RWKU`` schema, probe dump
2026-07-23): ``forget_target`` (train, 200 rows: {intro, target}) names the
deletion-request units; ``forget_level1``/``forget_level2`` (test) hold the
target-knowledge probes ({query, answer, subject, type, level}); and
``neighbor_level1``/``neighbor_level2`` add {neighbor} — knowledge adjacent to
the target that unlearning should retain.

Request semantics mirror TOFU's author requests:

- ``request_id = "rwku-t<idx:03d>"`` for row ``idx`` of ``forget_target``.
- forget set = every level-1/2 forget probe whose subject is that target.
- retained-candidate universe = the target's own neighbor probes (the
  susceptible locality; ``group = "neighbor-<entity>"``) plus the forget
  probes of a frozen pool of *other* targets (the remote mass;
  ``group = "target-<idx>"``). Groups are the fold-granularity unit exactly
  like TOFU's retained authors.
- the neighbor probes form the benchmark-native audit set
  (``native_audit_ids``), because RWKU defines them as the knowledge that
  must survive the removal.

Probes are tokenized with the prompt masked. ``type == "cloze"`` prompts end
at the ``___`` blank and the answer is the completion; every other type uses
the TOFU ``Question:/Answer:`` layout.
"""
from __future__ import annotations

import torch

from rsus.data.base import CandidateUniverse, Example, Request
from rsus.data.tofu import format_qa
from rsus.losses import IGNORE

TARGETS_TOTAL = 200
CLOZE_BLANK = "___"

FORGET_CONFIGS = ("forget_level1", "forget_level2")
NEIGHBOR_CONFIGS = ("neighbor_level1", "neighbor_level2")


def load_rwku_tables() -> dict[str, list[dict]]:
    """Load the needed RWKU configs (from the local HF cache when offline)."""
    from datasets import load_dataset

    tables: dict[str, list[dict]] = {}
    targets = load_dataset("jinzhuoran/RWKU", "forget_target")["train"]
    if len(targets) != TARGETS_TOTAL:
        raise ValueError(f"RWKU forget_target has {len(targets)} rows, expected {TARGETS_TOTAL}")
    tables["forget_target"] = list(targets)
    for config in (*FORGET_CONFIGS, *NEIGHBOR_CONFIGS):
        tables[config] = list(load_dataset("jinzhuoran/RWKU", config)["test"])
    return tables


def format_cloze(query: str, answer: str, tokenizer, max_length: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a cloze probe: prompt is the query up to the blank, labels are
    IGNORE on the prompt and real ids on the completion (+ EOS)."""
    blank = query.find(CLOZE_BLANK)
    prompt_text = (query[:blank] if blank >= 0 else query).rstrip()
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(f" {answer}", add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    n_prompt = min(len(prompt_ids), max_length)
    if n_prompt >= len(input_ids):
        raise ValueError("cloze completion fully truncated; raise max_length")
    labels = [IGNORE] * n_prompt + input_ids[n_prompt:]
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def _probe_example(row: dict, example_id: str, group: str, tokenizer, max_length: int) -> Example:
    query, answer = str(row["query"]), str(row["answer"])
    if str(row.get("type", "")).strip().casefold() == "cloze":
        ids, labels = format_cloze(query, answer, tokenizer, max_length)
        text = f"{query} -> {answer}"
    else:
        ids, labels = format_qa(query, answer, tokenizer, max_length)
        text = f"Question: {query}\nAnswer: {answer}"
    return Example(example_id=example_id, input_ids=ids, labels=labels, group=group, text=text)


def target_name(tables: dict[str, list[dict]], target_index: int) -> str:
    if not 0 <= target_index < TARGETS_TOTAL:
        raise ValueError(f"target_index {target_index} outside 0..{TARGETS_TOTAL - 1}")
    return str(tables["forget_target"][target_index]["target"])


def _rows_for_subject(tables: dict[str, list[dict]], configs: tuple[str, ...], subject: str) -> list[tuple[str, int, dict]]:
    picked = []
    for config in configs:
        for row_idx, row in enumerate(tables[config]):
            if str(row["subject"]) == subject:
                picked.append((config, row_idx, row))
    return picked


def rwku_request(
    tokenizer,
    target_index: int,
    candidate_targets: list[int],
    max_length: int = 256,
    tables: dict[str, list[dict]] | None = None,
) -> Request:
    """Build the Request for one RWKU target.

    ``candidate_targets`` is the frozen remote pool: indices of *other*
    targets whose forget probes act as far retained candidates, mirroring
    TOFU's frozen retained-author pools. It must not contain the request
    target and must not be empty.
    """
    if tables is None:
        tables = load_rwku_tables()
    if not candidate_targets:
        raise ValueError("candidate_targets must name at least one remote target")
    if target_index in candidate_targets:
        raise ValueError("the deletion target cannot appear in its own candidate pool")
    if len(set(candidate_targets)) != len(candidate_targets):
        raise ValueError("candidate_targets contains duplicates")

    subject = target_name(tables, target_index)
    request_id = f"rwku-t{target_index:03d}"

    forget = [
        _probe_example(row, f"{request_id}-{config}-{row_idx:05d}", f"target-{target_index:03d}",
                       tokenizer, max_length)
        for config, row_idx, row in _rows_for_subject(tables, FORGET_CONFIGS, subject)
    ]
    if not forget:
        raise ValueError(f"no forget probes found for RWKU target {subject!r}")

    universe: list[Example] = []
    native_audit_ids: set[str] = set()
    for config, row_idx, row in _rows_for_subject(tables, NEIGHBOR_CONFIGS, subject):
        neighbor = str(row.get("neighbor", "")).strip() or "unknown"
        example_id = f"{request_id}-{config}-{row_idx:05d}"
        universe.append(
            _probe_example(row, example_id, f"neighbor-{neighbor}", tokenizer, max_length)
        )
        native_audit_ids.add(example_id)
    if not native_audit_ids:
        raise ValueError(f"no neighbor probes found for RWKU target {subject!r}")

    for other in candidate_targets:
        other_subject = target_name(tables, other)
        rows = _rows_for_subject(tables, FORGET_CONFIGS, other_subject)
        if not rows:
            raise ValueError(f"remote pool target {other} ({other_subject!r}) has no probes")
        for config, row_idx, row in rows:
            universe.append(
                _probe_example(row, f"rwku-t{other:03d}-{config}-{row_idx:05d}",
                               f"target-{other:03d}", tokenizer, max_length)
            )

    return Request.build(
        request_id=request_id,
        forget=forget,
        universe=CandidateUniverse.freeze(universe),
        native_audit_ids=native_audit_ids,
    )
