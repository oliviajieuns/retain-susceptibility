"""WMDP-bio/MMLU adapter: hazardous-knowledge slices with an MMLU retain universe.

Layout (validated against the cached ``cais/wmdp`` and ``cais/mmlu`` schemas,
inventory 2026-07-23): ``wmdp-bio`` (test, 1,273 rows: {question, choices,
answer}) supplies the hazardous forget material and ``mmlu``/``all`` (test:
{question, choices, answer, subject}) supplies the retained-candidate
universe.  Both are 4-way multiple choice; examples are rendered as TOFU-style
``Question:/Answer:`` completions whose answer is the correct choice text.

Request semantics mirror the RWKU adapter:

- ``request_id = "wmdp-r<idx:03d>"`` for forget slice ``idx``: a contiguous
  ``chunk_size`` (default 40) block of the wmdp-bio question list after one
  frozen seeded shuffle (``SHUFFLE_SEED``), so slices are deterministic,
  disjoint, and independent of runtime state.
- forget set = the slice's questions (``group = "slice-<idx>"``).
- retained-candidate universe = MMLU questions grouped by ``subject``
  (``group = "mmlu-<subject>"``; the fold-granularity unit exactly like
  TOFU's retained authors).  ``candidate_subjects`` freezes the exact subject
  pool as indices into the sorted subject list; campaigns use disjoint
  development and audit pools so calibration never observes a candidate that
  later appears in a sealed audit.  Per subject the first ``per_subject``
  questions in dataset order are taken, capped at ``max_candidates`` — no
  unseeded randomness anywhere.
- ``native_audit_ids`` = every ``NATIVE_AUDIT_STRIDE``-th candidate of the
  frozen universe, the benchmark-native utility slice that must survive.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from rsus.data.base import CandidateUniverse, Example, Request
from rsus.data.tofu import format_qa

QUESTIONS_TOTAL = 1273
CHUNK_SIZE = 40
REQUESTS_TOTAL = QUESTIONS_TOTAL // CHUNK_SIZE  # 31 frozen forget slices
SHUFFLE_SEED = 20260723
PER_SUBJECT = 8
MAX_CANDIDATES = 480
NATIVE_AUDIT_STRIDE = 4

WMDP_REPO = "cais/wmdp"
WMDP_CACHE_REPO_DIR = "cais___wmdp"
WMDP_CONFIG = "wmdp-bio"
MMLU_REPO = "cais/mmlu"
MMLU_CACHE_REPO_DIR = "cais___mmlu"
MMLU_CONFIG = "all"
SPLIT = "test"


def find_cached_arrows(
    repo_dir: str, config: str, split: str, hf_home: str | Path | None = None
) -> list[Path]:
    """Arrow shards of one cached config split, newest fingerprint first.

    Reading the arrow files directly (``datasets.Dataset.from_file``) skips the
    library's builder FileLock, which fails with PermissionError on the shared
    read-only cache where lock files belong to another user.  Unlike RWKU, the
    mmlu ``all`` config caches several splits side by side, so shards are
    filtered to the requested split.
    """
    home = Path(hf_home or os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    config_dir = home / "datasets" / repo_dir / config
    if not config_dir.is_dir():
        return []
    infos = sorted(config_dir.glob("*/*/dataset_info.json"))
    if not infos:
        return []
    arrows = sorted(infos[-1].parent.glob("*.arrow"))
    return [
        arrow for arrow in arrows
        if arrow.stem.endswith(f"-{split}") or f"-{split}-" in arrow.stem
    ]


def _load_config_rows(repo: str, repo_dir: str, config: str, split: str) -> list[dict]:
    arrows = find_cached_arrows(repo_dir, config, split)
    if arrows:
        from datasets import Dataset

        rows: list[dict] = []
        for arrow in arrows:
            rows.extend(Dataset.from_file(str(arrow)))
        return rows
    # Cache miss: fall back to the normal loader (requires hub access or a
    # writable cache; on the cluster the shared cache should always hit above).
    from datasets import load_dataset

    ds = load_dataset(repo, config)
    return list(ds[split])


def load_wmdp_tables() -> dict[str, list[dict]]:
    """Load wmdp-bio and mmlu, preferring lock-free reads of the cache."""
    questions = _load_config_rows(WMDP_REPO, WMDP_CACHE_REPO_DIR, WMDP_CONFIG, SPLIT)
    if len(questions) != QUESTIONS_TOTAL:
        raise ValueError(
            f"wmdp-bio has {len(questions)} rows, expected {QUESTIONS_TOTAL}"
        )
    retain = _load_config_rows(MMLU_REPO, MMLU_CACHE_REPO_DIR, MMLU_CONFIG, SPLIT)
    if not retain:
        raise ValueError("MMLU retain table is empty")
    return {"wmdp_bio": questions, "mmlu": retain}


def forget_order() -> list[int]:
    """The one frozen shuffle assigning wmdp-bio rows to forget slices."""
    gen = torch.Generator().manual_seed(SHUFFLE_SEED)
    return torch.randperm(QUESTIONS_TOTAL, generator=gen).tolist()


def mmlu_subjects(tables: dict[str, list[dict]]) -> list[str]:
    """Sorted subject roster; candidate pools index into this list."""
    return sorted({str(row["subject"]) for row in tables["mmlu"]})


def _choice_example(row: dict, example_id: str, group: str, tokenizer, max_length: int) -> Example:
    question = str(row["question"]).strip()
    choices = list(row["choices"])
    answer_index = int(row["answer"])
    if not 0 <= answer_index < len(choices):
        raise ValueError(f"answer index {answer_index} outside choices for {example_id}")
    answer = str(choices[answer_index]).strip()
    ids, labels = format_qa(question, answer, tokenizer, max_length)
    return Example(example_id=example_id, input_ids=ids, labels=labels, group=group,
                   text=f"Question: {question}\nAnswer: {answer}")


def wmdp_request(
    tokenizer,
    request_index: int,
    candidate_subjects: list[int],
    max_length: int = 256,
    chunk_size: int = CHUNK_SIZE,
    per_subject: int = PER_SUBJECT,
    max_candidates: int = MAX_CANDIDATES,
    tables: dict[str, list[dict]] | None = None,
) -> Request:
    """Build the Request for one WMDP-bio forget slice.

    ``candidate_subjects`` is the frozen retain pool: indices into the sorted
    MMLU subject list whose leading ``per_subject`` questions form the
    retained-candidate universe, mirroring TOFU's frozen retained-author
    pools.  It must be non-empty and duplicate-free.
    """
    if tables is None:
        tables = load_wmdp_tables()
    if len(tables["wmdp_bio"]) != QUESTIONS_TOTAL:
        raise ValueError(
            f"wmdp-bio table has {len(tables['wmdp_bio'])} rows, expected {QUESTIONS_TOTAL}; "
            "the frozen slice shuffle covers the full question list"
        )
    if chunk_size < 1 or per_subject < 1 or max_candidates < 1:
        raise ValueError("chunk_size, per_subject and max_candidates must be positive")
    requests_total = QUESTIONS_TOTAL // chunk_size
    if not 0 <= request_index < requests_total:
        raise ValueError(
            f"request_index {request_index} outside 0..{requests_total - 1} "
            f"(chunk_size={chunk_size})"
        )
    if not candidate_subjects:
        raise ValueError("candidate_subjects must name at least one MMLU subject")
    if len(set(candidate_subjects)) != len(candidate_subjects):
        raise ValueError("candidate_subjects contains duplicates")
    subjects = mmlu_subjects(tables)
    invalid = [index for index in candidate_subjects if not 0 <= index < len(subjects)]
    if invalid:
        raise ValueError(
            f"candidate subject indices outside 0..{len(subjects) - 1}: {invalid}"
        )

    request_id = f"wmdp-r{request_index:03d}"
    order = forget_order()
    start = request_index * chunk_size
    forget = [
        _choice_example(tables["wmdp_bio"][row_idx], f"{request_id}-q{row_idx:05d}",
                        f"slice-{request_index:03d}", tokenizer, max_length)
        for row_idx in order[start:start + chunk_size]
    ]

    chosen = [subjects[index] for index in candidate_subjects]
    rows_by_subject: dict[str, list[tuple[int, dict]]] = {name: [] for name in chosen}
    for row_idx, row in enumerate(tables["mmlu"]):
        bucket = rows_by_subject.get(str(row["subject"]))
        if bucket is not None and len(bucket) < per_subject:
            bucket.append((row_idx, row))

    universe: list[Example] = []
    for name in chosen:
        if not rows_by_subject[name]:
            raise ValueError(f"MMLU subject {name!r} has no rows")
        for row_idx, row in rows_by_subject[name]:
            universe.append(
                _choice_example(row, f"mmlu-{row_idx:05d}", f"mmlu-{name}",
                                tokenizer, max_length)
            )
    universe = universe[:max_candidates]
    native_audit_ids = {
        example.example_id
        for position, example in enumerate(universe)
        if position % NATIVE_AUDIT_STRIDE == 0
    }

    return Request.build(
        request_id=request_id,
        forget=forget,
        universe=CandidateUniverse.freeze(universe),
        native_audit_ids=native_audit_ids,
    )
