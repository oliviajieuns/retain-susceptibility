"""Matched loss-shake fidelity/cost runner for Appendix ``tab:bwfree``.

The benchmark compares the paper's exact squared block-gradient energy with
dimension-corrected loss-shake energy on the same model, block, candidates,
precision, and device. It emits unit-level JSONL accepted by the
``lse_fidelity_cost`` artifact contract; it never inserts draft numbers into
LaTeX.

Examples
--------
CPU harness::

    python experiments/cost/bench.py --smoke --dirs 4,8 --repeats 2

Frozen GPU cell::

    python experiments/cost/bench.py \
      --model /group-volume/models/Qwen2.5-7B-Instruct --device cuda \
      --dtype float32 --candidate-authors 0-29 --dirs 16,32,64 \
      --norm-eta 3e-3 --out runs/paper/lse_qwen7.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.analysis.prediction import spearman, top_k_ids  # noqa: E402
from rsus.blocks import mlp_down_last_layers  # noqa: E402
from rsus.costs import CostRecord  # noqa: E402
from rsus.probe.base import ProbeSpec, get_scorer  # noqa: E402
from rsus.probe.fidelity import perturbation_report  # noqa: E402
from rsus.probe.finite_diff import aggregate_fd_norm  # noqa: E402


FIDELITY_PROTOCOL_FIELDS = (
    "directions",
    "repeats",
    "block_last_n",
    "norm_eta",
    "batch_size",
    "author",
    "candidate_authors",
    "n_candidates",
    "candidate_seed",
    "seed",
    "k",
    "min_rho",
    "min_overlap",
    "min_split_half",
    "min_perturbation_survival",
)


def _protocol_sha256(args: argparse.Namespace, directions: list[int]) -> str:
    values = vars(args)
    protocol = {
        field: (directions if field == "directions" else values[field])
        for field in FIDELITY_PROTOCOL_FIELDS
    }
    body = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _ints(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item < 2 or item % 2 for item in result):
        raise ValueError("--dirs must contain positive even integers")
    return result


def _int_ranges(value: str) -> list[int]:
    result: list[int] = []
    for raw in (item.strip() for item in value.split(",") if item.strip()):
        if "-" in raw:
            left, right = (int(item) for item in raw.split("-", 1))
            if right < left:
                raise ValueError(f"descending range {raw!r}")
            result.extend(range(left, right + 1))
        else:
            result.append(int(raw))
    if len(result) != len(set(result)):
        raise ValueError("--candidate-authors contains duplicates")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--author", type=int, default=198)
    parser.add_argument("--candidate-authors", default="0-29")
    parser.add_argument("--n-candidates", type=int, default=128)
    parser.add_argument("--candidate-seed", type=int, default=314159)
    parser.add_argument("--block-last-n", type=int, default=8)
    parser.add_argument("--norm-eta", type=float, default=3e-3)
    parser.add_argument("--dirs", default="16,32,64")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--min-rho", type=float, default=0.8)
    parser.add_argument("--min-overlap", type=float, default=0.7)
    parser.add_argument("--min-split-half", type=float, default=0.7)
    parser.add_argument("--min-perturbation-survival", type=float, default=0.9)
    parser.add_argument("--out", default="")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _build_world(args: argparse.Namespace):
    if args.smoke:
        sys.path.insert(0, str(ROOT / "tests"))
        from conftest import build_tiny, make_example
        from rsus.data.base import CandidateUniverse, Request

        model = build_tiny(0)
        generator = torch.Generator().manual_seed(7)
        forget = [make_example(generator, f"f{i:02d}") for i in range(4)]
        candidates = [make_example(generator, f"c{i:02d}") for i in range(16)]
        request = Request.build(
            "bench-smoke", forget, CandidateUniverse.freeze(candidates)
        )
        block = mlp_down_last_layers(model, 1)
        model_id = "tiny-random"
        precision = str(next(model.parameters()).dtype).replace("torch.", "")
        return model, request, block, model_id, precision

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rsus.data.base import CandidateUniverse, Request
    from rsus.data.tofu import load_tofu_examples, tofu_request

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager"
    ).to(args.device).eval()
    authors = _int_ranges(args.candidate_authors)
    full = tofu_request(
        args.author,
        load_tofu_examples(tokenizer),
        universe_authors=len(authors),
        candidate_authors=authors,
    )
    candidates = sorted(full.universe.examples, key=lambda item: item.example_id)
    generator = torch.Generator().manual_seed(args.candidate_seed)
    order = torch.randperm(len(candidates), generator=generator).tolist()
    candidates = [
        candidates[index]
        for index in order[: min(args.n_candidates, len(candidates))]
    ]
    request = Request.build(
        full.request_id, list(full.forget), CandidateUniverse.freeze(candidates)
    )
    block = mlp_down_last_layers(model, args.block_last_n)
    return (
        model,
        request,
        block,
        args.model_id or Path(args.model).name,
        args.dtype,
    )


def _agreement(
    exact: dict[str, float], estimate: dict[str, float], k: int
) -> tuple[float, float]:
    if set(exact) != set(estimate):
        raise ValueError("exact and loss-shake candidate support differs")
    candidate_ids = sorted(exact)
    rho = spearman(
        [exact[candidate_id] for candidate_id in candidate_ids],
        [estimate[candidate_id] for candidate_id in candidate_ids],
    )
    overlap = len(top_k_ids(exact, k) & top_k_ids(estimate, k)) / k
    return rho, overlap


def _cost_row(
    *,
    model_id: str,
    model_source: str,
    precision: str,
    block: str,
    protocol_sha256: str,
    profiler: str,
    directions: int,
    repeat: int,
    rho_exact: float,
    overlap_k: float,
    split_half_rho: float,
    perturbation_survival: float,
    record: CostRecord,
    valid: bool,
    candidate_backward: bool,
) -> dict[str, object]:
    return {
        "model": model_id,
        "model_source": model_source,
        "precision": precision,
        "block": block,
        "protocol_sha256": protocol_sha256,
        "profiler": profiler,
        "R": directions,
        "repeat": repeat,
        "rho_exact": rho_exact,
        "overlap_k": overlap_k,
        "split_half_rho": split_half_rho,
        "perturbation_survival": perturbation_survival,
        "time_seconds": record.wall_s,
        "peak_memory_bytes": int(record.peak_mem_bytes),
        "integrity_valid": valid,
        "candidate_backward": candidate_backward,
        "forward_passes": record.fwd_passes,
        "backward_passes": record.bwd_passes,
        "tokens_forward": record.tokens_fwd,
        "tokens_backward": record.tokens_bwd,
    }


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    directions = _ints(args.dirs)
    protocol_sha256 = _protocol_sha256(args, directions)
    model, request, block, model_id, precision = _build_world(args)
    candidate_count = len(request.universe.examples)
    k = args.k or max(1, round(0.1 * candidate_count))
    k = min(k, candidate_count)
    base_spec = ProbeSpec(
        block=block,
        eta=args.norm_eta,
        norm_eta=args.norm_eta,
        seed=args.seed,
        batch_size=args.batch_size,
        n_dirs=max(directions),
    )
    dimension = sum(parameter.numel() for parameter in block.select(model).values())
    rows: list[dict[str, object]] = []

    exact_scores: dict[str, float] | None = None
    for repeat in range(args.repeats):
        profile = get_scorer("grad_norm")(model, request, base_spec)
        exact_scores = exact_scores or profile.scores
        if profile.scores != exact_scores:
            raise RuntimeError("exact energy changed across a frozen repeated benchmark")
        rows.append(
            _cost_row(
                model_id=model_id,
                model_source="smoke" if args.smoke else args.model,
                precision=precision,
                block=block.pattern,
                protocol_sha256=protocol_sha256,
                profiler="exact_energy",
                directions=0,
                repeat=repeat,
                rho_exact=1.0,
                overlap_k=1.0,
                split_half_rho=1.0,
                perturbation_survival=1.0,
                record=profile.cost,
                valid=True,
                candidate_backward=True,
            )
        )
    assert exact_scores is not None

    perturbation = perturbation_report(model, base_spec, args.norm_eta)
    survival = min(
        float(perturbation["eff_over_eta"]),
        float(perturbation["frac_changed"]),
    )
    for direction_count in directions:
        spec = dataclasses.replace(base_spec, n_dirs=direction_count)
        for repeat in range(args.repeats):
            profile = get_scorer("fd_norm")(model, request, spec)
            responses = profile.artifacts["direction_responses"]
            first = aggregate_fd_norm(responses[: direction_count // 2], dimension)
            second = aggregate_fd_norm(responses[direction_count // 2 :], dimension)
            split_half = _agreement(first, second, k)[0]
            rho, overlap = _agreement(exact_scores, profile.scores, k)
            valid = (
                rho >= args.min_rho
                and overlap >= args.min_overlap
                and split_half >= args.min_split_half
                and survival >= args.min_perturbation_survival
            )
            rows.append(
                _cost_row(
                    model_id=model_id,
                    model_source="smoke" if args.smoke else args.model,
                    precision=precision,
                    block=block.pattern,
                    protocol_sha256=protocol_sha256,
                    profiler="loss_shake",
                    directions=direction_count,
                    repeat=repeat,
                    rho_exact=rho,
                    overlap_k=overlap,
                    split_half_rho=split_half,
                    perturbation_survival=survival,
                    record=profile.cost,
                    valid=valid,
                    candidate_backward=False,
                )
            )

    output = Path(args.out) if args.out else ROOT / "runs" / "paper" / f"lse_{model_id}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    loss_shake = [row for row in rows if row["profiler"] == "loss_shake"]
    print(f"wrote {len(rows)} rows to {output}")
    print(
        "loss-shake median time by R: "
        + ", ".join(
            f"{direction_count}={statistics.median(float(row['time_seconds']) for row in loss_shake if row['R'] == direction_count):.3f}s"
            for direction_count in directions
        )
    )


if __name__ == "__main__":
    main()
