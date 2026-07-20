"""Numerical fidelity and stability sweep (appendix tab:numerical-stability).

Reference: exact JVP in the preregistered late-MLP block. Variants change one
factor at a time — estimator (fd, chunked gradient-dot), probe radius
(eta/10, eta, 10*eta), block (late MLP, last layer, full trainable) — and are
compared on absolute error, relative error outside the zero tolerance, sign
agreement, Spearman rho, and Overlap@K. Precision arms (bf16 vs fp32
accumulation) are GPU-only and enabled via --dtype.

  python experiments/stability/sweep.py --smoke
  python experiments/stability/sweep.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.analysis.prediction import spearman, top_k_ids  # noqa: E402
from rsus.blocks import BlockSpec, mlp_down_last_layers  # noqa: E402
from rsus.probe.base import ProbeSpec, get_scorer  # noqa: E402

ZERO_TOL = 1e-8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--universe-authors", type=int, default=10)
    p.add_argument("--author", type=int, default=180)
    p.add_argument("--block-last-n", type=int, default=8)
    p.add_argument("--eta", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def build_world(a):
    if a.smoke:
        sys.path.insert(0, str(ROOT / "tests"))
        from conftest import build_tiny, make_example

        from rsus.data.base import CandidateUniverse, Request

        model = build_tiny(0)
        gen = torch.Generator().manual_seed(7)
        forget = [make_example(gen, f"f{i:02d}") for i in range(4)]
        cands = [make_example(gen, f"c{i:02d}") for i in range(16)]
        req = Request.build("stab-smoke", forget, CandidateUniverse.freeze(cands))
        return model, req, mlp_down_last_layers(model, 1)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rsus.data.tofu import load_tofu_examples, tofu_request

    tok = AutoTokenizer.from_pretrained(a.model)
    # eager attention: the jvp comparator drives forward-mode AD, which SDPA kernels lack
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32,
                                                 attn_implementation="eager")
    model = model.to(a.device).eval()
    req = tofu_request(a.author, load_tofu_examples(tok),
                       universe_authors=a.universe_authors, seed=a.seed)
    return model, req, mlp_down_last_layers(model, a.block_last_n)


def agreement(scores: dict[str, float], ref: dict[str, float], k: int) -> dict:
    ids = sorted(ref)
    s = torch.tensor([scores[c] for c in ids], dtype=torch.float64)
    r = torch.tensor([ref[c] for c in ids], dtype=torch.float64)
    err = (s - r).abs()
    nz = r.abs() > ZERO_TOL
    rel = (err[nz] / r[nz].abs()) if nz.any() else torch.zeros(1)
    return {
        "abs_err_max": float(err.max()),
        "rel_err_med": float(rel.median()),
        "sign_agree": float(((s.sign() == r.sign()) | ~nz).double().mean()),
        "rank_rho": spearman(s.tolist(), r.tolist()),
        "overlap_k": len(top_k_ids(scores, k) & top_k_ids(ref, k)) / k,
    }


def main():
    a = parse_args()
    model, req, block = build_world(a)
    k = min(a.k, max(3, len(req.universe) // 4))
    base_spec = ProbeSpec(block=block, eta=a.eta, batch_size=a.batch_size)
    ref = get_scorer("jvp")(model, req, base_spec).scores

    n_layers = model.config.num_hidden_layers
    variants: list[tuple[str, str, str, ProbeSpec]] = [
        ("Estimator", "finite difference", "fd", base_spec),
        ("Estimator", "chunked gradient-dot", "vmap_graddot", base_spec),
        ("Radius", "eta/10", "fd", dataclasses.replace(base_spec, eta=a.eta / 10)),
        ("Radius", "eta (default)", "fd", base_spec),
        ("Radius", "10*eta", "fd", dataclasses.replace(base_spec, eta=a.eta * 10)),
        ("Block", "late MLP (default)", "fd", base_spec),
        ("Block", "last layer", "fd",
         dataclasses.replace(base_spec, block=BlockSpec(r"lm_head\.weight"))),
        ("Block", "full trainable", "fd",
         dataclasses.replace(base_spec, block=BlockSpec(rf".*\.layers\.(?:{'|'.join(map(str, range(n_layers)))})\..*weight"))),
    ]

    out = ROOT / "runs" / ("stability_smoke" if a.smoke else f"stability_{a.model.split('/')[-1]}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for factor, label, scorer, spec in variants:
        try:
            prof = get_scorer(scorer)(model, req, spec)
        except ValueError as e:
            # e.g. 'last layer' targets lm_head.weight, which does not exist as a
            # standalone parameter on tied-embedding models (Qwen2.5): skip the
            # variant instead of losing the sweep.
            print({"factor": factor, "variant": label, "skipped": str(e)})
            continue
        # block variants target a different derivative; compare against their
        # own exact reference so the row measures estimator fidelity, and
        # against the default reference for coordinate dependence. Forward-mode
        # AD cannot pass through SDPA attention kernels, so blocks that include
        # attention weights fall back to the reverse-mode exact reference.
        if spec.block.pattern == block.pattern:
            own_ref = ref
        else:
            try:
                own_ref = get_scorer("jvp")(model, req, spec).scores
            except NotImplementedError:
                own_ref = get_scorer("streaming_backward")(model, req, spec).scores
        row = {"factor": factor, "variant": label,
               **{f"vs_own_{k_}": v for k_, v in agreement(prof.scores, own_ref, k).items()},
               "rank_rho_vs_default": spearman(
                   [prof.scores[c] for c in sorted(ref)], [ref[c] for c in sorted(ref)]
               )}
        rows.append(row)
        print({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in row.items()})
    with open(out / "stability.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'stability.csv'}")


if __name__ == "__main__":
    main()
