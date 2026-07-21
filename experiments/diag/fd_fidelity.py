"""Preflight fd_norm fidelity gate (A/B/C decomposition) on ~128 candidates.

Runs the decomposition over an R x eta x seed grid and prints, per cell, the
rank agreement and value-scale of A (exact ||g||^2), B (exact-gradient random
projection) and C (finite-difference == d*fd_norm). Read the columns as:

    rho(A,B) low   -> too few directions R, or direction-normalization
    rho(B,C) low   -> finite-difference / eta / bf16 perturbation underflow
    rho(A,C) high across the grid -> fd_norm is a faithful backward-free ||g||^2

Run it TWICE (--dtype float32 and --dtype bfloat16) on the same author to test
the bf16-underflow hypothesis directly: at eta=3e-4 a unit-direction coordinate
moves ~eta/sqrt(d) ~ 1e-7, far below the bf16 ULP, so C can collapse under bf16
while staying faithful under fp32.

  python experiments/diag/fd_fidelity.py \
      --model /group-volume/models/Qwen2.5-7B-Instruct --device cuda --dtype float32 \
      --n-cands 128 --dirs 16,32,64 --etas 3e-5,3e-4,3e-3 --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.analysis.prediction import spearman, top_k_ids  # noqa: E402
from rsus.blocks import mlp_down_last_layers  # noqa: E402
from rsus.data.base import CandidateUniverse, Request  # noqa: E402
from rsus.data.tofu import load_tofu_examples, tofu_request  # noqa: E402
from rsus.probe.base import ProbeSpec  # noqa: E402
from rsus.probe.fidelity import (  # noqa: E402
    B_scores,
    C_scores,
    direction_bank,
    exact_A_and_projsq,
)


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def agree(u: dict[str, float], v: dict[str, float], k: int) -> tuple[float, float]:
    ids = sorted(set(u) & set(v))
    rho = spearman([u[c] for c in ids], [v[c] for c in ids])
    ov = len(top_k_ids({c: u[c] for c in ids}, k) & top_k_ids({c: v[c] for c in ids}, k)) / k
    return rho, ov


def ratio_median(num: dict[str, float], den: dict[str, float]) -> float:
    xs = [num[c] / den[c] for c in num if c in den and den[c] > 0]
    return statistics.median(xs) if xs else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--author", type=int, default=180)
    p.add_argument("--universe-authors", type=int, default=30)
    p.add_argument("--n-cands", type=int, default=128)
    p.add_argument("--block-last-n", type=int, default=8)
    p.add_argument("--dirs", default="16,32,64")
    p.add_argument("--etas", default="3e-5,3e-4,3e-3")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--k", type=int, default=0, help="Overlap@k; 0 -> max(5, round(0.1*n))")
    p.add_argument("--out", default="")
    a = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(a.device).eval()
    full = tofu_request(a.author, load_tofu_examples(tok), universe_authors=a.universe_authors, seed=0)
    cands = list(full.universe.examples)[: a.n_cands]
    req = Request.build(full.request_id, list(full.forget), CandidateUniverse.freeze(cands))
    n = len(cands)
    k = a.k or max(5, round(0.1 * n))
    seeds, Rs, etas = _ints(a.seeds), _ints(a.dirs), _floats(a.etas)

    spec = ProbeSpec(block=mlp_down_last_layers(model, a.block_last_n), eta=etas[0],
                     batch_size=8, n_dirs=max(Rs))
    sel = spec.block.select(model)
    d = sum(pp.numel() for pp in sel.values())
    print(f"model={a.model} dtype={a.dtype} device={a.device} |cands|={n} dim(B)={d} k={k}")
    print(f"seeds={seeds} R={Rs} eta={etas}\n")

    bank = direction_bank(sel, seeds, max(Rs))
    A, projsq, d = exact_A_and_projsq(model, req, spec, bank)

    hdr = f"{'seed':>4} {'R':>4} {'eta':>8} | {'rho(A,B)':>9} {'rho(B,C)':>9} {'rho(A,C)':>9}" \
          f" | {'ov(A,C)':>8} | {'medB/A':>7} {'medC/A':>7} {'medC/B':>7}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for s in seeds:
        for R in Rs:
            B = B_scores(projsq, s, R, d)
            rab, _ = agree(A, B, k)
            for eta in etas:
                C = C_scores(model, req, spec, bank[s][:R], eta, d)
                rbc, _ = agree(B, C, k)
                rac, oac = agree(A, C, k)
                mba, mca, mcb = ratio_median(B, A), ratio_median(C, A), ratio_median(C, B)
                print(f"{s:>4} {R:>4} {eta:>8.0e} | {rab:>9.3f} {rbc:>9.3f} {rac:>9.3f}"
                      f" | {oac:>8.3f} | {mba:>7.2f} {mca:>7.2f} {mcb:>7.2f}")
                rows.append({"seed": s, "R": R, "eta": eta, "rho_AB": rab, "rho_BC": rbc,
                             "rho_AC": rac, "ov_AC": oac, "medB_A": mba, "medC_A": mca, "medC_B": mcb})

    best = max(rows, key=lambda r: (r["R"], r["rho_AC"]))
    print("\n=== VERDICT (best cell: seed={seed} R={R} eta={eta:.0e}) ===".format(**best))
    if best["rho_AB"] < 0.7:
        print(f"  rho(A,B)={best['rho_AB']:.3f} LOW -> Monte-Carlo (raise R) or direction-normalization; "
              "NOT a finite-difference problem.")
    elif best["rho_BC"] < 0.7:
        print(f"  rho(A,B)={best['rho_AB']:.3f} ok but rho(B,C)={best['rho_BC']:.3f} LOW -> "
              "finite-difference / eta / bf16 perturbation underflow. Re-run --dtype float32 "
              "and sweep eta; if C recovers under fp32, it was precision.")
    elif best["rho_AC"] >= 0.8:
        print(f"  rho(A,C)={best['rho_AC']:.3f} HIGH -> fd_norm is a FAITHFUL backward-free ||g||^2. "
              "Any damage-prediction gap is the analysis pipeline / fold, not the estimator. "
              "Proceed to the mechanism-matched campaign (no optimizer averaging).")
    else:
        print(f"  mixed: rho(A,B)={best['rho_AB']:.3f} rho(B,C)={best['rho_BC']:.3f} "
              f"rho(A,C)={best['rho_AC']:.3f} -- inspect the grid above.")

    out = Path(a.out) if a.out else ROOT / "runs" / f"fidelity_{a.model.split('/')[-1]}_{a.dtype}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
