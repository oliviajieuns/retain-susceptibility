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
import json
import math
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
    exact_A_and_projsq,
    perturbation_report,
)


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _int_ranges(raw: str) -> list[int]:
    values = []
    for item in (part.strip() for part in raw.split(",") if part.strip()):
        if "-" in item:
            lo_raw, hi_raw = item.split("-", 1)
            lo, hi = int(lo_raw), int(hi_raw)
            if hi < lo:
                raise ValueError(f"descending range: {item!r}")
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(item))
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate candidate author: {raw!r}")
    return values


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
    p.add_argument("--candidate-authors", default="",
                   help="fixed development-only retained-author pool")
    p.add_argument("--n-cands", type=int, default=64)
    p.add_argument("--candidate-seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-last-n", type=int, default=1,
                   help="fidelity MC variance is 2/R, independent of block dim; "
                        "1 layer is a valid, cheap estimator test")
    p.add_argument("--dirs", default="16,32,64")
    p.add_argument("--etas", default="3e-4,3e-3")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--k", type=int, default=0, help="Overlap@k; 0 -> max(5, round(0.1*n))")
    p.add_argument("--out", default="")
    p.add_argument("--certificate", default="")
    p.add_argument("--gate-r", type=int, default=64)
    p.add_argument("--gate-eta", type=float, default=3e-3)
    p.add_argument("--gate-seed", type=int, default=0)
    p.add_argument("--min-rho-ab", type=float, default=0.70)
    p.add_argument("--min-rho-bc", type=float, default=0.80)
    p.add_argument("--min-rho-ac", type=float, default=0.80)
    p.add_argument("--min-eff-ratio", type=float, default=0.90)
    p.add_argument("--min-frac-changed", type=float, default=0.90)
    p.add_argument("--enforce-gate", action="store_true")
    a = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(a.device).eval()
    candidate_authors = _int_ranges(a.candidate_authors) if a.candidate_authors else None
    full = tofu_request(
        a.author,
        load_tofu_examples(tok),
        universe_authors=a.universe_authors,
        seed=0,
        candidate_authors=candidate_authors,
    )
    all_cands = sorted(full.universe.examples, key=lambda example: example.example_id)
    candidate_gen = torch.Generator().manual_seed(a.candidate_seed)
    order = torch.randperm(len(all_cands), generator=candidate_gen).tolist()
    cands = [all_cands[index] for index in order[: a.n_cands]]
    req = Request.build(full.request_id, list(full.forget), CandidateUniverse.freeze(cands))
    n = len(cands)
    k = a.k or max(5, round(0.1 * n))
    seeds, Rs, etas = _ints(a.seeds), _ints(a.dirs), _floats(a.etas)

    if (a.gate_r not in Rs or a.gate_seed not in seeds
            or not any(math.isclose(a.gate_eta, eta, rel_tol=0.0, abs_tol=1e-15) for eta in etas)):
        p.error("the frozen --gate-r/--gate-seed/--gate-eta cell must be present in the grid")
    spec = ProbeSpec(block=mlp_down_last_layers(model, a.block_last_n), eta=etas[0],
                     batch_size=a.batch_size, n_dirs=max(Rs))
    sel = spec.block.select(model)
    d = sum(pp.numel() for pp in sel.values())
    print(f"model={a.model} dtype={a.dtype} device={a.device} |cands|={n} "
          f"block_last_n={a.block_last_n} dim(B)={d} k={k}")
    print(f"seeds={seeds} R={Rs} eta={etas}\n")

    # --- perturbation report FIRST: catch bf16 underflow before trusting any C ---
    print("=== realized perturbation (theta -> theta + eta*v, v unit) ===")
    print(f"{'eta':>8} | {'eff_norm':>10} {'eff/eta':>8} {'frac_changed':>13}")
    underflow = False
    perturbations = {}
    for eta in etas:
        r = perturbation_report(model, spec, eta)
        perturbations[eta] = r
        print(f"{eta:>8.0e} | {r['eff_norm']:>10.3e} {r['eff_over_eta']:>8.3f} {r['frac_changed']:>13.4f}")
        if r["eff_over_eta"] < 0.5 or r["frac_changed"] < 0.5:
            underflow = True
    if underflow:
        print("  !! perturbation UNDERFLOW (eff/eta << 1 or few params changed): C is unreliable "
              "at this dtype/eta. Re-run --dtype float32 and/or larger --etas.")
    print()

    A, projsq, d = exact_A_and_projsq(model, req, spec, seeds, max(Rs))

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
                C = C_scores(model, req, spec, s, R, eta, d)
                rbc, _ = agree(B, C, k)
                rac, oac = agree(A, C, k)
                mba, mca, mcb = ratio_median(B, A), ratio_median(C, A), ratio_median(C, B)
                print(f"{s:>4} {R:>4} {eta:>8.0e} | {rab:>9.3f} {rbc:>9.3f} {rac:>9.3f}"
                      f" | {oac:>8.3f} | {mba:>7.2f} {mca:>7.2f} {mcb:>7.2f}")
                rows.append({"seed": s, "R": R, "eta": eta, "rho_AB": rab, "rho_BC": rbc,
                             "rho_AC": rac, "ov_AC": oac, "medB_A": mba, "medC_A": mca,
                             "medC_B": mcb, **perturbations[eta]})

    gate = next(
        row for row in rows
        if row["R"] == a.gate_r and row["seed"] == a.gate_seed
        and math.isclose(row["eta"], a.gate_eta, rel_tol=0.0, abs_tol=1e-15)
    )
    print("\n=== FROZEN-CELL VERDICT (seed={seed} R={R} eta={eta:.0e}) ===".format(**gate))
    if gate["rho_AB"] < a.min_rho_ab:
        print(f"  rho(A,B)={gate['rho_AB']:.3f} LOW -> Monte-Carlo (raise R) or direction-normalization; "
              "NOT a finite-difference problem.")
    elif gate["rho_BC"] < a.min_rho_bc:
        print(f"  rho(A,B)={gate['rho_AB']:.3f} ok but rho(B,C)={gate['rho_BC']:.3f} LOW -> "
              "finite-difference / eta / bf16 perturbation underflow. Re-run --dtype float32 "
              "and sweep eta; if C recovers under fp32, it was precision.")
    elif gate["rho_AC"] >= a.min_rho_ac:
        print(f"  rho(A,C)={gate['rho_AC']:.3f} HIGH -> fd_norm is a FAITHFUL backward-free ||g||^2. "
              "Any damage-prediction gap is the analysis pipeline / fold, not the estimator. "
              "Proceed to the mechanism-matched campaign (no optimizer averaging).")
    else:
        print(f"  mixed: rho(A,B)={gate['rho_AB']:.3f} rho(B,C)={gate['rho_BC']:.3f} "
              f"rho(A,C)={gate['rho_AC']:.3f} -- inspect the grid above.")

    out = Path(a.out) if a.out else ROOT / "runs" / f"fidelity_{a.model.split('/')[-1]}_{a.dtype}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    passed = (
        gate["rho_AB"] >= a.min_rho_ab
        and gate["rho_BC"] >= a.min_rho_bc
        and gate["rho_AC"] >= a.min_rho_ac
        and gate["eff_over_eta"] >= a.min_eff_ratio
        and gate["frac_changed"] >= a.min_frac_changed
    )
    certificate = {
        "schema": "fd-fidelity-certificate-v1",
        "passed": passed,
        "model": a.model,
        "dtype": a.dtype,
        "author": a.author,
        "candidate_authors": candidate_authors,
        "candidate_seed": a.candidate_seed,
        "n_candidates": n,
        "block_last_n": a.block_last_n,
        "R": a.gate_r,
        "eta": a.gate_eta,
        "probe_seed": a.gate_seed,
        "metrics": gate,
        "thresholds": {
            "rho_AB": a.min_rho_ab,
            "rho_BC": a.min_rho_bc,
            "rho_AC": a.min_rho_ac,
            "eff_over_eta": a.min_eff_ratio,
            "frac_changed": a.min_frac_changed,
        },
    }
    cert_path = Path(a.certificate) if a.certificate else out.with_suffix(".certificate.json")
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    cert_path.write_text(json.dumps(certificate, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {cert_path}; passed={passed}")
    if a.enforce_gate and not passed:
        raise SystemExit("frozen fd_norm fidelity cell failed; do not start sealed audit")


if __name__ == "__main__":
    main()
