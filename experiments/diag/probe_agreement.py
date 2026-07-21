"""D1 probe-agreement diagnostic (scoring-free; reuses a completed gate run).

Central question: is fd_norm a faithful backward-free estimate of the exact
grad_norm it approximates, or is it measuring something else? The 7B gate
showed fd_norm (approx) BEATING grad_norm (exact) on average and disagreeing in
sign on rmu -- which is either K-noise, a bug, or a genuinely different quantity.
This tool reads the sealed audit-fold predictor scores and the saved generator
damage from an existing runs/gate_* directory and computes:

  1. the predictor x predictor rank-agreement matrix (Spearman rho, Overlap@k),
     with rho(fd_norm, grad_norm) as the headline;
  2. each predictor's rho vs each optimizer's realized damage, reproduced from
     the saved trajectories (a sanity check against table1.json).

No model, no GPU, no re-unlearning -- pure file reads, runs in seconds.

  python experiments/diag/probe_agreement.py --run-dir runs/gate_Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.analysis.prediction import per_cell_metrics, rank_agreement_matrix  # noqa: E402


def load_seals(run_dir: Path, request: str | None) -> tuple[str, dict[str, dict[str, float]]]:
    seals = run_dir / "seals"
    if not seals.is_dir():
        sys.exit(f"no seals/ under {run_dir}")
    reqs = sorted(p.name for p in seals.iterdir() if p.is_dir())
    if request is None:
        if len(reqs) != 1:
            sys.exit(f"seals/ holds requests {reqs}; pass --request")
        request = reqs[0]
    scores: dict[str, dict[str, float]] = {}
    for f in sorted((seals / request).glob("*.json")):
        scores[f.stem] = {k: float(v) for k, v in json.loads(f.read_text()).items()}
    if not scores:
        sys.exit(f"no sealed scores for request {request}")
    return request, scores


def load_damage(run_dir: Path, audit_ids: set[str]) -> dict[str, dict[str, float]]:
    dmg: dict[str, dict[str, float]] = {}
    for traj in sorted(run_dir.glob("traj_*")):
        dj = traj / "damage.json"
        if not dj.exists():
            continue
        payload = json.loads(dj.read_text())
        nll0 = payload["nll0"]
        term = payload["snapshots"][-1]["nll"]
        opt = payload.get("objective", traj.name[len("traj_"):])
        dmg[opt] = {c: term[c] - nll0[c] for c in audit_ids if c in term and c in nll0}
    return dmg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--request", default=None, help="request id under seals/ (auto if unique)")
    p.add_argument("--k", type=int, default=0, help="Overlap@k; 0 -> max(5, round(0.1*n))")
    a = p.parse_args()
    run_dir = Path(a.run_dir)

    request, scores = load_seals(run_dir, a.request)
    common = set.intersection(*(set(v) for v in scores.values()))
    scores = {pr: {c: s[c] for c in common} for pr, s in scores.items()}
    n = len(common)
    k = a.k or max(5, round(0.1 * n))
    preds = sorted(scores)
    print(f"request={request}  |audit|={n}  k={k}  predictors={preds}")

    agree = rank_agreement_matrix(scores, k)
    print("\n=== predictor x predictor rank agreement (Spearman rho) ===")
    print("".ljust(14) + "".join(pp[:12].rjust(13) for pp in preds))
    for a_ in preds:
        print(a_.ljust(14) + "".join(f"{agree[a_][b_]['rho']:13.3f}" for b_ in preds))
    if "fd_norm" in preds and "grad_norm" in preds:
        c = agree["fd_norm"]["grad_norm"]
        print(f"\n>>> D1 HEADLINE  rho(fd_norm, grad_norm) = {c['rho']:.3f}   "
              f"Overlap@{k}(fd_norm, grad_norm) = {c['overlap']:.3f}")
        verdict = ("FAITHFUL (fd_norm ~ grad_norm; disagreement w/ damage is real signal, not the estimator)"
                   if c["rho"] >= 0.8 else
                   "PARTIAL (bump K / check eta; fd_norm only loosely tracks grad_norm)"
                   if c["rho"] >= 0.5 else
                   "BROKEN (fd_norm does NOT track grad_norm -> K too small, wrong eta, or a bug)")
        print(f"    interpretation: {verdict}")

    dmg = load_damage(run_dir, common)
    if dmg:
        opts = sorted(dmg)
        print("\n=== predictor vs realized damage (from saved trajectories) ===")
        print("predictor".ljust(14) + "".join((o + "_rho")[:12].rjust(13) for o in opts))
        rows = []
        for pr in preds:
            cells = {}
            for o in opts:
                ids = sorted(set(dmg[o]) & set(scores[pr]))
                cells[o] = per_cell_metrics(
                    {c: scores[pr][c] for c in ids}, {c: dmg[o][c] for c in ids}, k
                )["rho"]
            rows.append((pr, cells))
            print(pr.ljust(14) + "".join(f"{cells[o]:13.3f}" for o in opts))

    out = run_dir / "diag_agreement.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pred_a", "pred_b", "rho", "overlap"])
        for a_ in preds:
            for b_ in preds:
                w.writerow([a_, b_, agree[a_][b_]["rho"], agree[a_][b_]["overlap"]])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
