"""G1 channel-separation report (scoring-free; reuses a completed gate run).

Reads sealed predictor scores and saved per-objective damage from a runs/gate_*
directory and emits the channel-conditioned analysis: the objective x predictor
matrix (rho / AUROC / Overlap@K / tail-rho, NO averaging over objectives) plus
the objective x family INTERACTION delta with a candidate-bootstrap CI. Channels
are DECLARED from objective definitions (channels.DECLARED_CHANNEL).

NOTE: if the run scored fd_norm at the alignment eta (3e-4) it is cancellation-
inflated; re-score fd_norm at --probe-norm-eta 3e-3 before trusting its row.

  python experiments/diag/channel_report.py --run-dir runs/gate_Qwen2.5-7B-Instruct_...
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.analysis.channels import (  # noqa: E402
    DECLARED_CHANNEL,
    HEADLINE_PROBE,
    PREDICTOR_FAMILY,
    bootstrap_interaction,
    cell_metrics,
    interaction_delta,
)

# reuse the D1 loaders
sys.path.insert(0, str(ROOT / "experiments" / "diag"))
from probe_agreement import load_damage, load_seals  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--request", default=None)
    p.add_argument("--k", type=int, default=0, help="Overlap@k; 0 -> max(5, round(0.1*n))")
    p.add_argument("--cvar-frac", type=float, default=0.05)
    p.add_argument("--grad-obj", default="graddiff")
    p.add_argument("--rep-obj", default="rmu")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    run_dir = Path(a.run_dir)

    request, scores = load_seals(run_dir, a.request)
    common = set.intersection(*(set(v) for v in scores.values()))
    scores = {pr: {c: s[c] for c in common} for pr, s in scores.items()}
    n = len(common)
    k = a.k or max(5, round(0.1 * n))
    dmg = load_damage(run_dir, common)
    objs = sorted(dmg)
    preds = sorted(scores)
    print(f"request={request} |audit|={n} k={k} objectives={objs}")
    print(f"declared channels: " + ", ".join(f"{o}->{DECLARED_CHANNEL.get(o,'?')}" for o in objs))

    # per (predictor, objective) cells; rho[pred][obj] for the interaction
    rho: dict[str, dict[str, float]] = {}
    rows = []
    for pr in preds:
        rho[pr] = {}
        for o in objs:
            m = cell_metrics(scores[pr], dmg[o], k, a.cvar_frac)
            rho[pr][o] = m["rho"]
            rows.append({"predictor": pr, "family": PREDICTOR_FAMILY.get(pr, "other"),
                         "objective": o, "channel": DECLARED_CHANNEL.get(o, "?"), **m})

    print("\n=== objective x predictor  (Spearman rho; families grouped) ===")
    order = (["grad_norm", "fd_norm"] + ["knn_feature", "knn_embed", "knn_lexical"]
             + ["fd", "one_sided", "last_layer", "random_rank", "random_dir"])
    shown = [pr for pr in order if pr in preds] + [pr for pr in preds if pr not in order]
    print("predictor".ljust(13) + "family".ljust(15) + "".join(o[:11].rjust(12) for o in objs))
    for pr in shown:
        print(pr.ljust(13) + PREDICTOR_FAMILY.get(pr, "other").ljust(15)
              + "".join(f"{rho[pr][o]:12.3f}" for o in objs))

    # interaction: headline probes and family-mean
    gp, rp = HEADLINE_PROBE["gradient"], HEADLINE_PROBE["representation"]
    out = {"request": request, "k": k, "cvar_frac": a.cvar_frac,
           "grad_obj": a.grad_obj, "rep_obj": a.rep_obj, "cells": rows}
    if all(x in rho for x in (gp, rp)) and all(o in objs for o in (a.grad_obj, a.rep_obj)):
        d_head = interaction_delta(rho, a.grad_obj, a.rep_obj, gp, rp)
        boot = bootstrap_interaction(scores, dmg, a.grad_obj, a.rep_obj, gp, rp, n_boot=a.n_boot)
        out["interaction_headline"] = {"grad_probe": gp, "rep_probe": rp, "delta": d_head, **boot}
        print(f"\n=== INTERACTION (channel matching) ===")
        print(f"  headline {gp} vs {rp} on {a.grad_obj}(loss-grad) vs {a.rep_obj}(representation):")
        print(f"    delta = {d_head:+.3f}   95% CI [{boot['lo']:+.3f}, {boot['hi']:+.3f}]"
              f"   (n={boot['n_cands']}, {boot['n_boot']} boot)")
        verdict = ("SUPPORTS channel matching (CI excludes 0)" if boot["lo"] > 0 else
                   "inconclusive at one request (CI includes 0) -> replicate over authors")
        print(f"    -> {verdict}")

    with open(run_dir / "channel_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    with open(run_dir / "channel_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {run_dir/'channel_report.json'} and channel_report.csv")


if __name__ == "__main__":
    main()
