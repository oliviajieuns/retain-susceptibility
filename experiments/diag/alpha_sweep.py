"""Channel-mixture score alpha sweep (offline; NO GPU, no new runs).

s_alpha(x) = alpha * rank(grad_probe) + (1 - alpha) * rank(rep_probe),
rank-normalized to [0, 1] per probe. Reads sealed per-candidate scores and
saved per-objective damage from a completed gate run, and reports
rho(s_alpha, damage) per objective across the alpha grid -- the continuous
generalization of the discrete channel router (alpha in {0,1} recovers the
two headline probes exactly).

  python experiments/diag/alpha_sweep.py --run-dir runs/gate_Qwen2.5-1.5B-Instruct_chanbal2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "diag"))

from rsus.analysis.channels import DECLARED_CHANNEL, HEADLINE_PROBE  # noqa: E402
from rsus.analysis.prediction import spearman  # noqa: E402
from probe_agreement import load_damage, load_seals  # noqa: E402


def rank01(scores: dict[str, float]) -> dict[str, float]:
    """Rank-normalize to [0,1] (ties broken by candidate id for determinism)."""
    order = sorted(scores, key=lambda c: (scores[c], c))
    n = len(order)
    return {c: (i / (n - 1) if n > 1 else 0.5) for i, c in enumerate(order)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--request", default=None)
    p.add_argument("--grad-probe", default=HEADLINE_PROBE["gradient"])
    p.add_argument("--rep-probe", default=HEADLINE_PROBE["representation"])
    p.add_argument("--alphas", default="0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1")
    a = p.parse_args()
    run_dir = Path(a.run_dir)
    alphas = [float(x) for x in a.alphas.split(",") if x.strip()]

    request, scores = load_seals(run_dir, a.request)
    for probe in (a.grad_probe, a.rep_probe):
        if probe not in scores:
            raise SystemExit(f"probe {probe!r} not in seals ({sorted(scores)})")
    common = sorted(set(scores[a.grad_probe]) & set(scores[a.rep_probe]))
    rg = rank01({c: scores[a.grad_probe][c] for c in common})
    rp = rank01({c: scores[a.rep_probe][c] for c in common})
    dmg = load_damage(run_dir, set(common))
    rank = {"loss_gradient": 0, "representation": 1}
    objs = sorted(dmg, key=lambda o: (rank.get(DECLARED_CHANNEL.get(o, "?"), 2), o))

    print(f"request={request} n={len(common)} grad_probe={a.grad_probe} rep_probe={a.rep_probe}")
    print("s_alpha = alpha*rank(grad) + (1-alpha)*rank(rep); cells = spearman rho vs damage\n")
    header = "alpha".ljust(8) + "".join(o[:11].rjust(12) for o in objs)
    print(header)
    rows = []
    curves: dict[str, list[float]] = {o: [] for o in objs}
    for al in alphas:
        s = [al * rg[c] + (1 - al) * rp[c] for c in common]
        line = f"{al:<8.3f}"
        for o in objs:
            r = spearman(s, [dmg[o][c] for c in common])
            curves[o].append(r)
            rows.append({"alpha": al, "objective": o,
                         "channel": DECLARED_CHANNEL.get(o, "?"), "rho": r})
            line += f"{r:12.3f}"
        print(line)

    print("\nper-objective argmax alpha (prediction-optimal mixture):")
    for o in objs:
        best = max(range(len(alphas)), key=lambda i: curves[o][i])
        print(f"  {o:18s} [{DECLARED_CHANNEL.get(o, '?'):15s}] "
              f"alpha*={alphas[best]:.3f}  rho={curves[o][best]:.3f}")

    out = run_dir / "alpha_sweep.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
