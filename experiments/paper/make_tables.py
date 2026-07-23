"""LEGACY diagnostic renderer for the superseded channel-interaction draft.

This file does not produce the current paper's claim-bearing tables. Use
``aggregate_raw.py`` followed by ``build_evidence.py`` for the authoritative
prediction/protection IUT and fixed-denominator pipeline. The explicit
``--legacy-diagnostic`` switch prevents an old CSV from being mistaken for a
current-paper result.

Table 1  channel x predictor-family matrix (from channel_report.csv):
         rows = predictors grouped by family, cols = objectives grouped by
         declared channel + per-channel family means; caption carries the
         preregistered interaction delta (from channel_report.json).
Table 1b headline probes only, secondary metrics (AUROC / Overlap@K / tail rho).
Table 2  crossed protection (from crossed.json): parent x selector, mean/CVaR
         audit dNLL at matched forgetting; matched-channel rows flagged.

  python experiments/paper/make_tables.py \
      --report runs/gate_.../channel_report.csv --crossed runs/xprot_.../crossed.json \
      --out docs/tables
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.analysis.channels import DECLARED_CHANNEL, HEADLINE_PROBE, PREDICTOR_FAMILY  # noqa: E402

OBJ_TEX = {"ga": "GA", "graddiff": "GradDiff", "npo": "NPO", "simnpo": "SimNPO",
           "idkdpo": "IdkDPO", "gru": "GRU", "rmu": "RMU", "repnoise": "RepNoise",
           "circuit_breakers": "CB"}
PRED_TEX = {"grad_norm": "Exact gradient norm", "fd_norm": r"Randomized FD norm (\textbf{ours})",
            "knn_feature": "Hidden-state kNN", "knn_embed": "Sentence-embedding kNN",
            "knn_lexical": "Lexical overlap", "fd": "Forget-direction FD",
            "one_sided": "One-sided FD", "last_layer": "Last-layer FD",
            "random_rank": "Random ranking", "random_dir": "One random direction"}
# per-candidate backward pass required?
NEEDS_BACKWARD = {"grad_norm"}
PRED_ORDER = ["grad_norm", "fd_norm", "knn_feature", "knn_embed", "knn_lexical",
              "fd", "one_sided", "last_layer", "random_rank", "random_dir"]
FAMILY_ORDER = ["gradient", "representation", "alignment", "control"]
FAMILY_TEX = {"gradient": "Gradient magnitude", "representation": "Representation proximity",
              "alignment": "Alignment (rejected)", "control": "Controls"}


def read_report(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def order_objectives(objs: set[str]) -> list[str]:
    rank = {"loss_gradient": 0, "representation": 1}
    return sorted(objs, key=lambda o: (rank.get(DECLARED_CHANNEL.get(o, "?"), 2), o))


def fmt(v: float | None, bold: bool = False) -> str:
    if v is None:
        return "--"
    s = f"{v:.2f}"
    return rf"\textbf{{{s}}}" if bold else s


# objective status flags -> column-header markers + caption note.
FLAG_MARK = {"notreached": r"$^{\dag}$", "collapsed": r"$^{\ddag}$"}
FLAG_NOTE = {
    "notreached": r"$^{\dag}$did not reach the preregistered forget criterion "
                  "within budget (weak-signal column)",
    "collapsed": r"$^{\ddag}$reached the criterion but with model-wide utility "
                 "collapse",
}


def table1(rows: list[dict], interaction: dict | None,
           flags: dict[str, str] | None = None,
           demote: set[str] | None = None) -> str:
    flags = flags or {}
    demote = demote or set()
    rho: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["predictor"] in demote:
            continue
        rho.setdefault(r["predictor"], {})[r["objective"]] = float(r["rho"])
    objs = order_objectives({r["objective"] for r in rows})
    lg = [o for o in objs if DECLARED_CHANNEL.get(o) == "loss_gradient"]
    rep = [o for o in objs if DECLARED_CHANNEL.get(o) == "representation"]
    preds = sorted(rho, key=lambda p: (FAMILY_ORDER.index(PREDICTOR_FAMILY.get(p, "control"))
                                       if PREDICTOR_FAMILY.get(p, "control") in FAMILY_ORDER else 9,
                                       PRED_ORDER.index(p) if p in PRED_ORDER else 99, p))
    best = {o: max(rho[p].get(o, float("-inf")) for p in preds) for o in objs}

    def chan_mean(p: str, chan_objs: list[str]) -> float | None:
        vals = [rho[p][o] for o in chan_objs if o in rho[p]]
        return sum(vals) / len(vals) if vals else None

    L = [r"\begin{tabular}{llc" + "c" * len(objs) + "cc}", r"\toprule",
         r" & & & \multicolumn{%d}{c}{\textbf{Output / loss-gradient channel}} & "
         r"\multicolumn{%d}{c}{\textbf{Representation channel}} & "
         r"\multicolumn{2}{c}{channel mean $\rho$} \\" % (len(lg), len(rep)),
         r"\cmidrule(lr){4-%d}\cmidrule(lr){%d-%d}\cmidrule(lr){%d-%d}"
         % (3 + len(lg), 4 + len(lg), 3 + len(objs), 4 + len(objs), 5 + len(objs)),
         "Family & Predictor & Bwd-free & "
         + " & ".join(OBJ_TEX.get(o, o) + FLAG_MARK.get(flags.get(o, ""), "")
                      for o in objs) + r" & LG & Rep \\", r"\midrule"]
    prev_fam = None
    for p in preds:
        fam = PREDICTOR_FAMILY.get(p, "control")
        if fam != prev_fam and prev_fam is not None:
            L.append(r"\addlinespace")
        fam_cell = FAMILY_TEX.get(fam, fam) if fam != prev_fam else ""
        prev_fam = fam
        bf = r"\xmark" if p in NEEDS_BACKWARD else r"\cmark"
        cells = " & ".join(fmt(rho[p].get(o), bold=rho[p].get(o) == best[o]) for o in objs)
        L.append(f"{fam_cell} & {PRED_TEX.get(p, p)} & {bf} & {cells}"
                 f" & {fmt(chan_mean(p, lg))} & {fmt(chan_mean(p, rep))} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    cap = ("% caption: Spearman rho(predictor score, realized audit dNLL) per objective, "
           "objectives grouped by DECLARED damage channel. No predictor wins everywhere; "
           "the winner flips with the channel.")
    if interaction:
        cap += ("\n% preregistered interaction delta = "
                f"{interaction['delta']:+.3f}, 95% CI [{interaction['lo']:+.3f}, "
                f"{interaction['hi']:+.3f}] (n={interaction['n_cands']})")
    used_flags = {v for v in flags.values() if v in FLAG_NOTE}
    if used_flags:
        cap += "\n% flag notes: " + "; ".join(FLAG_NOTE[f] for f in sorted(used_flags)) + "."
    return cap + "\n" + "\n".join(L) + "\n"


def table1c_controls(rows: list[dict], demote: set[str], flags: dict[str, str]) -> str:
    """Appendix variant holding the demoted control rows (space fallback)."""
    kept = [r for r in rows if r["predictor"] in demote]
    objs = order_objectives({r["objective"] for r in rows})
    rho: dict[str, dict[str, float]] = {}
    for r in kept:
        rho.setdefault(r["predictor"], {})[r["objective"]] = float(r["rho"])
    L = [r"\begin{tabular}{l" + "c" * len(objs) + "}", r"\toprule",
         "Control & " + " & ".join(OBJ_TEX.get(o, o) + FLAG_MARK.get(flags.get(o, ""), "")
                                   for o in objs) + r" \\", r"\midrule"]
    for p in sorted(rho, key=lambda p: (PRED_ORDER.index(p) if p in PRED_ORDER else 99, p)):
        L.append(PRED_TEX.get(p, p) + " & "
                 + " & ".join(fmt(rho[p].get(o)) for o in objs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return ("% caption: control predictors (demoted from Table 1 for space); "
            "same protocol and columns.\n" + "\n".join(L) + "\n")


def table1b(rows: list[dict]) -> str:
    heads = set(HEADLINE_PROBE.values())
    L = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Probe & Objective (channel) & $\rho$ & AUROC & Overlap@$K$ & tail $\rho$ \\",
         r"\midrule"]
    for r in sorted(rows, key=lambda r: (r["predictor"], DECLARED_CHANNEL.get(r["objective"], "?"))):
        if r["predictor"] not in heads:
            continue
        chan = "LG" if DECLARED_CHANNEL.get(r["objective"]) == "loss_gradient" else "Rep"
        L.append(f"{r['predictor'].replace('_', chr(92) + '_')} & "
                 f"{OBJ_TEX.get(r['objective'], r['objective'])} ({chan}) & "
                 f"{float(r['rho']):.2f} & {float(r['auroc']):.2f} & "
                 f"{float(r['overlap']):.2f} & {float(r['tail_rho']):.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def table2(crossed: dict) -> str:
    res = crossed["results"]
    L = [r"\begin{tabular}{llllcccc}", r"\toprule",
         r"Parent (channel) & Selector & Match & Reach & Forget $\downarrow$ & "
         r"Para & mean dNLL $\downarrow$ & CVaR$_{5\%}$ $\downarrow$ \\", r"\midrule"]
    parents = list(dict.fromkeys(r["parent"] for r in res))
    for parent in parents:
        rows_p = [r for r in res if r["parent"] == parent]
        best_cvar = min(r["cvar_dnll"] for r in rows_p)
        for i, r in enumerate(rows_p):
            pcell = f"{OBJ_TEX.get(parent, parent)} ({'LG' if r['channel']=='loss_gradient' else 'Rep'})" if i == 0 else ""
            match = {"matched": r"\textbf{matched}", "none": "--"}.get(r["match"], r["match"])
            para = "--" if r.get("para_recall") is None else f"{r['para_recall']:.2f}"
            cv = fmt(r["cvar_dnll"], bold=r["cvar_dnll"] == best_cvar)
            sel = r["selector"].replace("_", r"\_")
            L.append(f"{pcell} & {sel} & {match} & "
                     f"{'yes' if r['reached'] else 'NO'} & {r['forget_recall']:.2f} & {para} & "
                     f"{r['mean_dnll']:.3f} & {cv} \\\\")
        if parent != parents[-1]:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    wins = crossed.get("contrasts", {})
    cap = ("% caption: audit collateral damage at matched forgetting; success criterion "
           "(preregistered) = matched CVaR lowest per parent. "
           + " ".join(f"{p}: matched wins={c['matched_beats_mismatched']}" for p, c in wins.items()))
    return cap + "\n" + "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-diagnostic", action="store_true")
    ap.add_argument("--report", default="", help="channel_report.csv from channel_report.py")
    ap.add_argument("--crossed", default="", help="crossed.json from crossed_protection.py")
    ap.add_argument("--out", default=str(ROOT / "docs" / "tables"))
    ap.add_argument("--flags", default="",
                    help="objective status flags, e.g. 'ga=collapsed,idkdpo=notreached,"
                         "circuit_breakers=notreached' -> dag/ddag column markers")
    ap.add_argument("--demote", default="",
                    help="comma-separated predictors moved out of Table 1 into "
                         "table1c_controls.tex (space fallback)")
    a = ap.parse_args()
    if not a.legacy_diagnostic:
        ap.error(
            "legacy draft renderer; use experiments/paper/aggregate_raw.py and "
            "build_evidence.py, or pass --legacy-diagnostic intentionally"
        )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    flags = {k: v for k, v in (kv.split("=") for kv in a.flags.split(",") if kv.strip())}
    bad = {v for v in flags.values() if v not in FLAG_MARK}
    if bad:
        ap.error(f"unknown flag values {sorted(bad)}; use {sorted(FLAG_MARK)}")
    demote = {x.strip() for x in a.demote.split(",") if x.strip()}
    if a.report:
        rows = read_report(Path(a.report))
        jpath = Path(a.report).with_name("channel_report.json")
        inter = None
        if jpath.exists():
            inter = json.loads(jpath.read_text()).get("interaction_headline")
        (out / "table1_channel_matrix.tex").write_text(
            table1(rows, inter, flags, demote), encoding="utf-8")
        (out / "table1b_headline_secondary.tex").write_text(table1b(rows), encoding="utf-8")
        print(f"wrote {out}/table1_channel_matrix.tex and table1b_headline_secondary.tex")
        if demote:
            (out / "table1c_controls.tex").write_text(
                table1c_controls(rows, demote, flags), encoding="utf-8")
            print(f"wrote {out}/table1c_controls.tex ({len(demote)} demoted rows)")
    if a.crossed:
        crossed = json.loads(Path(a.crossed).read_text())
        (out / "table2_crossed_protection.tex").write_text(table2(crossed), encoding="utf-8")
        print(f"wrote {out}/table2_crossed_protection.tex")
    if not a.report and not a.crossed:
        ap.error("pass --report and/or --crossed")


if __name__ == "__main__":
    main()
