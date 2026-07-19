"""CPU pilot: fill miniature versions of Table 1 (prediction) and Table 2
(protection) on the controlled substrate with a tiny double-precision model.

Purpose: a cheap forecast of whether the main tables can succeed — does the
FD susceptibility profile out-predict similarity baselines on damage realized
by independent optimizers (T1), and does profile-guided protection improve
native damage at a matched forget criterion (T2)? Decoy candidates (surface-
similar, different answers) are included so similarity and update-conditioned
alignment genuinely disagree.

Run:  python experiments/pilot/pilot_t1t2.py
Writes runs/pilot_t1t2/{table1.csv,table2.csv} and prints both tables.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from rsus.blocks import mlp_down_last_layers  # noqa: E402
from rsus.data.base import collate  # noqa: E402
from rsus.data.substrate import make_substrate  # noqa: E402
from rsus.evalx.protection import evaluate_protection  # noqa: E402
from rsus.generators import TrajectoryConfig, run_trajectory  # noqa: E402
from rsus.generators.ours import OursConfig, run_ours_trajectory  # noqa: E402
from rsus.losses import seq_mean_answer_nll  # noqa: E402
from rsus.partition import PartitionParams, build_partition, make_folds  # noqa: E402
from rsus.probe.base import ProbeSpec, get_scorer  # noqa: E402
from rsus.probe.baselines import set_embed_encoder  # noqa: E402
from rsus.analysis.prediction import table1_rows  # noqa: E402
from rsus.stage1 import Stage1Config, calibrate_floor  # noqa: E402
from rsus.stage2 import Stage2Config  # noqa: E402

VOCAB = 128
SEEDS = [41, 42, 43]
PREDICTORS = ["fd", "knn_feature", "knn_embed", "knn_lexical", "grad_norm", "random_dir", "random_rank"]
GENERATORS = ["npo", "graddiff", "rmu"]
T2_METHODS = ["ga", "graddiff", "npo", "npo_transplant", "ours"]
RECALL_MAX = 0.10


def build_tiny(seed: int):
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
        pad_token_id=0,
    )
    return LlamaForCausalLM(cfg).double().eval()


def clone_from(state, seed: int):
    m = build_tiny(seed)
    m.load_state_dict(state)
    return m


def memorize(model, examples, target_loss=0.7, max_steps=400, lr=5e-3):
    """Memorize to a moderate loss, not to zero: fully saturated candidates
    have vanishing gradients, which is neither realistic (7B retained
    behavior sits at natural LM loss levels) nor probe-friendly."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    batch = collate(examples)
    for _ in range(max_steps):
        loss = seq_mean_answer_nll(model, batch).mean()
        if float(loss.detach()) <= target_loss:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.zero_grad(set_to_none=True)


def bag_encoder(examples):
    out = torch.zeros(len(examples), VOCAB)
    for i, e in enumerate(examples):
        for t in e.input_ids.tolist():
            out[i, t] += 1.0
    return out


def main():
    set_embed_encoder(bag_encoder)
    out_dir = ROOT / "runs" / "pilot_t1t2"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_by_pred: dict[str, dict[str, dict[str, float]]] = {p: {} for p in PREDICTORS}
    damage_by_opt: dict[str, dict[str, dict[str, float]]] = {g: {} for g in GENERATORS}
    t2_rows: dict[str, list] = {m: [] for m in T2_METHODS}
    audit_sizes = []

    for seed in SEEDS:
        req, truth = make_substrate(
            seed=seed, n_forget=4, n_adjacent=8, n_remote=10, n_decoy=6,
            answer_overlap=0.5,
        )
        rid = req.request_id
        base = build_tiny(seed)                      # pre-injection reference
        model0 = build_tiny(seed)
        memorize(model0, list(req.forget) + list(req.universe.examples))
        state0 = {k: v.clone() for k, v in model0.state_dict().items()}

        folds = make_folds({e.example_id: e.group for e in req.universe.examples}, 0.5, seed)
        audit_ids = {
            e.example_id for e in req.universe.examples if folds[e.group] == "audit"
        }
        audit_sizes.append(len(audit_ids))
        by_id = {e.example_id: e for e in req.universe.examples}
        disc_remote = [
            by_id[c] for c in sorted(by_id)
            if folds[by_id[c].group] == "discovery" and truth[c] in ("remote", "decoy")
        ]

        # --- T1: sealed scores + independent trajectories -------------------
        spec = ProbeSpec(block=mlp_down_last_layers(model0, 1), eta=1e-4, batch_size=8)
        for pred in PREDICTORS:
            prof = get_scorer(pred)(clone_from(state0, seed), req, spec)
            scores_by_pred[pred][rid] = {c: prof.scores[c] for c in audit_ids}
        for gen_name in GENERATORS:
            gmodel = clone_from(state0, seed)
            rec = run_trajectory(
                gmodel, gen_name, req, disc_remote,
                TrajectoryConfig(max_steps=30, checkpoint_every=5, lr=5e-4, seed=seed),
                out_dir=out_dir / f"traj_{rid}_{gen_name}",
            )
            damage_by_opt[gen_name][rid] = {
                c: v for c, v in rec.damage_at().items() if c in audit_ids
            }
            term = rec.terminal()
            rem_dmg = [term.nll[c] - rec.nll0[c] for c in audit_ids if truth[c] == "remote"]
            print(
                f"  [{gen_name}] terminal recall={term.forget_recall:.2f} "
                f"remote_dmg={sum(rem_dmg)/len(rem_dmg):+.3f}"
            )

        # --- T2: protection comparison --------------------------------------
        fd_prof = get_scorer("fd")(clone_from(state0, seed), req, spec)
        part = build_partition(
            fd_prof, req, folds,
            PartitionParams(pool_size=4, min_pool_size=3, tau_rem_abs_quantile=0.8, seed=seed),
        )
        protect = [by_id[c] for c in part.protect]
        remote_stream = [by_id[c] for c in part.remote_stream]
        native_audit = set(req.native_audit_ids) & audit_ids
        utility_audit = {c for c in audit_ids if truth[c] == "remote"}

        floor_m = calibrate_floor(base, req)
        tcfg = TrajectoryConfig(max_steps=80, checkpoint_every=10, lr=1e-3, seed=seed)
        for method in T2_METHODS:
            mmodel = clone_from(state0, seed)
            if method == "ours":
                ocfg = OursConfig(
                    stage1=Stage1Config(lr=2e-3, max_steps=800, eval_every=20, seed=seed),
                    stage2=Stage2Config(max_steps=80, refresh_k=1, delta_seq_sq=1e-2, delta_tok_sq=1e-1),
                    stage2_snapshots=4,
                )
                rec = run_ours_trajectory(
                    mmodel, mlp_down_last_layers(mmodel, 1), req, protect,
                    remote_stream, floor_m, ocfg,
                )
            elif method == "npo_transplant":
                rec = run_trajectory(mmodel, "npo", req, protect, tcfg)
            else:
                rec = run_trajectory(mmodel, method, req, disc_remote, tcfg)
            t2_rows[method].append(
                evaluate_protection(
                    rec, native_audit, utility_audit, RECALL_MAX,
                    mode="last" if method == "ours" else "first",
                )
            )
        print(f"[request {rid}] done")

    # --- Table 1 -------------------------------------------------------------
    k = max(3, round(0.25 * (sum(audit_sizes) / len(audit_sizes))))
    rows = table1_rows(scores_by_pred, damage_by_opt, k=k)
    t1_path = out_dir / "table1.csv"
    with open(t1_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["predictor"] + [f"{g}_rho" for g in GENERATORS] + ["auroc", "overlap@k", "tail_rho"])
        print("\n=== Mini Table 1: prediction (terminal checkpoint, audit fold) ===")
        hdr = f"{'predictor':18s}" + "".join(f"{g+'_rho':>13s}" for g in GENERATORS) + f"{'AUROC':>9s}{'Ovl@'+str(k):>9s}{'tail':>7s}"
        print(hdr)
        for pred in PREDICTORS:
            r = rows[pred]
            vals = [r[f"{g}_rho"]["mean"] for g in GENERATORS]
            tail = r["tail_rho"]["mean"] if r["tail_rho"] else float("nan")
            w.writerow([pred] + [f"{v:.3f}" for v in vals] + [f"{r['auroc']['mean']:.3f}", f"{r['overlap']['mean']:.3f}", f"{tail:.3f}"])
            print(
                f"{pred:18s}" + "".join(f"{v:13.3f}" for v in vals)
                + f"{r['auroc']['mean']:9.3f}{r['overlap']['mean']:9.3f}{tail:7.2f}"
            )

    # --- Table 2 -------------------------------------------------------------
    t2_path = out_dir / "table2.csv"
    with open(t2_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "reach", "step_med", "native_mean_dNLL", "native_cvar", "utility_ret"])
        print("\n=== Mini Table 2: protection at forget recall <= 0.10 (native = untouched ground-truth adjacents) ===")
        print(f"{'method':16s}{'reach':>7s}{'step':>7s}{'nat.mean':>10s}{'nat.CVaR':>10s}{'util.ret':>10s}")
        for method in T2_METHODS:
            outs = t2_rows[method]
            reached = [o for o in outs if o.reached]
            reach = f"{len(reached)}/{len(outs)}"
            if reached:
                med = sorted(o.step for o in reached)[len(reached) // 2]
                nm = sum(o.native_mean for o in reached) / len(reached)
                nc = sum(o.native_cvar for o in reached) / len(reached)
                ur = sum(o.utility_ret for o in reached) / len(reached)
                w.writerow([method, reach, med, f"{nm:.3f}", f"{nc:.3f}", f"{ur:.3f}"])
                print(f"{method:16s}{reach:>7s}{med:7d}{nm:10.3f}{nc:10.3f}{ur:10.3f}")
            else:
                w.writerow([method, reach, "", "", "", ""])
                print(f"{method:16s}{reach:>7s}{'--':>7s}{'--':>10s}{'--':>10s}{'--':>10s}")

    print(f"\nwrote {t1_path} and {t2_path}")


if __name__ == "__main__":
    main()
