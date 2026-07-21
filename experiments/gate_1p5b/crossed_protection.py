"""Crossed protection: does matching the protection selector to the unlearning
objective's DAMAGE CHANNEL reduce collateral damage?

For each parent unlearning method (a channel) x each protection SELECTOR
(a predictor family), run the parent to its criterion-reaching checkpoint,
then guarded-repair the protect partition chosen by that selector, and measure
audit collateral damage at MATCHED forgetting. Selectors include the matched
family, the mismatched family, random, and 'none' (parent alone).

The story is NOT 'we beat SoTA unlearners'. It is: matched-channel protection
reduces mean and (especially) worst-tail collateral damage more than random /
mismatched / no protection, at matched forgetting and comparable utility --
so the channel prediction is ACTIONABLE. Success = the interaction (matched
best per parent), not an absolute leaderboard.

  python experiments/gate_1p5b/crossed_protection.py --smoke
  python experiments/gate_1p5b/crossed_protection.py --model /group-volume/models/Qwen2.5-7B-Instruct \
      --device cuda --dtype float32 --parents graddiff,rmu \
      --selectors fd_norm,knn_feature --run-tag xprot
"""
from __future__ import annotations

import argparse
import dataclasses as _dc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from rsus.analysis.channels import DECLARED_CHANNEL, PREDICTOR_FAMILY  # noqa: E402
from rsus.analysis.prediction import cvar_upper  # noqa: E402
from rsus.blocks import mlp_down_last_layers  # noqa: E402
from rsus.costs import CostRecord  # noqa: E402
from rsus.data.base import collate  # noqa: E402
from rsus.data.tofu import FORGET10_FIRST_AUTHOR, load_tofu_examples, load_tofu_paraphrases, tofu_request  # noqa: E402
from rsus.evalx.metrics import mean_recall  # noqa: E402
from rsus.generators.base import TrajectoryConfig, run_trajectory  # noqa: E402
from rsus.generators.repaired import RepairedConfig, run_engine_repaired  # noqa: E402
from rsus.losses import seq_mean_answer_nll  # noqa: E402
from rsus.partition import PartitionParams, build_partition, make_folds  # noqa: E402
from rsus.probe.base import ProbeSpec, ScoreProfile, get_scorer  # noqa: E402
from rsus.stage1 import Stage1Config, calibrate_floor  # noqa: E402
from rsus.stage2 import Stage2Config  # noqa: E402

# gradient family <-> loss_gradient channel; representation family <-> representation channel.
FAMILY_CHANNEL = {"gradient": "loss_gradient", "representation": "representation"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--author", type=int, default=FORGET10_FIRST_AUTHOR)
    p.add_argument("--universe-authors", type=int, default=30)
    p.add_argument("--block-last-n", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eta", type=float, default=3e-4)
    p.add_argument("--probe-norm-eta", type=float, default=3e-3)
    p.add_argument("--probe-dirs", type=int, default=64)
    p.add_argument("--sft-lr", type=float, default=1e-5)
    p.add_argument("--sft-steps", type=int, default=400)
    p.add_argument("--sft-target-loss", type=float, default=0.8)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--parents", default="graddiff,rmu")
    p.add_argument("--selectors", default="fd_norm,knn_feature")
    p.add_argument("--gen-steps", type=int, default=60)
    p.add_argument("--gen-steps-per", default="")
    p.add_argument("--gen-lr", type=float, default=4e-6)
    p.add_argument("--gen-lr-per", default="")
    p.add_argument("--recall-max", type=float, default=0.10)
    p.add_argument("--pool-size", type=int, default=32)
    p.add_argument("--s2-eta2", type=float, default=3e-5)
    p.add_argument("--s2-steps", type=int, default=120)
    p.add_argument("--s2-delta-seq", type=float, default=1e-2)
    p.add_argument("--s2-delta-tok", type=float, default=1e-1)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run-tag", default="")
    return p.parse_args()


def apply_smoke(a):
    a.universe_authors = 4; a.block_last_n = 1; a.sft_steps = 60; a.sft_lr = 5e-3
    a.sft_target_loss = 1.2; a.gen_lr = 1e-3; a.gen_steps = 8; a.probe_dirs = 8
    a.s2_steps = 8; a.pool_size = 6; a.batch_size = 4; a.recall_max = 1.0


def load_model(a, tokenizer):
    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
    if a.smoke:
        torch.manual_seed(a.seed)
        cfg = Qwen2Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                          max_position_embeddings=512, pad_token_id=tokenizer.pad_token_id)
        return Qwen2ForCausalLM(cfg).to(dtype).eval()
    m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype)
    return m.to(a.device).eval()


def sft(model, examples, a, log):
    opt = torch.optim.AdamW(model.parameters(), lr=a.sft_lr, foreach=False)
    gen = torch.Generator().manual_seed(a.seed)
    for step in range(1, a.sft_steps + 1):
        idx = torch.randperm(len(examples), generator=gen)[: a.batch_size]
        batch = collate([examples[i] for i in idx.tolist()])
        loss = seq_mean_answer_nll(model, batch).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 20 == 0 and float(loss.detach()) <= a.sft_target_loss:
            break
    model.zero_grad(set_to_none=True)


def audit_metrics(rec, audit_ids, recall_max):
    snap = rec.snapshots[-1]
    dmg = [snap.nll[c] - rec.nll0[c] for c in sorted(audit_ids)]
    return {
        "reached": bool(snap.forget_recall <= recall_max),
        "forget_recall": float(snap.forget_recall),
        "mean_dnll": sum(dmg) / len(dmg),
        "cvar_dnll": cvar_upper(dmg, 0.05),
        "para_recall": (snap.extra or {}).get("para_recall"),
        "step": snap.step,
    }


def crossed_sweep(fresh, block, req, by_id, folds, audit_ids, retain_matched, floor_m,
                  parents, selectors, sel_scores, make_gcfg, s2, recall_max, pool_size,
                  seed, extra_eval=None, log=print):
    """Core (parent x selector) protection sweep. Returns {results, contrasts}.
    fresh() -> a model at the SFT'd state; make_gcfg(parent) -> TrajectoryConfig."""
    results = []
    for parent in parents:
        gcfg = make_gcfg(parent)
        pchan = DECLARED_CHANNEL.get(parent, "?")

        rec = run_trajectory(fresh(), parent, req, retain_matched, gcfg,
                             extra_eval=extra_eval, stop_at_recall=recall_max)
        m = audit_metrics(rec, audit_ids, recall_max)
        results.append({"parent": parent, "channel": pchan, "selector": "none", "match": "none", **m})
        log(f"{parent}[{pchan}] / none        reach={m['reached']} "
            f"mean={m['mean_dnll']:.3f} CVaR={m['cvar_dnll']:.3f}")

        for sel in list(selectors) + ["random"]:
            prof = ScoreProfile(req.request_id, sel, sel_scores[sel], None, CostRecord())
            try:
                part = build_partition(prof, req, folds,
                                       PartitionParams(pool_size=pool_size, min_pool_size=4,
                                                       tau_rem_abs_quantile=0.6, seed=seed))
            except Exception as e:  # noqa: BLE001
                log(f"{parent} / {sel}: partition failed ({type(e).__name__}: {e})")
                continue
            protect = [by_id[c] for c in part.protect]
            remote = [by_id[c] for c in part.remote_stream]
            rcfg = RepairedConfig(engine_cfg=gcfg, stage2=s2, recall_max=recall_max,
                                  batch_size=gcfg.batch_size, stage2_snapshots=4)
            rec = run_engine_repaired(fresh(), block, req, retain_matched, protect, remote,
                                      floor_m, parent, rcfg, extra_eval=extra_eval)
            m = audit_metrics(rec, audit_ids, recall_max)
            fam = PREDICTOR_FAMILY.get(sel, "control")
            match = ("random" if sel == "random"
                     else "matched" if FAMILY_CHANNEL.get(fam) == pchan else "mismatched")
            results.append({"parent": parent, "channel": pchan, "selector": sel, "match": match, **m})
            log(f"{parent}[{pchan}] / {sel:12s}[{match:10s}] reach={m['reached']} "
                f"mean={m['mean_dnll']:.3f} CVaR={m['cvar_dnll']:.3f}")

    log("\n=== crossed protection (audit collateral dNLL; lower = better) ===")
    contrasts = {}
    for parent in parents:
        rp = {r["match"]: r for r in results if r["parent"] == parent}
        if "matched" in rp and "mismatched" in rp:
            contrasts[parent] = {
                "cvar_matched": rp["matched"]["cvar_dnll"],
                "cvar_mismatched": rp["mismatched"]["cvar_dnll"],
                "cvar_none": rp.get("none", {}).get("cvar_dnll"),
                "cvar_random": rp.get("random", {}).get("cvar_dnll"),
                "matched_beats_mismatched": rp["matched"]["cvar_dnll"] < rp["mismatched"]["cvar_dnll"],
            }
            log(f"  {parent}: CVaR matched={rp['matched']['cvar_dnll']:.3f} "
                f"mismatched={rp['mismatched']['cvar_dnll']:.3f} "
                f"none={rp.get('none',{}).get('cvar_dnll','-')} "
                f"-> matched wins: {contrasts[parent]['matched_beats_mismatched']}")
    return {"results": results, "contrasts": contrasts}


def main():
    a = parse_args()
    if a.smoke:
        apply_smoke(a)
    tag = ("smoke" if a.smoke else a.model.split("/")[-1]) + (f"_{a.run_tag}" if a.run_tag else "")
    out = ROOT / "runs" / f"xprot_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    def log(m):
        print(m, flush=True)
        (out / "xprot.log").open("a").write(m + "\n")

    torch.manual_seed(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    examples = load_tofu_examples(tokenizer)
    req = tofu_request(a.author, examples, universe_authors=a.universe_authors, seed=a.seed)
    by_id = {e.example_id: e for e in req.universe.examples}
    base = load_model(a, tokenizer)
    model0 = load_model(a, tokenizer)
    sft(model0, list(req.forget) + list(req.universe.examples), a, log)
    state0 = {k: v.clone() for k, v in model0.state_dict().items()}

    def fresh():
        m = load_model(a, tokenizer); m.load_state_dict(state0); return m

    block = mlp_down_last_layers(model0, a.block_last_n)
    spec = ProbeSpec(block=block, eta=a.eta, batch_size=a.batch_size,
                     n_dirs=a.probe_dirs, norm_eta=a.probe_norm_eta)
    folds = make_folds({e.example_id: e.group for e in req.universe.examples}, 0.5, a.seed)
    audit_ids = {e.example_id for e in req.universe.examples if folds[e.group] == "audit"}
    disc_ids = sorted(set(by_id) - audit_ids)
    floor_m = calibrate_floor(base, req, a.batch_size)

    parents = [x.strip() for x in a.parents.split(",") if x.strip()]
    selectors = [x.strip() for x in a.selectors.split(",") if x.strip()]
    steps_per = {k: int(v) for k, v in (kv.split("=") for kv in a.gen_steps_per.split(",") if kv.strip())}
    lr_per = {k: float(v) for k, v in (kv.split("=") for kv in a.gen_lr_per.split(",") if kv.strip())}

    # score every selector once, over the full universe (partition uses discovery fold)
    log("scoring selectors ...")
    sel_scores = {s: get_scorer(s)(fresh(), req, spec).scores for s in selectors}
    sel_scores["random"] = get_scorer("random_rank")(fresh(), req, spec).scores

    retain_matched = [by_id[disc_ids[i]] for i in
                      torch.randperm(len(disc_ids), generator=torch.Generator().manual_seed(a.seed))
                      .tolist()[: a.pool_size]]
    try:
        paras = load_tofu_paraphrases(tokenizer)
        para_ex = [paras[e.example_id] for e in req.forget if e.example_id in paras]
        extra_eval = (lambda m: {"para_recall": mean_recall(m, para_ex, a.batch_size)}) if para_ex else None
    except Exception:
        extra_eval = None

    s2 = Stage2Config(max_steps=a.s2_steps, refresh_k=4, delta_seq_sq=a.s2_delta_seq,
                      delta_tok_sq=a.s2_delta_tok, batch_size=a.batch_size,
                      **({"eta2": a.s2_eta2} if a.s2_eta2 else {}))

    def make_gcfg(parent):
        return TrajectoryConfig(max_steps=steps_per.get(parent, a.gen_steps),
                                checkpoint_every=max(1, steps_per.get(parent, a.gen_steps) // 6),
                                lr=lr_per.get(parent, a.gen_lr), batch_size=a.batch_size,
                                seed=a.seed, **({"beta": a.beta} if a.beta else {}))

    payload = crossed_sweep(fresh, block, req, by_id, folds, audit_ids, retain_matched, floor_m,
                            parents, selectors, sel_scores, make_gcfg, s2, a.recall_max,
                            a.pool_size, a.seed, extra_eval=extra_eval, log=log)
    json.dump(payload, (out / "crossed.json").open("w"), indent=1)
    log(f"\nwrote {out/'crossed.json'}")


if __name__ == "__main__":
    main()
