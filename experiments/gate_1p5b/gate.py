"""1.5B TOFU gate experiment: the cheapest real-data forecast of Tables 1-2.

One forget10 author on Qwen2.5-1.5B-Instruct: SFT-memorize the request
universe, seal audit-fold scores, run independent generators (NPO/GradDiff/
RMU), unseal, fill a mini Table 1; then compare NPO / NPO+transplant / ours
for a mini Table 2 on the untouched random audit fold.

GPU (H100) run:   python experiments/gate_1p5b/gate.py --device cuda
CPU smoke:        python experiments/gate_1p5b/gate.py --smoke
Artifacts land in runs/gate_<tag>/ (tables + JSON + seal ledger).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from rsus.analysis.prediction import table1_rows  # noqa: E402
from rsus.blocks import mlp_down_last_layers  # noqa: E402
from rsus.data.base import collate  # noqa: E402
from rsus.data.tofu import (  # noqa: E402
    FORGET10_FIRST_AUTHOR,
    idk_variants,
    load_tofu_examples,
    load_tofu_paraphrases,
    tofu_request,
)
from rsus.evalx.metrics import mean_recall  # noqa: E402
from rsus.generators.s2s import S2SConfig, run_s2s_trajectory  # noqa: E402
from rsus.evalx.protection import evaluate_protection  # noqa: E402
from rsus.generators import TrajectoryConfig, run_trajectory  # noqa: E402
from rsus.generators.ours import OursConfig, run_ours_trajectory  # noqa: E402
from rsus.generators.repaired import RepairedConfig, run_engine_repaired  # noqa: E402
from rsus.losses import seq_mean_answer_nll  # noqa: E402
from rsus.partition import PartitionParams, build_partition, make_folds  # noqa: E402
from rsus.probe.base import ProbeSpec, get_scorer  # noqa: E402
from rsus.sealing import seal_scores, unseal  # noqa: E402
from rsus.stage1 import Stage1Config, calibrate_floor  # noqa: E402
from rsus.stage2 import Stage2Config  # noqa: E402

GENERATORS = ["npo", "graddiff", "rmu"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--author", type=int, default=FORGET10_FIRST_AUTHOR)
    p.add_argument("--universe-authors", type=int, default=30)
    p.add_argument("--block-last-n", type=int, default=8)
    p.add_argument("--eta", type=float, default=3e-4)
    p.add_argument("--probe-dirs", type=int, default=8,
                   help="random directions K for norm-estimating scorers (fd_norm): "
                        "2K forward sweeps, relative estimator variance 2/K")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sft-lr", type=float, default=1e-5)
    p.add_argument("--sft-steps", type=int, default=400)
    p.add_argument("--sft-target-loss", type=float, default=0.8)
    p.add_argument("--gen-lr", type=float, default=2e-6)
    p.add_argument("--gen-steps", type=int, default=60)
    p.add_argument("--gen-steps-per", default="",
                   help="per-generator step overrides, e.g. 'npo=240,rmu=120' "
                        "(unlisted generators keep --gen-steps)")
    p.add_argument("--gen-lr-per", default="",
                   help="per-generator lr overrides, e.g. 'npo=6e-6' "
                        "(unlisted generators keep --gen-lr)")
    p.add_argument("--extra-predictors", default="",
                   help="comma-separated additional registered scorers to run and "
                        "seal alongside the default roster, e.g. 'jvp,vmap_graddot,grad_cosine'")
    p.add_argument("--attn-impl", default="",
                   help="attention implementation passed to from_pretrained "
                        "(forced to 'eager' when jvp is requested: SDPA kernels "
                        "do not support forward-mode AD)")
    p.add_argument("--gen-ckpt-every", type=int, default=10)
    p.add_argument("--s1-lr", type=float, default=1e-5)
    p.add_argument("--s1-max-steps", type=int, default=600)
    p.add_argument("--partition-predictor", default="fd",
                   help="scored predictor whose profile builds the protect partition "
                        "for ours/npo_transplant (e.g. grad_norm, knn_feature). "
                        "fd is the preregistered default; it anticorrelates with damage "
                        "on 1.5B TOFU, mis-targeting the protect pool.")
    p.add_argument("--s1-recall-gate", type=float, default=0.0,
                   help="if >0, stage-1 (ours/s2s) must also push forget argmax recall "
                        "to <= this before exiting — aligns its stopping point with the "
                        "T2 common criterion (recall_max=0.10). 0 = floor-only exit.")
    p.add_argument("--s2-steps", type=int, default=80)
    p.add_argument("--t2-steps", type=int, default=0,
                   help="step budget for single-stage T2 arms (0 = inherit --gen-steps)")
    p.add_argument("--t2-lr", type=float, default=0.0,
                   help="lr for single-stage T2 arms (0 = inherit --gen-lr)")
    p.add_argument("--t2-lr-per", default="",
                   help="per-method T2 lr overrides, e.g. 'npo=8e-6,simnpo=8e-6' "
                        "(unlisted methods keep --t2-lr / --gen-lr)")
    p.add_argument("--beta", type=float, default=0.0,
                   help="NPO/SimNPO/IdkDPO temperature for T1 generators and T2 arms "
                        "(0 = keep TrajectoryConfig default 1.0; NPO paper standard is 0.1 — "
                        "beta=1.0 kills ascent pressure ~2 nats above reference)")
    p.add_argument("--pool-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument(
        "--t2-roster",
        default="ga,graddiff,npo,simnpo,idkdpo,rmu,gru,s2s,npo_transplant,ours",
        help="comma-separated Table-2 methods",
    )
    p.add_argument("--smoke", action="store_true", help="tiny random model, CPU-sized budgets")
    p.add_argument("--run-tag", default="", help="suffix for the run directory (rerun without clobbering seals)")
    return p.parse_args()


def apply_smoke(a):
    a.universe_authors = 4
    a.block_last_n = 1
    a.sft_steps = 60
    a.sft_lr = 5e-3
    a.sft_target_loss = 1.2
    a.gen_lr = 1e-3
    a.gen_steps = 10
    a.gen_ckpt_every = 5
    a.s1_lr = 2e-3
    a.s1_max_steps = 250
    a.s2_steps = 20
    a.pool_size = 8
    a.batch_size = 4


def load_model(a, tokenizer):
    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
    if a.smoke:
        torch.manual_seed(a.seed)
        cfg = Qwen2Config(
            vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=512, pad_token_id=tokenizer.pad_token_id,
        )
        return Qwen2ForCausalLM(cfg).to(dtype).eval()
    kw = {"torch_dtype": dtype}
    if a.attn_impl:
        kw["attn_implementation"] = a.attn_impl
    m = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    return m.to(a.device).eval()


def sft(model, examples, a, log):
    opt = torch.optim.AdamW(model.parameters(), lr=a.sft_lr)
    gen = torch.Generator().manual_seed(a.seed)
    for step in range(1, a.sft_steps + 1):
        idx = torch.randperm(len(examples), generator=gen)[: a.batch_size]
        batch = {k: v.to(a.device) if torch.is_tensor(v) else v
                 for k, v in collate([examples[i] for i in idx.tolist()]).items()}
        loss = seq_mean_answer_nll(model, batch).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 20 == 0:
            log(f"  sft step {step} loss {float(loss.detach()):.3f}")
            if float(loss.detach()) <= a.sft_target_loss:
                break
    model.zero_grad(set_to_none=True)


def main():
    a = parse_args()
    if a.smoke:
        apply_smoke(a)
    if "jvp" in a.extra_predictors and not a.attn_impl:
        a.attn_impl = "eager"  # SDPA has no forward-AD support (jvp would crash)
    tag = "smoke" if a.smoke else a.model.split("/")[-1]
    if a.run_tag:
        tag += f"_{a.run_tag}"
    out = ROOT / "runs" / f"gate_{tag}"
    if (out / "seal_ledger.jsonl").exists():
        sys.exit(
            f"{out} already contains sealed scores (seals are append-only by design). "
            "Pass --run-tag <name> for a fresh run directory."
        )
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "gate.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    torch.manual_seed(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    log(f"loading TOFU (universe_authors={a.universe_authors}) ...")
    examples = load_tofu_examples(tokenizer)
    req = tofu_request(a.author, examples, universe_authors=a.universe_authors, seed=a.seed)
    by_id = {e.example_id: e for e in req.universe.examples}
    log(f"request {req.request_id}: |Df|={len(req.forget)} |C|={len(req.universe)}")

    base = load_model(a, tokenizer)          # pre-injection reference
    model0 = load_model(a, tokenizer)
    log("SFT-memorizing the request universe ...")
    sft(model0, list(req.forget) + list(req.universe.examples), a, log)
    state0 = {k: v.clone() for k, v in model0.state_dict().items()}

    def fresh():
        m = load_model(a, tokenizer)
        m.load_state_dict(state0)
        return m

    folds = make_folds({e.example_id: e.group for e in req.universe.examples}, 0.5, a.seed)
    audit_ids = {e.example_id for e in req.universe.examples if folds[e.group] == "audit"}
    disc_ids = sorted(set(by_id) - audit_ids)
    log(f"folds: {len(audit_ids)} audit / {len(disc_ids)} discovery candidates")

    # ---- predictors, sealed on the audit fold --------------------------------
    spec = ProbeSpec(block=mlp_down_last_layers(model0, a.block_last_n), eta=a.eta,
                     batch_size=a.batch_size, n_dirs=a.probe_dirs)
    predictors = ["fd", "knn_feature", "knn_lexical", "grad_norm", "random_rank"]
    try:
        import sentence_transformers  # noqa: F401
        predictors.insert(2, "knn_embed")
    except ImportError:
        log("sentence-transformers missing: skipping knn_embed row")
    for extra in [x.strip() for x in a.extra_predictors.split(",") if x.strip()]:
        if extra not in predictors:
            predictors.insert(1, extra)  # next to fd for easy comparison
    scores_by_pred = {}
    for pred in predictors:
        log(f"scoring: {pred}")
        prof = get_scorer(pred)(fresh(), req, spec)
        scores_by_pred[pred] = prof.scores
        seal_scores(out / "seals", out / "seal_ledger.jsonl", req.request_id, pred,
                    {c: prof.scores[c] for c in audit_ids})

    # ---- independent generator trajectories ----------------------------------
    gen_cfg = TrajectoryConfig(max_steps=a.gen_steps, checkpoint_every=a.gen_ckpt_every,
                               lr=a.gen_lr, batch_size=a.batch_size, seed=a.seed,
                               **({"beta": a.beta} if a.beta else {}))
    import dataclasses as _dc
    gen_steps_per = {k: int(v) for k, v in
                     (kv.split("=") for kv in a.gen_steps_per.split(",") if kv.strip())}
    gen_lr_per = {k: float(v) for k, v in
                  (kv.split("=") for kv in a.gen_lr_per.split(",") if kv.strip())}
    retain_gen = [by_id[c] for c in disc_ids]
    markers = []
    damage_by_opt: dict[str, dict[str, dict[str, float]]] = {}
    for g in GENERATORS:
        log(f"generator: {g}")
        cfg_g = _dc.replace(gen_cfg, max_steps=gen_steps_per.get(g, a.gen_steps),
                            lr=gen_lr_per.get(g, a.gen_lr))
        rec = run_trajectory(fresh(), g, req, retain_gen, cfg_g, out_dir=out / f"traj_{g}")
        markers.append(out / f"traj_{g}" / "DONE")
        term = rec.terminal()
        dmg_all = rec.damage_at()
        rem = [dmg_all[c] for c in audit_ids]
        log(f"  terminal recall={term.forget_recall:.3f} mean_audit_dmg={sum(rem)/len(rem):+.3f}")
        damage_by_opt[g] = {req.request_id: {c: dmg_all[c] for c in audit_ids}}

    sealed = {
        pred: {req.request_id: unseal(out / "seals", out / "seal_ledger.jsonl",
                                      req.request_id, pred, markers)}
        for pred in predictors
    }
    k = max(5, round(0.1 * len(audit_ids)))
    rows = table1_rows(sealed, damage_by_opt, k=k)
    with open(out / "table1.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    log("\n=== Gate Table 1 (single request; audit fold) ===")
    log(f"{'predictor':14s}" + "".join(f"{g+'_rho':>14s}" for g in GENERATORS) + f"{'AUROC':>8s}{'Ovl@'+str(k):>8s}")
    for pred in predictors:
        r = rows[pred]
        log(f"{pred:14s}" + "".join(f"{r[f'{g}_rho']['mean']:14.3f}" for g in GENERATORS)
            + f"{r['auroc']['mean']:8.3f}{r['overlap']['mean']:8.3f}")

    # ---- protection (T2 mini) -------------------------------------------------
    part_scores = scores_by_pred[a.partition_predictor]
    from rsus.costs import CostRecord
    from rsus.probe.base import ScoreProfile

    part = build_partition(
        ScoreProfile(req.request_id, a.partition_predictor, part_scores, spec, CostRecord()),
        req, folds,
        PartitionParams(pool_size=a.pool_size, min_pool_size=4, tau_rem_abs_quantile=0.6,
                        seed=a.seed),
    )
    protect = [by_id[c] for c in part.protect]
    remote_stream = [by_id[c] for c in part.remote_stream]
    gen_r = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(len(disc_ids), generator=gen_r).tolist()
    retain_matched = [by_id[disc_ids[i]] for i in perm[: len(protect)]]  # size-matched

    floor_m = calibrate_floor(base, req, a.batch_size)
    log(f"\npartition: |P|={len(protect)} fallback={part.fallback}; floor m={floor_m:.2f}")

    # paraphrase-recall audit at every snapshot (weights are not persisted)
    extra_eval = None
    try:
        paras = load_tofu_paraphrases(tokenizer)
        para_ex = [paras[e.example_id] for e in req.forget if e.example_id in paras]
        if para_ex:
            extra_eval = lambda m: {"para_recall": mean_recall(m, para_ex, a.batch_size)}  # noqa: E731
    except Exception as e:
        log(f"paraphrase audit unavailable: {e}")

    s1_cfg = Stage1Config(lr=a.s1_lr, max_steps=a.s1_max_steps, eval_every=20,
                          batch_size=a.batch_size, seed=a.seed,
                          forget_recall_max=a.s1_recall_gate or None)
    s2_cfg = Stage2Config(max_steps=a.s2_steps, refresh_k=4,
                          delta_seq_sq=1e-2, delta_tok_sq=1e-1, batch_size=a.batch_size)
    idk = idk_variants(tokenizer, list(req.forget))

    t2 = {}
    t2_lr_per = {k: float(v) for k, v in
                 (kv.split("=") for kv in a.t2_lr_per.split(",") if kv.strip())}
    for method in [x.strip() for x in a.t2_roster.split(",") if x.strip()]:
        log(f"protection method: {method}")
        two_stage = method in ("ours", "s2s") or method.endswith("_repaired")
        try:
            m = fresh()
            if method == "ours":
                ocfg = OursConfig(stage1=s1_cfg, stage2=s2_cfg,
                                  batch_size=a.batch_size, stage2_snapshots=4)
                rec = run_ours_trajectory(m, mlp_down_last_layers(m, a.block_last_n), req,
                                          protect, remote_stream, floor_m, ocfg,
                                          extra_eval=extra_eval, log=log)
            elif method.endswith("_repaired"):
                import dataclasses as _dc

                engine = method[: -len("_repaired")]
                eng_cfg = _dc.replace(gen_cfg, max_steps=a.t2_steps or a.gen_steps,
                                      lr=t2_lr_per.get(engine, a.t2_lr or a.gen_lr))
                if engine == "idkdpo":
                    eng_cfg = _dc.replace(eng_cfg, idk_examples=idk)
                rcfg = RepairedConfig(engine_cfg=eng_cfg, stage2=s2_cfg,
                                      batch_size=a.batch_size)
                rec = run_engine_repaired(m, mlp_down_last_layers(m, a.block_last_n), req,
                                          retain_matched, protect, remote_stream, floor_m,
                                          engine, rcfg, extra_eval=extra_eval, log=log)
            elif method == "s2s":
                scfg = S2SConfig(stage1=s1_cfg, stage2=s2_cfg,
                                 partition=PartitionParams(pool_size=a.pool_size, min_pool_size=4,
                                                           tau_rem_abs_quantile=0.6, seed=a.seed),
                                 batch_size=a.batch_size)
                rec = run_s2s_trajectory(m, mlp_down_last_layers(m, a.block_last_n), req,
                                         folds, floor_m, scfg, extra_eval=extra_eval)
            else:
                import dataclasses as _dc

                objective = "npo" if method == "npo_transplant" else method
                retain = protect if method == "npo_transplant" else retain_matched
                cfg_m = _dc.replace(gen_cfg, max_steps=a.t2_steps or a.gen_steps,
                                    lr=t2_lr_per.get(method, a.t2_lr or a.gen_lr))
                if method == "idkdpo":
                    cfg_m = _dc.replace(cfg_m, idk_examples=idk)
                rec = run_trajectory(m, objective, req, retain, cfg_m, extra_eval=extra_eval)
            outp = evaluate_protection(rec, native_ids=audit_ids, utility_ids=set(),
                                       recall_max=0.10,
                                       mode="last" if two_stage else "first")
        except Exception as e:  # noqa: BLE001  (one arm must not sink the table)
            import traceback
            log(f"  ERROR {type(e).__name__}: {e}")
            log(traceback.format_exc())
            t2[method] = {"reached": False, "error": f"{type(e).__name__}: {e}"}
            continue
        t2[method] = vars(outp)
        para = (outp.extra or {}).get("para_recall")
        log(f"  reach={outp.reached} step={outp.step} "
            f"audit mean dNLL={outp.native_mean if outp.native_mean is None else round(outp.native_mean, 3)} "
            f"CVaR={outp.native_cvar if outp.native_cvar is None else round(outp.native_cvar, 3)}"
            + (f" para_recall={para:.3f}" if para is not None else ""))
    with open(out / "table2.json", "w", encoding="utf-8") as f:
        json.dump(t2, f, indent=1)
    log(f"\nartifacts in {out}")


if __name__ == "__main__":
    main()
