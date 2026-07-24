"""Score development prediction probes over the FULL candidate universe.

The alpha_protection development worker scores probes on the discovery fold
only (the allocation input), while the none-arm ``candidate_damage`` in
``results.json`` covers only the group-disjoint audit fold.  The frozen
prediction-weight selector (``select_prediction_freeze.py``) therefore needs
probe scores that also cover the audit fold.

This runner rebuilds the exact development Request a cell's worker builds —
same 600-candidate universe, same 300/300 discovery/audit group split, same
shas, via ``alpha_protection.build_dev_request`` — restores the identical
post-SFT entry checkpoint (through the shared SFT cache), and scores all 600
candidates with the config's gradient + proximity probes, plus ``knn_embed``
when sentence-transformers is available (skipped gracefully otherwise, like
gate.py).  One JSON per scorer is written to --out-dir.

These are development requests — NOT the sealed audit — so plain writes are
correct.  The runner refuses to run on any author outside
``alpha_protection.development.authors``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alpha_protection import (  # noqa: E402
    _enabled_models,
    _load_yaml,
    _validate_contract,
    build_dev_request,
)

SCHEMA = "dev-prediction-probe-v1"


def _output_root(cfg: dict) -> Path:
    root = Path(cfg["output_root"])
    return root if root.is_absolute() else (ROOT / root)


def _default_out_dir(cfg: dict, model_id: str, author: int) -> Path:
    return (_output_root(cfg) / "alpha_protection" / "prediction_probes"
            / model_id / f"tofu-a{author}")


def _existing_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise SystemExit(
            f"{path} exists with unexpected schema {payload.get('schema')!r}; "
            "move it aside before rerunning"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/channel_matrix/7b_tofu.yaml")
    parser.add_argument("--author", type=int, required=True,
                        help="development deletion-request author (never an audit author)")
    parser.add_argument("--model-id", default="",
                        help="required only when several models are enabled in the config")
    parser.add_argument(
        "--out-dir", default="",
        help="default: <output_root>/alpha_protection/prediction_probes/"
             "<model_id>/tofu-a<author>",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)
    _validate_contract(cfg)
    phase = cfg["alpha_protection"]

    development_authors = {int(value) for value in phase["development"]["authors"]}
    if int(args.author) not in development_authors:
        raise SystemExit(
            f"refusing author {args.author}: development prediction probes may only "
            f"run on development authors {sorted(development_authors)}, never the "
            "sealed audit roster"
        )

    models = _enabled_models(cfg, {args.model_id} if args.model_id else set())
    if len(models) != 1:
        raise SystemExit(
            f"several models enabled ({[model['id'] for model in models]}); pass --model-id"
        )
    model_cfg = models[0]
    model_id = model_cfg["id"]

    out_dir = (Path(args.out_dir).resolve() if args.out_dir
               else _default_out_dir(cfg, model_id, int(args.author)))
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)

    common = cfg["common"]
    seeds = [int(value) for value in phase["development"]["seeds"]]
    if len(seeds) != 1:
        raise SystemExit(
            f"development declares several seeds {seeds}; this runner scores the "
            "single-seed layout — extend it before adding development seeds"
        )
    seed = seeds[0]

    # Component-probe roster from the config; knn_embed joins only when the
    # optional sentence-transformers dependency is importable (gate.py rule).
    roster = list(dict.fromkeys([
        str(phase["probes"]["gradient"]),
        str(phase["probes"]["proximity"]),
    ]))
    knn_embed_available = False
    if "knn_embed" not in roster:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            log("sentence-transformers missing: skipping knn_embed probe")
        else:
            knn_embed_available = True
            roster.append("knn_embed")

    # Heavy imports after the cheap contract checks, mirroring the worker.
    import dataclasses as dc

    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT / "experiments" / "gate_1p5b"))
    import gate as gate_runtime  # noqa: E402

    from rsus.blocks import mlp_down_last_layers
    from rsus.probe.base import ProbeSpec, get_scorer

    runtime = SimpleNamespace(
        model=str(model_cfg["path"]),
        model_id=model_id,
        device=str(common.get("device", "cuda")),
        dtype=str(common.get("dtype", "float32")),
        attn_impl=str(common.get("attn_impl", "")),
        smoke=False,
        seed=seed,
        trainable_scope=str(common.get("trainable_scope", "probe_block")),
        sft_lr=float(common["sft_lr"]),
        sft_steps=int(common["sft_steps"]),
        sft_target_loss=float(common["sft_target_loss"]),
        sft_eval_every=int(common["sft_eval_every"]),
        batch_size=int(common["batch_size"]),
    )
    if runtime.trainable_scope != "probe_block":
        raise SystemExit("alpha protection requires trainable_scope=probe_block")

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(runtime.model)
    cell = build_dev_request(cfg, model_id, int(args.author), tokenizer, seed=seed)
    req = cell.req
    by_id = cell.by_id
    candidate_meta = {
        candidate_id: {
            "group": by_id[candidate_id].group,
            "fold": "audit" if candidate_id in cell.audit_ids else "discovery",
        }
        for candidate_id in sorted(by_id)
    }
    log(
        f"model={model_id} request={req.request_id} seed={seed} "
        f"universe={len(by_id)} discovery={len(cell.discovery_ids)} "
        f"audit={len(cell.audit_ids)} scorers={roster}"
    )

    # Idempotency: a scorer whose output already exists is verified and skipped.
    pending = []
    for scorer_name in roster:
        payload = _existing_payload(out_dir / f"{scorer_name}.json")
        if payload is None:
            pending.append(scorer_name)
            continue
        if payload.get("candidate_universe_sha") != req.universe.sha:
            raise SystemExit(
                f"{out_dir / (scorer_name + '.json')} was scored on another candidate "
                f"universe (sha {payload.get('candidate_universe_sha')!r} != "
                f"{req.universe.sha!r}); move it aside before rerunning"
            )
        log(f"SKIP existing verified probe output: {scorer_name}")
    if not pending:
        log(f"all probe outputs already present in {out_dir}")
        return

    if "knn_embed" in pending and knn_embed_available:
        try:
            from sentence_transformers import SentenceTransformer

            from rsus.probe.baselines import set_embed_encoder

            encoder_name = str(common.get(
                "sentence_encoder", "sentence-transformers/all-MiniLM-L6-v2"
            ))
            sentence_model = SentenceTransformer(encoder_name, device="cpu")

            def encode_sentences(examples):
                texts = [example.text for example in examples]
                if not all(texts):
                    raise ValueError("knn_embed requires non-empty Example.text")
                return torch.as_tensor(
                    sentence_model.encode(
                        texts,
                        batch_size=runtime.batch_size,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                )

            set_embed_encoder(encode_sentences)
        except Exception as error:  # encoder assets may be absent offline
            log(
                f"sentence encoder unavailable ({type(error).__name__}: {error}): "
                "skipping knn_embed probe"
            )
            pending.remove("knn_embed")
            if not pending:
                log(f"all remaining probe outputs already present in {out_dir}")
                return

    # Restore the identical post-SFT entry checkpoint the cell worker scored
    # at, reusing (or creating) the shared SFT cache.
    model0 = gate_runtime.load_model(runtime, tokenizer)
    probe_block = mlp_down_last_layers(model0, int(common["block_last_n"]))
    sft_examples = list(req.forget) + list(req.universe.examples)
    cache_path = (_output_root(cfg) / "sft_cache" / model_id
                  / f"tofu-a{args.author}_seed-{seed}.pt")
    contract = gate_runtime._sft_cache_contract(runtime, req, probe_block)
    sft_result = gate_runtime._load_sft_cache(model0, cache_path, contract, log)
    if sft_result is None:
        sft_result = gate_runtime.sft(model0, sft_examples, runtime, log, probe_block)
    if common.get("require_sft_target", True) and not sft_result["reached"]:
        raise RuntimeError(
            f"SFT gate failed: {sft_result['full_mean_nll']} > {sft_result['target']}"
        )
    state0 = {name: tensor.detach().cpu().clone()
              for name, tensor in model0.state_dict().items()}
    if not cache_path.exists():
        gate_runtime._write_sft_cache(cache_path, contract, sft_result, state0, log)
    del model0
    gate_runtime.clear_cuda_cache()

    def fresh():
        model = gate_runtime.load_model(runtime, tokenizer)
        model.load_state_dict(state0)
        return model

    probe_spec = ProbeSpec(
        block=probe_block,
        eta=3e-4,
        seed=int(common["probe_seed"]),
        batch_size=int(common["batch_size"]),
        n_dirs=int(common["probe_dirs"]),
        norm_eta=float(common["probe_norm_eta"]),
    )

    for scorer_name in pending:
        log(f"scoring full-universe prediction probe: {scorer_name}")
        scorer_model = fresh()
        try:
            profile = get_scorer(scorer_name)(scorer_model, req, probe_spec)
        finally:
            del scorer_model
            gate_runtime.clear_cuda_cache()
        missing = set(by_id) - set(profile.scores)
        if missing:
            raise RuntimeError(
                f"{scorer_name} scored {len(profile.scores)} of {len(by_id)} "
                f"candidates; missing e.g. {sorted(missing)[:5]}"
            )
        payload = {
            "schema": SCHEMA,
            "request": req.request_id,
            "model_id": model_id,
            "author": int(args.author),
            "seed": seed,
            "campaign_phase": "development",
            "scorer": scorer_name,
            "scoring_universe": "full",
            "candidate_universe_sha": req.universe.sha,
            "forget_sha": req.forget_sha,
            "scores": {cid: float(value) for cid, value in profile.scores.items()},
            "cost": dc.asdict(profile.cost),
            "candidate_meta": candidate_meta,
        }
        path = out_dir / f"{scorer_name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        log(f"wrote {path}")

    log(f"completed development prediction probes: {out_dir}")


if __name__ == "__main__":
    main()
