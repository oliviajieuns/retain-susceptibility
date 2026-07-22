"""Prospective 7B channel-mixture protection campaign.

Protocol order (enforced by code):

1. freeze parent-objective hyperparameters from outcome-independent calibration;
2. run the complete alpha grid on development deletion requests;
3. select alpha with ``select_alpha_freeze.py`` and commit that freeze;
4. evaluate the frozen alpha on sealed audit requests.

The worker ranks both component probes *inside the discovery fold*, builds
Top-K protect pools there, and evaluates damage only on the group-disjoint
audit fold.  Audit rows can never enter the alpha selector.  Endpoint alpha=0
is randomized sensitivity and endpoint alpha=1 is hidden-state proximity.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / "src"))


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha(value: object) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def _git_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    return {"code_commit": commit, "code_dirty": bool(dirty)}


def _expand_int_ranges(raw: str) -> list[int]:
    values: list[int] = []
    for item in (part.strip() for part in raw.split(",") if part.strip()):
        if "-" in item:
            lo_raw, hi_raw = item.split("-", 1)
            lo, hi = int(lo_raw), int(hi_raw)
            if hi < lo:
                raise ValueError(f"descending author range: {item!r}")
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(item))
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate author in {raw!r}")
    return values


def _resolve_config_path(config_path: Path, raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _enabled_models(cfg: dict, selected: set[str]) -> list[dict]:
    models = [model for model in cfg["models"] if model.get("enabled", True)]
    if selected:
        models = [model for model in models if model["id"] in selected]
    if not models:
        raise ValueError("no enabled model matched --model-id")
    return models


def _objective_freeze(config_path: Path, cfg: dict, models: list[dict]) -> tuple[Path, dict]:
    phase = cfg["alpha_protection"]
    path = _resolve_config_path(config_path, phase["objective_freeze"])
    freeze = _load_yaml(path)
    if freeze.get("status") != "frozen" or not freeze.get("frozen_before_audit"):
        raise RuntimeError(
            f"refusing alpha protection: parent objective freeze {path} is still draft"
        )
    if freeze.get("source_campaign") != cfg["campaign_id"]:
        raise ValueError("parent objective freeze belongs to another campaign")
    if freeze.get("unresolved"):
        raise ValueError(f"parent objective freeze is unresolved: {freeze['unresolved']}")
    for model in models:
        settings = freeze.get("models", {}).get(model["id"])
        if settings is None:
            raise ValueError(f"parent objective freeze has no model {model['id']}")
        for parent in phase["parents"]:
            spec = settings.get(parent)
            if not spec or spec.get("lr") is None or spec.get("steps") is None:
                raise ValueError(f"missing frozen settings for {model['id']}/{parent}")
    return path, freeze


def _alpha_freeze(config_path: Path, cfg: dict, models: list[dict]) -> tuple[Path, dict]:
    phase = cfg["alpha_protection"]
    path = _resolve_config_path(config_path, phase["alpha_freeze"])
    freeze = _load_yaml(path)
    if (freeze.get("status") != "frozen"
            or not freeze.get("frozen_before_alpha_audit")):
        raise RuntimeError(
            f"refusing alpha audit: {path} is not status=frozen with "
            "frozen_before_alpha_audit=true"
        )
    if freeze.get("source_campaign") != phase["campaign_id"]:
        raise ValueError("alpha freeze belongs to another protection campaign")
    if freeze.get("source_phase") != "development":
        raise ValueError("alpha freeze must declare source_phase=development")
    if not freeze.get("frozen_at_utc"):
        raise ValueError("alpha freeze needs a pre-audit UTC timestamp")
    if freeze.get("unresolved"):
        raise ValueError(f"alpha freeze remains unresolved: {freeze['unresolved']}")
    if freeze.get("normalization") != phase["normalization"]:
        raise ValueError("alpha freeze normalization differs from campaign config")
    if freeze.get("orientation") != phase["orientation"]:
        raise ValueError("alpha freeze orientation differs from campaign config")
    objective_path = _resolve_config_path(config_path, phase["objective_freeze"])
    if freeze.get("objective_freeze_sha256") != _sha256(objective_path):
        raise ValueError("alpha freeze was selected under another objective freeze")
    if freeze.get("campaign_config_sha256") != _sha256(config_path):
        raise ValueError("alpha freeze was selected under another campaign config")
    if not freeze.get("development_artifacts"):
        raise ValueError("alpha freeze must retain hashed development artifacts")
    grid = {float(value) for value in phase["alpha_grid"]}
    for model in models:
        selected = freeze.get("models", {}).get(model["id"])
        if set(selected or {}) != set(phase["parents"]):
            raise ValueError(f"alpha freeze parent roster mismatch for {model['id']}")
        for parent, value in selected.items():
            if value is None or float(value) not in grid:
                raise ValueError(
                    f"frozen alpha for {model['id']}/{parent} must be on {sorted(grid)}"
                )
    return path, freeze


def _fidelity_certificate(cfg: dict, model: dict) -> tuple[Path, dict]:
    declared = cfg["audit"].get("fidelity_certificates", {})
    if model["id"] not in declared:
        raise ValueError(f"no fidelity certificate declared for {model['id']}")
    path = Path(declared[model["id"]])
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"missing randomized-sensitivity fidelity certificate: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "fd-fidelity-certificate-v1",
        "passed": True,
        "model": str(model["path"]),
        "dtype": str(cfg["common"]["dtype"]),
        "block_last_n": int(cfg["common"]["block_last_n"]),
        "R": int(cfg["common"]["probe_dirs"]),
        "eta": float(cfg["common"]["probe_norm_eta"]),
        "probe_seed": int(cfg["common"]["probe_seed"]),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"fidelity certificate mismatch for {model['id']}/{key}: "
                f"expected {value!r}, got {payload.get(key)!r}"
            )
    expected_authors = set(_expand_int_ranges(
        str(cfg["common"]["candidate_author_pools"]["calibration"])
    ))
    if set(payload.get("candidate_authors") or []) != expected_authors:
        raise ValueError("fidelity certificate used another development candidate pool")
    if int(payload.get("n_candidates", 0)) < int(cfg["fidelity"]["n_candidates"]):
        raise ValueError("fidelity certificate has too few candidates")
    return path, payload


def _validate_contract(cfg: dict) -> None:
    from rsus.analysis.channels import DECLARED_CHANNEL
    from rsus.analysis.mixture import declared_alpha, validate_alpha

    phase = cfg["alpha_protection"]
    parents = list(phase["parents"])
    if len(parents) != len(set(parents)) or not parents:
        raise ValueError("alpha protection parents must be non-empty and unique")
    unknown = set(parents) - set(DECLARED_CHANNEL)
    if unknown:
        raise ValueError(f"alpha protection has undeclared parent channels: {sorted(unknown)}")
    grid = [validate_alpha(value) for value in phase["alpha_grid"]]
    if len(grid) != len(set(grid)) or 0.0 not in grid or 1.0 not in grid:
        raise ValueError("alpha protection grid must be unique and contain 0 and 1")
    if phase.get("normalization") != "rank01_discovery_only":
        raise ValueError("primary mixture normalization must remain rank01_discovery_only")
    if phase.get("orientation") != "(1-alpha)*gradient + alpha*proximity":
        raise ValueError("alpha orientation changed from the preregistered coordinate")
    for channel in {DECLARED_CHANNEL[parent] for parent in parents}:
        expected = declared_alpha(channel)
        actual = float(phase["declared_prior"][channel])
        if actual != expected:
            raise ValueError(f"declared prior for {channel} must remain {expected}")

    development_authors = set(phase["development"]["authors"])
    audit_authors = set(phase["audit"]["authors"])
    if development_authors & audit_authors:
        raise ValueError("alpha development and audit deletion requests overlap")
    if development_authors != set(cfg["calibration"]["authors"]):
        raise ValueError("alpha development requests must equal objective calibration requests")
    if audit_authors != set(cfg["audit"]["authors"]):
        raise ValueError("alpha audit requests must equal channel-matrix audit requests")

    pools = cfg["common"]["candidate_author_pools"]
    candidate_authors = set(_expand_int_ranges(str(pools["calibration"])))
    for raw in pools["audit"].values():
        candidate_authors |= set(_expand_int_ranges(str(raw)))
    utility = set(_expand_int_ranges(str(phase["utility_authors"])))
    deletion = development_authors | audit_authors
    if utility & candidate_authors:
        raise ValueError("ordinary utility authors overlap a damage-candidate pool")
    if utility & deletion:
        raise ValueError("ordinary utility authors overlap a deletion request")
    if int(phase["utility_examples_per_author"]) <= 0:
        raise ValueError("utility_examples_per_author must be positive")
    selection = phase["selection"]
    if selection.get("aggregation") != "minimax_across_development_requests_and_seeds":
        raise ValueError("adaptive alpha primary must remain development minimax CVaR")
    if selection.get("prohibited") != "audit_selection":
        raise ValueError("alpha protocol must explicitly prohibit audit selection")


def _candidate_pool(cfg: dict, phase: str, author: int) -> str:
    pools = cfg["common"]["candidate_author_pools"]
    if phase == "development":
        return str(pools["calibration"])
    return str(pools["audit"][str(author)])


def _output_dir(cfg: dict, phase: str, model_id: str, author: int, seed: int) -> Path:
    root = Path(cfg["output_root"])
    if not root.is_absolute():
        root = ROOT / root
    return (root / "alpha_protection" / phase / model_id
            / f"tofu-a{author}" / f"seed-{seed}")


def worker_commands(
    config_path: Path,
    cfg: dict,
    phase: str,
    models: list[dict],
) -> list[tuple[Path, list[str]]]:
    _validate_contract(cfg)
    _objective_freeze(config_path, cfg, models)
    if phase == "audit":
        _alpha_freeze(config_path, cfg, models)
    roster = cfg["alpha_protection"][phase]
    commands = []
    for model, author, seed in itertools.product(models, roster["authors"], roster["seeds"]):
        out = _output_dir(cfg, phase, model["id"], int(author), int(seed))
        commands.append((out, [
            sys.executable,
            str(SELF),
            "--config", str(config_path),
            "--phase", phase,
            "--model-id", str(model["id"]),
            "--author", str(author),
            "--seed", str(seed),
            "--worker",
            "--resume",
        ]))
    return commands


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _run_worker(
    config_path: Path,
    cfg: dict,
    phase_name: str,
    model_id: str,
    author: int,
    seed: int,
    resume: bool,
) -> None:
    # Heavy imports stay inside the worker so command/freeze tests run on a
    # controller node without torch, transformers, or model checkpoints.
    import dataclasses as dc
    import gc

    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT / "experiments" / "gate_1p5b"))
    import gate as gate_runtime  # noqa: E402

    from rsus.analysis.channels import DECLARED_CHANNEL
    from rsus.analysis.mixture import (
        alpha_label,
        channel_mixture_scores,
        declared_alpha,
        rank01,
    )
    from rsus.analysis.prediction import cvar_upper
    from rsus.blocks import load_params_, mlp_down_last_layers, save_params
    from rsus.costs import CostRecord
    from rsus.data.base import CandidateUniverse, Request
    from rsus.data.tofu import load_tofu_examples, load_tofu_paraphrases, tofu_request
    from rsus.evalx.metrics import mean_recall
    from rsus.generators import TrajectoryConfig, run_trajectory
    from rsus.generators.repaired import RepairedConfig, run_repair_from_reached
    from rsus.partition import PartitionParams, build_partition, make_folds
    from rsus.probe.base import ProbeSpec, ScoreProfile, get_scorer
    from rsus.stage1 import calibrate_floor
    from rsus.stage2 import Stage2Config

    _validate_contract(cfg)
    models = _enabled_models(cfg, {model_id})
    if len(models) != 1:
        raise ValueError(f"worker expected exactly one model, got {models}")
    model_cfg = models[0]
    if _git_state()["code_dirty"]:
        raise RuntimeError(f"refusing alpha {phase_name} worker from a dirty worktree")
    fidelity_path, fidelity = _fidelity_certificate(cfg, model_cfg)
    allowed = cfg["alpha_protection"][phase_name]
    if author not in allowed["authors"] or seed not in allowed["seeds"]:
        raise ValueError(f"worker cell {phase_name}/a{author}/s{seed} is outside the roster")
    objective_path, objective_freeze = _objective_freeze(config_path, cfg, models)
    alpha_path = None
    alpha_freeze = None
    if phase_name == "audit":
        alpha_path, alpha_freeze = _alpha_freeze(config_path, cfg, models)

    phase = cfg["alpha_protection"]
    common = cfg["common"]
    out = _output_dir(cfg, phase_name, model_id, author, seed)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "alpha_protection.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

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
        raise ValueError("7B alpha protection requires trainable_scope=probe_block")

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(runtime.model)
    examples = load_tofu_examples(tokenizer)
    candidate_authors = _expand_int_ranges(_candidate_pool(cfg, phase_name, author))
    req = tofu_request(
        author,
        examples,
        universe_authors=int(common["universe_authors"]),
        seed=seed,
        candidate_authors=candidate_authors,
    )
    by_id = {example.example_id: example for example in req.universe.examples}
    folds = make_folds(
        {example.example_id: example.group for example in req.universe.examples},
        0.5,
        seed,
    )
    audit_ids = {cid for cid, example in by_id.items() if folds[example.group] == "audit"}
    discovery_ids = set(by_id) - audit_ids
    if len(audit_ids) != 300 or len(discovery_ids) != 300:
        raise ValueError(
            f"expected 300/300 group split, got {len(discovery_ids)}/{len(audit_ids)}"
        )
    retain = [by_id[cid] for cid in sorted(discovery_ids)]
    scoring_request = Request.build(
        req.request_id,
        list(req.forget),
        CandidateUniverse.freeze(retain),
    )

    utility_authors = _expand_int_ranges(str(phase["utility_authors"]))
    per_author = int(phase["utility_examples_per_author"])
    utility = []
    for utility_author in utility_authors:
        group = f"author-{utility_author:03d}"
        members = sorted(
            (example for example in examples if example.group == group),
            key=lambda example: example.example_id,
        )
        if len(members) < per_author:
            raise ValueError(f"utility group {group} has only {len(members)} examples")
        utility.extend(members[:per_author])

    log(
        f"phase={phase_name} model={runtime.model} request={req.request_id} seed={seed} "
        f"discovery={len(discovery_ids)} audit={len(audit_ids)} utility={len(utility)}"
    )

    # Only a scalar floor is retained; base and SFT models never coexist on GPU.
    base = gate_runtime.load_model(runtime, tokenizer)
    try:
        floor_m = calibrate_floor(base, req, int(common["batch_size"]))
    finally:
        del base
        gate_runtime.clear_cuda_cache()

    model0 = gate_runtime.load_model(runtime, tokenizer)
    probe_block = mlp_down_last_layers(model0, int(common["block_last_n"]))
    sft_examples = list(req.forget) + list(req.universe.examples)
    root = Path(cfg["output_root"])
    if not root.is_absolute():
        root = ROOT / root
    cache_path = root / "sft_cache" / model_id / f"tofu-a{author}_seed-{seed}.pt"
    contract = gate_runtime._sft_cache_contract(runtime, req, probe_block)
    sft_result = gate_runtime._load_sft_cache(model0, cache_path, contract, log)
    if sft_result is None:
        sft_result = gate_runtime.sft(model0, sft_examples, runtime, log, probe_block)
    if common.get("require_sft_target", True) and not sft_result["reached"]:
        raise RuntimeError(
            f"SFT gate failed: {sft_result['full_mean_nll']} > {sft_result['target']}"
        )
    utility_nll0 = gate_runtime._mean_example_nll(
        model0, utility, int(common["batch_size"]), runtime.device
    )
    state0 = {name: tensor.detach().cpu().clone()
              for name, tensor in model0.state_dict().items()}
    if not cache_path.exists():
        gate_runtime._write_sft_cache(cache_path, contract, sft_result, state0, log)
    model_info = {
        "architecture": type(model0).__name__,
        "num_hidden_layers": int(model0.config.num_hidden_layers),
        "hidden_size": int(model0.config.hidden_size),
        "total_parameters": sum(parameter.numel() for parameter in model0.parameters()),
        "probe_block_parameters": sum(
            parameter.numel() for parameter in probe_block.select(model0).values()
        ),
    }
    del model0
    gate_runtime.clear_cuda_cache()

    def fresh():
        model = gate_runtime.load_model(runtime, tokenizer)
        model.load_state_dict(state0)
        return model

    try:
        paraphrases = load_tofu_paraphrases(tokenizer)
        para_examples = [paraphrases[example.example_id] for example in req.forget
                         if example.example_id in paraphrases]
    except Exception as error:  # dataset availability is checked before audit
        if phase_name == "audit":
            raise RuntimeError("paraphrase audit unavailable in offline alpha audit") from error
        log(f"development paraphrase evaluation unavailable: {type(error).__name__}: {error}")
        para_examples = []

    def extra_eval(model) -> dict[str, float]:
        current = gate_runtime._mean_example_nll(
            model, utility, int(common["batch_size"]), runtime.device
        )
        result = {
            "utility_mean_nll": current,
            "utility_retention": utility_nll0 / current if current > 0 else float("nan"),
        }
        if para_examples:
            result["para_recall"] = mean_recall(
                model, para_examples, int(common["batch_size"])
            )
        return result

    probe_spec = ProbeSpec(
        block=probe_block,
        eta=3e-4,
        seed=int(common["probe_seed"]),
        batch_size=int(common["batch_size"]),
        n_dirs=int(common["probe_dirs"]),
        norm_eta=float(common["probe_norm_eta"]),
    )
    score_names = [
        phase["probes"]["gradient"],
        phase["probes"]["proximity"],
        "grad_norm",
        "random_rank",
    ]
    scores: dict[str, dict[str, float]] = {}
    score_costs = {}
    for scorer_name in score_names:
        log(f"scoring discovery allocation component: {scorer_name}")
        scorer_model = fresh()
        try:
            profile = get_scorer(scorer_name)(scorer_model, scoring_request, probe_spec)
            scores[scorer_name] = profile.scores
            score_costs[scorer_name] = dc.asdict(profile.cost)
        finally:
            del scorer_model
            gate_runtime.clear_cuda_cache()

    gradient_scores = scores[phase["probes"]["gradient"]]
    proximity_scores = scores[phase["probes"]["proximity"]]
    if phase_name == "audit" and not phase["audit"].get("run_descriptive_grid", True):
        frozen_by_parent = alpha_freeze["models"][model_id]
        alphas_by_parent = {
            parent: sorted({
                0.0,
                1.0,
                declared_alpha(DECLARED_CHANNEL[parent]),
                float(frozen_by_parent[parent]),
            })
            for parent in phase["parents"]
        }
    else:
        alphas_by_parent = {
            parent: [float(value) for value in phase["alpha_grid"]]
            for parent in phase["parents"]
        }

    partition_params = PartitionParams(
        pool_size=int(phase["partition"]["pool_size"]),
        min_pool_size=int(phase["partition"]["min_pool_size"]),
        tau_rem_abs_quantile=float(phase["partition"]["tau_rem_abs_quantile"]),
        seed=seed,
    )

    # Build every unique allocation once. Audit candidates were never passed
    # to a scorer and cannot affect normalization or the remote-band quantile.
    selector_scores: dict[str, dict[str, float]] = {}
    selector_meta: dict[str, dict[str, object]] = {}
    all_alphas = sorted({value for values in alphas_by_parent.values() for value in values})
    for alpha in all_alphas:
        label = alpha_label(alpha)
        discovery = channel_mixture_scores(
            gradient_scores,
            proximity_scores,
            alpha,
            candidate_ids=discovery_ids,
        )
        selector_scores[label] = {
            cid: discovery[cid] if cid in discovery else 0.0 for cid in sorted(by_id)
        }
        selector_meta[label] = {
            "selector_type": "mixture",
            "alpha": alpha,
            "backward_free": True,
        }
    for label, raw, backward_free in (
        ("exact_grad_norm", scores["grad_norm"], False),
        ("random", scores["random_rank"], True),
    ):
        discovery = rank01({cid: raw[cid] for cid in discovery_ids})
        selector_scores[label] = {
            cid: discovery[cid] if cid in discovery else 0.0 for cid in sorted(by_id)
        }
        selector_meta[label] = {
            "selector_type": "exact_gradient_ceiling" if label == "exact_grad_norm" else "random",
            "alpha": None,
            "backward_free": backward_free,
        }

    partitions = {}
    partition_manifest = {}
    for selector, selector_score in selector_scores.items():
        profile = ScoreProfile(req.request_id, selector, selector_score, probe_spec, CostRecord())
        partition = build_partition(profile, req, folds, partition_params)
        partitions[selector] = partition
        partition_manifest[selector] = {
            **selector_meta[selector],
            "manifest_sha": partition.manifest_sha,
            "fallback": partition.fallback,
            "protect": list(partition.protect),
            "remote_stream": list(partition.remote_stream),
            "discovery_score_sha256": _json_sha(
                {cid: selector_score[cid] for cid in sorted(discovery_ids)}
            ),
        }

    manifest = {
        "schema": "channel-mixture-protection-run-v1",
        "campaign_id": phase["campaign_id"],
        "campaign_phase": phase_name,
        "host": platform.node(),
        "model": runtime.model,
        "model_id": model_id,
        "model_info": model_info,
        "request": req.request_id,
        "author": author,
        "seed": seed,
        "dtype": runtime.dtype,
        "candidate_authors": candidate_authors,
        "candidate_universe_sha": req.universe.sha,
        "forget_sha": req.forget_sha,
        "discovery_ids_sha256": _json_sha(sorted(discovery_ids)),
        "audit_ids_sha256": _json_sha(sorted(audit_ids)),
        "normalization_scope": "discovery_only",
        "scored_candidate_count": len(scoring_request.universe),
        "utility_authors": utility_authors,
        "utility_examples": len(utility),
        "utility_ids_sha256": _json_sha([example.example_id for example in utility]),
        "utility_mean_nll_at_theta0": utility_nll0,
        "sft": sft_result,
        "probe_config": {
            "gradient": phase["probes"]["gradient"],
            "proximity": phase["probes"]["proximity"],
            "R": int(common["probe_dirs"]),
            "eta": float(common["probe_norm_eta"]),
            "seed": int(common["probe_seed"]),
            "block": probe_block.pattern,
        },
        "score_costs": score_costs,
        "alpha_grid_executed": all_alphas,
        "parents": phase["parents"],
        "objective_freeze": str(objective_path),
        "objective_freeze_id": objective_freeze["freeze_id"],
        "objective_freeze_sha256": _sha256(objective_path),
        "fidelity_certificate": str(fidelity_path),
        "fidelity_certificate_sha256": _sha256(fidelity_path),
        "fidelity_gate_metrics": fidelity.get("metrics"),
        "alpha_freeze": str(alpha_path) if alpha_path else None,
        "alpha_freeze_id": alpha_freeze.get("freeze_id") if alpha_freeze else None,
        "alpha_freeze_sha256": _sha256(alpha_path) if alpha_path else None,
        "campaign_config_sha256": _sha256(config_path),
        "partitions": partition_manifest,
        **_git_state(),
    }
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable_keys = (
            "schema", "campaign_id", "campaign_phase", "model_id", "request", "seed",
            "candidate_universe_sha", "forget_sha", "objective_freeze_sha256",
            "fidelity_certificate_sha256", "alpha_freeze_sha256",
            "campaign_config_sha256", "partitions",
            "code_commit",
        )
        mismatched = [key for key in immutable_keys if old.get(key) != manifest.get(key)]
        if mismatched:
            raise RuntimeError(
                f"resume manifest mismatch at {out}: {mismatched}; preserve the directory"
            )
        if not resume:
            raise RuntimeError(f"pre-existing alpha protection run at {out}; pass --resume")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    result_log = out / "results.partial.jsonl"
    prior_rows = _read_jsonl(result_log)
    completed = {(row["parent"], row["selector"]) for row in prior_rows}

    settings = objective_freeze["models"][model_id]
    parent_cfgs = {}
    for parent in phase["parents"]:
        spec = settings[parent]
        parent_cfgs[parent] = TrajectoryConfig(
            max_steps=int(spec["steps"]),
            checkpoint_every=int(phase["parent"]["checkpoint_every"]),
            batch_size=int(common["batch_size"]),
            lr=float(spec["lr"]),
            seed=seed,
            beta=float(spec.get("beta", 1.0)),
            rmu_alpha=float(spec.get("rmu_alpha", 10.0)),
            rmu_c=float(spec.get("rmu_c", 3.0)),
            trainable_pattern=probe_block.pattern,
            forget_weight=float(spec.get("forget_weight", 1.0)),
            retain_weight=float(spec.get("retain_weight", 1.0)),
            representation_retain_mode=str(common["representation_retain_mode"]),
        )

    stage2 = Stage2Config(
        eta2=float(phase["stage2"]["eta2"]),
        mu_v=float(phase["stage2"]["mu_v"]),
        refresh_k=int(phase["stage2"]["refresh_k"]),
        rho_g=float(phase["stage2"]["rho_g"]),
        delta_seq_sq=float(phase["stage2"]["delta_seq_sq"]),
        delta_tok_sq=float(phase["stage2"]["delta_tok_sq"]),
        shrink=float(phase["stage2"]["shrink"]),
        max_steps=int(phase["stage2"]["max_steps"]),
        batch_size=int(common["batch_size"]),
    )
    recall_max = float(phase["parent"]["recall_max"])

    def metrics(record) -> dict[str, object]:
        if not record.snapshots:
            return {
                "reached": False,
                "step": None,
                "forget_recall": None,
                "mean_dnll": None,
                "cvar05_dnll": None,
                "utility_retention": None,
                "utility_mean_nll": None,
                "para_recall": None,
            }
        terminal = record.snapshots[-1]
        damage = [terminal.nll[cid] - record.nll0[cid] for cid in sorted(audit_ids)]
        extra = terminal.extra or {}
        return {
            "reached": bool(terminal.forget_recall <= recall_max),
            "step": int(terminal.step),
            "forget_recall": float(terminal.forget_recall),
            "mean_dnll": sum(damage) / len(damage),
            "cvar05_dnll": cvar_upper(damage, 0.05),
            "utility_retention": extra.get("utility_retention"),
            "utility_mean_nll": extra.get("utility_mean_nll"),
            "para_recall": extra.get("para_recall"),
        }

    def row_base(parent: str, selector: str) -> dict[str, object]:
        meta = selector_meta.get(selector, {
            "selector_type": "none", "alpha": None, "backward_free": True,
        })
        channel = DECLARED_CHANNEL[parent]
        selected_alpha = (
            float(alpha_freeze["models"][model_id][parent]) if alpha_freeze else None
        )
        return {
            "campaign_phase": phase_name,
            "model_id": model_id,
            "request": req.request_id,
            "seed": seed,
            "parent": parent,
            "channel": channel,
            "selector": selector,
            **meta,
            "declared_prior": (
                meta.get("alpha") is not None
                and float(meta["alpha"]) == declared_alpha(channel)
            ),
            "deployed": (
                selected_alpha is not None and meta.get("alpha") is not None
                and float(meta["alpha"]) == selected_alpha
            ),
        }

    for parent in phase["parents"]:
        required_selectors = [alpha_label(alpha) for alpha in alphas_by_parent[parent]]
        required_selectors += ["exact_grad_norm", "random"]
        expected_keys = {(parent, "none"), *((parent, value) for value in required_selectors)}
        if expected_keys <= completed:
            log(f"SKIP complete parent {parent}")
            continue

        log(f"parent {parent}: run once to first reaching checkpoint")
        parent_model = fresh()
        try:
            parent_record = run_trajectory(
                parent_model,
                parent,
                req,
                retain,
                parent_cfgs[parent],
                extra_eval=extra_eval,
                stop_at_recall=recall_max,
            )
            saved_parent_block = {
                name: tensor.detach().cpu().clone()
                for name, tensor in save_params(probe_block.select(parent_model)).items()
            }
        finally:
            del parent_model
            gc.collect()
            gate_runtime.clear_cuda_cache()

        parent_metrics = metrics(parent_record)
        if (parent, "none") not in completed:
            row = {**row_base(parent, "none"), "executed": True, **parent_metrics}
            _append_jsonl(result_log, row)
            completed.add((parent, "none"))
            log(
                f"  none reached={row['reached']} CVaR={row['cvar05_dnll']} "
                f"utility={row['utility_retention']}"
            )

        if not parent_metrics["reached"]:
            for selector in required_selectors:
                if (parent, selector) in completed:
                    continue
                row = {
                    **row_base(parent, selector),
                    "executed": False,
                    "error": "parent_did_not_reach_forget_criterion",
                    **parent_metrics,
                }
                _append_jsonl(result_log, row)
                completed.add((parent, selector))
            del parent_record, saved_parent_block
            continue

        repaired_cfg = RepairedConfig(
            engine_cfg=parent_cfgs[parent],
            stage2=stage2,
            recall_max=recall_max,
            batch_size=int(common["batch_size"]),
            stage2_snapshots=int(phase["stage2"]["snapshots"]),
        )
        for selector in required_selectors:
            if (parent, selector) in completed:
                log(f"  SKIP completed selector {selector}")
                continue
            partition = partitions[selector]
            protect = [by_id[cid] for cid in partition.protect]
            remote = [by_id[cid] for cid in partition.remote_stream]
            log(f"  repair selector={selector} protect={len(protect)} remote={len(remote)}")
            repair_model = fresh()
            try:
                load_params_(probe_block.select(repair_model), saved_parent_block)
                record = run_repair_from_reached(
                    repair_model,
                    probe_block,
                    req,
                    protect,
                    remote,
                    floor_m,
                    parent,
                    repaired_cfg,
                    parent_record,
                    extra_eval=extra_eval,
                    log=log,
                )
                result_metrics = metrics(record)
            finally:
                del repair_model
                if "record" in locals():
                    del record
                gc.collect()
                gate_runtime.clear_cuda_cache()
            row = {
                **row_base(parent, selector),
                "executed": True,
                "partition_sha": partition.manifest_sha,
                **result_metrics,
            }
            _append_jsonl(result_log, row)
            completed.add((parent, selector))
            log(
                f"    reached={row['reached']} CVaR={row['cvar05_dnll']} "
                f"utility={row['utility_retention']}"
            )
        del parent_record, saved_parent_block
        gate_runtime.clear_cuda_cache()

    rows = _read_jsonl(result_log)
    expected = set()
    for parent in phase["parents"]:
        expected.add((parent, "none"))
        expected |= {(parent, alpha_label(alpha)) for alpha in alphas_by_parent[parent]}
        expected |= {(parent, "exact_grad_norm"), (parent, "random")}
    actual = {(row["parent"], row["selector"]) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError(
            "alpha protection result grid incomplete/non-unique: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}, "
            f"rows={len(rows)} expected={len(expected)}"
        )
    payload = {"manifest": manifest, "results": rows}
    (out / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "DONE").touch()
    log(f"completed {phase_name} alpha-protection cell: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/channel_matrix/7b_tofu.yaml")
    parser.add_argument("--phase", required=True, choices=["development", "audit"])
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--author", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)
    _validate_contract(cfg)
    models = _enabled_models(cfg, set(args.model_id))
    if args.worker:
        if len(models) != 1 or args.author is None or args.seed is None:
            parser.error("worker mode needs exactly one --model-id, --author, and --seed")
        _run_worker(
            config_path, cfg, args.phase, models[0]["id"], args.author, args.seed, args.resume
        )
        return

    git_state = _git_state()
    if not args.dry_run and git_state["code_dirty"]:
        raise RuntimeError(
            f"refusing alpha {args.phase} from a dirty worktree; commit the protocol first"
        )
    commands = worker_commands(config_path, cfg, args.phase, models)
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    if args.phase == "audit":
        env.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        })
    count = 0
    for out, command in commands:
        if args.resume and (out / "DONE").exists():
            print(f"SKIP complete: {out}")
            continue
        if out.exists() and any(out.iterdir()) and not args.resume:
            raise RuntimeError(f"pre-existing run directory {out}; preserve it or use --resume")
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        count += 1
        if args.limit and count >= args.limit:
            break
    print(
        f"alpha protection phase={args.phase}: "
        f"{'planned' if args.dry_run else 'completed'} {count} run(s)"
    )


if __name__ == "__main__":
    main()
