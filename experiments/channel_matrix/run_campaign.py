"""Launch a sealed, multi-model 7B channel-matrix campaign.

The launcher has two deliberately separate phases:

``calibration``
    Runs one objective at a time on development-only TOFU authors, without
    computing susceptibility probes.  Learning rate and horizon may be chosen
    from these runs using forget reach and ordinary utility only.

``audit``
    Requires an immutable objective-freeze YAML, then computes and seals all
    probes before running every frozen objective on each audit author/seed.
    The launcher never selects a setting from audit outcomes.

``fidelity``
    Evaluates the already frozen randomized-sensitivity operating point on the
    disjoint development pool and writes the certificate required by audit.

Examples (on the H100 host)::

    python experiments/channel_matrix/run_campaign.py \
      --config configs/channel_matrix/7b_tofu.yaml --phase calibration --dry-run

    python experiments/channel_matrix/run_campaign.py \
      --config configs/channel_matrix/7b_tofu.yaml --phase audit
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "experiments" / "gate_1p5b" / "gate.py"
REPORT = ROOT / "experiments" / "diag" / "channel_report.py"
FIDELITY = ROOT / "experiments" / "diag" / "fd_fidelity.py"


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    dirty_output = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    return {"code_commit": commit, "code_dirty": bool(dirty_output)}


def _csv(values) -> str:
    return ",".join(str(v) for v in values)


def _expand_int_ranges(raw: str) -> set[int]:
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
        raise ValueError(f"duplicate candidate author in {raw!r}")
    return set(values)


def _validate_campaign(cfg: dict) -> None:
    calibration_authors = set(cfg["calibration"]["authors"])
    audit_authors = set(cfg["audit"]["authors"])
    if calibration_authors & audit_authors:
        raise ValueError("calibration and audit deletion-request authors overlap")
    pools = cfg["common"].get("candidate_author_pools", {})
    if set(pools) != {"calibration", "audit"}:
        raise ValueError("candidate_author_pools must declare calibration and audit")
    development_pool = _expand_int_ranges(str(pools["calibration"]))
    if not isinstance(pools["audit"], dict):
        raise ValueError("audit candidate_author_pools must map deletion author -> fixed pool")
    if set(pools["audit"]) != {str(author) for author in audit_authors}:
        raise ValueError("audit candidate pools do not match the audit deletion-request roster")
    audit_pools = {
        int(author): _expand_int_ranges(str(raw))
        for author, raw in pools["audit"].items()
    }
    all_seen = set(development_pool)
    for author, pool in sorted(audit_pools.items()):
        overlap = all_seen & pool
        if overlap:
            raise ValueError(
                f"retained-candidate pool for audit author {author} overlaps an earlier pool: "
                f"{sorted(overlap)}"
            )
        all_seen |= pool
    expected_size = int(cfg["common"]["universe_authors"])
    if (len(development_pool) != expected_size
            or any(len(pool) != expected_size for pool in audit_pools.values())):
        raise ValueError(
            f"each candidate pool must contain universe_authors={expected_size} authors"
        )
    if calibration_authors & development_pool:
        raise ValueError("a calibration forget author appears in its retained-candidate pool")
    if any(author in pool for author, pool in audit_pools.items()):
        raise ValueError("an audit forget author appears in its own retained-candidate pool")
    objective_grid = cfg["calibration"]["objective_grid"]
    audit_roster = set(cfg["audit"]["objectives"]) | set(cfg["audit"].get("stress_objectives", []))
    if set(objective_grid) != audit_roster:
        raise ValueError("calibration and audit objective rosters disagree")
    if set(cfg["audit"]["objectives"]) & set(cfg["audit"].get("stress_objectives", [])):
        raise ValueError("core and stress objective rosters overlap")
    for objective, settings in objective_grid.items():
        ids = [setting.get("id") for setting in settings]
        if not settings or None in ids or len(ids) != len(set(ids)):
            raise ValueError(f"{objective} calibration settings need unique explicit ids")


def _kv(settings: dict[str, dict], key: str) -> str:
    return ",".join(
        f"{objective}={spec[key]}"
        for objective, spec in settings.items()
        if key in spec and spec[key] is not None
    )


def _request_dirname(cfg: dict, author: int) -> str:
    """Request directory stem matching the dataset's request_id convention."""
    dataset = cfg.get("dataset", "tofu")
    if dataset == "rwku":
        return f"rwku-t{author:03d}"
    if dataset == "wmdp_bio_mmlu":
        return f"wmdp-r{author:03d}"
    return f"tofu-a{author}"


def _common_gate_args(
    cfg: dict, model: dict, author: int, seed: int, out_dir: Path, phase: str
) -> list[str]:
    common = cfg["common"]
    args = [
        sys.executable,
        str(GATE),
        "--dataset", str(cfg.get("dataset", "tofu")),
        "--model", str(model["path"]),
        "--model-id", str(model["id"]),
        "--device", str(common.get("device", "cuda")),
        "--dtype", str(common.get("dtype", "float32")),
        "--author", str(author),
        "--seed", str(seed),
        "--probe-seed", str(common.get("probe_seed", 0)),
        "--universe-authors", str(common.get("universe_authors", 30)),
        "--block-last-n", str(common.get("block_last_n", 8)),
        "--trainable-scope", str(common.get("trainable_scope", "probe_block")),
        "--batch-size", str(common.get("batch_size", 4)),
        "--sft-lr", str(common.get("sft_lr", 1e-5)),
        "--sft-steps", str(common.get("sft_steps", 400)),
        "--sft-target-loss", str(common.get("sft_target_loss", 0.8)),
        "--sft-eval-every", str(common.get("sft_eval_every", 100)),
        "--probe-dirs", str(common.get("probe_dirs", 64)),
        "--probe-norm-eta", str(common.get("probe_norm_eta", 3e-3)),
        "--gen-rep-retain-mode", str(common.get("representation_retain_mode", "stream_cached")),
        "--t2-roster", "",
        "--out-dir", str(out_dir),
    ]
    if common.get("require_sft_target", True):
        args.append("--require-sft-target")
    pool = common.get("candidate_author_pools", {}).get(phase)
    if isinstance(pool, dict):
        pool = pool.get(str(author))
    if pool:
        args += ["--candidate-authors", str(pool)]
    if common.get("attn_impl"):
        args += ["--attn-impl", str(common["attn_impl"])]
    if common.get("sentence_encoder"):
        args += ["--sentence-encoder", str(common["sentence_encoder"])]
    return args


def _run(cmd: list[str], dry_run: bool, env: dict[str, str]) -> None:
    print(shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _annotate_manifest(out: Path, metadata: dict) -> None:
    path = out / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(metadata)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _enabled_models(cfg: dict, selected: set[str]) -> list[dict]:
    models = [m for m in cfg["models"] if m.get("enabled", True)]
    if selected:
        models = [m for m in models if m["id"] in selected]
    if not models:
        raise ValueError("no enabled model matched --model-id")
    return models


def _filter_authors(
    roster: list[int], selected: set[int] | None, phase: str
) -> list[int]:
    """Apply an execution-only request shard without changing the frozen roster."""
    authors = [int(author) for author in roster]
    if selected is None:
        return authors
    unknown = selected - set(authors)
    if unknown:
        raise ValueError(
            f"--only-authors contains author(s) outside the {phase} roster: "
            f"{sorted(unknown)}; allowed={authors}"
        )
    if not selected:
        raise ValueError("--only-authors resolved to an empty shard")
    return [author for author in authors if author in selected]


def calibration_commands(
    cfg: dict,
    models: list[dict],
    output_root: Path,
    selected_authors: set[int] | None = None,
):
    phase = cfg["calibration"]
    authors = _filter_authors(phase["authors"], selected_authors, "calibration")
    for model, author, seed, objective in itertools.product(
        models, authors, phase["seeds"], phase["objective_grid"]
    ):
        for index, setting in enumerate(phase["objective_grid"][objective]):
            setting_id = setting.get("id", f"g{index:02d}")
            out = output_root / "calibration" / model["id"] / _request_dirname(cfg, author) / f"seed-{seed}" / objective / setting_id
            cmd = _common_gate_args(cfg, model, author, seed, out, "calibration")
            sft_cache = output_root / "sft_cache" / model["id"] / f"{_request_dirname(cfg, author)}_seed-{seed}.pt"
            cmd += [
                "--generators", objective,
                "--predictors", "",
                "--gen-lr", str(setting["lr"]),
                "--gen-steps", str(setting["steps"]),
                "--sft-cache", str(sft_cache),
            ]
            if setting.get("beta") is not None:
                cmd += ["--beta", str(setting["beta"])]
            if setting.get("forget_weight") is not None:
                cmd += ["--gen-forget-weight-per", f"{objective}={setting['forget_weight']}"]
            if setting.get("retain_weight") is not None:
                cmd += ["--gen-retain-weight-per", f"{objective}={setting['retain_weight']}"]
            if setting.get("rmu_alpha") is not None:
                cmd += ["--gen-rmu-alpha-per", f"{objective}={setting['rmu_alpha']}"]
            if setting.get("rmu_c") is not None:
                cmd += ["--gen-rmu-c-per", f"{objective}={setting['rmu_c']}"]
            yield out, cmd


def fidelity_commands(cfg: dict, models: list[dict], output_root: Path):
    common = cfg["common"]
    phase = cfg["fidelity"]
    declared = cfg["audit"]["fidelity_certificates"]
    for model in models:
        csv_path = output_root / "fidelity" / f"{model['id']}.csv"
        certificate = Path(declared[model["id"]])
        if not certificate.is_absolute():
            certificate = (ROOT / certificate).resolve()
        cmd = [
            sys.executable,
            str(FIDELITY),
            "--dataset", str(cfg.get("dataset", "tofu")),
            "--model", str(model["path"]),
            "--device", str(common.get("device", "cuda")),
            "--dtype", str(common["dtype"]),
            "--author", str(phase["author"]),
            "--universe-authors", str(common["universe_authors"]),
            "--candidate-authors", str(common["candidate_author_pools"]["calibration"]),
            "--n-cands", str(phase["n_candidates"]),
            "--candidate-seed", str(phase["candidate_seed"]),
            "--batch-size", str(common["batch_size"]),
            "--block-last-n", str(common["block_last_n"]),
            "--dirs", str(common["probe_dirs"]),
            "--etas", str(common["probe_norm_eta"]),
            "--seeds", str(common["probe_seed"]),
            "--gate-r", str(common["probe_dirs"]),
            "--gate-eta", str(common["probe_norm_eta"]),
            "--gate-seed", str(common["probe_seed"]),
            "--min-rho-ab", str(phase["min_rho_ab"]),
            "--min-rho-bc", str(phase["min_rho_bc"]),
            "--min-rho-ac", str(phase["min_rho_ac"]),
            "--min-eff-ratio", str(phase["min_eff_ratio"]),
            "--min-frac-changed", str(phase["min_frac_changed"]),
            "--out", str(csv_path),
            "--certificate", str(certificate),
            "--enforce-gate",
        ]
        yield csv_path, certificate, cmd


def _validate_settings(settings: dict, expected: set[str], label: str) -> None:
    if set(settings) != expected:
        raise ValueError(
            f"objective freeze mismatch for {label}: expected {sorted(expected)}, "
            f"got {sorted(settings)}"
        )
    required = {"lr", "steps"}
    for objective, spec in settings.items():
        missing = required - set(spec)
        if missing or any(spec.get(key) is None for key in required):
            raise ValueError(f"{label}/{objective} lacks frozen lr/steps")


def _load_freeze(config_path: Path, cfg: dict, models: list[dict]) -> tuple[Path, dict]:
    freeze_path = Path(cfg["audit"]["objective_freeze"])
    if not freeze_path.is_absolute():
        freeze_path = (config_path.parent / freeze_path).resolve()
    freeze = _load_yaml(freeze_path)
    if freeze.get("status") != "frozen" or not freeze.get("frozen_before_audit"):
        raise RuntimeError(
            f"refusing audit: {freeze_path} is not marked status=frozen and "
            "frozen_before_audit=true"
        )
    if freeze.get("source_campaign") != cfg["campaign_id"]:
        raise ValueError(
            f"objective freeze belongs to {freeze.get('source_campaign')!r}, "
            f"not {cfg['campaign_id']!r}"
        )
    if not freeze.get("frozen_at_utc"):
        raise ValueError("objective freeze must record frozen_at_utc before audit")
    if freeze.get("unresolved"):
        raise ValueError(f"objective freeze still has unresolved arms: {freeze['unresolved']}")
    expected = set(cfg["audit"]["objectives"]) | set(cfg["audit"].get("stress_objectives", []))
    if "models" in freeze:
        for model in models:
            if model["id"] not in freeze["models"]:
                raise ValueError(f"objective freeze has no settings for model {model['id']}")
            _validate_settings(freeze["models"][model["id"]], expected, model["id"])
    else:
        _validate_settings(freeze.get("objectives", {}), expected, "global")
    return freeze_path, freeze


def _load_fidelity_certificates(config_path: Path, cfg: dict, models: list[dict]) -> dict[str, dict]:
    declared = cfg["audit"].get("fidelity_certificates", {})
    development_pool = _expand_int_ranges(
        str(cfg["common"]["candidate_author_pools"]["calibration"])
    )
    certificates = {}
    for model in models:
        if model["id"] not in declared:
            raise ValueError(f"no fidelity certificate declared for model {model['id']}")
        path = Path(declared[model["id"]])
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"missing fidelity certificate for {model['id']}: {path}; run the "
                "development-only frozen-cell fidelity command before audit"
            )
        cert = json.loads(path.read_text(encoding="utf-8"))
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
            if cert.get(key) != value:
                raise ValueError(
                    f"fidelity certificate mismatch for {model['id']}/{key}: "
                    f"expected {value!r}, got {cert.get(key)!r}"
                )
        if set(cert.get("candidate_authors") or []) != development_pool:
            raise ValueError(
                f"fidelity certificate for {model['id']} did not use the frozen "
                "development candidate pool"
            )
        minimum = int(cfg.get("fidelity", {}).get("n_candidates", 128))
        if int(cert.get("n_candidates", 0)) < minimum:
            raise ValueError(
                f"fidelity certificate for {model['id']} has too few candidates: "
                f"{cert.get('n_candidates')} < {minimum}"
            )
        certificates[model["id"]] = {
            "path": str(path),
            "sha256": _sha256(path),
            "payload": cert,
        }
    return certificates


def audit_commands(
    config_path: Path,
    cfg: dict,
    models: list[dict],
    output_root: Path,
    selected_authors: set[int] | None = None,
):
    freeze_path, freeze = _load_freeze(config_path, cfg, models)
    fidelity = _load_fidelity_certificates(config_path, cfg, models)
    phase = cfg["audit"]
    authors = _filter_authors(phase["authors"], selected_authors, "audit")
    core_objectives = phase["objectives"]
    stress_objectives = phase.get("stress_objectives", [])
    objectives = core_objectives + stress_objectives
    predictors = phase["predictors"]
    for model, author, seed in itertools.product(models, authors, phase["seeds"]):
        settings = (freeze["models"][model["id"]]
                    if "models" in freeze else freeze["objectives"])
        out = output_root / "audit" / model["id"] / _request_dirname(cfg, author) / f"seed-{seed}"
        cmd = _common_gate_args(cfg, model, author, seed, out, "audit")
        cmd += [
            "--generators", _csv(objectives),
            "--predictors", _csv(predictors),
            "--gen-lr-per", _kv(settings, "lr"),
            "--gen-steps-per", _kv(settings, "steps"),
            "--gen-beta-per", _kv(settings, "beta"),
            "--gen-forget-weight-per", _kv(settings, "forget_weight"),
            "--gen-retain-weight-per", _kv(settings, "retain_weight"),
            "--gen-rmu-alpha-per", _kv(settings, "rmu_alpha"),
            "--gen-rmu-c-per", _kv(settings, "rmu_c"),
            "--require-all-predictors",
        ]
        metadata = {
            "campaign_id": cfg["campaign_id"],
            "campaign_phase": "audit",
            "objective_freeze": str(freeze_path),
            "objective_freeze_id": freeze.get("freeze_id"),
            "objective_freeze_sha256": _sha256(freeze_path),
            "campaign_config_sha256": _sha256(config_path),
            "objective_acceptance_rule": cfg["calibration"]["selection"],
            "core_objectives": core_objectives,
            "stress_objectives": stress_objectives,
            "fidelity_certificate": fidelity[model["id"]]["path"],
            "fidelity_certificate_sha256": fidelity[model["id"]]["sha256"],
        }
        yield out, cmd, metadata


def _trajectories_complete(out: Path, objectives: list[str]) -> bool:
    return all((out / f"traj_{name}" / "DONE").exists() for name in objectives)


def _predictors_opened(out: Path) -> bool:
    manifest_path = out / "run_manifest.json"
    ledger_path = out / "seal_ledger.jsonl"
    if not manifest_path.exists() or not ledger_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    opened = {(row.get("request"), row.get("scorer")) for row in entries
              if row.get("status") == "opened"}
    return all((manifest["request"], predictor) in opened
               for predictor in manifest["predictors"])


def _complete(out: Path, objectives: list[str], audit: bool) -> bool:
    if not _trajectories_complete(out, objectives):
        return False
    return not audit or (_predictors_opened(out) and (out / "channel_report.csv").exists())


def _has_artifacts(out: Path) -> bool:
    return out.exists() and any(out.iterdir())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--phase", required=True, choices=["fidelity", "calibration", "audit"])
    p.add_argument("--model-id", action="append", default=[], help="run only this model alias")
    p.add_argument(
        "--only-authors",
        default="",
        help=(
            "execution-only deletion-request shard (comma list/ranges); must be an "
            "exact subset of the phase's frozen author roster"
        ),
    )
    p.add_argument(
        "--only-objectives",
        default="",
        help=(
            "execution-only calibration shard (comma list); must be a subset of "
            "the configured objective grid — lets one queue unit per objective "
            "spread a calibration wave across GPUs"
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="skip complete run directories")
    p.add_argument("--limit", type=int, default=0, help="execute/print at most N runs (0=all)")
    a = p.parse_args()

    config_path = Path(a.config).resolve()
    cfg = _load_yaml(config_path)
    _validate_campaign(cfg)
    models = _enabled_models(cfg, set(a.model_id))
    if a.phase == "fidelity" and a.only_authors:
        p.error("--only-authors does not apply to the single frozen fidelity cell")
    selected_authors = (
        _expand_int_ranges(a.only_authors) if a.only_authors else None
    )
    selected_objectives = {
        part.strip() for part in a.only_objectives.split(",") if part.strip()
    } or None
    if selected_objectives is not None:
        if a.phase != "calibration":
            p.error("--only-objectives applies only to the calibration phase")
        grid_objectives = set(cfg["calibration"]["objective_grid"])
        unknown_objectives = selected_objectives - grid_objectives
        if unknown_objectives:
            p.error(
                f"--only-objectives outside the configured grid: {sorted(unknown_objectives)}; "
                f"allowed={sorted(grid_objectives)}"
            )
    output_root = Path(cfg["output_root"])
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    if a.phase == "audit" and cfg["audit"].get("offline", True):
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    git_state = _git_state()

    if a.phase == "audit" and not a.dry_run and git_state["code_dirty"]:
        raise RuntimeError(
            "refusing sealed audit from a dirty worktree; commit the campaign config, "
            "objective freeze, and code first"
        )

    if not a.dry_run:
        for model in models:
            path = Path(model["path"])
            if path.is_absolute() and not path.exists():
                raise FileNotFoundError(
                    f"model {model['id']} is enabled but missing at {path}; provision it or set enabled:false"
                )

    n = 0
    if a.phase == "fidelity":
        for csv_path, certificate, cmd in fidelity_commands(cfg, models, output_root):
            if a.resume and csv_path.exists() and certificate.exists():
                payload = json.loads(certificate.read_text(encoding="utf-8"))
                if payload.get("passed"):
                    print(f"SKIP passed fidelity certificate: {certificate}")
                    continue
            if not a.dry_run and (csv_path.exists() or certificate.exists()):
                raise RuntimeError(
                    f"pre-existing fidelity artifact for {csv_path.stem}; preserve it and "
                    "choose a new campaign/output_root rather than overwriting"
                )
            _run(cmd, a.dry_run, env)
            n += 1
            if a.limit and n >= a.limit:
                break
    elif a.phase == "calibration":
        for out, cmd in calibration_commands(
            cfg, models, output_root, selected_authors=selected_authors
        ):
            objective = cmd[cmd.index("--generators") + 1]
            if selected_objectives is not None and objective not in selected_objectives:
                continue
            if a.resume and _complete(out, [objective], audit=False):
                print(f"SKIP complete: {out}")
                continue
            if _has_artifacts(out):
                raise RuntimeError(
                    f"partial or pre-existing calibration directory: {out}. Preserve it for "
                    "forensics and choose a new campaign/output_root; do not overwrite it."
                )
            _run(cmd, a.dry_run, env)
            if not a.dry_run:
                _annotate_manifest(out, {
                    "campaign_id": cfg["campaign_id"],
                    "campaign_phase": "calibration",
                    "campaign_config_sha256": _sha256(config_path),
                    "objective_acceptance_rule": cfg["calibration"]["selection"],
                    **git_state,
                })
            n += 1
            if a.limit and n >= a.limit:
                break
    else:
        for out, cmd, metadata in audit_commands(
            config_path, cfg, models, output_root,
            selected_authors=selected_authors,
        ):
            objectives = cfg["audit"]["objectives"] + cfg["audit"].get("stress_objectives", [])
            if a.resume and _complete(out, objectives, audit=True):
                print(f"SKIP complete: {out}")
                continue
            if (a.resume and _trajectories_complete(out, objectives)
                    and _predictors_opened(out)):
                print(f"RESUME report-only: {out}")
                _run(
                    [sys.executable, str(REPORT), "--run-dir", str(out), "--n-boot", "2000"],
                    a.dry_run,
                    env,
                )
                n += 1
                if a.limit and n >= a.limit:
                    break
                continue
            if _has_artifacts(out):
                raise RuntimeError(
                    f"partial or pre-existing sealed run: {out}. The append-only seal cannot "
                    "be overwritten; preserve it and choose a new campaign/output_root."
                )
            _run(cmd, a.dry_run, env)
            if not a.dry_run:
                _annotate_manifest(out, {**metadata, **git_state})
                _run(
                    [sys.executable, str(REPORT), "--run-dir", str(out), "--n-boot", "2000"],
                    False,
                    env,
                )
            n += 1
            if a.limit and n >= a.limit:
                break

    print(f"campaign phase={a.phase}: {'planned' if a.dry_run else 'completed'} {n} run(s)")


if __name__ == "__main__":
    main()
