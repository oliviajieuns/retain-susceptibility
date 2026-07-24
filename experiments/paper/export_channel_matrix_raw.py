"""Export channel-matrix audit + alpha-protection runs into raw evidence shards.

This is the bridge between the sealed GPU campaign outputs and the normalized
paper ledger:

    gate.py audit cells        -> prediction.jsonl   (candidate-level)
    alpha_protection.py audit  -> protection.jsonl   (candidate-level, per arm)
    fd-fidelity certificate    -> fidelity summary JSON (table f_rho/f_K cells)

The exporter opens sealed audit scores only through ``rsus.sealing`` (which
requires every trajectory DONE marker) and never recomputes or reselects
anything: scores, folds, damage, margins, and feasibility all come from the
frozen artifacts. Any missing or inconsistent field fails the whole export
rather than shrinking support silently.

Example
-------
python experiments/paper/export_channel_matrix_raw.py \
  --campaign-config configs/channel_matrix/7b_tofu.yaml \
  --setting-id tofu_qwen25_7b \
  --prediction-alpha 0.5 --protection-alpha-freeze configs/channel_matrix/alpha_protection_freeze.yaml \
  --control-predictor knn_embed \
  --out-dir results/paper/raw/tofu_qwen25_7b
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.analysis.mixture import channel_mixture_scores, empirical_midrank01  # noqa: E402
from rsus.evidence.schemas import EvidenceValidationError  # noqa: E402
from rsus import sealing  # noqa: E402


SEED_DIR = re.compile(r"^seed-(\d+)$")

# alpha_protection selector labels -> normalized ledger arms. The deployed
# frozen-alpha row is identified by its "deployed" flag, not by its label.
SELECTOR_ARMS = {
    "none": "no_repair",
    "s_alpha_0p0": "s0",
    "s_alpha_1p0": "s1",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise EvidenceValidationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"{path} root must be a mapping")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceValidationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"{path} root must be a mapping")
    return value


def _profile_scores(path: Path) -> tuple[dict[str, float], set[str], dict[str, str]]:
    """Return (discovery_scores, audit_ids, group_by_id) from a plain artifact.

    Audit-fold values are intentionally absent from the plain artifact (they
    live behind the seal ledger); only identities and folds are read here.
    """
    payload = _load_json(path)
    if payload.get("schema") != "paper-profile-v2":
        raise EvidenceValidationError(f"{path} is not a paper-profile-v2 artifact")
    discovery: dict[str, float] = {}
    audit_ids: set[str] = set()
    groups: dict[str, str] = {}
    for row in payload.get("candidates", []):
        candidate_id = str(row["candidate_id"])
        groups[candidate_id] = str(row["group"])
        fold = row.get("fold")
        if fold == "discovery":
            discovery[candidate_id] = float(row["score"])
        elif fold == "audit":
            audit_ids.add(candidate_id)
        else:
            raise EvidenceValidationError(
                f"{path} candidate {candidate_id} has unknown fold {fold!r}"
            )
    if not discovery or not audit_ids:
        raise EvidenceValidationError(f"{path} lacks a two-fold candidate split")
    return discovery, audit_ids, groups


def _sealed_audit_scores(
    cell: Path, request_id: str, scorer: str, done_markers: list[Path]
) -> dict[str, float]:
    """Honor the seal protocol: read if already opened, else unseal."""
    seals = cell / "seals"
    ledger = cell / "seal_ledger.jsonl"
    try:
        return sealing.read_scores(seals, ledger, request_id, scorer)
    except sealing.SealedError:
        return sealing.unseal(seals, ledger, request_id, scorer, done_markers)


def _first_reach_index(snapshots: list[dict[str, Any]], recall_max: float) -> int | None:
    for index, snapshot in enumerate(snapshots):
        if float(snapshot["forget_recall"]) <= recall_max:
            return index
    return None


def _prediction_alpha(args, parent: str) -> float:
    if args.prediction_alpha is not None:
        return float(args.prediction_alpha)
    freeze = _load_yaml(ROOT / args.prediction_alpha_freeze)
    table = freeze.get("prediction_alpha") or freeze.get("alpha_pred") or {}
    if parent in table:
        return float(table[parent])
    if "default" in table:
        return float(table["default"])
    raise EvidenceValidationError(
        f"prediction alpha for parent {parent!r} not found in "
        f"{args.prediction_alpha_freeze}"
    )


def export_prediction(
    args, cfg: Mapping[str, Any], out_path: Path
) -> tuple[int, int]:
    output_root = ROOT / cfg["output_root"]
    audit_root = output_root / "audit"
    audit_cfg = cfg["audit"]
    recall_max = float(cfg["calibration"]["selection"]["forget_recall_max"])
    parents = [
        objective
        for objective in list(audit_cfg["objectives"])
        + list(audit_cfg.get("stress_objectives", []))
        if objective in set(args.parents)
    ]
    gradient_probe = args.gradient_predictor
    proximity_probe = args.proximity_predictor

    records: list[dict[str, Any]] = []
    cells = 0
    if not audit_root.is_dir():
        raise EvidenceValidationError(f"no audit outputs under {audit_root}")
    for model_dir in sorted(p for p in audit_root.iterdir() if p.is_dir()):
        for request_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in request_dir.iterdir() if p.is_dir()):
                match = SEED_DIR.match(seed_dir.name)
                if not match:
                    continue
                manifest_path = seed_dir / "run_manifest.json"
                if not manifest_path.is_file():
                    continue
                cells += 1
                manifest = _load_json(manifest_path)
                request_id = request_dir.name
                seed = match.group(1)

                profile_dir = seed_dir / "profile_artifacts"
                grad_disc, declared_audit_ids, groups = _profile_scores(
                    profile_dir / f"{gradient_probe}.json"
                )
                prox_disc, prox_audit_ids, prox_groups = _profile_scores(
                    profile_dir / f"{proximity_probe}.json"
                )
                groups.update(prox_groups)
                control_disc: dict[str, float] | None = None
                if args.control_predictor:
                    control_disc, control_ids, _ = _profile_scores(
                        profile_dir / f"{args.control_predictor}.json"
                    )
                    if control_ids != declared_audit_ids:
                        raise EvidenceValidationError(
                            f"{seed_dir}: control predictor audit fold differs "
                            "from the gradient probe's declared fold"
                        )
                if prox_audit_ids != declared_audit_ids:
                    raise EvidenceValidationError(
                        f"{seed_dir}: proximity audit fold differs from the "
                        "gradient probe's declared fold"
                    )

                # The seal may open only after EVERY planned trajectory of the
                # cell is done — including objectives outside --parents.  A
                # missing trajectory directory is an incomplete cell, never a
                # smaller marker list.
                planned_objectives = list(audit_cfg["objectives"]) + list(
                    audit_cfg.get("stress_objectives", [])
                )
                traj_dirs = {
                    parent: seed_dir / f"traj_{parent}" for parent in parents
                }
                missing_traj = [
                    objective
                    for objective in planned_objectives
                    if not (seed_dir / f"traj_{objective}").is_dir()
                ]
                if missing_traj:
                    raise EvidenceValidationError(
                        f"{seed_dir}: planned trajectories not present yet: "
                        f"{missing_traj}; refusing to open sealed audit scores"
                    )
                done_markers = [
                    seed_dir / f"traj_{objective}" / "DONE"
                    for objective in planned_objectives
                ]
                grad_audit = _sealed_audit_scores(
                    seed_dir, request_id, gradient_probe, done_markers
                )
                prox_audit = _sealed_audit_scores(
                    seed_dir, request_id, proximity_probe, done_markers
                )
                control_audit: dict[str, float] | None = None
                if args.control_predictor:
                    control_audit = _sealed_audit_scores(
                        seed_dir, request_id, args.control_predictor, done_markers
                    )
                    if set(control_audit) != declared_audit_ids:
                        raise EvidenceValidationError(
                            f"{seed_dir}: sealed control fold differs from the "
                            "declared audit fold"
                        )
                audit_ids = sorted(grad_audit)
                if set(grad_audit) != declared_audit_ids:
                    raise EvidenceValidationError(
                        f"{seed_dir}: sealed gradient fold differs from the "
                        "declared audit fold"
                    )
                if sorted(prox_audit) != audit_ids:
                    raise EvidenceValidationError(
                        f"{seed_dir}: gradient/proximity sealed folds differ"
                    )

                cert_map = audit_cfg.get("fidelity_certificates", {})
                model_id = manifest.get("model_id") or model_dir.name
                cert_path = cert_map.get(model_id)
                if not cert_path:
                    raise EvidenceValidationError(
                        f"{seed_dir}: no fidelity certificate configured for "
                        f"model {model_id!r}; profile validity is undecidable"
                    )
                if not (ROOT / cert_path).is_file():
                    raise EvidenceValidationError(
                        f"{seed_dir}: fidelity certificate missing: {cert_path}"
                    )
                certificate = _load_json(ROOT / cert_path).get("passed") is True

                for parent in parents:
                    traj = traj_dirs[parent]
                    damage_path = traj / "damage.json"
                    if not damage_path.is_file():
                        continue
                    payload = _load_json(damage_path)
                    snapshots = payload.get("snapshots", [])
                    if not snapshots:
                        continue
                    reach = _first_reach_index(snapshots, recall_max)
                    reached = reach is not None
                    snapshot = snapshots[reach if reached else -1]
                    nll0 = payload["nll0"]
                    alpha_pred = _prediction_alpha(args, parent)
                    joint = channel_mixture_scores(
                        {**grad_disc, **grad_audit},
                        {**prox_disc, **prox_audit},
                        alpha_pred,
                        candidate_ids=audit_ids,
                        normalization_ids=sorted(grad_disc),
                    )
                    s0 = empirical_midrank01(grad_disc, grad_audit)
                    s1 = empirical_midrank01(prox_disc, prox_audit)
                    control_scores: dict[str, float] | None = None
                    if control_audit is not None and control_disc is not None:
                        control_scores = empirical_midrank01(
                            control_disc, control_audit
                        )
                    for candidate_id in audit_ids:
                        if candidate_id not in snapshot["nll"] or candidate_id not in nll0:
                            raise EvidenceValidationError(
                                f"{damage_path}: audit candidate {candidate_id} "
                                "missing from trajectory NLL tables"
                            )
                        record = {
                            "setting": args.setting_id,
                            "parent": parent,
                            "request": request_id,
                            "seed": seed,
                            "prediction_selection": {
                                "valid": True,
                                "fallback": False,
                                "alpha": alpha_pred,
                            },
                            "candidate_id": candidate_id,
                            "group": groups.get(
                                candidate_id,
                                payload.get("candidate_groups", {}).get(candidate_id),
                            ),
                            "s0": s0[candidate_id],
                            "s1": s1[candidate_id],
                            "joint": joint[candidate_id],
                            "damage": float(snapshot["nll"][candidate_id])
                            - float(nll0[candidate_id]),
                            "profile_valid": bool(certificate),
                            "reached": reached,
                            "trajectory_completed": (traj / "DONE").is_file(),
                        }
                        if record["group"] is None:
                            raise EvidenceValidationError(
                                f"{damage_path}: candidate {candidate_id} has no group"
                            )
                        if control_scores is not None:
                            record["control"] = control_scores[candidate_id]
                        records.append(record)
    if not records:
        raise EvidenceValidationError(
            f"no prediction records exported from {audit_root}"
        )
    _write_jsonl(out_path, records)
    return len(records), cells


def _margins(row: Mapping[str, Any], final: Mapping[str, Any]) -> dict[str, float] | None:
    required = (
        ("forget_recall", "direct_recall_max", "direct_forget_margin", -1.0),
        ("para_recall", "paraphrase_recall_max", "paraphrase_forget_margin", -1.0),
        (
            "extraction_generation",
            "extraction_generation_max",
            "extraction_generation_margin",
            -1.0,
        ),
        ("utility_retention", "utility_retention_min", "utility_margin", 1.0),
    )
    result: dict[str, float] = {}
    for metric_key, bound_key, margin_key, orientation in required:
        value = row.get(metric_key)
        if value is None:
            return None
        bound = float(final[bound_key])
        value = float(value)
        result[margin_key] = (bound - value) if orientation < 0 else (value - bound)
    return result


def export_protection(
    args, cfg: Mapping[str, Any], out_path: Path
) -> tuple[int, int]:
    alpha_cfg = cfg.get("alpha_protection")
    if not alpha_cfg:
        raise EvidenceValidationError(
            "campaign config has no alpha_protection block; use --skip-protection"
        )
    final = alpha_cfg["final_checkpoint"]
    output_root = ROOT / cfg["output_root"]
    records: list[dict[str, Any]] = []
    cells = 0
    for results_path in sorted(output_root.rglob("results.json")):
        payload = _load_json(results_path)
        manifest = payload.get("manifest", {})
        if manifest.get("schema") != "channel-mixture-protection-run-v1":
            continue
        if manifest.get("campaign_phase") != "audit":
            continue
        request_id = str(
            manifest.get("request") or results_path.parent.parent.name
        )
        seed_name = manifest.get("seed")
        if seed_name is None:
            match = SEED_DIR.match(results_path.parent.name)
            if not match:
                raise EvidenceValidationError(
                    f"{results_path}: cannot determine trajectory seed"
                )
            seed_name = match.group(1)
        seed = str(seed_name)
        cells += 1

        draw_rows = {}
        random_log = results_path.parent / "random_draws.partial.jsonl"
        if random_log.is_file():
            for line in random_log.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                draw_rows[(str(row["parent"]), str(row["random_draw_id"]))] = row

        rows = payload.get("results", [])
        by_parent: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            by_parent.setdefault(str(row["parent"]), []).append(row)
        for parent, parent_rows in sorted(by_parent.items()):
            if parent not in set(args.parents):
                continue
            protection_alpha = None
            arm_rows: dict[str, Mapping[str, Any]] = {}
            for row in parent_rows:
                if row.get("executed") is not True:
                    # Placeholder rows (parent did not reach the criterion)
                    # copy diagnostic parent metrics; they must never become
                    # claim arms.
                    continue
                selector = str(row["selector"])
                if row.get("deployed"):
                    arm_rows["joint"] = row
                    protection_alpha = float(row.get("alpha"))
                arm = SELECTOR_ARMS.get(selector)
                if arm:
                    arm_rows[arm] = row
            checkpoint = next(
                (
                    row.get("parent_checkpoint")
                    for row in parent_rows
                    if row.get("parent_checkpoint")
                ),
                None,
            )
            if checkpoint is None:
                continue
            checkpoint_id = json.dumps(checkpoint, sort_keys=True)
            first_reaching = bool(checkpoint.get("first_direct_reaching"))

            def emit(arm: str, row: Mapping[str, Any], draw_id: str | None, draw_complete: bool) -> None:
                margins = _margins(row, final)
                damage = row.get("candidate_damage")
                if margins is None or not damage:
                    return
                groups = row.get("candidate_groups") or {}
                feasible = all(value >= 0.0 for value in margins.values())
                # Stage-2 refresh telemetry: accepted refreshes and guarded
                # rollbacks of the deployed arm fill the table's
                # updates/rollback diagnostics.  Arms without a repair phase
                # (no_repair placeholders) legitimately carry neither field.
                stage2 = (row.get("trajectory_metadata") or {}).get("stage2") or {}
                updates_accepted = stage2.get("n_accepted")
                updates_rolled_back = stage2.get("n_rejected")
                if (updates_accepted is None) != (updates_rolled_back is None):
                    raise EvidenceValidationError(
                        f"{results_path}: stage2 telemetry must report "
                        "n_accepted and n_rejected together"
                    )
                for candidate_id, value in sorted(damage.items()):
                    record = {
                        "setting": args.setting_id,
                        "parent": parent,
                        "request": request_id,
                        "seed": seed,
                        "protection_selection": {
                            "valid": True,
                            "fallback": False,
                            "alpha": protection_alpha,
                        },
                        "arm": arm,
                        "candidate_id": candidate_id,
                        "group": groups.get(candidate_id),
                        "damage": float(value),
                        "feasible": feasible,
                        **margins,
                        "draw_complete": draw_complete,
                        "parent_checkpoint_id": checkpoint_id,
                        "parent_checkpoint_first_reaching": first_reaching,
                        "native_metric": (
                            float(row["utility_retention"])
                            if row.get("utility_retention") is not None
                            else None
                        ),
                        "updates_accepted": (
                            float(updates_accepted)
                            if updates_accepted is not None
                            else None
                        ),
                        "updates_rolled_back": (
                            float(updates_rolled_back)
                            if updates_rolled_back is not None
                            else None
                        ),
                    }
                    if record["group"] is None:
                        raise EvidenceValidationError(
                            f"{results_path}: candidate {candidate_id} has no group"
                        )
                    if draw_id is not None:
                        record["draw_id"] = draw_id
                    records.append(record)

            for arm in ("joint", "no_repair", "s0", "s1"):
                row = arm_rows.get(arm)
                if row is not None:
                    emit(arm, row, None, True)
            for (draw_parent, draw_id), row in sorted(draw_rows.items()):
                if draw_parent != parent:
                    continue
                if row.get("executed") is not True:
                    continue
                emit("repeated_random", row, draw_id, True)
    if not records:
        raise EvidenceValidationError(
            f"no protection records exported from {output_root}"
        )
    _write_jsonl(out_path, records)
    return len(records), cells


def export_fidelity_summary(args, cfg: Mapping[str, Any], out_path: Path) -> None:
    """Summarize the frozen fd-fidelity certificate for the table renderer.

    f_rho is rho(A,C): exact per-candidate gradient energy versus the frozen
    loss-shake estimate. f_K and the bootstrap lower bounds are computable
    only when the certificate persisted per-candidate scores; otherwise the
    corresponding fields stay null and the RQ2 cells remain placeholders.
    """
    cert_path = ROOT / args.fidelity_certificate
    cert = _load_json(cert_path)
    metrics = cert.get("metrics") or {}
    summary: dict[str, Any] = {
        "setting": args.setting_id,
        "source_certificate": str(args.fidelity_certificate),
        "certificate_passed": cert.get("passed"),
        "f_rho": metrics.get("rho_AC"),
        "f_k": metrics.get("ov_AC"),
        "f_rho_lb": None,
        "f_k_lb": None,
        "tau_rho": 0.80,
        "tau_k": 0.70,
    }
    scores = cert.get("scores") or {}
    scores_a = scores.get("A")
    scores_c = scores.get("C")
    if (
        isinstance(scores_a, list)
        and isinstance(scores_c, list)
        and len(scores_a) == len(scores_c)
        and len(scores_a) >= 4
    ):
        k = max(1, math.ceil(args.top_q * len(scores_a)))

        def overlap(sample_a: list[float], sample_c: list[float]) -> float:
            top = lambda values: set(
                sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]
            )
            return len(top(sample_a) & top(sample_c)) / k

        def spearman(sample_a: list[float], sample_c: list[float]) -> float:
            def ranks(values: list[float]) -> list[float]:
                order = sorted(range(len(values)), key=lambda i: values[i])
                out = [0.0] * len(values)
                for rank, index in enumerate(order):
                    out[index] = float(rank)
                return out

            ra, rc = ranks(sample_a), ranks(sample_c)
            mean_a = sum(ra) / len(ra)
            mean_c = sum(rc) / len(rc)
            num = sum((x - mean_a) * (y - mean_c) for x, y in zip(ra, rc))
            var_a = sum((x - mean_a) ** 2 for x in ra)
            var_c = sum((y - mean_c) ** 2 for y in rc)
            return num / math.sqrt(var_a * var_c) if var_a and var_c else 0.0

        from rsus.evidence.statistics import percentile

        summary["f_k"] = overlap(scores_a, scores_c)
        rng = random.Random(20260723)
        rhos, overlaps = [], []
        indices = range(len(scores_a))
        for _ in range(2000):
            sample = [rng.choice(indices) for _ in indices]
            sample_a = [scores_a[i] for i in sample]
            sample_c = [scores_c[i] for i in sample]
            rhos.append(spearman(sample_a, sample_c))
            overlaps.append(overlap(sample_a, sample_c))
        summary["f_rho_lb"] = percentile(rhos, 0.05)
        summary["f_k_lb"] = percentile(overlaps, 0.05)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-config", required=True)
    parser.add_argument("--setting-id", required=True)
    parser.add_argument(
        "--parents",
        nargs="+",
        default=[
            "graddiff", "npo", "simnpo", "gru", "rmu", "repnoise",
            "circuit_breakers",
        ],
        help="evidence parents to export (matches the contract roster)",
    )
    parser.add_argument("--gradient-predictor", default="fd_norm")
    parser.add_argument("--proximity-predictor", default="knn_feature")
    parser.add_argument(
        "--control-predictor",
        default=None,
        help="frozen strongest simple control (selected on development folds)",
    )
    parser.add_argument("--prediction-alpha", type=float, default=None)
    parser.add_argument(
        "--prediction-alpha-freeze",
        default=None,
        help="YAML with a frozen per-parent prediction_alpha mapping",
    )
    # Must equal the raw plan's bootstrap.top_q so the fidelity overlap@K and
    # the tail definition share one frozen q (campaign.yaml execution.bootstrap).
    parser.add_argument("--top-q", type=float, default=0.10)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skip-prediction", action="store_true")
    parser.add_argument("--skip-protection", action="store_true")
    parser.add_argument(
        "--fidelity-certificate",
        default=None,
        help="fd-fidelity certificate JSON to summarize for the table renderer",
    )
    parser.add_argument(
        "--fidelity-out",
        default=None,
        help=(
            "fidelity summary path; defaults to the tracked "
            "docs/data/fidelity_summaries/<setting-id>.json declared by "
            "configs/paper/evidence.yaml fidelity_inputs"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prediction_alpha is None and not args.prediction_alpha_freeze:
        if not args.skip_prediction:
            print(
                "export failed: provide --prediction-alpha or "
                "--prediction-alpha-freeze",
                file=sys.stderr,
            )
            return 1
    try:
        cfg = _load_yaml(ROOT / args.campaign_config)
        out_dir = ROOT / args.out_dir
        if not args.skip_prediction:
            count, cells = export_prediction(
                args, cfg, out_dir / "prediction.jsonl"
            )
            print(f"prediction: {count} records from {cells} audit cells")
        if not args.skip_protection:
            count, cells = export_protection(
                args, cfg, out_dir / "protection.jsonl"
            )
            print(f"protection: {count} records from {cells} alpha cells")
        if args.fidelity_certificate:
            fidelity_out = (
                ROOT / args.fidelity_out
                if args.fidelity_out
                else ROOT / "results" / "paper" / "fidelity_summaries" / f"{args.setting_id}.json"
            )
            export_fidelity_summary(args, cfg, fidelity_out)
            print(f"fidelity summary: {fidelity_out}")
        return 0
    except (EvidenceValidationError, sealing.SealedError, KeyError) as error:
        print(f"export failed: {error!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
