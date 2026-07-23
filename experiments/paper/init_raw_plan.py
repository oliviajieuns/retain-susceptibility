"""Freeze the immutable raw-evidence denominator from paper campaign configs.

The command runs after development-only prediction/protection selections have
been frozen and before any target outcome is opened. It expands exact target
request/seed/parent units and all five appendix evidence-block contracts into the
JSON consumed by :mod:`experiments.paper.aggregate_raw`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.evidence.raw import raw_plan_from_mapping  # noqa: E402
from rsus.evidence.schemas import EvidenceValidationError, Selection  # noqa: E402


FIDELITY_PROTOCOL_FIELDS = (
    "directions",
    "repeats",
    "block_last_n",
    "norm_eta",
    "batch_size",
    "author",
    "candidate_authors",
    "n_candidates",
    "candidate_seed",
    "seed",
    "k",
    "min_rho",
    "min_overlap",
    "min_split_half",
    "min_perturbation_survival",
)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceValidationError(f"missing contract: {path}")
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise EvidenceValidationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"contract root must be a mapping: {path}")
    return value


def _selection_mapping(
    raw: object, *, name: str
) -> dict[str, object]:
    selection = Selection.from_mapping(raw, name=name)
    if selection.valid == selection.fallback:
        raise EvidenceValidationError(
            f"{name} must be exactly one of valid selection or declared fallback"
        )
    if selection.valid and selection.alpha is None:
        raise EvidenceValidationError(f"{name} valid selection requires alpha")
    return {
        "valid": selection.valid,
        "fallback": selection.fallback,
        "alpha": selection.alpha,
    }


def _fidelity_block_pattern(num_hidden_layers: int, block_last_n: int) -> str:
    if not 0 < block_last_n <= num_hidden_layers:
        raise EvidenceValidationError(
            "fidelity block_last_n must be positive and no larger than the model depth"
        )
    indices = "|".join(
        str(index)
        for index in range(num_hidden_layers - block_last_n, num_hidden_layers)
    )
    # Must remain byte-for-byte identical to mlp_down_last_layers().pattern;
    # the cost row carries this string as a frozen artifact key.
    return rf".*\.layers\.(?:{indices})\.mlp\.down_proj\.weight"


def _fidelity_protocol_sha256(raw: Mapping[str, Any]) -> str:
    protocol = {field: raw[field] for field in FIDELITY_PROTOCOL_FIELDS}
    body = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _validate_execution(execution: object) -> Mapping[str, Any]:
    if not isinstance(execution, Mapping):
        raise EvidenceValidationError("campaign.execution is missing")
    seeds = execution.get("seeds")
    draws = execution.get("repeated_random_draws")
    bootstrap = execution.get("bootstrap")
    if (
        not isinstance(seeds, list)
        or not seeds
        or len({str(seed) for seed in seeds}) != len(seeds)
    ):
        raise EvidenceValidationError(
            "campaign execution seeds must be a non-empty unique list"
        )
    if (
        not isinstance(draws, list)
        or not draws
        or any(not isinstance(draw, str) or not draw.strip() for draw in draws)
        or len(set(draws)) != len(draws)
    ):
        raise EvidenceValidationError(
            "campaign execution repeated_random_draws must be unique non-empty ids"
        )
    if not isinstance(bootstrap, Mapping):
        raise EvidenceValidationError("campaign execution bootstrap is missing")

    fidelity = execution.get("fidelity")
    if not isinstance(fidelity, Mapping):
        raise EvidenceValidationError("campaign execution fidelity is missing")
    directions = fidelity.get("directions")
    if (
        not isinstance(directions, list)
        or not directions
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 2
            or value % 2
            for value in directions
        )
        or len(set(directions)) != len(directions)
    ):
        raise EvidenceValidationError(
            "fidelity directions must be unique positive even integers"
        )
    positive_integer_fields = ("repeats", "block_last_n", "batch_size", "n_candidates")
    for field in positive_integer_fields:
        value = fidelity.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise EvidenceValidationError(
                f"campaign execution fidelity.{field} must be a positive integer"
            )
    numeric_positive_fields = ("norm_eta",)
    numeric_unit_fields = (
        "min_rho",
        "min_overlap",
        "min_split_half",
        "min_perturbation_survival",
    )
    for field in numeric_positive_fields:
        value = fidelity.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise EvidenceValidationError(
                f"campaign execution fidelity.{field} must be positive"
            )
    for field in numeric_unit_fields:
        value = fidelity.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise EvidenceValidationError(
                f"campaign execution fidelity.{field} must be in [0, 1]"
            )
    candidate_authors = fidelity.get("candidate_authors")
    if not isinstance(candidate_authors, str) or not candidate_authors.strip():
        raise EvidenceValidationError(
            "campaign execution fidelity.candidate_authors must be non-empty text"
        )
    for field in ("author", "candidate_seed", "seed", "k"):
        value = fidelity.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or (field in {"author", "k"} and value < 0)
        ):
            raise EvidenceValidationError(
                f"campaign execution fidelity.{field} must be a valid integer"
            )

    fractions = execution.get("budget_fractions")
    if (
        not isinstance(fractions, list)
        or not fractions
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) < 1.0
            for value in fractions
        )
        or len({float(value) for value in fractions}) != len(fractions)
    ):
        raise EvidenceValidationError(
            "budget_fractions must be unique numeric values in (0, 1)"
        )
    controls = execution.get("specificity_controls")
    if (
        not isinstance(controls, list)
        or not controls
        or any(not isinstance(value, str) or not value.strip() for value in controls)
        or len(set(controls)) != len(controls)
    ):
        raise EvidenceValidationError(
            "specificity_controls must be unique non-empty ids"
        )
    return execution


def selection_template(evidence: Mapping[str, Any], campaign_id: str) -> dict[str, Any]:
    settings = evidence.get("settings")
    if not isinstance(settings, list):
        raise EvidenceValidationError("evidence settings must be a list")
    return {
        "schema_version": 1,
        "status": "draft",
        "freeze_id": "PENDING-DEVELOPMENT-SELECTIONS",
        "source_campaign": campaign_id,
        "frozen_before_target": False,
        "selections": {
            str(setting["id"]): {
                str(parent): {
                    "prediction": {"valid": False, "fallback": False, "alpha": None},
                    "protection": {"valid": False, "fallback": False, "alpha": None},
                }
                for parent in setting["parents"]
            }
            for setting in settings
        },
    }


def _metric_specs() -> dict[str, dict[str, dict[str, str]]]:
    return {
        "reference_roster": {
            "rho_all": {"type": "number", "aggregate": "mean"},
            "rho_output": {"type": "number", "aggregate": "mean"},
            "rho_representation": {"type": "number", "aggregate": "mean"},
            "top_q_recall": {"type": "number", "aggregate": "mean"},
            "common_support": {"type": "integer", "aggregate": "sum"},
        },
        "construction_checks": {
            "joint_rho": {"type": "number", "aggregate": "mean"},
            "min_gain": {"type": "number", "aggregate": "mean"},
            "min_lower_bound": {"type": "number", "aggregate": "min"},
            "top_q_recall": {"type": "number", "aggregate": "mean"},
            "common_support": {"type": "integer", "aggregate": "sum"},
            "eligible": {"type": "boolean", "aggregate": "all_true"},
            "claim_pass": {"type": "boolean", "aggregate": "all_true"},
        },
        "lse_fidelity_cost": {
            "rho_exact": {"type": "number", "aggregate": "mean"},
            "overlap_k": {"type": "number", "aggregate": "mean"},
            "split_half_rho": {"type": "number", "aggregate": "range"},
            "perturbation_survival": {"type": "number", "aggregate": "min"},
            "time_seconds": {"type": "number", "aggregate": "median"},
            "peak_memory_bytes": {"type": "integer", "aggregate": "max"},
            "integrity_valid": {"type": "boolean", "aggregate": "count_true"},
            "candidate_backward": {"type": "boolean", "aggregate": "all_true"},
        },
        "protection_budget_sweep": {
            "worst_effect": {"type": "number", "aggregate": "max"},
            "max_upper_bound": {"type": "number", "aggregate": "max"},
            "bottleneck": {"type": "text", "aggregate": "first"},
            "eligible": {"type": "boolean", "aggregate": "count_true"},
            "claim_pass": {"type": "boolean", "aggregate": "count_true"},
            "min_forget_margin": {"type": "number", "aggregate": "min"},
            "min_utility_margin": {"type": "number", "aggregate": "min"},
            "accepted_updates": {"type": "integer", "aggregate": "sum"},
            "common_support": {"type": "integer", "aggregate": "sum"},
            "random_draws_complete": {"type": "boolean", "aggregate": "all_true"},
        },
        "specificity_negative_controls": {
            "rho_g": {"type": "number", "aggregate": "mean"},
            "rho_h": {"type": "number", "aggregate": "mean"},
            "rho_joint": {"type": "number", "aggregate": "mean"},
            "top_q_lift": {"type": "number", "aggregate": "mean"},
            "displacement_matched": {"type": "boolean", "aggregate": "all_true"},
            "common_support": {"type": "integer", "aggregate": "sum"},
        },
    }


def _artifact_contracts(
    evidence: Mapping[str, Any],
    campaign: Mapping[str, Any],
    selected_settings: list[Mapping[str, Any]],
    units: list[dict[str, object]],
) -> dict[str, object]:
    execution = _validate_execution(campaign.get("execution"))
    metrics = _metric_specs()
    primary = next(
        (setting for setting in selected_settings if setting.get("role") == "primary"),
        None,
    )
    scope_rows = []
    for setting in selected_settings:
        dataset = campaign["datasets"][setting["dataset"]]
        model = campaign["models"][setting["model"]]
        target = dataset["rosters"]["target"]
        setting_units = [unit for unit in units if unit["setting"] == setting["id"]]
        scope_rows.append(
            {
                "setting": setting["id"],
                "dataset_role": f"{setting['dataset']} / {setting['role']}",
                "model_precision": f"{setting['model']} / {model['dtype']}",
                "folds": {
                    name: len(dataset["rosters"][name])
                    for name in ("D_cal", "D_pred", "D_prot")
                },
                "target_requests": list(target),
                "parents": len(setting["parents"]),
                "seeds": list(execution["seeds"]),
                "candidate_counts": "from frozen request manifests",
                "planned_cells": len(setting_units),
            }
        )
    feasibility_rows = [
        {
            "audit": "direct forgetting",
            "metric": "teacher-forced answer-token recall",
            "direction": "lower",
            "boundary": "frozen setting threshold",
            "stopping_role": "parent entry + final constraint",
            "reported_slack": True,
        },
        {
            "audit": "paraphrase forgetting",
            "metric": "teacher-forced paraphrase-token recall",
            "direction": "lower",
            "boundary": "frozen setting threshold",
            "stopping_role": "final constraint",
            "reported_slack": True,
        },
        {
            "audit": "extraction / generation",
            "metric": "greedy autoregressive gold-token recall",
            "direction": "lower",
            "boundary": "frozen setting threshold",
            "stopping_role": "final constraint",
            "reported_slack": True,
        },
        {
            "audit": "general utility",
            "metric": "retention ratio",
            "direction": "higher",
            "boundary": "frozen setting floor",
            "stopping_role": "final floor",
            "reported_slack": True,
        },
        {
            "audit": "native retain",
            "metric": "mean/CVaR95 delta NLL",
            "direction": "lower",
            "boundary": "outcome only",
            "stopping_role": "none",
            "reported_slack": False,
        },
    ]
    contracts: dict[str, object] = {
        "campaign_manifest": {
            "kind": "plan_manifest",
            "scope_rows": scope_rows,
            "feasibility_rows": feasibility_rows,
        }
    }
    if primary is None:
        return contracts

    reference_rows = ["exact_energy", "directional_fd", "shuffled_proximity", "initial_nll", "random"]
    construction_rows = ["representation_layer", "pooling", "loss_shake_block", "mixture", "horizon"]
    contracts["tail_structure"] = {
        "kind": "tail_from_prediction",
        "rows": [
            {"setting": primary["id"], "parent": parent}
            for parent in primary["parents"]
        ],
        "q": execution["bootstrap"]["top_q"],
        "cvar_q": execution["bootstrap"]["cvar_q"],
        "permutations": execution["bootstrap"]["replicates"],
        "supplementary_blocks": {
            "reference_roster": {
                "key_fields": ["block", "row"],
                "group_by": ["block", "row"],
                "planned": [
                    {"block": "reference_roster", "row": row}
                    for row in reference_rows
                ],
                "metrics": metrics["reference_roster"],
            },
            "construction_checks": {
                "key_fields": ["block", "row"],
                "group_by": ["block", "row"],
                "planned": [
                    {"block": "construction_checks", "row": row}
                    for row in construction_rows
                ],
                "metrics": metrics["construction_checks"],
            },
        },
    }

    fidelity_planned = []
    fidelity = execution["fidelity"]
    repeats = range(int(fidelity["repeats"]))
    directions = [int(value) for value in fidelity["directions"]]
    block_last_n = int(fidelity["block_last_n"])
    protocol_sha256 = _fidelity_protocol_sha256(fidelity)
    for model_name, model in campaign["models"].items():
        if not model.get("provisioned"):
            continue
        depth = model.get("num_hidden_layers")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise EvidenceValidationError(
                f"campaign model {model_name} needs num_hidden_layers for fidelity keys"
            )
        block = _fidelity_block_pattern(depth, block_last_n)
        precision = model.get("dtype")
        model_source = model.get("source")
        if not isinstance(model_source, str) or not model_source.strip():
            raise EvidenceValidationError(
                f"campaign model {model_name} needs a frozen source path"
            )
        if precision not in {"float32", "bfloat16"}:
            raise EvidenceValidationError(
                f"campaign model {model_name} has unsupported fidelity dtype {precision!r}"
            )
        for repeat in repeats:
            fidelity_planned.append(
                {
                    "model": model_name,
                    "model_source": model_source,
                    "precision": precision,
                    "block": block,
                    "protocol_sha256": protocol_sha256,
                    "profiler": "exact_energy",
                    "R": 0,
                    "repeat": repeat,
                }
            )
            fidelity_planned.extend(
                {
                    "model": model_name,
                    "model_source": model_source,
                    "precision": precision,
                    "block": block,
                    "protocol_sha256": protocol_sha256,
                    "profiler": "loss_shake",
                    "R": count,
                    "repeat": repeat,
                }
                for count in directions
            )
    contracts["lse_fidelity_cost"] = {
        "kind": "measurements",
        # Non-key runner arguments remain inside the immutable plan hash even
        # though only runtime-emitted identity fields form a raw-row key.
        "protocol": dict(fidelity),
        "key_fields": [
            "model",
            "model_source",
            "precision",
            "block",
            "protocol_sha256",
            "profiler",
            "R",
            "repeat",
        ],
        "group_by": [
            "model",
            "model_source",
            "precision",
            "block",
            "protocol_sha256",
            "profiler",
            "R",
        ],
        "planned": fidelity_planned,
        "metrics": metrics["lse_fidelity_cost"],
    }
    contracts["protection_budget_sweep"] = {
        "kind": "measurements",
        "key_fields": ["fraction", "parent"],
        "group_by": ["fraction", "parent"],
        "planned": [
            {"fraction": fraction, "parent": parent}
            for fraction in execution["budget_fractions"]
            for parent in primary["parents"]
        ],
        "metrics": metrics["protection_budget_sweep"],
    }
    contracts["specificity_negative_controls"] = {
        "kind": "measurements",
        "key_fields": ["motion"],
        "group_by": ["motion"],
        "planned": [
            {"motion": motion}
            for motion in execution["specificity_controls"]
        ],
        "metrics": metrics["specificity_negative_controls"],
    }
    return contracts


def build_plan(
    evidence: Mapping[str, Any],
    campaign: Mapping[str, Any],
    freeze: Mapping[str, Any],
    *,
    setting_ids: set[str] | None = None,
) -> dict[str, Any]:
    if freeze.get("schema_version") != 1 or freeze.get("status") != "frozen":
        raise EvidenceValidationError("selection freeze must have schema_version 1 and status=frozen")
    freeze_id = freeze.get("freeze_id")
    if (
        not isinstance(freeze_id, str)
        or not freeze_id.strip()
        or freeze_id.strip().upper().startswith("PENDING")
    ):
        raise EvidenceValidationError("selection freeze requires a final non-PENDING freeze_id")
    if freeze.get("source_campaign") != campaign.get("campaign_id"):
        raise EvidenceValidationError("selection freeze source_campaign mismatch")
    if freeze.get("frozen_before_target") is not True:
        raise EvidenceValidationError("selection freeze was not frozen before target")
    settings = evidence.get("settings")
    if not isinstance(settings, list) or not settings:
        raise EvidenceValidationError("evidence settings must be non-empty")
    chosen = [
        setting for setting in settings
        if setting_ids is None or setting.get("id") in setting_ids
    ]
    if setting_ids is not None and {setting["id"] for setting in chosen} != setting_ids:
        raise EvidenceValidationError("--setting includes an unknown evidence setting")
    execution = _validate_execution(campaign.get("execution"))
    seeds = execution.get("seeds")
    draws = execution.get("repeated_random_draws")
    bootstrap = execution.get("bootstrap")
    assert isinstance(seeds, list) and isinstance(draws, list)
    assert isinstance(bootstrap, Mapping)
    frozen = freeze.get("selections")
    if not isinstance(frozen, Mapping):
        raise EvidenceValidationError("selection freeze lacks selections")

    units: list[dict[str, object]] = []
    for setting in chosen:
        setting_id = str(setting["id"])
        model = campaign.get("models", {}).get(setting.get("model"))
        if not isinstance(model, Mapping):
            raise EvidenceValidationError(f"campaign model missing for {setting_id}")
        if model.get("provisioned") is not True:
            raise EvidenceValidationError(f"campaign model is not provisioned for {setting_id}")
        parent_availability = model.get("parents")
        if not isinstance(parent_availability, Mapping) or any(
            parent_availability.get(parent) is not True for parent in setting["parents"]
        ):
            raise EvidenceValidationError(
                f"campaign model lacks a required parent for {setting_id}"
            )
        dataset = campaign.get("datasets", {}).get(setting["dataset"])
        if not isinstance(dataset, Mapping):
            raise EvidenceValidationError(f"campaign dataset missing for {setting_id}")
        target = dataset.get("rosters", {}).get("target")
        if not isinstance(target, list) or not target or any(
            not isinstance(request, str) or request.startswith("TBD") for request in target
        ):
            raise EvidenceValidationError(f"target roster unresolved for {setting_id}")
        by_parent = frozen.get(setting_id)
        if not isinstance(by_parent, Mapping):
            raise EvidenceValidationError(f"selection freeze lacks setting {setting_id}")
        for parent in setting["parents"]:
            selection = by_parent.get(parent)
            if not isinstance(selection, Mapping):
                raise EvidenceValidationError(
                    f"selection freeze lacks {setting_id}/{parent}"
                )
            prediction = _selection_mapping(
                selection.get("prediction"),
                name=f"selections.{setting_id}.{parent}.prediction",
            )
            protection = _selection_mapping(
                selection.get("protection"),
                name=f"selections.{setting_id}.{parent}.protection",
            )
            for request in target:
                for seed in seeds:
                    units.append(
                        {
                            "setting": setting_id,
                            "parent": parent,
                            "request": request,
                            "seed": seed,
                            "prediction_selection": prediction,
                            "protection_selection": protection,
                            "repeated_random_draws": list(draws),
                        }
                    )
    raw_margins = campaign.get("execution", {}).get("native_margins", {}) or {}
    if not isinstance(raw_margins, dict):
        raise EvidenceValidationError("execution.native_margins must be a mapping")
    native_margins = {}
    for setting in chosen:
        margin = raw_margins.get(setting["id"], 0.0)
        try:
            margin = float(margin)
        except (TypeError, ValueError) as error:
            raise EvidenceValidationError(
                f"execution.native_margins.{setting['id']} must be numeric"
            ) from error
        native_margins[setting["id"]] = margin
    plan = {
        "schema_version": 1,
        "campaign_id": campaign.get("campaign_id"),
        "selection_freeze_id": freeze_id.strip(),
        "bootstrap": dict(bootstrap),
        "native_margins": native_margins,
        "units": units,
    }
    plan["artifact_contracts"] = _artifact_contracts(
        evidence, campaign, chosen, units
    )
    raw_plan_from_mapping(plan)
    return plan


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "configs/paper/evidence.yaml")
    parser.add_argument("--campaign", type=Path, default=ROOT / "configs/paper/campaign.yaml")
    parser.add_argument("--selection-freeze", type=Path, default=ROOT / "configs/paper/selection_freeze.yaml")
    parser.add_argument("--setting", action="append", default=[])
    parser.add_argument("--out", type=Path, default=ROOT / "results/paper/raw_plan.json")
    parser.add_argument("--write-selection-template", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        evidence = _load(args.evidence.resolve())
        campaign = _load(args.campaign.resolve())
        if args.write_selection_template is not None:
            template = selection_template(evidence, str(campaign.get("campaign_id", "")))
            target = args.write_selection_template.resolve()
            _atomic_write(target, yaml.safe_dump(template, sort_keys=False))
            print(f"wrote draft selection template: {target}")
            return 0
        freeze = _load(args.selection_freeze.resolve())
        plan = build_plan(
            evidence,
            campaign,
            freeze,
            setting_ids=set(args.setting) or None,
        )
        target = args.out.resolve()
        _atomic_write(target, json.dumps(plan, indent=2, sort_keys=True) + "\n")
        print(f"wrote immutable raw plan: {target}")
        print(f"planned target units: {len(plan['units'])}")
        return 0
    except (EvidenceValidationError, OSError) as error:
        print(f"raw plan initialization failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
