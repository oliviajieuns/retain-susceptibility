"""Export sealed alpha-protection results to the paper candidate JSONL schema.

This is a strict bridge, not a relabeling utility. The immutable raw plan is
the denominator: request/seed/parent cells, frozen protection alphas, repeated
random draw IDs, model source, candidate support, four feasibility margins,
and the shared first-reaching parent checkpoint must all match exactly.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.analysis.mixture import alpha_label  # noqa: E402
from rsus.evidence.raw import (  # noqa: E402
    RawPlan,
    aggregate_raw_evidence,
    load_raw_plan,
)
from rsus.evidence.schemas import EvidenceValidationError  # noqa: E402


CONSTRAINTS = (
    "direct_forgetting",
    "paraphrase_forgetting",
    "extraction_generation",
    "utility",
)


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise EvidenceValidationError(f"cannot read contract {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"contract root must be a mapping: {path}")
    return value


def _read_payloads(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("**/results.json"))
    if not paths:
        raise EvidenceValidationError(f"no results.json below {root}")
    payloads = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvidenceValidationError(f"cannot read {path}: {error}") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise EvidenceValidationError(f"invalid alpha-protection payload: {path}")
        payload["_source_path"] = str(path)
        payloads.append(payload)
    return payloads


def _selection_mapping(unit) -> dict[str, object]:
    selection = unit.protection_selection
    return {
        "valid": selection.valid,
        "fallback": selection.fallback,
        "alpha": selection.alpha,
    }


def _constraint_margins(row: Mapping[str, Any], *, where: str) -> dict[str, float]:
    decision = row.get("final_checkpoint_decision")
    if not isinstance(decision, Mapping):
        raise EvidenceValidationError(f"{where} lacks final_checkpoint_decision")
    trace = decision.get("trace")
    if not isinstance(trace, list):
        raise EvidenceValidationError(f"{where} has invalid final checkpoint trace")
    step = row.get("step")
    entries = [entry for entry in trace if isinstance(entry, Mapping) and entry.get("step") == step]
    if len(entries) != 1:
        raise EvidenceValidationError(
            f"{where} must identify exactly one reported checkpoint trace entry"
        )
    raw_slacks = entries[0].get("slacks")
    if not isinstance(raw_slacks, Mapping):
        raise EvidenceValidationError(f"{where} reported checkpoint lacks slacks")
    slacks: dict[str, float] = {}
    for name in CONSTRAINTS:
        value = raw_slacks.get(name)
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise EvidenceValidationError(
                f"{where} lacks finite {name} slack"
            ) from error
        if not math.isfinite(number):
            raise EvidenceValidationError(f"{where} has non-finite {name} slack")
        slacks[name] = number
    expected_feasible = all(value >= 0.0 for value in slacks.values())
    if type(row.get("feasible")) is not bool or row["feasible"] != expected_feasible:
        raise EvidenceValidationError(
            f"{where}.feasible disagrees with the four checkpoint slacks"
        )
    if expected_feasible and row.get("selected_checkpoint_step") != step:
        raise EvidenceValidationError(
            f"{where} did not report its selected fully feasible checkpoint"
        )
    return slacks


def _source_support(
    row: Mapping[str, Any], *, where: str
) -> tuple[dict[str, float], dict[str, str]]:
    if row.get("executed") is not True:
        raise EvidenceValidationError(f"{where} was not executed")
    damage = row.get("candidate_damage")
    groups = row.get("candidate_groups")
    if not isinstance(damage, Mapping) or not damage:
        raise EvidenceValidationError(f"{where} lacks candidate damage")
    if not isinstance(groups, Mapping) or set(groups) != set(damage):
        raise EvidenceValidationError(f"{where} candidate/group support differs")
    parsed: dict[str, float] = {}
    parsed_groups: dict[str, str] = {}
    for candidate_id, value in damage.items():
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise EvidenceValidationError(
                f"{where} has nonnumeric damage for {candidate_id}"
            ) from error
        if not math.isfinite(number):
            raise EvidenceValidationError(
                f"{where} has non-finite damage for {candidate_id}"
            )
        group = groups[candidate_id]
        if not isinstance(candidate_id, str) or not candidate_id or not isinstance(group, str) or not group:
            raise EvidenceValidationError(f"{where} has invalid candidate/group ID")
        parsed[candidate_id] = number
        parsed_groups[candidate_id] = group
    return parsed, parsed_groups


def export_records(
    plan: RawPlan,
    payloads: Iterable[Mapping[str, Any]],
    *,
    setting: str,
    expected_model_source: str,
) -> list[dict[str, object]]:
    units = [unit for unit in plan.units.values() if unit.key[0] == setting]
    if not units:
        raise EvidenceValidationError(f"raw plan has no units for setting {setting!r}")
    expected_cells = {(unit.key[2], unit.key[3]) for unit in units}
    parents = sorted({unit.key[1] for unit in units})
    unit_by_key = {unit.key: unit for unit in units}
    by_cell: dict[tuple[str, str], Mapping[str, Any]] = {}
    for payload in payloads:
        manifest = payload.get("manifest")
        if not isinstance(manifest, Mapping):
            raise EvidenceValidationError("runner payload lacks manifest")
        source = str(payload.get("_source_path", "runner payload"))
        if manifest.get("campaign_phase") != "audit":
            raise EvidenceValidationError(f"{source} is not an audit artifact")
        if str(manifest.get("model")) != expected_model_source:
            raise EvidenceValidationError(
                f"{source} model source does not match paper setting {setting}"
            )
        cell = (str(manifest.get("request")), str(manifest.get("seed")))
        if cell in by_cell:
            raise EvidenceValidationError(f"duplicate runner cell {cell}")
        by_cell[cell] = payload
    actual_cells = set(by_cell)
    if actual_cells != expected_cells:
        raise EvidenceValidationError(
            "runner roster differs from immutable paper plan; refusing relabel: "
            f"missing={sorted(expected_cells - actual_cells)}, "
            f"extra={sorted(actual_cells - expected_cells)}"
        )

    output: list[dict[str, object]] = []
    for request, seed in sorted(expected_cells):
        payload = by_cell[(request, seed)]
        rows = payload["results"]
        by_parent: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise EvidenceValidationError(f"{request}/{seed} contains a non-row result")
            if str(row.get("request")) != request or str(row.get("seed")) != seed:
                raise EvidenceValidationError(f"{request}/{seed} result identity changed")
            by_parent.setdefault(str(row.get("parent")), []).append(row)
        if set(by_parent) != set(parents):
            raise EvidenceValidationError(
                f"{request}/{seed} parent roster differs from immutable plan"
            )
        for parent in parents:
            unit = unit_by_key[(setting, parent, request, seed)]
            selection = _selection_mapping(unit)
            if not selection["valid"] or selection["fallback"] or selection["alpha"] is None:
                raise EvidenceValidationError(
                    f"{setting}/{parent} protection selection is not a valid frozen alpha"
                )
            alpha = float(selection["alpha"])
            members = by_parent[parent]
            selector_rows = {str(row.get("selector")): row for row in members}
            if len(selector_rows) != len(members):
                raise EvidenceValidationError(f"{request}/{seed}/{parent} has duplicate selectors")
            deployed = [
                row for row in members
                if row.get("selector_type") == "mixture" and row.get("deployed") is True
            ]
            if len(deployed) != 1 or not math.isclose(
                float(deployed[0].get("alpha")), alpha, abs_tol=1e-12
            ):
                raise EvidenceValidationError(
                    f"{request}/{seed}/{parent} deployed alpha differs from raw plan"
                )
            required = {
                "none",
                "random",
                alpha_label(0.0),
                alpha_label(1.0),
            }
            if not required <= set(selector_rows):
                raise EvidenceValidationError(
                    f"{request}/{seed}/{parent} lacks a claim arm: "
                    f"{sorted(required - set(selector_rows))}"
                )
            sources: list[tuple[str, str | None, Mapping[str, Any]]] = [
                ("joint", None, deployed[0]),
                ("no_repair", None, selector_rows["none"]),
                ("s0", None, selector_rows[alpha_label(0.0)]),
                ("s1", None, selector_rows[alpha_label(1.0)]),
            ]
            random_summary = selector_rows["random"]
            raw_draws = random_summary.get("random_draws")
            if not isinstance(raw_draws, list):
                raise EvidenceValidationError(
                    f"{request}/{seed}/{parent} lacks repeated-random raw draws"
                )
            draws = {str(row.get("random_draw_id")): row for row in raw_draws}
            expected_draws = set(unit.repeated_random_draws)
            if set(draws) != expected_draws or len(draws) != len(raw_draws):
                raise EvidenceValidationError(
                    f"{request}/{seed}/{parent} random draw roster differs from raw plan"
                )
            sources.extend(
                ("repeated_random", draw_id, draws[draw_id])
                for draw_id in unit.repeated_random_draws
            )

            parsed_sources = []
            reference_ids: set[str] | None = None
            reference_groups: dict[str, str] | None = None
            checkpoint_ids = set()
            for arm, draw_id, source_row in sources:
                where = f"{request}/{seed}/{parent}/{arm}/{draw_id or '-'}"
                damage, groups = _source_support(source_row, where=where)
                slacks = _constraint_margins(source_row, where=where)
                checkpoint = source_row.get("parent_checkpoint")
                if not isinstance(checkpoint, Mapping):
                    raise EvidenceValidationError(f"{where} lacks parent checkpoint identity")
                if checkpoint.get("first_direct_reaching") is not True:
                    raise EvidenceValidationError(f"{where} parent checkpoint is not first-reaching")
                checkpoint_id = checkpoint.get("block_sha256")
                if not isinstance(checkpoint_id, str) or len(checkpoint_id) != 64:
                    raise EvidenceValidationError(f"{where} has invalid parent block hash")
                checkpoint_ids.add(checkpoint_id)
                if reference_ids is None:
                    reference_ids = set(damage)
                    reference_groups = groups
                elif set(damage) != reference_ids or groups != reference_groups:
                    raise EvidenceValidationError(
                        f"{where} does not share exact candidate/group support"
                    )
                parsed_sources.append((arm, draw_id, damage, groups, slacks, source_row, checkpoint_id))
            if len(checkpoint_ids) != 1:
                raise EvidenceValidationError(
                    f"{request}/{seed}/{parent} claim arms mix parent checkpoints"
                )

            for arm, draw_id, damage, groups, slacks, source_row, checkpoint_id in parsed_sources:
                for candidate_id in sorted(damage):
                    record: dict[str, object] = {
                        "setting": setting,
                        "parent": parent,
                        "request": request,
                        "seed": seed,
                        "candidate_id": candidate_id,
                        "group": groups[candidate_id],
                        "arm": arm,
                        "damage": damage[candidate_id],
                        "feasible": bool(source_row["feasible"]),
                        "direct_forget_margin": slacks["direct_forgetting"],
                        "paraphrase_forget_margin": slacks["paraphrase_forgetting"],
                        "extraction_generation_margin": slacks["extraction_generation"],
                        "utility_margin": slacks["utility"],
                        "parent_checkpoint_id": checkpoint_id,
                        "parent_checkpoint_first_reaching": True,
                        "protection_selection": selection,
                    }
                    if arm == "repeated_random":
                        record["draw_id"] = draw_id
                        record["draw_complete"] = True
                    output.append(record)

    # Reuse the authoritative parser as the final schema/common-support check.
    aggregate_raw_evidence(plan, [], output)
    return output


def _setting_contract(
    evidence: Mapping[str, Any], campaign: Mapping[str, Any], setting_id: str
) -> tuple[str, str]:
    settings = evidence.get("settings")
    match = next(
        (item for item in settings or [] if isinstance(item, Mapping) and item.get("id") == setting_id),
        None,
    )
    if not isinstance(match, Mapping):
        raise EvidenceValidationError(f"unknown paper setting {setting_id!r}")
    if match.get("dataset") != "TOFU":
        raise EvidenceValidationError(
            "the alpha-protection exporter accepts the real TOFU runner only"
        )
    model_name = str(match.get("model"))
    model = campaign.get("models", {}).get(model_name)
    if not isinstance(model, Mapping) or not isinstance(model.get("source"), str):
        raise EvidenceValidationError(f"paper model contract missing for {model_name}")
    return model_name, str(model["source"])


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--evidence", type=Path, default=ROOT / "configs/paper/evidence.yaml")
    parser.add_argument("--campaign", type=Path, default=ROOT / "configs/paper/campaign.yaml")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = load_raw_plan(args.plan.resolve())
        evidence = _load_mapping(args.evidence.resolve())
        campaign = _load_mapping(args.campaign.resolve())
        model_name, model_source = _setting_contract(evidence, campaign, args.setting)
        records = export_records(
            plan,
            _read_payloads(args.runner_root.resolve()),
            setting=args.setting,
            expected_model_source=model_source,
        )
        target = args.out.resolve()
        _atomic_jsonl(target, records)
        print(f"validated paper setting: {args.setting} ({model_name})")
        print(f"wrote {len(records)} protection candidate rows: {target}")
        return 0
    except (EvidenceValidationError, OSError, ValueError) as error:
        print(f"alpha-protection export failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
