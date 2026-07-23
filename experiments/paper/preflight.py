"""Fail-closed readiness check for every setting in the paper campaign.

This command performs no model or dataset loading.  It verifies, before a GPU
is allocated, that every evidence setting names a real adapter, an explicitly
provisioned model and dtype, all seven parent implementations, and exact
pairwise-disjoint ``D_cal``/``D_pred``/``D_prot``/``target`` rosters.

Exit codes:
  0 -- every setting and stage is ready
  1 -- a contract is structurally invalid
  2 -- the contract is valid but at least one item is unresolved/not ready
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rsus.data.registry import (  # noqa: E402
    AdapterNotFoundError,
    DatasetAdapterRegistry,
    ADAPTERS,
)
from rsus.generators import objective_names  # noqa: E402


ROSTER_NAMES = ("D_cal", "D_pred", "D_prot", "target")
VALID_DTYPES = frozenset({"float32", "bfloat16", "float16"})
EXECUTOR_MARKER = "PAPER_STAGE_CONTRACT"
EXECUTOR_FLAGS = (
    "consumes_campaign_config",
    "uses_adapter_registry",
    "consumes_exact_roster",
)
STAGE_EXECUTOR_FLAGS = {
    "calibration": ("emits_selection_inputs",),
    "prediction": ("emits_candidate_level_prediction_raw",),
    "protection": ("emits_candidate_level_protection_raw",),
    "target_evaluation": (
        "emits_candidate_level_prediction_raw",
        "emits_candidate_level_protection_raw",
    ),
}


class PreflightConfigError(ValueError):
    """A malformed evidence or campaign contract cannot be audited."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PreflightConfigError(f"missing config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PreflightConfigError(f"config must be a mapping: {path}")
    if raw.get("schema_version") != 1:
        raise PreflightConfigError(f"unsupported schema_version in {path}")
    return raw


def _is_tbd(value: Any) -> bool:
    return isinstance(value, str) and value.strip().casefold().startswith("tbd")


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _digest_roster(values: list[str]) -> str:
    body = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _paper_contract(campaign: Mapping[str, Any]) -> dict[str, Any]:
    raw = campaign.get("paper_contract")
    if not isinstance(raw, dict):
        raise PreflightConfigError("campaign.paper_contract must be a mapping")
    setting_ids = raw.get("setting_ids")
    stage_ids = raw.get("stage_ids")
    dtype = raw.get("confirmatory_dtype")
    if not isinstance(setting_ids, list) or len(setting_ids) != 9 or any(
        not _nonempty_string(value) for value in setting_ids
    ):
        raise PreflightConfigError("paper_contract must freeze exactly nine setting_ids")
    if len(set(setting_ids)) != len(setting_ids):
        raise PreflightConfigError("paper_contract setting_ids must be unique")
    if not isinstance(stage_ids, list) or len(stage_ids) != 4 or any(
        not _nonempty_string(value) for value in stage_ids
    ):
        raise PreflightConfigError("paper_contract must freeze exactly four stage_ids")
    if len(set(stage_ids)) != len(stage_ids):
        raise PreflightConfigError("paper_contract stage_ids must be unique")
    if dtype not in VALID_DTYPES:
        raise PreflightConfigError("paper_contract confirmatory_dtype is invalid")
    return {
        "setting_ids": tuple(setting_ids),
        "stage_ids": tuple(stage_ids),
        "confirmatory_dtype": dtype,
    }


def _validate_stage_contract(
    campaign: Mapping[str, Any], expected_stage_ids: tuple[str, ...]
) -> dict[str, dict[str, str]]:
    raw = campaign.get("stages")
    if not isinstance(raw, dict) or not raw:
        raise PreflightConfigError("campaign stages must be a non-empty mapping")
    stages: dict[str, dict[str, str]] = {}
    for stage, spec in raw.items():
        if not _nonempty_string(stage) or not isinstance(spec, dict):
            raise PreflightConfigError("each campaign stage needs a mapping")
        roster = spec.get("roster")
        capability = spec.get("adapter_capability")
        executor = spec.get("executor")
        if roster not in ROSTER_NAMES:
            raise PreflightConfigError(
                f"stage {stage!r} roster must be one of {list(ROSTER_NAMES)}"
            )
        if not _nonempty_string(capability):
            raise PreflightConfigError(f"stage {stage!r} lacks adapter_capability")
        if not _nonempty_string(executor):
            raise PreflightConfigError(f"stage {stage!r} lacks executor")
        stages[stage] = {
            "roster": roster,
            "adapter_capability": capability,
            "executor": executor,
        }
    if tuple(stages) != expected_stage_ids:
        raise PreflightConfigError(
            "campaign stages must exactly match paper_contract.stage_ids in order"
        )
    used_rosters = {spec["roster"] for spec in stages.values()}
    if used_rosters != set(ROSTER_NAMES):
        raise PreflightConfigError(
            "campaign stages must cover D_cal, D_pred, D_prot, and target exactly"
        )
    return stages


def _executor_report(stage: str, entrypoint: str) -> dict[str, Any]:
    reasons: list[str] = []
    path: Path | None = None
    marker: object = None
    if _is_tbd(entrypoint):
        reasons.append(f"executor is unresolved: {entrypoint}")
    else:
        path = Path(entrypoint)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not path.is_file():
            reasons.append(f"executor entrypoint is missing: {path}")
        else:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in tree.body:
                    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                        continue
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if any(
                        isinstance(target, ast.Name) and target.id == EXECUTOR_MARKER
                        for target in targets
                    ):
                        marker = ast.literal_eval(node.value)
                        break
            except (OSError, SyntaxError, ValueError) as error:
                reasons.append(f"cannot inspect executor contract: {error}")
    if path is not None and path.is_file():
        if not isinstance(marker, dict):
            reasons.append(f"executor lacks literal {EXECUTOR_MARKER} marker")
        else:
            if marker.get("schema_version") != 1:
                reasons.append("executor marker schema_version is not 1")
            supported = marker.get("stages")
            if not isinstance(supported, (list, tuple)) or stage not in supported:
                reasons.append(f"executor marker does not support stage {stage!r}")
            for flag in EXECUTOR_FLAGS + STAGE_EXECUTOR_FLAGS.get(stage, ()):
                if marker.get(flag) is not True:
                    reasons.append(f"executor marker does not certify {flag}")
    return {
        "ready": not reasons,
        "entrypoint": entrypoint,
        "resolved_path": str(path) if path is not None else None,
        "marker": marker if isinstance(marker, dict) else None,
        "reasons": reasons,
    }


def _selection_freeze_report(
    evidence: Mapping[str, Any], campaign: Mapping[str, Any]
) -> dict[str, Any]:
    reasons: list[str] = []
    execution = campaign.get("execution")
    configured = execution.get("selection_freeze") if isinstance(execution, dict) else None
    path: Path | None = None
    raw: object = None
    if not _nonempty_string(configured):
        reasons.append("execution.selection_freeze is missing")
    elif _is_tbd(configured):
        reasons.append(f"selection freeze is unresolved: {configured}")
    else:
        path = Path(configured)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not path.is_file():
            reasons.append(f"selection freeze is missing: {path}")
        else:
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as error:
                reasons.append(f"cannot read selection freeze: {error}")
    completed = 0
    expected = 0
    if isinstance(raw, dict):
        if raw.get("schema_version") != 1 or raw.get("status") != "frozen":
            reasons.append("selection freeze must have schema_version 1 and status=frozen")
        if raw.get("source_campaign") != campaign.get("campaign_id"):
            reasons.append("selection freeze source_campaign mismatch")
        if raw.get("frozen_before_target") is not True:
            reasons.append("selection freeze was not frozen before target")
        selections = raw.get("selections")
        settings = evidence.get("settings")
        if not isinstance(selections, dict) or not isinstance(settings, list):
            reasons.append("selection freeze lacks selections mapping")
        else:
            expected_settings = {setting["id"] for setting in settings}
            if set(selections) != expected_settings:
                reasons.append("selection freeze setting roster is not exact")
            for setting in settings:
                parents = setting.get("parents", [])
                expected += len(parents) * 2
                by_parent = selections.get(setting["id"])
                if not isinstance(by_parent, dict) or set(by_parent) != set(parents):
                    reasons.append(
                        f"selection freeze parent roster is not exact for {setting['id']}"
                    )
                    continue
                for parent in parents:
                    pair = by_parent[parent]
                    if not isinstance(pair, dict):
                        reasons.append(
                            f"selection freeze lacks {setting['id']}/{parent} mapping"
                        )
                        continue
                    for claim in ("prediction", "protection"):
                        selection = pair.get(claim)
                        where = f"{setting['id']}/{parent}/{claim}"
                        if not isinstance(selection, dict):
                            reasons.append(f"selection freeze lacks {where}")
                            continue
                        valid = selection.get("valid")
                        fallback = selection.get("fallback")
                        alpha = selection.get("alpha")
                        resolved = (valid is True) != (fallback is True)
                        numeric_alpha = (
                            not isinstance(alpha, bool)
                            and isinstance(alpha, (int, float))
                            and 0.0 <= float(alpha) <= 1.0
                        )
                        if not resolved or (valid is True and not numeric_alpha):
                            reasons.append(f"selection freeze has unresolved {where}")
                            continue
                        completed += 1
    elif raw is not None:
        reasons.append("selection freeze root must be a mapping")
    return {
        "ready": not reasons and expected > 0 and completed == expected,
        "configured_path": configured,
        "resolved_path": str(path) if path is not None else None,
        "selections_resolved": completed,
        "selections_expected": expected,
        "reasons": list(dict.fromkeys(reasons)),
    }


def _required_parents(campaign: Mapping[str, Any]) -> tuple[str, ...]:
    raw = campaign.get("required_parents")
    if not isinstance(raw, list) or any(not _nonempty_string(item) for item in raw):
        raise PreflightConfigError("required_parents must be a list of names")
    parents = tuple(raw)
    if len(parents) != 7 or len(set(parents)) != 7:
        raise PreflightConfigError(
            "paper campaign requires exactly seven unique parent objectives"
        )
    return parents


def _dataset_report(
    dataset: str,
    raw: Any,
    stages: Mapping[str, Mapping[str, str]],
    registry: DatasetAdapterRegistry,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(raw, dict):
        return {
            "ready": False,
            "adapter": None,
            "adapter_registered": False,
            "capabilities": None,
            "stage_support": {stage: False for stage in stages},
            "roster_unit": None,
            "rosters": {},
            "pairwise_disjoint": False,
            "overlaps": {},
            "reasons": [f"dataset {dataset!r} is absent from campaign config"],
        }

    adapter_name = raw.get("adapter")
    adapter = None
    if not _nonempty_string(adapter_name):
        reasons.append("adapter name is missing")
    elif _is_tbd(adapter_name):
        reasons.append(f"adapter is unresolved: {adapter_name}")
    else:
        try:
            adapter = registry.resolve(adapter_name)
        except AdapterNotFoundError:
            reasons.append(f"adapter {adapter_name!r} is not registered")

    roster_unit = raw.get("roster_unit")
    if not _nonempty_string(roster_unit):
        reasons.append("roster_unit is missing")
    elif _is_tbd(roster_unit):
        reasons.append(f"roster_unit is unresolved: {roster_unit}")
    elif adapter is not None and roster_unit != adapter.capabilities.roster_unit:
        reasons.append(
            f"roster_unit {roster_unit!r} disagrees with adapter contract "
            f"{adapter.capabilities.roster_unit!r}"
        )

    raw_rosters = raw.get("rosters")
    if not isinstance(raw_rosters, dict):
        raw_rosters = {}
        reasons.append("rosters mapping is missing")

    roster_reports: dict[str, dict[str, Any]] = {}
    valid_sets: dict[str, set[str]] = {}
    for name in ROSTER_NAMES:
        values = raw_rosters.get(name)
        roster_reasons: list[str] = []
        clean: list[str] = []
        if not isinstance(values, list) or not values:
            roster_reasons.append("must be a non-empty explicit list")
        else:
            for value in values:
                if not _nonempty_string(value):
                    roster_reasons.append("contains a non-string or empty id")
                    continue
                clean.append(value.strip())
            tbd = sorted(value for value in clean if _is_tbd(value))
            if tbd:
                roster_reasons.append(f"contains unresolved ids: {', '.join(tbd)}")
            duplicates = sorted(
                {value for value in clean if clean.count(value) > 1 and not _is_tbd(value)}
            )
            if duplicates:
                roster_reasons.append(f"contains duplicate ids: {', '.join(duplicates)}")
            if adapter is not None:
                if adapter.roster_id_validator is None:
                    roster_reasons.append("adapter has no roster-id validator")
                else:
                    invalid = sorted(
                        value
                        for value in clean
                        if not _is_tbd(value) and not adapter.accepts_roster_id(value)
                    )
                    if invalid:
                        roster_reasons.append(
                            f"ids rejected by adapter contract: {', '.join(invalid)}"
                        )
        exact = not roster_reasons
        if exact:
            valid_sets[name] = set(clean)
        roster_reports[name] = {
            "ids": clean,
            "count": len(clean),
            "exact": exact,
            "sha256": _digest_roster(clean) if exact else None,
            "reasons": roster_reasons,
        }

    overlaps: dict[str, list[str]] = {}
    for left_index, left in enumerate(ROSTER_NAMES):
        for right in ROSTER_NAMES[left_index + 1 :]:
            shared = sorted(valid_sets.get(left, set()) & valid_sets.get(right, set()))
            if shared:
                overlaps[f"{left}__{right}"] = shared
    pairwise_disjoint = len(valid_sets) == len(ROSTER_NAMES) and not overlaps
    if overlaps:
        reasons.append(
            "rosters overlap: "
            + "; ".join(f"{pair}={ids}" for pair, ids in overlaps.items())
        )
    elif len(valid_sets) != len(ROSTER_NAMES):
        reasons.append("one or more rosters are not exact")

    stage_support: dict[str, bool] = {}
    for stage, spec in stages.items():
        stage_support[stage] = bool(
            adapter is not None
            and adapter.capabilities.supports(spec["adapter_capability"])
        )

    return {
        "ready": not reasons and pairwise_disjoint,
        "adapter": adapter_name,
        "adapter_registered": adapter is not None,
        "resolved_adapter": adapter.key if adapter is not None else None,
        "capabilities": adapter.capabilities.as_dict() if adapter is not None else None,
        "roster_id_validation": bool(
            adapter is not None and adapter.roster_id_validator is not None
        ),
        "stage_support": stage_support,
        "roster_unit": roster_unit,
        "rosters": roster_reports,
        "pairwise_disjoint": pairwise_disjoint,
        "overlaps": overlaps,
        "reasons": reasons,
    }


def _model_report(
    model: str,
    raw: Any,
    required_parents: tuple[str, ...],
    implemented_parents: frozenset[str],
    confirmatory_dtype: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(raw, dict):
        return {
            "ready": False,
            "source": None,
            "source_kind": None,
            "source_exists": False,
            "dtype": None,
            "provisioned": False,
            "parent_availability": {parent: False for parent in required_parents},
            "parents_available": 0,
            "parents_required": len(required_parents),
            "reasons": [f"model {model!r} is absent from campaign config"],
        }

    source = raw.get("source")
    source_kind = raw.get("source_kind")
    source_exists = False
    if not _nonempty_string(source):
        reasons.append("model source is missing")
    elif _is_tbd(source):
        reasons.append(f"model source is unresolved: {source}")
    elif source_kind != "local_path":
        reasons.append("confirmatory model source_kind must be 'local_path'")
    else:
        source_path = Path(source).expanduser()
        source_exists = source_path.exists()
        if not source_exists:
            reasons.append(f"provisioned model path is not present on this host: {source}")

    dtype = raw.get("dtype")
    if dtype not in VALID_DTYPES:
        reasons.append(
            f"dtype must be explicit and one of {sorted(VALID_DTYPES)}; got {dtype!r}"
        )
    elif dtype != confirmatory_dtype:
        reasons.append(
            f"dtype {dtype!r} violates confirmatory precision {confirmatory_dtype!r}"
        )

    provisioned = raw.get("provisioned")
    if provisioned is not True:
        reasons.append("model is not explicitly provisioned")

    parent_raw = raw.get("parents")
    if not isinstance(parent_raw, dict):
        parent_raw = {}
        reasons.append("parent availability mapping is missing")
    availability = {
        parent: parent_raw.get(parent) is True and parent in implemented_parents
        for parent in required_parents
    }
    missing = [parent for parent, available in availability.items() if not available]
    if missing:
        reasons.append(f"parent implementations unavailable: {', '.join(missing)}")
    extra = sorted(set(parent_raw) - set(required_parents))
    if extra:
        reasons.append(f"undeclared parent implementations present: {', '.join(extra)}")

    return {
        "ready": not reasons,
        "source": source,
        "source_kind": source_kind,
        "source_exists": source_exists,
        "dtype": dtype,
        "provisioned": provisioned is True,
        "parent_availability": availability,
        "parents_available": sum(availability.values()),
        "parents_required": len(required_parents),
        "implemented_parents": sorted(implemented_parents),
        "reasons": reasons,
    }


def build_preflight_report(
    evidence: Mapping[str, Any],
    campaign: Mapping[str, Any],
    *,
    registry: DatasetAdapterRegistry = ADAPTERS,
) -> dict[str, Any]:
    """Build a JSON-serialisable readiness report without loading a model."""

    paper_contract = _paper_contract(campaign)
    stages = _validate_stage_contract(campaign, paper_contract["stage_ids"])
    required_parents = _required_parents(campaign)
    implemented_parents = frozenset(objective_names())
    missing_implementations = sorted(set(required_parents) - implemented_parents)
    if missing_implementations:
        raise PreflightConfigError(
            "required parent objectives are not registered in code: "
            + ", ".join(missing_implementations)
        )

    settings_raw = evidence.get("settings")
    if not isinstance(settings_raw, list) or not settings_raw:
        raise PreflightConfigError("evidence settings must be a non-empty list")
    setting_ids = [item.get("id") for item in settings_raw if isinstance(item, dict)]
    if len(setting_ids) != len(settings_raw) or any(
        not _nonempty_string(setting_id) for setting_id in setting_ids
    ):
        raise PreflightConfigError("every evidence setting needs a string id")
    if len(set(setting_ids)) != len(setting_ids):
        raise PreflightConfigError("evidence setting ids must be unique")
    if tuple(setting_ids) != paper_contract["setting_ids"]:
        raise PreflightConfigError(
            "evidence settings must exactly match paper_contract.setting_ids in order"
        )

    dataset_configs = campaign.get("datasets")
    model_configs = campaign.get("models")
    if not isinstance(dataset_configs, dict):
        dataset_configs = {}
    if not isinstance(model_configs, dict):
        model_configs = {}

    dataset_names = sorted(
        {str(item.get("dataset")) for item in settings_raw if isinstance(item, dict)}
    )
    model_names = sorted(
        {str(item.get("model")) for item in settings_raw if isinstance(item, dict)}
    )
    dataset_reports = {
        name: _dataset_report(name, dataset_configs.get(name), stages, registry)
        for name in dataset_names
    }
    model_reports = {
        name: _model_report(
            name,
            model_configs.get(name),
            required_parents,
            implemented_parents,
            paper_contract["confirmatory_dtype"],
        )
        for name in model_names
    }
    executor_reports = {
        stage: _executor_report(stage, spec["executor"])
        for stage, spec in stages.items()
    }
    selection_freeze = _selection_freeze_report(evidence, campaign)

    setting_reports: list[dict[str, Any]] = []
    ready_stage_count = 0
    for raw in settings_raw:
        setting_id = raw["id"]
        dataset = raw.get("dataset")
        model = raw.get("model")
        if not _nonempty_string(dataset) or not _nonempty_string(model):
            raise PreflightConfigError(
                f"evidence setting {setting_id!r} needs dataset and model names"
            )
        declared_parents = raw.get("parents")
        if not isinstance(declared_parents, list):
            declared_parents = []
        parent_contract_exact = tuple(declared_parents) == required_parents
        parent_reason = None
        if not parent_contract_exact:
            parent_reason = (
                "evidence parent roster must exactly match the campaign's ordered "
                "seven-parent roster"
            )

        dataset_report = dataset_reports[dataset]
        model_report = model_reports[model]
        stage_reports: dict[str, dict[str, Any]] = {}
        for stage, spec in stages.items():
            reasons: list[str] = []
            executor_report = executor_reports[stage]
            roster_report = dataset_report["rosters"].get(spec["roster"], {})
            if not dataset_report["adapter_registered"]:
                reasons.append(f"dataset adapter {dataset_report['adapter']!r} is unavailable")
            elif not dataset_report["stage_support"].get(stage, False):
                reasons.append(
                    f"adapter does not support capability {spec['adapter_capability']!r}"
                )
            if not roster_report.get("exact", False):
                reasons.extend(
                    f"{spec['roster']}: {reason}"
                    for reason in roster_report.get("reasons", ["roster unavailable"])
                )
            if not dataset_report["pairwise_disjoint"]:
                reasons.append("dataset rosters are not exact pairwise-disjoint sets")
            if _is_tbd(dataset_report.get("roster_unit")):
                reasons.append("dataset roster_unit is unresolved")
            if not dataset_report["ready"]:
                reasons.extend(dataset_report["reasons"])
            if not model_report["ready"]:
                reasons.extend(model_report["reasons"])
            if parent_reason is not None:
                reasons.append(parent_reason)
            if not executor_report["ready"]:
                reasons.extend(executor_report["reasons"])
            if stage == "target_evaluation" and not selection_freeze["ready"]:
                reasons.extend(selection_freeze["reasons"])
            ready = not reasons
            ready_stage_count += int(ready)
            stage_reports[stage] = {
                "ready": ready,
                "adapter_capability": spec["adapter_capability"],
                "executor": executor_report["entrypoint"],
                "executor_ready": executor_report["ready"],
                "roster": spec["roster"],
                "roster_count": roster_report.get("count", 0),
                "roster_sha256": roster_report.get("sha256"),
                "model_dtype": model_report["dtype"],
                "parents_available": model_report["parents_available"],
                "parents_required": model_report["parents_required"],
                "reasons": list(dict.fromkeys(reasons)),
            }

        setting_reports.append(
            {
                "id": setting_id,
                "dataset": dataset,
                "model": model,
                "role": raw.get("role"),
                "parents": list(declared_parents),
                "parent_contract_exact": parent_contract_exact,
                "ready": all(item["ready"] for item in stage_reports.values()),
                "stages": stage_reports,
            }
        )

    total_stages = len(setting_reports) * len(stages)
    ready = ready_stage_count == total_stages
    return {
        "schema_version": 1,
        "campaign_id": campaign.get("campaign_id"),
        "ready": ready,
        "paper_contract": {
            "setting_ids": list(paper_contract["setting_ids"]),
            "stage_ids": list(paper_contract["stage_ids"]),
            "confirmatory_dtype": paper_contract["confirmatory_dtype"],
        },
        "required_parents": list(required_parents),
        "implemented_parent_objectives": sorted(implemented_parents),
        "registered_adapters": {
            adapter.key: adapter.as_dict() for adapter in registry.adapters()
        },
        "stages": stages,
        "executors": executor_reports,
        "selection_freeze": selection_freeze,
        "datasets": dataset_reports,
        "models": model_reports,
        "settings": setting_reports,
        "summary": {
            "settings_ready": sum(item["ready"] for item in setting_reports),
            "settings_total": len(setting_reports),
            "stages_ready": ready_stage_count,
            "stages_total": total_stages,
            "missing_adapter_datasets": sorted(
                name
                for name, item in dataset_reports.items()
                if not item["adapter_registered"]
            ),
            "unresolved_roster_datasets": sorted(
                name for name, item in dataset_reports.items() if not item["ready"]
            ),
            "unprovisioned_models": sorted(
                name for name, item in model_reports.items() if not item["provisioned"]
            ),
            "missing_model_paths": sorted(
                name for name, item in model_reports.items() if not item["source_exists"]
            ),
            "unready_executors": sorted(
                stage for stage, item in executor_reports.items() if not item["ready"]
            ),
            "selection_freeze_ready": selection_freeze["ready"],
        },
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _print_report(report: Mapping[str, Any], output: Path) -> None:
    marker = "READY" if report["ready"] else "NOT READY"
    summary = report["summary"]
    print(
        f"[{marker}] {summary['settings_ready']}/{summary['settings_total']} settings; "
        f"{summary['stages_ready']}/{summary['stages_total']} stages"
    )
    for setting in report["settings"]:
        print(f"  {setting['id']}: {'ready' if setting['ready'] else 'not ready'}")
        for stage, result in setting["stages"].items():
            if result["ready"]:
                continue
            print(f"    {stage}: " + "; ".join(result["reasons"]))
    print(f"wrote {output}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-config",
        type=Path,
        default=ROOT / "configs/paper/evidence.yaml",
    )
    parser.add_argument(
        "--campaign-config",
        type=Path,
        default=ROOT / "configs/paper/campaign.yaml",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        evidence = _load_yaml(args.evidence_config.resolve())
        campaign = _load_yaml(args.campaign_config.resolve())
        report = build_preflight_report(evidence, campaign)
        output = args.out
        if output is None:
            configured = campaign.get("outputs", {}).get(
                "preflight_json", "results/paper/campaign_preflight.json"
            )
            output = ROOT / configured
        output = output.resolve()
        _atomic_write_json(output, report)
        _print_report(report, output)
        return 0 if report["ready"] else 2
    except (OSError, yaml.YAMLError, PreflightConfigError) as error:
        print(f"preflight contract error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
