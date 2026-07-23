"""Load and validate the predeclared paper evidence registry."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schemas import EvidenceValidationError


CLAIMS = ("prediction", "protection")


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _string_list(value: object, *, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        suffix = "" if allow_empty else " and non-empty"
        raise EvidenceValidationError(f"{name} must be a list{suffix}")
    result = tuple(_nonempty_string(item, name=f"{name}[]") for item in value)
    if len(set(result)) != len(result):
        raise EvidenceValidationError(f"{name} contains duplicates")
    return result


@dataclass(frozen=True)
class SettingSpec:
    setting_id: str
    dataset: str
    model: str
    role: str
    parents: tuple[str, ...]


@dataclass(frozen=True)
class TableSpec:
    table_id: str
    label: str
    location: str
    settings: tuple[str, ...]
    claims: tuple[str, ...]
    artifacts: tuple[str, ...]
    producer: str


@dataclass(frozen=True)
class RuleGroup:
    group_id: str
    settings: tuple[str, ...]
    minimum_pass: int


@dataclass(frozen=True)
class ParentGroupRule:
    group_id: str
    parents: tuple[str, ...]
    minimum_joint_pass: int
    multiplicity: str


@dataclass(frozen=True)
class MultiSettingRule:
    rule_id: str
    primary_required: tuple[str, ...]
    groups: tuple[RuleGroup, ...]
    parent_groups: tuple[ParentGroupRule, ...]
    stress_excluded: tuple[str, ...]
    require_both_claims: bool


@dataclass(frozen=True)
class EvidenceContract:
    schema_version: int
    alpha: float
    minimum_support_units: int
    ledger_path: str
    readiness_output: str
    tex_output: str
    core_table_output: str
    robustness_table_output: str
    fidelity_inputs: Mapping[str, str]
    settings: Mapping[str, SettingSpec]
    artifacts: tuple[str, ...]
    tables: Mapping[str, TableSpec]
    multi_setting: MultiSettingRule

    @property
    def planned_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (setting_id, parent)
            for setting_id, setting in self.settings.items()
            for parent in setting.parents
        )


def _parse_settings(raw: object) -> dict[str, SettingSpec]:
    if not isinstance(raw, list) or not raw:
        raise EvidenceValidationError("settings must be a non-empty list")
    result: dict[str, SettingSpec] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise EvidenceValidationError(f"settings[{index}] must be a mapping")
        setting_id = _nonempty_string(item.get("id"), name=f"settings[{index}].id")
        if setting_id in result:
            raise EvidenceValidationError(f"duplicate setting id {setting_id!r}")
        result[setting_id] = SettingSpec(
            setting_id=setting_id,
            dataset=_nonempty_string(
                item.get("dataset"), name=f"settings[{index}].dataset"
            ),
            model=_nonempty_string(item.get("model"), name=f"settings[{index}].model"),
            role=_nonempty_string(item.get("role"), name=f"settings[{index}].role"),
            parents=_string_list(
                item.get("parents"), name=f"settings[{index}].parents"
            ),
        )
    return result


def _parse_tables(
    raw: object,
    *,
    settings: Mapping[str, SettingSpec],
    artifacts: set[str],
) -> dict[str, TableSpec]:
    if not isinstance(raw, list) or not raw:
        raise EvidenceValidationError("tables must be a non-empty list")
    result: dict[str, TableSpec] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise EvidenceValidationError(f"tables[{index}] must be a mapping")
        table_id = _nonempty_string(item.get("id"), name=f"tables[{index}].id")
        if table_id in result:
            raise EvidenceValidationError(f"duplicate table id {table_id!r}")
        table_settings = _string_list(
            item.get("settings", []),
            name=f"tables[{index}].settings",
            allow_empty=True,
        )
        unknown_settings = set(table_settings) - set(settings)
        if unknown_settings:
            raise EvidenceValidationError(
                f"table {table_id!r} references unknown settings {sorted(unknown_settings)}"
            )
        claims = _string_list(
            item.get("claims", []),
            name=f"tables[{index}].claims",
            allow_empty=True,
        )
        unknown_claims = set(claims) - set(CLAIMS)
        if unknown_claims:
            raise EvidenceValidationError(
                f"table {table_id!r} has unknown claims {sorted(unknown_claims)}"
            )
        required_artifacts = _string_list(
            item.get("artifacts", []),
            name=f"tables[{index}].artifacts",
            allow_empty=True,
        )
        unknown_artifacts = set(required_artifacts) - artifacts
        if unknown_artifacts:
            raise EvidenceValidationError(
                f"table {table_id!r} references unknown artifacts "
                f"{sorted(unknown_artifacts)}"
            )
        result[table_id] = TableSpec(
            table_id=table_id,
            label=_nonempty_string(item.get("label"), name=f"tables[{index}].label"),
            location=_nonempty_string(
                item.get("location"), name=f"tables[{index}].location"
            ),
            settings=table_settings,
            claims=claims,
            artifacts=required_artifacts,
            producer=_nonempty_string(
                item.get("producer"), name=f"tables[{index}].producer"
            ),
        )
    return result


def _parse_multi_setting(
    raw: object, *, settings: Mapping[str, SettingSpec]
) -> MultiSettingRule:
    if not isinstance(raw, Mapping):
        raise EvidenceValidationError("multi_setting_rule must be a mapping")
    primary = _string_list(
        raw.get("primary_required"), name="multi_setting_rule.primary_required"
    )
    stress = _string_list(
        raw.get("stress_excluded", []),
        name="multi_setting_rule.stress_excluded",
        allow_empty=True,
    )
    known = set(settings)
    unknown = (set(primary) | set(stress)) - known
    if unknown:
        raise EvidenceValidationError(
            f"multi-setting rule references unknown settings {sorted(unknown)}"
        )
    raw_groups = raw.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise EvidenceValidationError("multi_setting_rule.groups must be non-empty")
    groups: list[RuleGroup] = []
    group_ids: set[str] = set()
    for index, item in enumerate(raw_groups):
        if not isinstance(item, Mapping):
            raise EvidenceValidationError(
                f"multi_setting_rule.groups[{index}] must be a mapping"
            )
        group_id = _nonempty_string(
            item.get("id"), name=f"multi_setting_rule.groups[{index}].id"
        )
        if group_id in group_ids:
            raise EvidenceValidationError(f"duplicate rule group id {group_id!r}")
        group_ids.add(group_id)
        members = _string_list(
            item.get("settings"),
            name=f"multi_setting_rule.groups[{index}].settings",
        )
        unknown_members = set(members) - known
        if unknown_members:
            raise EvidenceValidationError(
                f"rule group {group_id!r} has unknown settings {sorted(unknown_members)}"
            )
        if set(members) & set(stress):
            raise EvidenceValidationError(
                f"stress settings cannot rescue rule group {group_id!r}"
            )
        minimum = item.get("minimum_pass")
        if isinstance(minimum, bool) or not isinstance(minimum, int):
            raise EvidenceValidationError(
                f"rule group {group_id!r} minimum_pass must be an integer"
            )
        if not 1 <= minimum <= len(members):
            raise EvidenceValidationError(
                f"rule group {group_id!r} minimum_pass must be in [1, {len(members)}]"
            )
        groups.append(
            RuleGroup(group_id=group_id, settings=members, minimum_pass=minimum)
        )
    raw_parent_groups = raw.get("parent_groups")
    if not isinstance(raw_parent_groups, list) or not raw_parent_groups:
        raise EvidenceValidationError(
            "multi_setting_rule.parent_groups must be non-empty"
        )
    known_parents = set.intersection(
        *(set(setting.parents) for setting in settings.values())
    )
    parent_groups: list[ParentGroupRule] = []
    parent_group_ids: set[str] = set()
    assigned_parents: set[str] = set()
    for index, item in enumerate(raw_parent_groups):
        if not isinstance(item, Mapping):
            raise EvidenceValidationError(
                f"multi_setting_rule.parent_groups[{index}] must be a mapping"
            )
        group_id = _nonempty_string(
            item.get("id"),
            name=f"multi_setting_rule.parent_groups[{index}].id",
        )
        if group_id in parent_group_ids:
            raise EvidenceValidationError(
                f"duplicate parent-group rule id {group_id!r}"
            )
        parent_group_ids.add(group_id)
        parents = _string_list(
            item.get("parents"),
            name=f"multi_setting_rule.parent_groups[{index}].parents",
        )
        unknown_parents = set(parents) - known_parents
        if unknown_parents:
            raise EvidenceValidationError(
                f"parent group {group_id!r} has unknown/non-common parents "
                f"{sorted(unknown_parents)}"
            )
        overlap = assigned_parents & set(parents)
        if overlap:
            raise EvidenceValidationError(
                f"parent groups overlap on {sorted(overlap)}"
            )
        assigned_parents.update(parents)
        minimum = item.get("minimum_joint_pass")
        if isinstance(minimum, bool) or not isinstance(minimum, int):
            raise EvidenceValidationError(
                f"parent group {group_id!r} minimum_joint_pass must be an integer"
            )
        if not 1 <= minimum <= len(parents):
            raise EvidenceValidationError(
                f"parent group {group_id!r} minimum_joint_pass must be in "
                f"[1, {len(parents)}]"
            )
        multiplicity = str(item.get("multiplicity", "bonferroni")).strip()
        if multiplicity != "bonferroni":
            raise EvidenceValidationError(
                f"parent group {group_id!r} multiplicity must be 'bonferroni'"
            )
        parent_groups.append(
            ParentGroupRule(
                group_id=group_id,
                parents=parents,
                minimum_joint_pass=minimum,
                multiplicity=multiplicity,
            )
        )
    if set(primary) & set(stress):
        raise EvidenceValidationError("primary_required cannot contain stress settings")
    both = raw.get("require_both_claims", True)
    if type(both) is not bool:
        raise EvidenceValidationError(
            "multi_setting_rule.require_both_claims must be a boolean"
        )
    return MultiSettingRule(
        rule_id=_nonempty_string(raw.get("id"), name="multi_setting_rule.id"),
        primary_required=primary,
        groups=tuple(groups),
        parent_groups=tuple(parent_groups),
        stress_excluded=stress,
        require_both_claims=both,
    )


def load_contract(path: str | Path) -> EvidenceContract:
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise EvidenceValidationError(f"cannot read evidence config {source}: {error}") from error
    if not isinstance(raw, Mapping):
        raise EvidenceValidationError("evidence config root must be a mapping")
    if raw.get("schema_version") != 1:
        raise EvidenceValidationError("config schema_version must be 1")
    decision = raw.get("decision", {})
    if not isinstance(decision, Mapping):
        raise EvidenceValidationError("decision must be a mapping")
    alpha = decision.get("alpha", 0.05)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise EvidenceValidationError("decision.alpha must be numeric")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise EvidenceValidationError("decision.alpha must be in (0, 1)")
    support = decision.get("minimum_support_units")
    if isinstance(support, bool) or not isinstance(support, int) or support < 1:
        raise EvidenceValidationError(
            "decision.minimum_support_units must be a positive integer"
        )
    prediction_iut = decision.get("prediction_iut")
    if not isinstance(prediction_iut, Mapping):
        raise EvidenceValidationError("decision.prediction_iut must be a mapping")
    if prediction_iut.get("contrasts") != ["joint_minus_s0", "joint_minus_s1"]:
        raise EvidenceValidationError(
            "prediction IUT must contain paired joint_minus_s0 and joint_minus_s1"
        )
    if prediction_iut.get("favorable_sign") != "positive":
        raise EvidenceValidationError("prediction IUT favorable_sign must be positive")
    protection_iut = decision.get("protection_iut")
    if not isinstance(protection_iut, Mapping):
        raise EvidenceValidationError("decision.protection_iut must be a mapping")
    expected_protection = {
        "comparators": ["no_repair", "repeated_random", "s0", "s1"],
        "outcomes": ["mean", "cvar95"],
        "favorable_sign": "negative",
        "common_arms": ["joint", "no_repair", "repeated_random", "s0", "s1"],
        "exact_norm_role": "descriptive_same_estimand_reference_outside_iut",
    }
    for key, expected in expected_protection.items():
        if protection_iut.get(key) != expected:
            raise EvidenceValidationError(
                f"decision.protection_iut.{key} must equal {expected!r}"
            )
    settings = _parse_settings(raw.get("settings"))
    artifacts = _string_list(raw.get("artifacts"), name="artifacts")
    tables = _parse_tables(
        raw.get("tables"), settings=settings, artifacts=set(artifacts)
    )
    multi = _parse_multi_setting(raw.get("multi_setting_rule"), settings=settings)
    outputs = raw.get("outputs", {})
    if not isinstance(outputs, Mapping):
        raise EvidenceValidationError("outputs must be a mapping")
    raw_tex_tables = outputs.get("tex_tables", {}) or {}
    if not isinstance(raw_tex_tables, Mapping):
        raise EvidenceValidationError("outputs.tex_tables must be a mapping")
    unknown_tables = set(raw_tex_tables) - {"core", "robustness"}
    if unknown_tables:
        raise EvidenceValidationError(
            f"outputs.tex_tables has unknown keys {sorted(unknown_tables)}"
        )
    core_table_output = (
        _nonempty_string(raw_tex_tables["core"], name="outputs.tex_tables.core")
        if "core" in raw_tex_tables
        else "sections/generated/table_core_evidence.tex"
    )
    robustness_table_output = (
        _nonempty_string(
            raw_tex_tables["robustness"], name="outputs.tex_tables.robustness"
        )
        if "robustness" in raw_tex_tables
        else "sections/generated/table_robustness.tex"
    )
    raw_fidelity = raw.get("fidelity_inputs", {}) or {}
    if not isinstance(raw_fidelity, Mapping):
        raise EvidenceValidationError("fidelity_inputs must be a mapping")
    fidelity_inputs: dict[str, str] = {}
    for setting_id, path in raw_fidelity.items():
        if str(setting_id) not in settings:
            raise EvidenceValidationError(
                f"fidelity_inputs references unknown setting {setting_id!r}"
            )
        fidelity_inputs[str(setting_id)] = _nonempty_string(
            path, name=f"fidelity_inputs.{setting_id}"
        )
    return EvidenceContract(
        schema_version=1,
        alpha=alpha,
        minimum_support_units=support,
        ledger_path=_nonempty_string(raw.get("ledger"), name="ledger"),
        readiness_output=_nonempty_string(
            outputs.get("readiness_json"), name="outputs.readiness_json"
        ),
        tex_output=_nonempty_string(
            outputs.get("tex_macros"), name="outputs.tex_macros"
        ),
        core_table_output=core_table_output,
        robustness_table_output=robustness_table_output,
        fidelity_inputs=fidelity_inputs,
        settings=settings,
        artifacts=artifacts,
        tables=tables,
        multi_setting=multi,
    )
