"""Normalized schema for paper-facing evidence rows.

The ledger is intentionally an aggregate boundary.  Upstream runners retain
candidate/request data and bootstrap draws; this schema records the paired
effects, decision bounds, and coverage funnels needed to license (or reject)
paper claims without silently intersecting whatever artifacts happened to
finish.
"""
from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import json


class EvidenceValidationError(ValueError):
    """Raised when a ledger could make an ambiguous or unsafe claim."""


def _as_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:  # bool is intentionally stricter than truthy.
        raise EvidenceValidationError(f"{field_name} must be a boolean")
    return value


def _as_count(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceValidationError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _optional_number(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise EvidenceValidationError(f"{field_name} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise EvidenceValidationError(f"{field_name} must be numeric") from error
    if not math.isfinite(number):
        raise EvidenceValidationError(f"{field_name} must be finite")
    return number


@dataclass(frozen=True)
class Effect:
    estimate: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    p_one_sided: float | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None, *, name: str) -> "Effect":
        data = raw or {}
        if not isinstance(data, Mapping):
            raise EvidenceValidationError(f"{name} must be a mapping")
        effect = cls(
            estimate=_optional_number(data.get("estimate"), field_name=f"{name}.estimate"),
            lower_bound=_optional_number(
                data.get("lower_bound"), field_name=f"{name}.lower_bound"
            ),
            upper_bound=_optional_number(
                data.get("upper_bound"), field_name=f"{name}.upper_bound"
            ),
            p_one_sided=_optional_number(
                data.get("p_one_sided"), field_name=f"{name}.p_one_sided"
            ),
        )
        if effect.p_one_sided is not None and not 0.0 <= effect.p_one_sided <= 1.0:
            raise EvidenceValidationError(f"{name}.p_one_sided must be in [0, 1]")
        return effect

    def complete_for_gain(self) -> bool:
        return all(
            value is not None
            for value in (self.estimate, self.lower_bound, self.p_one_sided)
        )

    def complete_for_reduction(self) -> bool:
        return all(
            value is not None
            for value in (self.estimate, self.upper_bound, self.p_one_sided)
        )


@dataclass(frozen=True)
class Selection:
    valid: bool = False
    fallback: bool = False
    alpha: float | None = None

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any] | None, *, name: str
    ) -> "Selection":
        data = raw or {}
        if not isinstance(data, Mapping):
            raise EvidenceValidationError(f"{name} must be a mapping")
        valid = _as_bool(data.get("valid", False), field_name=f"{name}.valid")
        fallback = _as_bool(
            data.get("fallback", False), field_name=f"{name}.fallback"
        )
        alpha = _optional_number(data.get("alpha"), field_name=f"{name}.alpha")
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise EvidenceValidationError(f"{name}.alpha must be in [0, 1]")
        if fallback and valid:
            raise EvidenceValidationError(
                f"{name} cannot be both a valid selection and a fallback"
            )
        return cls(valid=valid, fallback=fallback, alpha=alpha)


@dataclass(frozen=True)
class Funnel:
    profiles_planned: int = 0
    profiles_valid: int = 0
    trajectories_planned: int = 0
    trajectories_attempted: int = 0
    trajectories_completed: int = 0
    trajectories_reached: int = 0
    reached_with_valid_profile: int = 0
    prediction_common: int = 0
    protection_feasible_all_arms: int = 0
    protection_common: int = 0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None, *, name: str) -> "Funnel":
        data = raw or {}
        if not isinstance(data, Mapping):
            raise EvidenceValidationError(f"{name} must be a mapping")
        values = {
            field_name: _as_count(
                data.get(field_name, 0), field_name=f"{name}.{field_name}"
            )
            for field_name in cls.__dataclass_fields__
        }
        result = cls(**values)
        result.validate(name=name)
        return result

    def validate(self, *, name: str = "funnel") -> None:
        if self.profiles_valid > self.profiles_planned:
            raise EvidenceValidationError(
                f"{name}: profiles_valid exceeds profiles_planned"
            )
        if not (
            self.trajectories_reached
            <= self.trajectories_completed
            <= self.trajectories_attempted
            <= self.trajectories_planned
        ):
            raise EvidenceValidationError(
                f"{name}: require reached <= completed <= attempted <= planned"
            )
        if self.reached_with_valid_profile > self.trajectories_reached:
            raise EvidenceValidationError(
                f"{name}: reached_with_valid_profile exceeds reached"
            )
        if self.prediction_common > self.reached_with_valid_profile:
            raise EvidenceValidationError(
                f"{name}: prediction_common exceeds reached_with_valid_profile"
            )
        if self.protection_feasible_all_arms > self.reached_with_valid_profile:
            raise EvidenceValidationError(
                f"{name}: protection feasible count exceeds reached/valid count"
            )
        if self.protection_common > self.protection_feasible_all_arms:
            raise EvidenceValidationError(
                f"{name}: protection_common exceeds all-arm feasible count"
            )


def _optional_count(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_count(value, field_name=field_name)


@dataclass(frozen=True)
class PredictionEvidence:
    paired: bool = False
    joint_rho: float | None = None
    top_q_recall: float | None = None
    joint: Effect = field(default_factory=Effect)
    vs_s0: Effect = field(default_factory=Effect)
    vs_s1: Effect = field(default_factory=Effect)
    vs_control: Effect = field(default_factory=Effect)
    tail_lift: Effect = field(default_factory=Effect)
    tail_eligible_n: int | None = None
    tail_total_n: int | None = None

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any] | None, *, name: str
    ) -> "PredictionEvidence":
        data = raw or {}
        if not isinstance(data, Mapping):
            raise EvidenceValidationError(f"{name} must be a mapping")
        evidence = cls(
            paired=_as_bool(
                data.get("paired", False), field_name=f"{name}.paired"
            ),
            joint_rho=_optional_number(
                data.get("joint_rho"), field_name=f"{name}.joint_rho"
            ),
            top_q_recall=_optional_number(
                data.get("top_q_recall"), field_name=f"{name}.top_q_recall"
            ),
            joint=Effect.from_mapping(data.get("joint"), name=f"{name}.joint"),
            vs_s0=Effect.from_mapping(data.get("vs_s0"), name=f"{name}.vs_s0"),
            vs_s1=Effect.from_mapping(data.get("vs_s1"), name=f"{name}.vs_s1"),
            vs_control=Effect.from_mapping(
                data.get("vs_control"), name=f"{name}.vs_control"
            ),
            tail_lift=Effect.from_mapping(
                data.get("tail_lift"), name=f"{name}.tail_lift"
            ),
            tail_eligible_n=_optional_count(
                data.get("tail_eligible_n"), field_name=f"{name}.tail_eligible_n"
            ),
            tail_total_n=_optional_count(
                data.get("tail_total_n"), field_name=f"{name}.tail_total_n"
            ),
        )
        if evidence.joint_rho is not None and not -1.0 <= evidence.joint_rho <= 1.0:
            raise EvidenceValidationError(f"{name}.joint_rho must be in [-1, 1]")
        if evidence.top_q_recall is not None and not 0.0 <= evidence.top_q_recall <= 1.0:
            raise EvidenceValidationError(f"{name}.top_q_recall must be in [0, 1]")
        if (
            evidence.tail_eligible_n is not None
            and evidence.tail_total_n is not None
            and evidence.tail_eligible_n > evidence.tail_total_n
        ):
            raise EvidenceValidationError(
                f"{name}.tail_eligible_n exceeds tail_total_n"
            )
        if (evidence.tail_eligible_n is None) != (evidence.tail_total_n is None):
            raise EvidenceValidationError(
                f"{name} tail counts must be reported together"
            )
        return evidence

    def tail_coverage(self) -> float | None:
        if not self.tail_total_n:
            return None
        assert self.tail_eligible_n is not None
        return self.tail_eligible_n / self.tail_total_n


PROTECTION_COMPARATORS = ("no_repair", "repeated_random", "s0", "s1")
PROTECTION_OUTCOMES = ("mean", "cvar95")
PROTECTION_ABSOLUTE_ARMS = ("joint", "no_repair")


def _absolute_outcomes(
    raw: object, *, name: str
) -> dict[str, dict[str, float]]:
    data = raw or {}
    if not isinstance(data, Mapping):
        raise EvidenceValidationError(f"{name} must be a mapping")
    unknown = set(data) - set(PROTECTION_ABSOLUTE_ARMS)
    if unknown:
        raise EvidenceValidationError(
            f"{name} has unknown arms {sorted(unknown)}"
        )
    result: dict[str, dict[str, float]] = {}
    for arm, raw_outcomes in data.items():
        if not isinstance(raw_outcomes, Mapping):
            raise EvidenceValidationError(f"{name}.{arm} must be a mapping")
        unknown_outcomes = set(raw_outcomes) - set(PROTECTION_OUTCOMES)
        if unknown_outcomes:
            raise EvidenceValidationError(
                f"{name}.{arm} has unknown outcomes {sorted(unknown_outcomes)}"
            )
        result[str(arm)] = {}
        for outcome, value in raw_outcomes.items():
            number = _optional_number(
                value, field_name=f"{name}.{arm}.{outcome}"
            )
            if number is None:
                raise EvidenceValidationError(
                    f"{name}.{arm}.{outcome} must be numeric"
                )
            result[str(arm)][str(outcome)] = number
    return result


UPDATE_DIAGNOSTIC_KEYS = ("accepted", "rolled_back")


def _update_diagnostics(raw: object, *, name: str) -> dict[str, float]:
    data = raw or {}
    if not isinstance(data, Mapping):
        raise EvidenceValidationError(f"{name} must be a mapping")
    unknown = set(data) - set(UPDATE_DIAGNOSTIC_KEYS)
    if unknown:
        raise EvidenceValidationError(f"{name} has unknown keys {sorted(unknown)}")
    result: dict[str, float] = {}
    for key, value in data.items():
        number = _optional_number(value, field_name=f"{name}.{key}")
        if number is None or number < 0.0:
            raise EvidenceValidationError(
                f"{name}.{key} must be a non-negative number"
            )
        result[str(key)] = number
    return result


@dataclass(frozen=True)
class ProtectionEvidence:
    paired: bool = False
    comparisons: Mapping[str, Mapping[str, Effect]] = field(default_factory=dict)
    exact_norm: Mapping[str, Effect] = field(default_factory=dict)
    absolute: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    native: Mapping[str, Effect] = field(default_factory=dict)
    update_diagnostics: Mapping[str, float] = field(default_factory=dict)
    min_forget_margin: float | None = None
    min_utility_margin: float | None = None

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any] | None, *, name: str
    ) -> "ProtectionEvidence":
        data = raw or {}
        if not isinstance(data, Mapping):
            raise EvidenceValidationError(f"{name} must be a mapping")
        raw_comparisons = data.get("comparisons", {}) or {}
        if not isinstance(raw_comparisons, Mapping):
            raise EvidenceValidationError(f"{name}.comparisons must be a mapping")
        comparisons: dict[str, dict[str, Effect]] = {}
        for comparator, raw_outcomes in raw_comparisons.items():
            if comparator not in PROTECTION_COMPARATORS:
                raise EvidenceValidationError(
                    f"{name}.comparisons has unknown comparator {comparator!r}"
                )
            if not isinstance(raw_outcomes, Mapping):
                raise EvidenceValidationError(
                    f"{name}.comparisons.{comparator} must be a mapping"
                )
            unknown = set(raw_outcomes) - set(PROTECTION_OUTCOMES)
            if unknown:
                raise EvidenceValidationError(
                    f"{name}.comparisons.{comparator} has unknown outcomes {sorted(unknown)}"
                )
            comparisons[str(comparator)] = {
                outcome: Effect.from_mapping(
                    raw_outcomes.get(outcome),
                    name=f"{name}.comparisons.{comparator}.{outcome}",
                )
                for outcome in PROTECTION_OUTCOMES
            }
        raw_exact = data.get("exact_norm", {}) or {}
        if not isinstance(raw_exact, Mapping):
            raise EvidenceValidationError(f"{name}.exact_norm must be a mapping")
        unknown_exact = set(raw_exact) - set(PROTECTION_OUTCOMES)
        if unknown_exact:
            raise EvidenceValidationError(
                f"{name}.exact_norm has unknown outcomes {sorted(unknown_exact)}"
            )
        exact = {
            outcome: Effect.from_mapping(
                raw_exact.get(outcome), name=f"{name}.exact_norm.{outcome}"
            )
            for outcome in PROTECTION_OUTCOMES
            if outcome in raw_exact
        }
        raw_native = data.get("native", {}) or {}
        if not isinstance(raw_native, Mapping):
            raise EvidenceValidationError(f"{name}.native must be a mapping")
        unknown_native = set(raw_native) - set(PROTECTION_COMPARATORS)
        if unknown_native:
            raise EvidenceValidationError(
                f"{name}.native has unknown comparators {sorted(unknown_native)}"
            )
        native = {
            str(comparator): Effect.from_mapping(
                value, name=f"{name}.native.{comparator}"
            )
            for comparator, value in raw_native.items()
        }
        return cls(
            paired=_as_bool(
                data.get("paired", False), field_name=f"{name}.paired"
            ),
            comparisons=comparisons,
            exact_norm=exact,
            absolute=_absolute_outcomes(
                data.get("absolute"), name=f"{name}.absolute"
            ),
            native=native,
            update_diagnostics=_update_diagnostics(
                data.get("update_diagnostics"), name=f"{name}.update_diagnostics"
            ),
            min_forget_margin=_optional_number(
                data.get("min_forget_margin"),
                field_name=f"{name}.min_forget_margin",
            ),
            min_utility_margin=_optional_number(
                data.get("min_utility_margin"),
                field_name=f"{name}.min_utility_margin",
            ),
        )


@dataclass(frozen=True)
class EvidenceRow:
    setting: str
    parent: str
    attempted: bool
    completed: bool
    prediction_selection: Selection
    protection_selection: Selection
    funnel: Funnel
    prediction: PredictionEvidence
    protection: ProtectionEvidence

    @property
    def key(self) -> tuple[str, str]:
        return self.setting, self.parent

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, index: int) -> "EvidenceRow":
        if not isinstance(raw, Mapping):
            raise EvidenceValidationError(f"rows[{index}] must be a mapping")
        setting = str(raw.get("setting", "")).strip()
        parent = str(raw.get("parent", "")).strip()
        if not setting or not parent:
            raise EvidenceValidationError(
                f"rows[{index}] requires non-empty setting and parent"
            )
        attempted = _as_bool(
            raw.get("attempted", False), field_name=f"rows[{index}].attempted"
        )
        completed = _as_bool(
            raw.get("completed", False), field_name=f"rows[{index}].completed"
        )
        funnel = Funnel.from_mapping(raw.get("funnel"), name=f"rows[{index}].funnel")
        if completed and not attempted:
            raise EvidenceValidationError(
                f"rows[{index}]: completed row was not attempted"
            )
        if attempted != (funnel.trajectories_attempted > 0):
            raise EvidenceValidationError(
                f"rows[{index}]: attempted flag disagrees with trajectory funnel"
            )
        if completed and (
            funnel.trajectories_completed != funnel.trajectories_planned
            or funnel.trajectories_attempted != funnel.trajectories_planned
        ):
            raise EvidenceValidationError(
                f"rows[{index}]: completed requires every planned trajectory to be "
                "attempted and completed"
            )
        return cls(
            setting=setting,
            parent=parent,
            attempted=attempted,
            completed=completed,
            prediction_selection=Selection.from_mapping(
                raw.get("prediction_selection"),
                name=f"rows[{index}].prediction_selection",
            ),
            protection_selection=Selection.from_mapping(
                raw.get("protection_selection"),
                name=f"rows[{index}].protection_selection",
            ),
            funnel=funnel,
            prediction=PredictionEvidence.from_mapping(
                raw.get("prediction"), name=f"rows[{index}].prediction"
            ),
            protection=ProtectionEvidence.from_mapping(
                raw.get("protection"), name=f"rows[{index}].protection"
            ),
        )


@dataclass(frozen=True)
class ArtifactStatus:
    completed: bool
    path: str | None = None
    sha256: str | None = None
    headline_tex: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None, *, name: str) -> "ArtifactStatus":
        data = raw or {}
        if not isinstance(data, Mapping):
            raise EvidenceValidationError(f"artifact {name!r} must be a mapping")
        completed = _as_bool(
            data.get("completed", False), field_name=f"artifacts.{name}.completed"
        )
        path = data.get("path")
        sha = data.get("sha256")
        headline = data.get("headline_tex")
        if path is not None and not isinstance(path, str):
            raise EvidenceValidationError(f"artifacts.{name}.path must be a string")
        if sha is not None and not isinstance(sha, str):
            raise EvidenceValidationError(f"artifacts.{name}.sha256 must be a string")
        if headline is not None and not isinstance(headline, str):
            raise EvidenceValidationError(
                f"artifacts.{name}.headline_tex must be a string"
            )
        if completed and (not path or not sha):
            raise EvidenceValidationError(
                f"completed artifact {name!r} requires path and sha256"
            )
        if completed and not (
            isinstance(sha, str)
            and len(sha) == 64
            and all(character in "0123456789abcdefABCDEF" for character in sha)
        ):
            raise EvidenceValidationError(
                f"completed artifact {name!r} requires a hexadecimal SHA-256"
            )
        return cls(
            completed=completed,
            path=path,
            sha256=sha,
            headline_tex=headline,
        )


@dataclass(frozen=True)
class EvidenceLedger:
    schema_version: int
    rows: Mapping[tuple[str, str], EvidenceRow]
    artifacts: Mapping[str, ArtifactStatus]

    @classmethod
    def empty(cls) -> "EvidenceLedger":
        return cls(schema_version=1, rows={}, artifacts={})

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvidenceLedger":
        if not isinstance(raw, Mapping):
            raise EvidenceValidationError("ledger root must be a mapping")
        version = raw.get("schema_version")
        if version != 1:
            raise EvidenceValidationError(
                f"unsupported ledger schema_version {version!r}; expected 1"
            )
        rows: dict[tuple[str, str], EvidenceRow] = {}
        raw_rows = raw.get("rows", [])
        if not isinstance(raw_rows, list):
            raise EvidenceValidationError("ledger.rows must be a list")
        for index, item in enumerate(raw_rows):
            row = EvidenceRow.from_mapping(item, index=index)
            if row.key in rows:
                raise EvidenceValidationError(f"duplicate ledger row {row.key!r}")
            rows[row.key] = row
        raw_artifacts = raw.get("artifacts", {}) or {}
        if not isinstance(raw_artifacts, Mapping):
            raise EvidenceValidationError("ledger.artifacts must be a mapping")
        artifacts = {
            str(name): ArtifactStatus.from_mapping(value, name=str(name))
            for name, value in raw_artifacts.items()
        }
        return cls(schema_version=1, rows=rows, artifacts=artifacts)

    @classmethod
    def read(cls, path: str | Path) -> "EvidenceLedger":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvidenceValidationError(f"cannot read ledger {source}: {error}") from error
        return cls.from_mapping(raw)


def validate_artifact_files(
    ledger: EvidenceLedger, *, repository_root: str | Path
) -> None:
    """Verify every completed artifact's hash and paper-table payload shape.

    A digest proves byte identity, not that the bytes are evidence.  Completed
    artifacts therefore also need the schema emitted by ``aggregate_raw.py``;
    an arbitrary JSON file with a matching self-declared hash fails closed.
    """
    root = Path(repository_root).resolve()
    for name, artifact in ledger.artifacts.items():
        if not artifact.completed:
            continue
        assert artifact.path is not None and artifact.sha256 is not None
        declared = Path(artifact.path)
        path = declared if declared.is_absolute() else root / declared
        path = path.resolve()
        if not path.is_file():
            raise EvidenceValidationError(
                f"completed artifact {name!r} does not exist: {path}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest.lower() != artifact.sha256.lower():
            raise EvidenceValidationError(
                f"completed artifact {name!r} sha256 mismatch: "
                f"declared {artifact.sha256}, observed {digest}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EvidenceValidationError(
                f"completed artifact {name!r} is not valid JSON: {error}"
            ) from error
        if not isinstance(payload, Mapping):
            raise EvidenceValidationError(
                f"completed artifact {name!r} root must be a mapping"
            )
        if payload.get("schema_version") != 1 or payload.get("artifact_id") != name:
            raise EvidenceValidationError(
                f"completed artifact {name!r} has the wrong schema/artifact_id"
            )
        if payload.get("complete") is not True:
            raise EvidenceValidationError(
                f"completed artifact {name!r} payload is not complete"
            )
        source_sha = payload.get("source_plan_sha256")
        if not (
            isinstance(source_sha, str)
            and len(source_sha) == 64
            and all(character in "0123456789abcdefABCDEF" for character in source_sha)
        ):
            raise EvidenceValidationError(
                f"completed artifact {name!r} lacks a valid source-plan SHA-256"
            )

        kind = payload.get("kind")
        if name == "campaign_manifest":
            if kind != "plan_manifest" or not all(
                isinstance(payload.get(field), list) and payload[field]
                for field in ("scope_rows", "feasibility_rows", "planned_units")
            ):
                raise EvidenceValidationError(
                    "completed campaign_manifest lacks scope, feasibility, or planned units"
                )
        elif name == "tail_structure":
            supplementary = payload.get("supplementary")
            if (
                kind != "tail_from_prediction"
                or not isinstance(payload.get("rows"), list)
                or not payload["rows"]
                or not isinstance(supplementary, Mapping)
                or set(supplementary) != {"reference_roster", "construction_checks"}
                or any(
                    not isinstance(block, Mapping) or block.get("complete") is not True
                    for block in supplementary.values()
                )
            ):
                raise EvidenceValidationError(
                    "completed tail_structure lacks a complete row/supplementary schema"
                )
        else:
            cells = payload.get("cells")
            planned = payload.get("planned_units")
            observed = payload.get("observed_units")
            missing = payload.get("missing_units")
            if (
                kind != "measurements"
                or isinstance(planned, bool)
                or not isinstance(planned, int)
                or planned < 1
                or observed != planned
                or missing != []
                or not isinstance(cells, list)
                or not cells
                or any(
                    not isinstance(cell, Mapping)
                    or not isinstance(cell.get("source_keys"), list)
                    or not cell["source_keys"]
                    or not isinstance(cell.get("values"), Mapping)
                    for cell in cells
                )
            ):
                raise EvidenceValidationError(
                    f"completed artifact {name!r} lacks complete measurement cells"
                )
