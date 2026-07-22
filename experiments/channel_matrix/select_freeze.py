"""Recommend objective settings from development-only calibration runs.

This script never reads predictor seals or correlations.  It applies the
selection rule already present in the campaign YAML, writes a *draft*
recommendation, and leaves the explicit review/commit/freeze step to the
researcher before any audit run can start.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cvar(values: list[float], frac: float = 0.05) -> float:
    count = max(1, math.ceil(frac * len(values)))
    return sum(sorted(values, reverse=True)[:count]) / count


def _run_summary(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if len(manifest["objectives"]) != 1:
        raise ValueError(f"calibration run must contain one objective: {run_dir}")
    objective = manifest["objectives"][0]
    payload = json.loads(
        (run_dir / f"traj_{objective}" / "damage.json").read_text(encoding="utf-8")
    )
    terminal = payload["snapshots"][-1]
    damage = [float(terminal["nll"][key] - payload["nll0"][key]) for key in payload["nll0"]]
    raw = manifest["objective_configs"][objective]
    setting = {
        "lr": float(raw["lr"]),
        "steps": int(raw["max_steps"]),
    }
    for key in ("beta", "forget_weight", "retain_weight", "rmu_alpha", "rmu_c"):
        if raw.get(key) is not None:
            setting[key] = raw[key]
    return {
        "path": str(run_dir),
        "model": manifest["model_id"],
        "objective": objective,
        "setting_id": run_dir.name,
        "setting": setting,
        "forget_recall": float(terminal["forget_recall"]),
        "mean_dnll": sum(damage) / len(damage),
        "cvar05_dnll": _cvar(damage),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--root", required=True, help="campaign calibration root")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    config_path = Path(a.config).resolve()
    cfg = _load_yaml(config_path)
    rule = cfg["calibration"]["selection"]
    paths = sorted(path.parent for path in Path(a.root).glob("**/run_manifest.json"))
    if not paths:
        p.error(f"no calibration manifests under {a.root}")
    runs = [_run_summary(path) for path in paths]

    expected_campaign = cfg["campaign_id"]
    enabled_models = [m["id"] for m in cfg["models"] if m.get("enabled", True)]
    for path in paths:
        manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("campaign_id") != expected_campaign:
            raise ValueError(
                f"foreign calibration run under root: {path} "
                f"({manifest.get('campaign_id')!r} != {expected_campaign!r})"
            )
        if manifest.get("campaign_phase") != "calibration":
            raise ValueError(f"non-calibration manifest under calibration root: {path}")

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for run in runs:
        groups.setdefault(
            (run["model"], run["objective"], run["setting_id"]), []
        ).append(run)

    expected_groups = {
        (model, objective, setting["id"])
        for model in enabled_models
        for objective, settings in cfg["calibration"]["objective_grid"].items()
        for setting in settings
    }
    actual_groups = set(groups)
    if actual_groups != expected_groups:
        raise ValueError(
            "calibration grid is not complete and exact; "
            f"missing={sorted(expected_groups - actual_groups)}, "
            f"extra={sorted(actual_groups - expected_groups)}"
        )

    expected_runs = len(cfg["calibration"]["authors"]) * len(cfg["calibration"]["seeds"])
    eligible: dict[tuple[str, str], list[dict]] = {}
    diagnostics = []
    for (model, objective, setting_id), members in groups.items():
        complete = len(members) == expected_runs
        each_ok = all(
            row["forget_recall"] <= rule["forget_recall_max"]
            and row["mean_dnll"] <= rule["mean_dnll_max"]
            and row["cvar05_dnll"] <= rule["cvar05_dnll_max"]
            for row in members
        )
        ok = complete and (each_ok if rule.get("require_every_development_run", True) else any(
            row["forget_recall"] <= rule["forget_recall_max"] for row in members
        ))
        aggregate = {
            "model": model,
            "objective": objective,
            "setting_id": setting_id,
            "setting": members[0]["setting"],
            "n_runs": len(members),
            "eligible": ok,
            "forget_recall_max": max(row["forget_recall"] for row in members),
            "mean_dnll": sum(row["mean_dnll"] for row in members) / len(members),
            "cvar05_dnll": sum(row["cvar05_dnll"] for row in members) / len(members),
            "runs": members,
        }
        diagnostics.append(aggregate)
        if ok:
            eligible.setdefault((model, objective), []).append(aggregate)

    models = {}
    unresolved = []
    for model in enabled_models:
        models[model] = {}
        stress_objectives = set(cfg["audit"].get("stress_objectives", []))
        for objective in cfg["audit"]["objectives"] + cfg["audit"].get("stress_objectives", []):
            choices = eligible.get((model, objective), [])
            if not choices and objective not in stress_objectives:
                unresolved.append(f"{model}/{objective}")
                models[model][objective] = {"lr": None, "steps": None}
                continue
            if objective in stress_objectives:
                complete_choices = [
                    row for row in diagnostics
                    if row["model"] == model and row["objective"] == objective
                    and row["n_runs"] == expected_runs
                ]
                if not complete_choices:
                    unresolved.append(f"{model}/{objective}")
                    models[model][objective] = {"lr": None, "steps": None}
                    continue
                winner = min(
                    complete_choices,
                    key=lambda row: (
                        row["forget_recall_max"] > rule["forget_recall_max"],
                        row["forget_recall_max"],
                        row["mean_dnll"],
                        row["cvar05_dnll"],
                        row["setting"]["steps"],
                        row["setting"]["lr"],
                    ),
                )
                models[model][objective] = winner["setting"]
                continue
            winner = min(
                choices,
                key=lambda row: (
                    row["mean_dnll"],
                    row["cvar05_dnll"],
                    row["setting"]["steps"],
                    row["setting"]["lr"],
                ),
            )
            models[model][objective] = winner["setting"]

    payload = {
        "freeze_id": "PENDING-REVIEW-AND-COMMIT",
        "status": "draft",
        "frozen_before_audit": False,
        "frozen_at_utc": None,
        "selection_rule": rule,
        "source_campaign": cfg["campaign_id"],
        "models": models,
        "unresolved": unresolved,
        "development_diagnostics": diagnostics,
    }
    Path(a.out).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"wrote draft recommendation {a.out}")
    if unresolved:
        print("UNRESOLVED (expand the development grid; do not run audit):")
        for item in unresolved:
            print(f"  {item}")
    else:
        print("All core and stress settings are resolved. Review them, assign a dated "
              "freeze_id, set status=frozen and frozen_before_audit=true, then commit it "
              "before launching audit.")


if __name__ == "__main__":
    main()
