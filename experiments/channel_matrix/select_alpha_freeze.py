"""Create a draft alpha freeze from complete development-only protection runs.

This script has no audit input.  It rejects any result carrying a phase other
than ``development`` and never scans the alpha-protection audit directory.
The output remains a draft until a researcher reviews, timestamps, commits,
and explicitly marks it frozen before the audit launcher will accept it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alpha_protection import (  # noqa: E402
    _enabled_models,
    _load_yaml,
    _objective_freeze,
    _validate_contract,
)
from rsus.analysis.channels import DECLARED_CHANNEL  # noqa: E402
from rsus.analysis.mixture import declared_alpha, select_development_alpha  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True, help="development result root only")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)
    _validate_contract(cfg)
    phase = cfg["alpha_protection"]
    models = _enabled_models(cfg, set())
    objective_path, objective_freeze = _objective_freeze(config_path, cfg, models)

    root = Path(args.root).resolve()
    result_paths = sorted(root.glob("**/results.json"))
    if not result_paths:
        parser.error(f"no development results below {root}")

    expected_cells = {
        (model["id"], f"tofu-a{author}", int(seed))
        for model in models
        for author in phase["development"]["authors"]
        for seed in phase["development"]["seeds"]
    }
    cells: dict[tuple[str, str, int], dict] = {}
    artifact_hashes = []
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = payload.get("manifest", {})
        if manifest.get("campaign_phase") != "development":
            raise ValueError(
                f"alpha selection accepts development artifacts only; {path} declares "
                f"{manifest.get('campaign_phase')!r}"
            )
        if manifest.get("campaign_id") != phase["campaign_id"]:
            raise ValueError(f"foreign alpha-protection campaign artifact: {path}")
        if manifest.get("objective_freeze_sha256") != _sha256(objective_path):
            raise ValueError(f"development result used another objective freeze: {path}")
        if manifest.get("normalization_scope") != "discovery_only":
            raise ValueError(f"development result did not use discovery-only ranks: {path}")
        key = (
            str(manifest.get("model_id")),
            str(manifest.get("request")),
            int(manifest.get("seed")),
        )
        if key in cells:
            raise ValueError(f"duplicate development cell {key}: {path}")
        cells[key] = payload
        artifact_hashes.append({"path": str(path), "sha256": _sha256(path)})
    actual_cells = set(cells)
    if actual_cells != expected_cells:
        raise ValueError(
            "development design is not complete and exact; "
            f"missing={sorted(expected_cells - actual_cells)}, "
            f"extra={sorted(actual_cells - expected_cells)}"
        )

    expected_selectors = len(phase["alpha_grid"]) + 3  # mixture grid + none/random/exact
    all_rows = []
    for key, payload in cells.items():
        rows = payload.get("results", [])
        by_parent: dict[str, list[dict]] = {}
        for row in rows:
            if row.get("campaign_phase") != "development":
                raise ValueError(f"non-development row reached alpha selector: {key}")
            by_parent.setdefault(str(row.get("parent")), []).append(row)
            all_rows.append(row)
        if set(by_parent) != set(phase["parents"]):
            raise ValueError(f"parent roster mismatch in development cell {key}")
        for parent, parent_rows in by_parent.items():
            if len(parent_rows) != expected_selectors:
                raise ValueError(
                    f"incomplete selector grid in {key}/{parent}: "
                    f"{len(parent_rows)} != {expected_selectors}"
                )

    expected_run_keys = {
        (f"tofu-a{author}", int(seed))
        for author in phase["development"]["authors"]
        for seed in phase["development"]["seeds"]
    }
    selections = {}
    frozen_models = {}
    unresolved = []
    for model in models:
        model_id = model["id"]
        selections[model_id] = {}
        frozen_models[model_id] = {}
        for parent in phase["parents"]:
            rows = [row for row in all_rows
                    if row["model_id"] == model_id and row["parent"] == parent]
            result = select_development_alpha(
                rows,
                alpha_grid=phase["alpha_grid"],
                expected_run_keys=expected_run_keys,
                prior_alpha=declared_alpha(DECLARED_CHANNEL[parent]),
                recall_max=float(phase["parent"]["recall_max"]),
                utility_retention_min=float(phase["selection"]["utility_retention_min"]),
            )
            selections[model_id][parent] = result
            frozen_models[model_id][parent] = result["alpha"]
            if not result["resolved"]:
                unresolved.append(f"{model_id}/{parent}")

    output = {
        "freeze_id": "PENDING-REVIEW-AND-COMMIT",
        "status": "draft",
        "frozen_before_alpha_audit": False,
        "frozen_at_utc": None,
        "source_campaign": phase["campaign_id"],
        "source_phase": "development",
        "selection_rule": "minimax_cvar05_subject_to_every_run_reach_and_utility",
        "normalization": phase["normalization"],
        "orientation": phase["orientation"],
        "objective_freeze_id": objective_freeze["freeze_id"],
        "objective_freeze_sha256": _sha256(objective_path),
        "campaign_config_sha256": _sha256(config_path),
        "models": frozen_models,
        "unresolved": unresolved,
        "development_artifacts": artifact_hashes,
        "development_diagnostics": selections,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    print(f"wrote development-only draft alpha freeze {out}")
    if unresolved:
        print("UNRESOLVED (do not run alpha audit):")
        for item in unresolved:
            print(f"  {item}")
    else:
        print(
            "All model/parent alpha values resolved. Review diagnostics, assign a dated "
            "freeze_id and frozen_at_utc, set status=frozen and "
            "frozen_before_alpha_audit=true, then commit before audit."
        )


if __name__ == "__main__":
    main()
