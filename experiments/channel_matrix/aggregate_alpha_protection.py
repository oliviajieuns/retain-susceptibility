"""Aggregate the complete frozen alpha-protection audit.

The deployable method is the row marked ``deployed`` by the pre-audit alpha
freeze.  The remaining alpha grid is a descriptive response curve.  Paired
contrasts are reported only where both arms reach the forget criterion and
meet the frozen ordinary-utility floor; missing or failed cells remain counts,
not silently dropped successes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alpha_protection import (  # noqa: E402
    _alpha_freeze,
    _enabled_models,
    _load_yaml,
    _validate_contract,
)
from rsus.analysis.mixture import alpha_label  # noqa: E402


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _hierarchical_paired_ci(rows: list[dict], n_boot: int, seed: int) -> tuple[float, float]:
    """Resample model -> request -> seed while preserving paired differences."""
    by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_model[row["model_id"]][row["request"]].append(float(row["difference"]))
    rng = random.Random(seed)
    model_names = sorted(by_model)
    draws = []
    for _ in range(n_boot):
        values = []
        for _model_index in model_names:
            model = model_names[rng.randrange(len(model_names))]
            requests = sorted(by_model[model])
            for _request_index in requests:
                request = requests[rng.randrange(len(requests))]
                seeds = by_model[model][request]
                for _seed_index in seeds:
                    values.append(seeds[rng.randrange(len(seeds))])
        if values:
            draws.append(sum(values) / len(values))
    lo = _percentile(draws, 0.025)
    hi = _percentile(draws, 0.975)
    return (
        float("nan") if lo is None else lo,
        float("nan") if hi is None else hi,
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)
    _validate_contract(cfg)
    phase = cfg["alpha_protection"]
    models = _enabled_models(cfg, set())
    _, frozen = _alpha_freeze(config_path, cfg, models)
    utility_min = float(phase["selection"]["utility_retention_min"])

    paths = sorted(Path(args.root).glob("**/results.json"))
    if not paths:
        parser.error(f"no alpha audit results under {args.root}")
    expected_cells = {
        (model["id"], f"tofu-a{author}", int(seed))
        for model in models
        for author in phase["audit"]["authors"]
        for seed in phase["audit"]["seeds"]
    }
    cells = {}
    rows = []
    alpha_freeze_ids = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = payload["manifest"]
        if manifest.get("campaign_phase") != "audit":
            raise ValueError(f"non-audit artifact below audit root: {path}")
        if manifest.get("campaign_id") != phase["campaign_id"]:
            raise ValueError(f"foreign campaign artifact: {path}")
        key = (manifest["model_id"], manifest["request"], int(manifest["seed"]))
        if key in cells:
            raise ValueError(f"duplicate alpha audit cell: {key}")
        cells[key] = path
        alpha_freeze_ids.add(manifest.get("alpha_freeze_id"))
        rows.extend(payload["results"])
    if set(cells) != expected_cells:
        raise ValueError(
            "alpha audit is not complete and balanced; "
            f"missing={sorted(expected_cells - set(cells))}, "
            f"extra={sorted(set(cells) - expected_cells)}"
        )
    if alpha_freeze_ids != {frozen["freeze_id"]}:
        raise ValueError(f"audit cells used inconsistent alpha freezes: {alpha_freeze_ids}")

    expected_selectors = {"none", "random", "exact_grad_norm"} | {
        alpha_label(value) for value in phase["alpha_grid"]
    }
    by_cell_parent: dict[tuple[str, str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("campaign_phase") != "audit":
            raise ValueError("non-audit result row reached audit aggregation")
        key = (row["model_id"], row["request"], int(row["seed"]), row["parent"])
        by_cell_parent[key].append(row)
    for key, parent_rows in by_cell_parent.items():
        selectors = [row["selector"] for row in parent_rows]
        if set(selectors) != expected_selectors or len(selectors) != len(expected_selectors):
            raise ValueError(f"incomplete/non-unique selector grid for {key}")
        deployed = [row for row in parent_rows if row.get("deployed")]
        if len(deployed) != 1 or deployed[0].get("selector_type") != "mixture":
            raise ValueError(f"expected exactly one frozen deployed mixture for {key}")

    curve_rows = []
    for parent in phase["parents"]:
        for alpha in phase["alpha_grid"]:
            selector = alpha_label(float(alpha))
            members = [row for row in rows
                       if row["parent"] == parent and row["selector"] == selector]
            eligible = [row for row in members
                        if row.get("reached")
                        and row.get("utility_retention") is not None
                        and float(row["utility_retention"]) >= utility_min]
            deployed_models = sorted({
                row["model_id"] for row in members if row.get("deployed")
            })
            cvar_metric_rows = [{
                "model_id": row["model_id"],
                "request": row["request"],
                "seed": row["seed"],
                "difference": float(row["cvar05_dnll"]),
            } for row in eligible]
            curve_lo, curve_hi = _hierarchical_paired_ci(
                cvar_metric_rows, args.n_boot, args.seed
            )
            curve_rows.append({
                "parent": parent,
                "channel": members[0]["channel"],
                "alpha": float(alpha),
                "declared_prior": bool(members[0]["declared_prior"]),
                "deployed_model_count": len(deployed_models),
                "deployed_models": ",".join(deployed_models),
                "n_total": len(members),
                "n_reach": sum(bool(row.get("reached")) for row in members),
                "n_utility_eligible": len(eligible),
                "mean_cvar05_dnll_eligible": _mean([
                    float(row["cvar05_dnll"]) for row in eligible
                ]),
                "cvar05_ci95_lo": curve_lo if math.isfinite(curve_lo) else None,
                "cvar05_ci95_hi": curve_hi if math.isfinite(curve_hi) else None,
                "mean_dnll_eligible": _mean([
                    float(row["mean_dnll"]) for row in eligible
                ]),
                "mean_utility_retention_reached": _mean([
                    float(row["utility_retention"]) for row in members
                    if row.get("reached") and row.get("utility_retention") is not None
                ]),
            })

    contrast_rows = []
    comparator_selectors = {
        "none", "random", "exact_grad_norm", alpha_label(0.0), alpha_label(1.0)
    }
    for parent in phase["parents"]:
        parent_cells = {
            key: value for key, value in by_cell_parent.items() if key[-1] == parent
        }
        declared_selector = alpha_label(
            0.0 if next(iter(parent_cells.values()))[0]["channel"] == "loss_gradient" else 1.0
        )
        for comparator in sorted(comparator_selectors):
            pairs = []
            for key, members in parent_cells.items():
                deployed = next(row for row in members if row.get("deployed"))
                compare = next(row for row in members if row["selector"] == comparator)
                if deployed["selector"] == comparator:
                    continue
                both_eligible = all(
                    row.get("reached")
                    and row.get("utility_retention") is not None
                    and float(row["utility_retention"]) >= utility_min
                    and row.get("cvar05_dnll") is not None
                    for row in (deployed, compare)
                )
                if both_eligible:
                    pairs.append({
                        "model_id": key[0],
                        "request": key[1],
                        "seed": key[2],
                        "difference": (
                            float(deployed["cvar05_dnll"])
                            - float(compare["cvar05_dnll"])
                        ),
                    })
            differences = [row["difference"] for row in pairs]
            lo, hi = _hierarchical_paired_ci(pairs, args.n_boot, args.seed)
            contrast_rows.append({
                "parent": parent,
                "deployed_alpha_by_model": json.dumps({
                    model["id"]: float(frozen["models"][model["id"]][parent])
                    for model in models
                }, sort_keys=True),
                "comparator": comparator,
                "comparator_is_declared_prior": comparator == declared_selector,
                "n_paired_eligible": len(pairs),
                "mean_cvar_difference_deployed_minus_comparator": _mean(differences),
                "ci95_lo": lo if math.isfinite(lo) else None,
                "ci95_hi": hi if math.isfinite(hi) else None,
                "deployed_better_on_all_paired": bool(differences)
                    and all(value < 0 for value in differences),
            })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "alpha_protection_curve.csv", curve_rows)
    _write_csv(out / "alpha_protection_contrasts.csv", contrast_rows)
    summary = {
        "schema": "channel-mixture-protection-aggregate-v1",
        "campaign_id": phase["campaign_id"],
        "alpha_freeze_id": frozen["freeze_id"],
        "n_cells": len(cells),
        "utility_retention_min": utility_min,
        "curve": curve_rows,
        "paired_contrasts": contrast_rows,
        "interpretation": (
            "negative deployed-minus-comparator CVaR favors the frozen adaptive method; "
            "contrasts are conditional on both arms reaching forgetting and utility"
        ),
    }
    (out / "alpha_protection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote alpha-protection aggregate to {out}")


if __name__ == "__main__":
    main()
