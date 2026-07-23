"""Aggregate sealed channel-matrix runs with hierarchical uncertainty.

Point estimates give every model/request/seed run equal weight. Confidence
intervals resample models, requests within model, seeds within request, and
candidates within run. The script reports both the predeclared roster-level
difference-in-differences and every output/representation objective pair.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.analysis.channels import DECLARED_CHANNEL, HEADLINE_PROBE  # noqa: E402
from rsus.analysis.prediction import auroc, spearman, top_k_ids  # noqa: E402
from rsus.sealing import read_scores  # noqa: E402


@dataclass
class Run:
    path: Path
    model: str
    request: str
    seed: int
    scores: dict[str, dict[str, float]]
    damage: dict[str, dict[str, float]]
    terminal_recall: dict[str, float]


def _first_reaching_damage(
    run_dir: Path, objectives: list[str], recall_max: float
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    out = {}
    recall = {}
    for objective in objectives:
        path = run_dir / f"traj_{objective}" / "damage.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        nll0 = payload["nll0"]
        snapshots = payload["snapshots"]
        reaching = next(
            (snapshot for snapshot in snapshots
             if float(snapshot["forget_recall"]) <= recall_max),
            None,
        )
        # Preserve a non-reaching row for coverage diagnostics, but never
        # substitute its terminal outcome for the primary first-reach horizon.
        selected = reaching if reaching is not None else snapshots[-1]
        selected_nll = selected["nll"]
        out[objective] = {
            key: float(selected_nll[key] - nll0[key])
            for key in selected_nll if key in nll0
        }
        recall[objective] = float(payload["snapshots"][-1]["forget_recall"])
    return out, recall


def _load_run(path: Path) -> tuple[Run, dict]:
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    request = manifest["request"]
    ledger = path / "seal_ledger.jsonl"
    scores = {
        predictor: read_scores(path / "seals", ledger, request, predictor)
        for predictor in manifest["predictors"]
    }
    if not scores:
        raise ValueError(f"run has no opened predictor scores: {path}")
    recall_max = float(
        manifest.get("objective_acceptance_rule", {}).get("forget_recall_max", 0.10)
    )
    damage, terminal_recall = _first_reaching_damage(
        path, manifest["objectives"], recall_max
    )
    sizes = {predictor: len(values) for predictor, values in scores.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"predictor seals have unequal candidate counts in {path}: {sizes}")
    score_ids = {predictor: set(values) for predictor, values in scores.items()}
    first_ids = next(iter(score_ids.values()))
    if any(ids != first_ids for ids in score_ids.values()):
        raise ValueError(f"predictor seals have different candidate ids in {path}")
    for predictor, values in scores.items():
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError(f"non-finite score in {path}/{predictor}")
    for objective, values in damage.items():
        if not first_ids <= set(values):
            raise ValueError(f"damage misses sealed candidates in {path}/{objective}")
        if not all(math.isfinite(float(values[key])) for key in first_ids):
            raise ValueError(f"non-finite audit damage in {path}/{objective}")
    return Run(
        path=path,
        model=manifest["model_id"],
        request=request,
        seed=int(manifest["seed"]),
        scores=scores,
        damage=damage,
        terminal_recall=terminal_recall,
    ), manifest


def _common_ids(run: Run, predictor: str, objective: str) -> list[str]:
    score_ids = set(run.scores[predictor])
    missing = score_ids - set(run.damage[objective])
    if missing:
        raise ValueError(
            f"incomplete effect support in {run.path}/{predictor}/{objective}: "
            f"missing {sorted(missing)[:5]}"
        )
    ids = sorted(score_ids)
    if len(ids) < 20:
        raise ValueError(f"too few common candidates in {run.path}: {len(ids)}")
    return ids


def _metrics(run: Run, predictor: str, objective: str, k_frac: float, cvar_frac: float) -> dict:
    ids = _common_ids(run, predictor, objective)
    score = run.scores[predictor]
    damage = run.damage[objective]
    k = max(5, round(k_frac * len(ids)))
    realized = top_k_ids({key: damage[key] for key in ids}, k)
    pred_top = top_k_ids({key: score[key] for key in ids}, k)
    n_tail = max(3, math.ceil(cvar_frac * len(ids)))
    tail = sorted(ids, key=lambda key: -damage[key])[:n_tail]
    return {
        "rho": spearman([score[key] for key in ids], [damage[key] for key in ids]),
        "auroc": auroc([score[key] for key in ids], [key in realized for key in ids]),
        "overlap": len(pred_top & realized) / k,
        "tail_rho": spearman([score[key] for key in tail], [damage[key] for key in tail]),
        "n": len(ids),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, int(q * len(sorted_values))))
    return sorted_values[index]


def _cvar(values: list[float], frac: float) -> float:
    count = max(1, math.ceil(frac * len(values)))
    return _mean(sorted(values, reverse=True)[:count])


def _hierarchical_sample(runs: list[Run], rng: random.Random) -> list[Run]:
    by_model: dict[str, dict[str, list[Run]]] = {}
    for run in runs:
        by_model.setdefault(run.model, {}).setdefault(run.request, []).append(run)
    models = sorted(by_model)
    selected = []
    for _ in models:
        model = models[rng.randrange(len(models))]
        requests = sorted(by_model[model])
        for _ in requests:
            request = requests[rng.randrange(len(requests))]
            seeds = by_model[model][request]
            for _ in seeds:
                selected.append(seeds[rng.randrange(len(seeds))])
    return selected


def _bootstrap_rho(runs: list[Run], predictor: str, objective: str, n_boot: int,
                   rng: random.Random) -> tuple[float, float]:
    values = []
    for _ in range(n_boot):
        cells = []
        for run in _hierarchical_sample(runs, rng):
            ids = _common_ids(run, predictor, objective)
            sampled = [ids[rng.randrange(len(ids))] for _ in ids]
            cells.append(spearman(
                [run.scores[predictor][key] for key in sampled],
                [run.damage[objective][key] for key in sampled],
            ))
        values.append(_mean(cells))
    values.sort()
    return _quantile(values, 0.025), _quantile(values, 0.975)


def _delta(run: Run, output_objectives: list[str], rep_objectives: list[str],
           sampled_ids: list[str] | None = None) -> float:
    grad = HEADLINE_PROBE["gradient"]
    prox = HEADLINE_PROBE["representation"]

    def rho(predictor: str, objective: str) -> float:
        ids = sampled_ids or _common_ids(run, predictor, objective)
        ids = [key for key in ids if key in run.scores[predictor] and key in run.damage[objective]]
        return spearman(
            [run.scores[predictor][key] for key in ids],
            [run.damage[objective][key] for key in ids],
        )

    output_gap = _mean([rho(grad, obj) - rho(prox, obj) for obj in output_objectives])
    rep_gap = _mean([rho(grad, obj) - rho(prox, obj) for obj in rep_objectives])
    return output_gap - rep_gap


def _bootstrap_delta(runs: list[Run], output_objectives: list[str], rep_objectives: list[str],
                     n_boot: int, rng: random.Random) -> tuple[float, float]:
    grad = HEADLINE_PROBE["gradient"]
    prox = HEADLINE_PROBE["representation"]
    all_objs = output_objectives + rep_objectives
    values = []
    for _ in range(n_boot):
        cells = []
        for run in _hierarchical_sample(runs, rng):
            common = set(run.scores[grad]) & set(run.scores[prox])
            for objective in all_objs:
                common &= set(run.damage[objective])
            ids = sorted(common)
            sampled = [ids[rng.randrange(len(ids))] for _ in ids]
            cells.append(_delta(run, output_objectives, rep_objectives, sampled))
        values.append(_mean(cells))
    values.sort()
    return _quantile(values, 0.025), _quantile(values, 0.975)


def _validate(manifests: list[dict]) -> tuple[list[str], list[str], list[str]]:
    fields = [
        "campaign_id",
        "campaign_config_sha256",
        "objective_freeze_id",
        "objective_freeze_sha256",
        "dtype",
        "trainable_scope",
        "probe_seed",
        "objective_acceptance_rule",
        "probe_config",
        "implementation_variants",
        "code_commit",
        "code_dirty",
        "core_objectives",
        "stress_objectives",
        "sentence_encoder",
    ]
    for field in fields:
        values = {json.dumps(m.get(field), sort_keys=True) for m in manifests}
        if len(values) != 1:
            raise ValueError(f"runs disagree on {field}: {sorted(values)}")
    all_objectives = manifests[0]["objectives"]
    core_objectives = manifests[0]["core_objectives"]
    stress_objectives = manifests[0]["stress_objectives"]
    if all_objectives != core_objectives + stress_objectives:
        raise ValueError("manifest objective order is not core + stress")
    predictors = manifests[0]["predictors"]
    for manifest in manifests[1:]:
        if manifest["objectives"] != all_objectives:
            raise ValueError("runs disagree on objective roster/order")
        if manifest["predictors"] != predictors:
            raise ValueError("runs disagree on predictor roster/order")
    if not set(HEADLINE_PROBE.values()) <= set(predictors):
        raise ValueError("headline probes missing from campaign")
    if manifests[0].get("code_dirty"):
        raise ValueError("sealed audit manifest reports a dirty code worktree")
    for model in {manifest["model_id"] for manifest in manifests}:
        shas = {
            manifest.get("fidelity_certificate_sha256")
            for manifest in manifests
            if manifest["model_id"] == model
        }
        if len(shas) != 1 or None in shas:
            raise ValueError(f"runs for {model} disagree on fidelity certificate: {shas}")
    pools_by_request = {}
    for request in {manifest["request"] for manifest in manifests}:
        pools = {
            tuple(manifest.get("candidate_authors") or [])
            for manifest in manifests
            if manifest["request"] == request
        }
        if len(pools) != 1:
            raise ValueError(f"runs for {request} disagree on retained-candidate pool")
        pools_by_request[request] = set(next(iter(pools)))
    requests = sorted(pools_by_request)
    for index, first in enumerate(requests):
        for second in requests[index + 1:]:
            overlap = pools_by_request[first] & pools_by_request[second]
            if overlap:
                raise ValueError(
                    f"audit request candidate pools overlap ({first}, {second}): {sorted(overlap)}"
                )
    return core_objectives, stress_objectives, predictors


def _validate_balanced(runs: list[Run]) -> None:
    models = sorted({run.model for run in runs})
    requests = sorted({run.request for run in runs})
    seeds = sorted({run.seed for run in runs})
    actual = [(run.model, run.request, run.seed) for run in runs]
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate model/request/seed run in aggregate input")
    expected = {(model, request, seed) for model in models for request in requests for seed in seeds}
    missing = sorted(expected - set(actual))
    extra = sorted(set(actual) - expected)
    if missing or extra:
        raise ValueError(f"unbalanced campaign; missing={missing}, extra={extra}")


def _objective_status(
    runs: list[Run], objective: str, rule: dict, cvar_frac: float
) -> dict:
    recall_max = float(rule["forget_recall_max"])
    mean_max = float(rule["mean_dnll_max"])
    cvar_max = float(rule["cvar05_dnll_max"])
    per_run = []
    for run in runs:
        audit_ids = set(next(iter(run.scores.values())))
        values = [run.damage[objective][key] for key in sorted(audit_ids)]
        mean_damage = _mean(values)
        cvar_damage = _cvar(values, cvar_frac)
        recall = run.terminal_recall[objective]
        per_run.append({
            "model": run.model,
            "request": run.request,
            "seed": run.seed,
            "forget_recall": recall,
            "mean_dnll": mean_damage,
            "cvar_dnll": cvar_damage,
            "reached": recall <= recall_max,
            "collapsed": mean_damage > mean_max or cvar_damage > cvar_max,
        })
    return {
        "reach_rate": _mean([float(row["reached"]) for row in per_run]),
        "failed_runs": sum(not row["reached"] for row in per_run),
        "collapsed_runs": sum(row["collapsed"] for row in per_run),
        "mean_audit_dnll": _mean([row["mean_dnll"] for row in per_run]),
        "mean_audit_cvar": _mean([row["cvar_dnll"] for row in per_run]),
        "thresholds": {
            "forget_recall_max": recall_max,
            "mean_dnll_max": mean_max,
            "cvar_dnll_max": cvar_max,
        },
        "runs": per_run,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="audit root containing run_manifest.json files")
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--k-frac", type=float, default=0.10)
    p.add_argument("--cvar-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    root = Path(a.root)
    run_paths = sorted(path.parent for path in root.glob("**/run_manifest.json"))
    if not run_paths:
        p.error(f"no run manifests under {root}")
    loaded = [_load_run(path) for path in run_paths]
    runs = [item[0] for item in loaded]
    manifests = [item[1] for item in loaded]
    objectives, stress_objectives, predictors = _validate(manifests)
    all_objectives = objectives + stress_objectives
    _validate_balanced(runs)
    output_objectives = [o for o in objectives if DECLARED_CHANNEL.get(o) == "loss_gradient"]
    rep_objectives = [o for o in objectives if DECLARED_CHANNEL.get(o) == "representation"]
    if not output_objectives or not rep_objectives:
        raise ValueError("campaign must contain both declared channels")

    rng = random.Random(a.seed)
    rows = []
    for predictor in predictors:
        for objective in all_objectives:
            cells = [_metrics(run, predictor, objective, a.k_frac, a.cvar_frac) for run in runs]
            lo, hi = _bootstrap_rho(runs, predictor, objective, a.n_boot, rng)
            rows.append({
                "predictor": predictor,
                "objective": objective,
                "channel": DECLARED_CHANNEL[objective],
                "rho": _mean([cell["rho"] for cell in cells]),
                "rho_lo": lo,
                "rho_hi": hi,
                "auroc": _mean([cell["auroc"] for cell in cells]),
                "overlap": _mean([cell["overlap"] for cell in cells]),
                "tail_rho": _mean([cell["tail_rho"] for cell in cells]),
                "n_runs": len(cells),
                "n_candidates_min": min(cell["n"] for cell in cells),
            })

    point = _mean([_delta(run, output_objectives, rep_objectives) for run in runs])
    lo, hi = _bootstrap_delta(runs, output_objectives, rep_objectives, a.n_boot, rng)
    pairwise = []
    for output_obj in output_objectives:
        for rep_obj in rep_objectives:
            pair_point = _mean([_delta(run, [output_obj], [rep_obj]) for run in runs])
            pair_lo, pair_hi = _bootstrap_delta(
                runs, [output_obj], [rep_obj], a.n_boot, rng
            )
            pairwise.append({
                "output_objective": output_obj,
                "representation_objective": rep_obj,
                "delta": pair_point,
                "lo": pair_lo,
                "hi": pair_hi,
            })

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with open(out / "pooled_channel_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema": "pooled-channel-report-v1",
        "campaign_id": manifests[0].get("campaign_id"),
        "objective_freeze_id": manifests[0].get("objective_freeze_id"),
        "models": sorted({run.model for run in runs}),
        "requests": sorted({run.request for run in runs}),
        "seeds": sorted({run.seed for run in runs}),
        "n_runs": len(runs),
        "protocol": {
            "dtype": manifests[0]["dtype"],
            "trainable_scope": manifests[0]["trainable_scope"],
            "probe_config": manifests[0]["probe_config"],
            "audit_candidates_per_run": sorted({
                len(next(iter(run.scores.values()))) for run in runs
            }),
        },
        "objectives": objectives,
        "stress_objectives": stress_objectives,
        "all_objectives": all_objectives,
        "predictors": predictors,
        "objective_status": {
            objective: _objective_status(
                runs,
                objective,
                manifests[0]["objective_acceptance_rule"],
                a.cvar_frac,
            )
            for objective in all_objectives
        },
        "roster_interaction": {
            "definition": "mean_OC(rho_fd_norm-rho_knn_feature) - mean_RC(rho_fd_norm-rho_knn_feature)",
            "delta": point,
            "lo": lo,
            "hi": hi,
            "n_boot": a.n_boot,
        },
        "pairwise_interactions": pairwise,
    }
    (out / "pooled_channel_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    model_rows = []
    for model in sorted({run.model for run in runs}):
        subset = [run for run in runs if run.model == model]
        for predictor in predictors:
            for objective in all_objectives:
                cells = [_metrics(run, predictor, objective, a.k_frac, a.cvar_frac) for run in subset]
                model_rows.append({
                    "model": model,
                    "predictor": predictor,
                    "objective": objective,
                    "rho": _mean([cell["rho"] for cell in cells]),
                    "n_runs": len(cells),
                })
    with open(out / "model_channel_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(model_rows[0]))
        writer.writeheader()
        writer.writerows(model_rows)

    print(
        f"wrote {out}; pooled roster interaction={point:+.3f} "
        f"95% CI [{lo:+.3f}, {hi:+.3f}] across {len(runs)} runs"
    )


if __name__ == "__main__":
    main()
