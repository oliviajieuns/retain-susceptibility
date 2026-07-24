"""Select the frozen prediction weight from development alpha cells only.

The protection selector consumes repair outcomes; this one consumes only the
entry-checkpoint (no-repair) damage of the development requests, ranking the
frozen mixture S_alpha against realized damage per request and picking the
grid point by equal-request Spearman, then top-tail recall, then midpoint
distance (rsus.analysis.mixture.select_prediction_alpha).  It never reads an
audit directory and refuses rows from any phase but development.

Output is a draft: review, set status: frozen, and commit before the raw-plan
/ exporter step consumes it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alpha_protection import _enabled_models, _load_yaml, _validate_contract  # noqa: E402
from rsus.analysis.mixture import channel_mixture_scores, select_prediction_alpha  # noqa: E402


def _midranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    return ranks


def _spearman(left, right):
    ra, rb = _midranks(left), _midranks(right)
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    num = sum((a - ma) * (b - mb) for a, b in zip(ra, rb))
    va = sum((a - ma) ** 2 for a in ra)
    vb = sum((b - mb) ** 2 for b in rb)
    return num / math.sqrt(va * vb) if va > 0 and vb > 0 else None


def _top_q_recall(scores, damage, ids, q):
    count = max(1, math.ceil(q * len(ids)))
    top = lambda values: set(
        sorted(range(len(ids)), key=lambda i: (-values[i], ids[i]))[:count]
    )
    return len(top(scores) & top(damage)) / count


def _profile_scores(cell: Path, scorer: str) -> tuple[dict[str, float], list[str]]:
    payload = json.loads((cell / "profile_artifacts" / f"{scorer}.json").read_text())
    scores, discovery = {}, []
    for row in payload["candidates"]:
        cid = str(row["candidate_id"])
        if row.get("score") is not None:
            scores[cid] = float(row["score"])
        if row.get("fold") == "discovery":
            discovery.append(cid)
    return scores, discovery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True, help="development result root only")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-q", type=float, default=0.10)
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config).resolve())
    _validate_contract(cfg)
    phase = cfg["alpha_protection"]
    models = _enabled_models(cfg, set())
    grid = [float(a) for a in phase["alpha_grid"]]
    probes = phase["probes"]

    output = {"models": {}, "diagnostics": {}}
    for model in models:
        model_id = model["id"]
        per_parent: dict[str, float | None] = {}
        for parent in phase["parents"]:
            rows = []
            for author in phase["development"]["authors"]:
                for seed in phase["development"]["seeds"]:
                    cell = (Path(args.root).resolve() / model_id
                            / f"tofu-a{author}" / f"seed-{seed}")
                    results = json.loads((cell / "results.json").read_text())["results"]
                    none_row = next(
                        row for row in results
                        if row["parent"] == parent and row["selector"] == "none"
                    )
                    if str(none_row.get("campaign_phase")) != "development":
                        raise SystemExit("refusing non-development rows")
                    damage = {
                        str(cid): float(value)
                        for cid, value in (none_row.get("candidate_damage") or {}).items()
                    }
                    grad, discovery = _profile_scores(cell, probes["gradient"])
                    prox, _ = _profile_scores(cell, probes["proximity"])
                    ids = sorted(set(damage) & set(grad) & set(prox))
                    if len(ids) < 2:
                        continue
                    for alpha in grid:
                        mixture = channel_mixture_scores(
                            grad, prox, alpha,
                            candidate_ids=ids,
                            normalization_ids=[c for c in discovery if c in grad],
                        )
                        scores = [mixture[c] for c in ids]
                        target = [damage[c] for c in ids]
                        rho = _spearman(scores, target)
                        rows.append({
                            "selector_type": "mixture",
                            "alpha": alpha,
                            "request": f"tofu-a{author}",
                            "seed": int(seed),
                            "campaign_phase": "development",
                            "reached": bool(none_row.get("reached")),
                            "spearman": rho,
                            "top_q_recall": _top_q_recall(scores, target, ids, args.top_q),
                        })
            expected = [
                (f"tofu-a{author}", int(seed))
                for author in phase["development"]["authors"]
                for seed in phase["development"]["seeds"]
            ]
            selection = select_prediction_alpha(
                rows,
                alpha_grid=grid,
                expected_run_keys=expected,
                min_reached_requests=len(phase["development"]["authors"]),
            )
            per_parent[parent] = selection["alpha"] if selection["resolved"] else None
            output["diagnostics"].setdefault(model_id, {})[parent] = selection
        output["models"][model_id] = per_parent

    resolved = {
        f"{model_id}/{parent}"
        for model_id, parents in output["models"].items()
        for parent, alpha in parents.items()
        if alpha is not None
    }
    payload = {
        "freeze_id": "PENDING-REVIEW-AND-COMMIT",
        "status": "draft",
        "selection_rule": "equal_request_spearman_then_top_q_midpoint_smaller",
        "damage_source": "development none-arm entry-checkpoint damage only",
        "prediction_alpha": {
            parent: alpha
            for parents in output["models"].values()
            for parent, alpha in parents.items()
            if alpha is not None
        },
        **output,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"wrote {out}")
    print(f"resolved: {sorted(resolved) or 'NONE'}")


if __name__ == "__main__":
    main()
