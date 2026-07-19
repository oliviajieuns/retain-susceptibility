"""Table-1 prediction metrics (appendix 'Result Variables and Checkpoint
Rules'): per-cell Spearman rho, AUROC for realized top-K damage membership,
Overlap@K; request-level Tail rho; CVaR summaries. Dependency-free
implementations on plain dicts so the analysis layer runs anywhere.
"""
from __future__ import annotations

import math


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("need two equal-length vectors of size >= 2")
    ra, rb = _ranks(a), _ranks(b)
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va > 0 and vb > 0 else 0.0


def auroc(scores: list[float], labels: list[bool]) -> float:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        raise ValueError("AUROC needs both classes")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def top_k_ids(values: dict[str, float], k: int) -> set[str]:
    return set(sorted(values, key=lambda c: (-values[c], c))[:k])


def cvar_upper(values: list[float], frac: float = 0.05) -> float:
    if not values:
        raise ValueError("empty values")
    n = max(1, math.ceil(frac * len(values)))
    return sum(sorted(values, reverse=True)[:n]) / n


def per_cell_metrics(
    scores: dict[str, float], damage: dict[str, float], k: int
) -> dict[str, float]:
    """One (request, optimizer) cell over a common candidate id set."""
    ids = sorted(scores)
    if set(ids) != set(damage):
        raise ValueError("score/damage id mismatch")
    s = [scores[c] for c in ids]
    d = [damage[c] for c in ids]
    realized = top_k_ids(damage, k)
    return {
        "rho": spearman(s, d),
        "auroc": auroc(s, [c in realized for c in ids]),
        "overlap": len(top_k_ids(scores, k) & realized) / k,
    }


def tail_rho(request_pairs: list[tuple[float, float]]) -> float:
    """Correlate score-CVaR with damage-CVaR across requests (one optimizer)."""
    return spearman([p[0] for p in request_pairs], [p[1] for p in request_pairs])


def macro_with_range(values: list[float]) -> dict[str, float]:
    return {"mean": sum(values) / len(values), "lo": min(values), "hi": max(values)}


def table1_rows(
    scores_by_predictor: dict[str, dict[str, dict[str, float]]],
    damage_by_optimizer: dict[str, dict[str, dict[str, float]]],
    k: int,
    cvar_frac: float = 0.05,
) -> dict[str, dict[str, object]]:
    """Assemble Table-1 rows.

    scores_by_predictor[predictor][request_id] -> {cand: score}
    damage_by_optimizer[optimizer][request_id] -> {cand: damage}
    Returns row[predictor] = {"<opt>_rho": macro over requests,
    "auroc"/"overlap": macro over all cells, "tail_rho": macro over
    optimizers, each with (lo, hi) ranges where applicable}.
    """
    rows: dict[str, dict[str, object]] = {}
    for pred, by_req in scores_by_predictor.items():
        row: dict[str, object] = {}
        all_cells: list[dict[str, float]] = []
        tails: list[float] = []
        for opt, dmg_by_req in damage_by_optimizer.items():
            cells = []
            pairs = []
            for rid, dmg in dmg_by_req.items():
                sc = {c: by_req[rid][c] for c in dmg}
                cells.append(per_cell_metrics(sc, dmg, k))
                pairs.append(
                    (cvar_upper(list(sc.values()), cvar_frac), cvar_upper(list(dmg.values()), cvar_frac))
                )
            row[f"{opt}_rho"] = macro_with_range([c["rho"] for c in cells])
            all_cells.extend(cells)
            if len(pairs) >= 2:
                tails.append(tail_rho(pairs))
        row["auroc"] = macro_with_range([c["auroc"] for c in all_cells])
        row["overlap"] = macro_with_range([c["overlap"] for c in all_cells])
        row["tail_rho"] = macro_with_range(tails) if tails else None
        rows[pred] = row
    return rows
