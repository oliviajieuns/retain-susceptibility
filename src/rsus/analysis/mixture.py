"""Rank-normalized soft channel routing and development-only alpha selection.

The public score convention is

    s_alpha = (1 - alpha) * rank01(gradient) + alpha * rank01(proximity),

so alpha=0 is the output/loss-gradient endpoint and alpha=1 is the hidden-
representation endpoint.  Selection helpers in this module are deliberately
dependency-light and reject sealed-audit rows: a deployable alpha may be fit
only on the development requests named by the prospective campaign contract.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


ALPHA_ORIENTATION = {
    "loss_gradient": 0.0,
    "representation": 1.0,
    "hybrid": 0.5,
}


def _finite_score(value: object, candidate_id: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"non-numeric score for {candidate_id!r}: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"non-finite score for {candidate_id!r}: {number!r}")
    return number


def validate_alpha(alpha: float) -> float:
    """Return a finite alpha in [0,1], otherwise fail loudly."""
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"alpha must be finite and in [0,1], got {alpha!r}")
    return value


def rank01(scores: Mapping[str, float]) -> dict[str, float]:
    """Map scores to [0,1] with candidate-ID tie breaking.

    This is an ordinal transform rather than a mid-rank transform.  It makes
    every protect ranking total and reproducible, and guarantees that each
    endpoint induces exactly the corresponding raw-score ranking under the
    same ``(-score, candidate_id)`` allocation rule.
    """
    if not scores:
        raise ValueError("cannot rank-normalize an empty score mapping")
    clean = {str(cid): _finite_score(value, str(cid)) for cid, value in scores.items()}
    if len(clean) != len(scores):
        raise ValueError("candidate ids must be unique after string normalization")
    # Allocate rank 1 to the first element under the same descending-score,
    # ascending-ID order used by ScoreProfile.ranking().
    order = sorted(clean, key=lambda cid: (-clean[cid], cid))
    if len(order) == 1:
        return {order[0]: 0.5}
    scale = len(order) - 1
    return {cid: 1.0 - index / scale for index, cid in enumerate(order)}


def channel_mixture_scores(
    gradient_scores: Mapping[str, float],
    proximity_scores: Mapping[str, float],
    alpha: float,
    *,
    candidate_ids: Iterable[str] | None = None,
) -> dict[str, float]:
    """Compute the frozen rank mixture on an exact candidate set.

    ``candidate_ids`` is important for protection: normalization is performed
    inside the discovery fold only, so neither audit outcomes nor audit-side
    score distributions can influence a protect pool.  Prediction analysis
    should analogously pass the sealed audit IDs being evaluated.
    """
    weight = validate_alpha(alpha)
    g_keys, h_keys = set(gradient_scores), set(proximity_scores)
    if g_keys != h_keys:
        raise ValueError(
            "gradient/proximity candidate coverage differs: "
            f"gradient_only={sorted(g_keys - h_keys)[:5]}, "
            f"proximity_only={sorted(h_keys - g_keys)[:5]}"
        )
    ids = set(g_keys if candidate_ids is None else (str(cid) for cid in candidate_ids))
    missing = ids - g_keys
    if missing:
        raise ValueError(f"mixture candidate set has missing scores: {sorted(missing)[:5]}")
    if not ids:
        raise ValueError("mixture candidate set is empty")
    rg = rank01({cid: gradient_scores[cid] for cid in ids})
    rh = rank01({cid: proximity_scores[cid] for cid in ids})
    return {cid: (1.0 - weight) * rg[cid] + weight * rh[cid] for cid in sorted(ids)}


def alpha_label(alpha: float) -> str:
    """Canonical filesystem/selector label, e.g. ``s_alpha_0p25``."""
    value = validate_alpha(alpha)
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return "s_alpha_" + text.replace(".", "p")


def declared_alpha(channel: str) -> float:
    try:
        return ALPHA_ORIENTATION[channel]
    except KeyError:
        raise ValueError(f"no declared alpha prior for channel {channel!r}") from None


def select_development_alpha(
    rows: Sequence[Mapping[str, object]],
    *,
    alpha_grid: Sequence[float],
    expected_run_keys: Iterable[tuple[str, int]],
    prior_alpha: float,
    recall_max: float,
    utility_retention_min: float,
) -> dict[str, object]:
    """Minimax development selection under forgetting and utility constraints.

    Rows must belong to the ``development`` phase and one model/parent pair.
    A weight is feasible only when every prospective (request, seed) run is
    present, reached the forgetting criterion, and retained ordinary utility.
    The winner minimizes worst-run audit-fold CVaR, then mean CVaR, then
    distance to the declared channel prior.  If no weight is feasible the
    result is unresolved; there is no best-effort fallback to be audited.
    """
    grid = tuple(validate_alpha(value) for value in alpha_grid)
    if len(set(grid)) != len(grid) or not grid:
        raise ValueError("alpha_grid must be non-empty and unique")
    expected = {(str(request), int(seed)) for request, seed in expected_run_keys}
    if not expected:
        raise ValueError("expected_run_keys must be non-empty")
    prior = validate_alpha(prior_alpha)

    foreign = sorted({str(row.get("campaign_phase")) for row in rows
                      if row.get("campaign_phase") != "development"})
    if foreign:
        raise ValueError(
            "alpha selection accepts development rows only; "
            f"found phases {foreign}. Sealed audit must never select alpha."
        )

    diagnostics: list[dict[str, object]] = []
    feasible: list[dict[str, object]] = []
    for alpha in grid:
        members = [row for row in rows
                   if row.get("selector_type") == "mixture"
                   and math.isclose(float(row.get("alpha")), alpha, abs_tol=1e-12)]
        keyed: dict[tuple[str, int], Mapping[str, object]] = {}
        duplicates: list[tuple[str, int]] = []
        for row in members:
            key = (str(row.get("request")), int(row.get("seed")))
            if key in keyed:
                duplicates.append(key)
            keyed[key] = row
        actual = set(keyed)
        complete = actual == expected and not duplicates
        constraints_ok = complete and all(
            bool(row.get("reached"))
            and float(row.get("forget_recall")) <= recall_max
            and row.get("utility_retention") is not None
            and float(row.get("utility_retention")) >= utility_retention_min
            and row.get("cvar05_dnll") is not None
            and math.isfinite(float(row.get("cvar05_dnll")))
            for row in keyed.values()
        )
        cvars = [float(row["cvar05_dnll"]) for row in keyed.values()
                 if row.get("cvar05_dnll") is not None]
        diagnostic: dict[str, object] = {
            "alpha": alpha,
            "n_runs": len(members),
            "complete": complete,
            "feasible": constraints_ok,
            "missing_runs": sorted(expected - actual),
            "extra_runs": sorted(actual - expected),
            "duplicate_runs": sorted(set(duplicates)),
            "worst_cvar05_dnll": max(cvars) if cvars else None,
            "mean_cvar05_dnll": sum(cvars) / len(cvars) if cvars else None,
            "min_utility_retention": (
                min(float(row["utility_retention"]) for row in keyed.values()
                    if row.get("utility_retention") is not None)
                if any(row.get("utility_retention") is not None for row in keyed.values())
                else None
            ),
            "max_forget_recall": (
                max(float(row["forget_recall"]) for row in keyed.values())
                if keyed else None
            ),
        }
        diagnostics.append(diagnostic)
        if constraints_ok:
            feasible.append(diagnostic)

    if not feasible:
        return {
            "resolved": False,
            "alpha": None,
            "prior_alpha": prior,
            "selection_rule": "minimax_cvar05_subject_to_every_run_reach_and_utility",
            "diagnostics": diagnostics,
        }
    winner = min(
        feasible,
        key=lambda item: (
            float(item["worst_cvar05_dnll"]),
            float(item["mean_cvar05_dnll"]),
            abs(float(item["alpha"]) - prior),
            float(item["alpha"]),
        ),
    )
    return {
        "resolved": True,
        "alpha": float(winner["alpha"]),
        "prior_alpha": prior,
        "selection_rule": "minimax_cvar05_subject_to_every_run_reach_and_utility",
        "winner": winner,
        "diagnostics": diagnostics,
    }
