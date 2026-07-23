"""Rank-normalized soft channel routing and development-only alpha selection.

The public score convention is

    s_alpha = (1 - alpha) * midrank01(gradient) + alpha * midrank01(proximity),

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


def empirical_midrank01(
    reference_scores: Mapping[str, float],
    target_scores: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Apply the empirical midrank CDF fitted on ``reference_scores``.

    For a value ``z`` this returns the fraction of reference values strictly
    below ``z`` plus half the fraction tied with it.  Tied scores therefore
    remain tied: candidate IDs are reserved for the final top-K allocation
    boundary and never alter correlations.  Passing ``target_scores`` applies
    the frozen reference transform to another fold without refitting it.
    """
    if not reference_scores:
        raise ValueError("cannot fit a midrank transform on empty scores")
    reference = {
        str(cid): _finite_score(value, str(cid))
        for cid, value in reference_scores.items()
    }
    if len(reference) != len(reference_scores):
        raise ValueError("candidate ids must be unique after string normalization")
    targets = reference if target_scores is None else {
        str(cid): _finite_score(value, str(cid))
        for cid, value in target_scores.items()
    }
    if len(targets) != len(target_scores or reference_scores):
        raise ValueError("candidate ids must be unique after string normalization")

    ordered = sorted(reference.values())
    n_reference = len(ordered)
    # bisect is intentionally local to keep this dependency-light module fast
    # for large candidate pools.
    from bisect import bisect_left, bisect_right

    return {
        cid: (
            bisect_left(ordered, value)
            + 0.5 * (bisect_right(ordered, value) - bisect_left(ordered, value))
        ) / n_reference
        for cid, value in targets.items()
    }


def rank01(scores: Mapping[str, float]) -> dict[str, float]:
    """Backward-compatible name for the empirical midrank transform."""
    if not scores:
        raise ValueError("cannot rank-normalize an empty score mapping")
    return empirical_midrank01(scores)


def channel_mixture_scores(
    gradient_scores: Mapping[str, float],
    proximity_scores: Mapping[str, float],
    alpha: float,
    *,
    candidate_ids: Iterable[str] | None = None,
    normalization_ids: Iterable[str] | None = None,
) -> dict[str, float]:
    """Compute the frozen rank mixture on an exact candidate set.

    ``candidate_ids`` chooses the returned fold. ``normalization_ids`` chooses
    the discovery fold on which both empirical transforms are fitted.  When it
    is omitted, the returned fold is also the normalization fold, preserving
    the protection-path behavior.  Prediction should pass audit IDs as
    ``candidate_ids`` and discovery IDs as ``normalization_ids``.
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
    reference_ids = set(
        ids if normalization_ids is None else (str(cid) for cid in normalization_ids)
    )
    missing = (ids | reference_ids) - g_keys
    if missing:
        raise ValueError(f"mixture candidate set has missing scores: {sorted(missing)[:5]}")
    if not ids:
        raise ValueError("mixture candidate set is empty")
    if not reference_ids:
        raise ValueError("mixture normalization set is empty")
    rg = empirical_midrank01(
        {cid: gradient_scores[cid] for cid in reference_ids},
        {cid: gradient_scores[cid] for cid in ids},
    )
    rh = empirical_midrank01(
        {cid: proximity_scores[cid] for cid in reference_ids},
        {cid: proximity_scores[cid] for cid in ids},
    )
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


def select_prediction_alpha(
    rows: Sequence[Mapping[str, object]],
    *,
    alpha_grid: Sequence[float],
    expected_run_keys: Iterable[tuple[str, int]],
    min_reached_requests: int,
    fallback_alpha: float = 0.5,
) -> dict[str, object]:
    """Select a prediction weight on development requests only.

    Seeds are averaged within each request and requests receive equal weight.
    The primary criterion is within-request Spearman correlation, followed by
    top-tail recall, distance to the readout-independent midpoint 0.5, and the
    smaller weight.  Missing expected cells are never discarded.  If no grid
    point has the required complete, criterion-reaching request support, the
    declared fallback is returned and marked unresolved/claim-ineligible.
    """
    grid = tuple(validate_alpha(value) for value in alpha_grid)
    if len(set(grid)) != len(grid) or not grid:
        raise ValueError("alpha_grid must be non-empty and unique")
    expected = {(str(request), int(seed)) for request, seed in expected_run_keys}
    if not expected:
        raise ValueError("expected_run_keys must be non-empty")
    if int(min_reached_requests) < 1:
        raise ValueError("min_reached_requests must be positive")
    fallback = validate_alpha(fallback_alpha)
    foreign = sorted({
        str(row.get("campaign_phase"))
        for row in rows
        if row.get("campaign_phase") != "development"
    })
    if foreign:
        raise ValueError(
            "prediction alpha selection accepts development rows only; "
            f"found phases {foreign}. Sealed audit must never select alpha."
        )

    expected_seeds: dict[str, set[int]] = {}
    for request, seed in expected:
        expected_seeds.setdefault(request, set()).add(seed)

    diagnostics: list[dict[str, object]] = []
    eligible: list[dict[str, object]] = []
    for alpha in grid:
        members = [
            row for row in rows
            if row.get("selector_type") == "mixture"
            and row.get("alpha") is not None
            and math.isclose(float(row["alpha"]), alpha, abs_tol=1e-12)
        ]
        keyed: dict[tuple[str, int], Mapping[str, object]] = {}
        duplicates: list[tuple[str, int]] = []
        for row in members:
            key = (str(row.get("request")), int(row.get("seed")))
            if key in keyed:
                duplicates.append(key)
            keyed[key] = row
        actual = set(keyed)
        complete = actual == expected and not duplicates

        request_rho: list[float] = []
        request_recall: list[float] = []
        reached_requests: list[str] = []
        for request, seeds in sorted(expected_seeds.items()):
            request_rows = [keyed.get((request, seed)) for seed in sorted(seeds)]
            request_complete = all(row is not None for row in request_rows)
            request_valid = request_complete and all(
                bool(row.get("reached"))
                and row.get("spearman") is not None
                and math.isfinite(float(row["spearman"]))
                and row.get("top_q_recall") is not None
                and math.isfinite(float(row["top_q_recall"]))
                for row in request_rows if row is not None
            )
            if request_valid:
                reached_requests.append(request)
                request_rho.append(sum(float(row["spearman"]) for row in request_rows) / len(request_rows))
                request_recall.append(sum(float(row["top_q_recall"]) for row in request_rows) / len(request_rows))

        support_ok = complete and len(reached_requests) >= int(min_reached_requests)
        diagnostic: dict[str, object] = {
            "alpha": alpha,
            "complete": complete,
            "support_ok": support_ok,
            "n_expected": len(expected),
            "n_observed": len(actual),
            "n_reached_requests": len(reached_requests),
            "reached_requests": reached_requests,
            "missing_runs": sorted(expected - actual),
            "extra_runs": sorted(actual - expected),
            "duplicate_runs": sorted(set(duplicates)),
            "equal_request_mean_spearman": (
                sum(request_rho) / len(request_rho) if request_rho else None
            ),
            "equal_request_mean_top_q_recall": (
                sum(request_recall) / len(request_recall) if request_recall else None
            ),
        }
        diagnostics.append(diagnostic)
        if support_ok:
            eligible.append(diagnostic)

    if not eligible:
        return {
            "resolved": False,
            "fallback": True,
            "claim_eligible": False,
            "alpha": fallback,
            "selection_rule": "equal_request_spearman_then_top_q_midpoint_smaller",
            "diagnostics": diagnostics,
        }
    winner = min(
        eligible,
        key=lambda item: (
            -float(item["equal_request_mean_spearman"]),
            -float(item["equal_request_mean_top_q_recall"]),
            abs(float(item["alpha"]) - 0.5),
            float(item["alpha"]),
        ),
    )
    return {
        "resolved": True,
        "fallback": False,
        "claim_eligible": True,
        "alpha": float(winner["alpha"]),
        "selection_rule": "equal_request_spearman_then_top_q_midpoint_smaller",
        "winner": winner,
        "diagnostics": diagnostics,
    }


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
    distance to the readout-independent midpoint 0.5.  If no weight is feasible the
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
            # New protection artifacts make the full direct/paraphrase/
            # utility conjunction explicit. Legacy development artifacts lack
            # this field and retain their prior direct+utility interpretation.
            and (row.get("feasible") is None or bool(row.get("feasible")))
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
            "selection_rule": "minimax_cvar05_then_mean_midpoint_smaller_subject_to_all_constraints",
            "diagnostics": diagnostics,
        }
    winner = min(
        feasible,
        key=lambda item: (
            float(item["worst_cvar05_dnll"]),
            float(item["mean_cvar05_dnll"]),
            abs(float(item["alpha"]) - 0.5),
            float(item["alpha"]),
        ),
    )
    return {
        "resolved": True,
        "alpha": float(winner["alpha"]),
        "prior_alpha": prior,
        "selection_rule": "minimax_cvar05_then_mean_midpoint_smaller_subject_to_all_constraints",
        "winner": winner,
        "diagnostics": diagnostics,
    }
