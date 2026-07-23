"""Small, dependency-free statistical helpers used by evidence aggregators.

The experiment runners may use richer hierarchical resampling, but their
paper-facing summaries must obey the same signs and finite-sample correction
implemented here.  Prediction is beneficial when a paired gain is positive;
protection is beneficial when mixture-minus-comparator damage is negative.
"""
from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence


def _finite(values: Iterable[float], *, name: str) -> list[float]:
    clean = [float(value) for value in values]
    if not clean:
        raise ValueError(f"{name} must be non-empty")
    if not all(math.isfinite(value) for value in clean):
        raise ValueError(f"{name} contains a non-finite value")
    return clean


def percentile(values: Sequence[float], probability: float) -> float:
    """Linearly interpolated empirical percentile on a finite sample."""
    clean = sorted(_finite(values, name="values"))
    q = float(probability)
    if not 0.0 <= q <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if len(clean) == 1:
        return clean[0]
    position = q * (len(clean) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def paired_differences(
    treatment: Mapping[Hashable, float], comparator: Mapping[Hashable, float]
) -> dict[Hashable, float]:
    """Form differences only on exact common support; never take an intersection."""
    treatment_keys = set(treatment)
    comparator_keys = set(comparator)
    if treatment_keys != comparator_keys:
        raise ValueError(
            "paired effects require identical unit keys; "
            f"treatment_only={len(treatment_keys - comparator_keys)}, "
            f"comparator_only={len(comparator_keys - treatment_keys)}"
        )
    if not treatment_keys:
        raise ValueError("paired effects require at least one unit")
    result: dict[Hashable, float] = {}
    for key in treatment:
        left = _finite([treatment[key]], name=f"treatment[{key!r}]")[0]
        right = _finite([comparator[key]], name=f"comparator[{key!r}]")[0]
        result[key] = left - right
    return result


def finite_sample_one_sided_p(
    bootstrap_effects: Sequence[float], *, beneficial: str
) -> float:
    """Finite-sample corrected one-sided bootstrap p-value.

    ``beneficial='positive'`` tests an endpoint gain against zero and counts
    bootstrap values ``<= 0``.  ``beneficial='negative'`` tests a damage
    reduction and counts values ``>= 0``.
    """
    effects = _finite(bootstrap_effects, name="bootstrap_effects")
    if beneficial == "positive":
        adverse = sum(value <= 0.0 for value in effects)
    elif beneficial == "negative":
        adverse = sum(value >= 0.0 for value in effects)
    else:
        raise ValueError("beneficial must be 'positive' or 'negative'")
    return (1.0 + adverse) / (len(effects) + 1.0)


def summarize_bootstrap_effect(
    estimate: float,
    bootstrap_effects: Sequence[float],
    *,
    beneficial: str,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Return a one-sided bound and corrected p-value for a paired effect."""
    point = float(estimate)
    if not math.isfinite(point):
        raise ValueError("estimate must be finite")
    level = float(alpha)
    if not 0.0 < level < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    effects = _finite(bootstrap_effects, name="bootstrap_effects")
    result = {
        "estimate": point,
        "p_one_sided": finite_sample_one_sided_p(
            effects, beneficial=beneficial
        ),
    }
    if beneficial == "positive":
        result["lower_bound"] = percentile(effects, level)
    elif beneficial == "negative":
        result["upper_bound"] = percentile(effects, 1.0 - level)
    else:
        raise ValueError("beneficial must be 'positive' or 'negative'")
    return result


def intersection_union_p(p_values: Iterable[float]) -> float:
    """IUT p-value: the maximum elementary one-sided p-value."""
    values = _finite(p_values, name="p_values")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must lie in [0, 1]")
    return max(values)
