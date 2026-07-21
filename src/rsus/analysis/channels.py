"""Channel-conditioned susceptibility analysis (G1).

Damage channel is DECLARED from the objective's loss definition, before any
result is seen (anti post-hoc): an objective whose loss is a function of model
OUTPUTS (token likelihoods / preferences) damages retained candidates through
the loss-gradient channel; an objective whose loss is a function of internal
REPRESENTATIONS (hidden states) damages through the representation channel.
Each channel is read by a predictor family -- gradient magnitude vs
representation proximity. The headline statistic is NOT an average leaderboard
(which hides the channel structure) but the objective x family INTERACTION.
"""
from __future__ import annotations

import math

from rsus.analysis.prediction import auroc, spearman, top_k_ids

# Declared from each objective's loss definition (not from results).
DECLARED_CHANNEL: dict[str, str] = {
    "ga": "loss_gradient", "graddiff": "loss_gradient", "npo": "loss_gradient",
    "simnpo": "loss_gradient", "idkdpo": "loss_gradient", "gru": "loss_gradient",
    "rmu": "representation", "repnoise": "representation", "circuit_breakers": "representation",
}
PREDICTOR_FAMILY: dict[str, str] = {
    "grad_norm": "gradient", "fd_norm": "gradient",
    "knn_feature": "representation", "knn_embed": "representation", "knn_lexical": "representation",
    "fd": "alignment", "one_sided": "alignment", "last_layer": "alignment",
    "random_rank": "control", "random_dir": "control",
}
# One headline probe per family for the 2x2 core / main figure.
HEADLINE_PROBE: dict[str, str] = {"gradient": "fd_norm", "representation": "knn_feature"}


def tail_region_rho(scores: dict[str, float], damage: dict[str, float], cvar_frac: float) -> float:
    """Spearman restricted to the worst-cvar_frac damaged candidates: does the
    predictor rank WITHIN the damage tail (the concentrated-failure regime)?"""
    ids = sorted(set(scores) & set(damage))
    n_tail = max(3, math.ceil(cvar_frac * len(ids)))
    tail = sorted(ids, key=lambda c: -damage[c])[:n_tail]
    if len(tail) < 2:
        return 0.0
    return spearman([scores[c] for c in tail], [damage[c] for c in tail])


def cell_metrics(scores: dict[str, float], damage: dict[str, float], k: int,
                 cvar_frac: float = 0.05) -> dict[str, float]:
    """One (predictor, objective) cell: full-ranking rho, top-K damage AUROC,
    Overlap@K, and tail-region rho. No averaging over objectives."""
    ids = sorted(set(scores) & set(damage))
    s = [scores[c] for c in ids]
    dmg = {c: damage[c] for c in ids}
    realized = top_k_ids(dmg, k)
    return {
        "rho": spearman(s, [dmg[c] for c in ids]),
        "auroc": auroc(s, [c in realized for c in ids]),
        "overlap": len(top_k_ids({c: scores[c] for c in ids}, k) & realized) / k,
        "tail_rho": tail_region_rho(scores, dmg, cvar_frac),
    }


def interaction_delta(rho: dict[str, dict[str, float]], grad_obj: str, rep_obj: str,
                      grad_probe: str, rep_probe: str) -> float:
    """Difference-in-differences: how much more the gradient probe out-ranks the
    representation probe on a loss-gradient objective than on a representation
    objective. rho[predictor][objective] -> Spearman. Delta>0 => channel matching.
    Robust to 'some probe is generally better' and 'some objective is more
    predictable' -- both cancel in the double difference."""
    return ((rho[grad_probe][grad_obj] - rho[rep_probe][grad_obj])
            - (rho[grad_probe][rep_obj] - rho[rep_probe][rep_obj]))


def bootstrap_interaction(scores_by_pred: dict[str, dict[str, float]],
                          damage_by_obj: dict[str, dict[str, float]],
                          grad_obj: str, rep_obj: str, grad_probe: str, rep_probe: str,
                          n_boot: int = 1000, rng=None) -> dict[str, float]:
    """Candidate-resampled CI for the interaction delta. rng: a random.Random."""
    import random as _random
    rng = rng or _random.Random(0)
    ids = sorted(set(damage_by_obj[grad_obj]) & set(damage_by_obj[rep_obj])
                 & set(scores_by_pred[grad_probe]) & set(scores_by_pred[rep_probe]))

    def delta_on(sample: list[str]) -> float:
        def rho(pred: str, obj: str) -> float:
            return spearman([scores_by_pred[pred][c] for c in sample],
                            [damage_by_obj[obj][c] for c in sample])
        return ((rho(grad_probe, grad_obj) - rho(rep_probe, grad_obj))
                - (rho(grad_probe, rep_obj) - rho(rep_probe, rep_obj)))

    point = delta_on(ids)
    boots = []
    n = len(ids)
    for _ in range(n_boot):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        try:
            boots.append(delta_on(sample))
        except ValueError:
            continue
    boots.sort()
    lo = boots[int(0.025 * len(boots))] if boots else float("nan")
    hi = boots[int(0.975 * len(boots))] if boots else float("nan")
    return {"delta": point, "lo": lo, "hi": hi, "n_cands": n, "n_boot": len(boots)}
