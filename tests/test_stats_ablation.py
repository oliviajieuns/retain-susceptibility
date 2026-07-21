"""Hierarchical bootstrap, LOTO, matched-parent ablation math."""
import pytest

from rsus.analysis.ablation import ablation_row, matched_delta
from rsus.analysis.stats import hierarchical_bootstrap, leave_one_target_out


VALUES = {
    "t1": {"r1": 0.5, "r2": 0.7},
    "t2": {"r1": 0.6, "r2": 0.8},
    "t3": {"r1": 0.4, "r2": 0.6},
}


def test_bootstrap_point_and_ci():
    out = hierarchical_bootstrap(VALUES, n_boot=500, seed=0)
    assert out["mean"] == pytest.approx(0.6)
    assert out["lo"] <= out["mean"] <= out["hi"]
    # deterministic under the same seed
    assert out == hierarchical_bootstrap(VALUES, n_boot=500, seed=0)
    with pytest.raises(ValueError):
        hierarchical_bootstrap({"t1": {}})


def test_loto_range():
    out = leave_one_target_out(VALUES)
    assert out["lo"] == pytest.approx(0.55)   # drop t2
    assert out["hi"] == pytest.approx(0.65)   # drop t3
    with pytest.raises(ValueError):
        leave_one_target_out({"t1": {"r1": 1.0}})


def test_matched_delta_paired():
    cond = {"r1": 0.9, "r2": 0.7}
    par = {"r1": 0.8, "r2": 0.5}
    out = matched_delta(cond, par, n_boot=200, seed=0)
    assert out["mean"] == pytest.approx(0.15)
    assert out["per_request"] == {"r1": pytest.approx(0.1), "r2": pytest.approx(0.2)}
    with pytest.raises(ValueError):
        matched_delta(cond, {"r1": 0.8})


def test_ablation_row_cvar_gain_sign():
    # damage: condition 0.2, parent 0.5 -> gain = parent - condition = +0.3
    row = ablation_row(
        "Partition", "susc vs random",
        damage=({"r1": 0.2}, {"r1": 0.5}),
    )
    assert row["cvar_gain"]["mean"] == pytest.approx(0.3)
    assert row["d_pred_rho"] is None


def test_rank_agreement_matrix():
    from rsus.analysis.prediction import rank_agreement_matrix

    base = {f"c{i}": float(i) for i in range(10)}
    rev = {f"c{i}": float(-i) for i in range(10)}
    perturbed = {f"c{i}": float(i) + (0.3 if i % 2 else -0.3) for i in range(10)}
    m = rank_agreement_matrix({"a": base, "b": rev, "c": perturbed}, k=3)
    assert m["a"]["a"]["rho"] == 1.0 and m["a"]["a"]["overlap"] == 1.0
    assert m["a"]["b"]["rho"] == -1.0            # exact reversal
    assert m["a"]["c"]["rho"] > 0.9              # small perturbation preserves rank
    assert m["a"]["b"]["overlap"] == 0.0         # top-3 disjoint from bottom-3
    assert m["a"]["c"]["rho"] == m["c"]["a"]["rho"]  # symmetric


def test_channel_interaction_delta():
    """Difference-in-differences isolates channel matching: a gradient probe that
    tracks the gradient-objective damage but is blind to the representation
    objective yields a large positive delta, even when a representation probe
    also partly predicts the gradient objective."""
    from rsus.analysis.channels import cell_metrics, interaction_delta

    cands = [f"c{i:02d}" for i in range(40)]
    g = {c: (i % 7) / 7.0 for i, c in enumerate(cands)}      # gradient-magnitude signal
    r = {c: (i % 5) / 5.0 for i, c in enumerate(cands)}      # representation signal
    scores = {"fd_norm": g, "knn_feature": r}
    damage = {
        "graddiff": {c: 0.7 * g[c] + 0.3 * r[c] for c in cands},  # both channels
        "rmu": {c: r[c] for c in cands},                          # representation only
    }
    rho = {p: {o: cell_metrics(scores[p], damage[o], k=5)["rho"] for o in damage} for p in scores}
    delta = interaction_delta(rho, "graddiff", "rmu", "fd_norm", "knn_feature")
    assert delta > 0.3            # gradient probe's edge concentrates on the gradient objective
    assert rho["fd_norm"]["rmu"] < rho["knn_feature"]["rmu"]  # gradient probe blind to representation
