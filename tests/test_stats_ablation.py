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
