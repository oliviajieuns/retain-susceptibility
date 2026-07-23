"""CPU fixture tests for the Figure 2 evidence-chain renderer."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "paper" / "plot_evidence_chain.py"


def _write_channel_report(path: Path) -> None:
    rows = []
    for predictor, base in (("fd_norm", 0.45), ("knn_feature", 0.30),
                            ("random_rank", 0.01)):
        for objective, channel in (("graddiff", "loss_gradient"),
                                   ("rmu", "representation")):
            rows.append({
                "predictor": predictor, "objective": objective,
                "channel": channel, "rho": base, "rho_lo": base - 0.1,
                "rho_hi": base + 0.1, "auroc": 0.6, "overlap": 0.2,
                "tail_rho": 0.1, "n_runs": 6, "n_candidates_min": 300,
            })
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_certificate(path: Path, passed: bool) -> None:
    path.write_text(json.dumps({
        "schema": "fd-fidelity-certificate-v1",
        "passed": passed,
        "metrics": {"rho_AB": 0.95, "rho_BC": 0.91, "rho_AC": 0.90,
                    "frac_changed": 0.99 if passed else 0.002,
                    "eff_over_eta": 0.97 if passed else 0.08},
        "thresholds": {"rho_AB": 0.70, "rho_BC": 0.80, "rho_AC": 0.80,
                       "frac_changed": 0.90, "eff_over_eta": 0.90},
    }), encoding="utf-8")


def _write_protection(root: Path) -> None:
    contrasts = [
        {"parent": "graddiff", "deployed_alpha_by_model": "{}",
         "comparator": comparator, "comparator_is_declared_prior": False,
         "n_paired_eligible": 6,
         "mean_cvar_difference_deployed_minus_comparator": diff,
         "ci95_lo": diff - 0.5, "ci95_hi": diff + 0.5,
         "deployed_better_on_all_paired": diff < 0}
        for comparator, diff in (("none", -2.0), ("random", -1.2),
                                 ("exact_grad_norm", -0.4),
                                 ("alpha0.0", -0.8), ("alpha1.0", -0.3))
    ]
    with open(root / "alpha_protection_contrasts.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    curve = [
        {"parent": "graddiff", "channel": "loss_gradient", "alpha": alpha,
         "declared_prior": alpha in (0.0,), "deployed_model_count": deployed,
         "deployed_models": "qwen25_7b" if deployed else "",
         "n_total": 6, "n_reach": 6, "n_utility_eligible": 5,
         "mean_cvar05_dnll_eligible": 4.0 - alpha,
         "cvar05_ci95_lo": 3.0, "cvar05_ci95_hi": 5.0,
         "mean_dnll_eligible": 0.5,
         "mean_utility_retention_reached": 0.95}
        for alpha, deployed in ((0.0, 0), (0.5, 1), (1.0, 0))
    ]
    with open(root / "alpha_protection_curve.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_full_inputs_render(tmp_path: Path) -> None:
    report = tmp_path / "pooled_channel_report.csv"
    _write_channel_report(report)
    cert = tmp_path / "cert.json"
    _write_certificate(cert, passed=True)
    _write_protection(tmp_path)
    out = tmp_path / "fig2.png"
    proc = _run([
        "--channel-report", f"TOFU-7B={report}",
        "--fidelity", f"7B fp32={cert}",
        "--protection", f"TOFU-7B={tmp_path}",
        "--out", str(out),
    ])
    assert proc.returncode == 0, proc.stderr
    assert out.exists() and out.stat().st_size > 0
    assert "4/4 input groups present" in proc.stdout


def test_allow_partial_renders_placeholders(tmp_path: Path) -> None:
    cert = tmp_path / "cert_fail.json"
    _write_certificate(cert, passed=False)
    out = tmp_path / "fig2_partial.pdf"
    proc = _run([
        "--channel-report", f"TOFU-7B={tmp_path / 'missing.csv'}",
        "--fidelity", f"7B bf16={cert}",
        "--protection", f"TOFU-7B={tmp_path / 'missing_dir'}",
        "--allow-partial",
        "--out", str(out),
    ])
    assert proc.returncode == 0, proc.stderr
    assert out.exists() and out.stat().st_size > 0


def test_missing_input_fails_closed_without_allow_partial(tmp_path: Path) -> None:
    proc = _run([
        "--channel-report", f"X={tmp_path / 'missing.csv'}",
        "--out", str(tmp_path / "fig.png"),
    ])
    assert proc.returncode != 0
    assert "missing channel report" in proc.stderr


def test_tikz_output_is_balanced_and_carries_data(tmp_path: Path) -> None:
    report = tmp_path / "pooled_channel_report.csv"
    _write_channel_report(report)
    cert = tmp_path / "cert.json"
    _write_certificate(cert, passed=True)
    _write_protection(tmp_path)
    tikz = tmp_path / "fig2.tex"
    proc = _run([
        "--channel-report", f"TOFU-7B={report}",
        "--fidelity", f"7B fp32={cert}",
        "--protection", f"TOFU-7B={tmp_path}",
        "--tikz", str(tikz),
    ])
    assert proc.returncode == 0, proc.stderr
    text = tikz.read_text(encoding="utf-8")
    assert text.count("\\begin{tikzpicture}") == 1
    assert text.count("\\begin{axis}") == text.count("\\end{axis}") == 6
    assert text.count("{") == text.count("}")
    assert "0.45" in text            # fixture fd_norm rho reaches the coordinates
    assert "-2" in text              # contrast difference vs "none"
    assert "pending" not in text     # full inputs leave no placeholder
    # every fixture label with an underscore-free form must be TeX-escaped
    assert "fd_norm" not in text.replace("fd\\_norm", "")


def test_tikz_partial_renders_placeholders(tmp_path: Path) -> None:
    tikz = tmp_path / "fig2_partial.tex"
    proc = _run(["--allow-partial", "--tikz", str(tikz)])
    assert proc.returncode == 0, proc.stderr
    text = tikz.read_text(encoding="utf-8")
    assert text.count("\\begin{tikzpicture}") == 1
    assert "pending" in text
    assert text.count("{") == text.count("}")


def test_rejects_unlabeled_input(tmp_path: Path) -> None:
    proc = _run([
        "--channel-report", "no_label_here.csv",
        "--out", str(tmp_path / "fig.png"),
    ])
    assert proc.returncode != 0
    assert "expects LABEL=PATH" in proc.stderr
