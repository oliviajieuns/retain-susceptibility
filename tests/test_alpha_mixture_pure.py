"""Dependency-light contracts for soft channel routing and alpha freezing."""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsus.analysis.mixture import (  # noqa: E402
    alpha_label,
    channel_mixture_scores,
    rank01,
    select_development_alpha,
)


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


alpha_campaign = _module(
    "alpha_protection_campaign", "experiments/channel_matrix/alpha_protection.py"
)


class MixtureScoreTest(unittest.TestCase):
    def test_endpoints_recover_component_rankings(self):
        gradient = {"a": 4.0, "b": 1.0, "c": 3.0, "d": 2.0}
        proximity = {"a": -2.0, "b": 8.0, "c": 0.0, "d": 4.0}
        self.assertEqual(
            channel_mixture_scores(gradient, proximity, 0.0), rank01(gradient)
        )
        self.assertEqual(
            channel_mixture_scores(gradient, proximity, 1.0), rank01(proximity)
        )
        self.assertEqual(alpha_label(0.25), "s_alpha_0p25")

    def test_tie_break_matches_descending_score_then_candidate_id(self):
        ranked = rank01({"b": 1.0, "a": 1.0, "c": 0.0})
        self.assertGreater(ranked["a"], ranked["b"])
        self.assertGreater(ranked["b"], ranked["c"])

    def test_discovery_normalization_is_invariant_to_audit_scores(self):
        discovery = {"d0", "d1", "d2"}
        gradient_a = {"d0": 1, "d1": 2, "d2": 3, "a0": -1e9}
        gradient_b = {**gradient_a, "a0": 1e9}
        proximity_a = {"d0": 3, "d1": 1, "d2": 2, "a0": 1e9}
        proximity_b = {**proximity_a, "a0": -1e9}
        self.assertEqual(
            channel_mixture_scores(
                gradient_a, proximity_a, 0.5, candidate_ids=discovery
            ),
            channel_mixture_scores(
                gradient_b, proximity_b, 0.5, candidate_ids=discovery
            ),
        )

    def test_minimax_selects_only_feasible_development_alpha(self):
        rows = []
        cvars = {
            0.0: [4.0, 5.0],
            0.5: [2.0, 3.0],
            1.0: [1.0, 8.0],
        }
        for alpha, values in cvars.items():
            for request, value in zip(("tofu-a198", "tofu-a199"), values):
                rows.append({
                    "campaign_phase": "development",
                    "selector_type": "mixture",
                    "alpha": alpha,
                    "request": request,
                    "seed": 2025,
                    "reached": True,
                    "forget_recall": 0.05,
                    "utility_retention": 0.95,
                    "cvar05_dnll": value,
                })
        result = select_development_alpha(
            rows,
            alpha_grid=[0, 0.5, 1],
            expected_run_keys={("tofu-a198", 2025), ("tofu-a199", 2025)},
            prior_alpha=0,
            recall_max=0.10,
            utility_retention_min=0.90,
        )
        self.assertTrue(result["resolved"])
        self.assertEqual(result["alpha"], 0.5)  # minimax: 3 < 5 < 8

    def test_selector_rejects_any_audit_row(self):
        with self.assertRaisesRegex(ValueError, "audit"):
            select_development_alpha(
                [{
                    "campaign_phase": "audit",
                    "selector_type": "mixture",
                    "alpha": 0.0,
                    "request": "tofu-a181",
                    "seed": 2025,
                    "reached": True,
                    "forget_recall": 0.0,
                    "utility_retention": 1.0,
                    "cvar05_dnll": 0.0,
                }],
                alpha_grid=[0.0],
                expected_run_keys={("tofu-a181", 2025)},
                prior_alpha=0.0,
                recall_max=0.1,
                utility_retention_min=0.9,
            )


class AlphaCampaignFreezeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_path = ROOT / "configs/channel_matrix/7b_tofu.yaml"
        cls.base = yaml.safe_load(cls.base_path.read_text(encoding="utf-8"))

    def test_contract_has_disjoint_ordinary_utility_pool(self):
        alpha_campaign._validate_contract(self.base)

    def test_draft_alpha_freeze_blocks_audit(self):
        models = alpha_campaign._enabled_models(self.base, {"qwen25_7b"})
        with self.assertRaises(RuntimeError):
            alpha_campaign._alpha_freeze(self.base_path, self.base, models)

    def test_worker_grid_requires_both_freezes_and_is_exact(self):
        cfg = copy.deepcopy(self.base)
        models = alpha_campaign._enabled_models(cfg, {"qwen25_7b"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parents = cfg["alpha_protection"]["parents"]
            all_objectives = cfg["audit"]["objectives"] + cfg["audit"]["stress_objectives"]
            objective = {
                "freeze_id": "OBJ-TEST",
                "status": "frozen",
                "frozen_before_audit": True,
                "frozen_at_utc": "2026-07-22T00:00:00Z",
                "source_campaign": cfg["campaign_id"],
                "unresolved": [],
                "models": {"qwen25_7b": {
                    name: {"lr": 1e-6, "steps": 10} for name in all_objectives
                }},
            }
            alpha = {
                "freeze_id": "ALPHA-TEST",
                "status": "frozen",
                "frozen_before_alpha_audit": True,
                "frozen_at_utc": "2026-07-22T01:00:00Z",
                "source_campaign": cfg["alpha_protection"]["campaign_id"],
                "source_phase": "development",
                "normalization": cfg["alpha_protection"]["normalization"],
                "orientation": cfg["alpha_protection"]["orientation"],
                "objective_freeze_sha256": None,
                "campaign_config_sha256": None,
                "development_artifacts": [{"path": "dev.json", "sha256": "test"}],
                "models": {"qwen25_7b": {name: 0.5 for name in parents}},
                "unresolved": [],
            }
            (root / "objective.yaml").write_text(yaml.safe_dump(objective), encoding="utf-8")
            alpha["objective_freeze_sha256"] = alpha_campaign._sha256(root / "objective.yaml")
            (root / "alpha.yaml").write_text(yaml.safe_dump(alpha), encoding="utf-8")
            cfg["alpha_protection"]["objective_freeze"] = "objective.yaml"
            cfg["alpha_protection"]["alpha_freeze"] = "alpha.yaml"
            config_path = root / "campaign.yaml"
            config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            alpha["campaign_config_sha256"] = alpha_campaign._sha256(config_path)
            (root / "alpha.yaml").write_text(yaml.safe_dump(alpha), encoding="utf-8")

            development = alpha_campaign.worker_commands(
                config_path, cfg, "development", models
            )
            audit = alpha_campaign.worker_commands(config_path, cfg, "audit", models)
            self.assertEqual(
                len(development),
                len(cfg["alpha_protection"]["development"]["authors"])
                * len(cfg["alpha_protection"]["development"]["seeds"]),
            )
            self.assertEqual(
                len(audit),
                len(cfg["alpha_protection"]["audit"]["authors"])
                * len(cfg["alpha_protection"]["audit"]["seeds"]),
            )
            self.assertTrue(all("--worker" in command for _, command in audit))

            selected = {cfg["alpha_protection"]["development"]["authors"][0]}
            shard = alpha_campaign.worker_commands(
                config_path,
                cfg,
                "development",
                models,
                selected_authors=selected,
            )
            self.assertEqual(
                len(shard), len(cfg["alpha_protection"]["development"]["seeds"])
            )
            self.assertTrue(all(
                f"tofu-a{next(iter(selected))}" in str(out) for out, _ in shard
            ))

    def test_alpha_author_shard_rejects_out_of_roster_request(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            alpha_campaign._filter_authors([198, 199], {181}, "development")

    def test_development_results_to_draft_freeze_end_to_end(self):
        cfg = copy.deepcopy(self.base)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg["output_root"] = str(root / "runs")
            cfg["alpha_protection"]["objective_freeze"] = "objective.yaml"
            cfg["alpha_protection"]["alpha_freeze"] = "alpha.yaml"
            all_objectives = cfg["audit"]["objectives"] + cfg["audit"]["stress_objectives"]
            objective = {
                "freeze_id": "OBJ-E2E",
                "status": "frozen",
                "frozen_before_audit": True,
                "frozen_at_utc": "2026-07-22T00:00:00Z",
                "source_campaign": cfg["campaign_id"],
                "unresolved": [],
                "models": {"qwen25_7b": {
                    name: {"lr": 1e-6, "steps": 10} for name in all_objectives
                }},
            }
            objective_path = root / "objective.yaml"
            objective_path.write_text(yaml.safe_dump(objective), encoding="utf-8")
            config_path = root / "campaign.yaml"
            config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            objective_sha = alpha_campaign._sha256(objective_path)

            development_root = root / "development"
            for author in cfg["alpha_protection"]["development"]["authors"]:
                request = f"tofu-a{author}"
                run = development_root / "qwen25_7b" / request / "seed-2025"
                run.mkdir(parents=True)
                rows = []
                for parent in cfg["alpha_protection"]["parents"]:
                    for alpha in cfg["alpha_protection"]["alpha_grid"]:
                        rows.append({
                            "campaign_phase": "development",
                            "model_id": "qwen25_7b",
                            "request": request,
                            "seed": 2025,
                            "parent": parent,
                            "selector": alpha_label(alpha),
                            "selector_type": "mixture",
                            "alpha": float(alpha),
                            "reached": True,
                            "forget_recall": 0.05,
                            "utility_retention": 0.95,
                            "cvar05_dnll": (float(alpha) - 0.5) ** 2 + author * 1e-6,
                        })
                    for selector, selector_type in (
                        ("none", "none"),
                        ("random", "random"),
                        ("exact_grad_norm", "exact_gradient_ceiling"),
                    ):
                        rows.append({
                            "campaign_phase": "development",
                            "model_id": "qwen25_7b",
                            "request": request,
                            "seed": 2025,
                            "parent": parent,
                            "selector": selector,
                            "selector_type": selector_type,
                            "alpha": None,
                            "reached": True,
                            "forget_recall": 0.05,
                            "utility_retention": 0.95,
                            "cvar05_dnll": 1.0,
                        })
                payload = {
                    "manifest": {
                        "campaign_phase": "development",
                        "campaign_id": cfg["alpha_protection"]["campaign_id"],
                        "model_id": "qwen25_7b",
                        "request": request,
                        "seed": 2025,
                        "objective_freeze_sha256": objective_sha,
                        "normalization_scope": "discovery_only",
                    },
                    "results": rows,
                }
                (run / "results.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            out = root / "recommended.yaml"
            subprocess.run([
                sys.executable,
                str(ROOT / "experiments/channel_matrix/select_alpha_freeze.py"),
                "--config", str(config_path),
                "--root", str(development_root),
                "--out", str(out),
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            selected = yaml.safe_load(out.read_text(encoding="utf-8"))
            self.assertEqual(selected["status"], "draft")
            self.assertEqual(selected["unresolved"], [])
            self.assertEqual(
                set(selected["models"]["qwen25_7b"].values()), {0.5}
            )
            self.assertEqual(len(selected["development_artifacts"]), 2)

    def test_frozen_audit_aggregate_end_to_end(self):
        cfg = copy.deepcopy(self.base)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg["output_root"] = str(root / "runs")
            cfg["alpha_protection"]["objective_freeze"] = "objective.yaml"
            cfg["alpha_protection"]["alpha_freeze"] = "alpha.yaml"
            all_objectives = cfg["audit"]["objectives"] + cfg["audit"]["stress_objectives"]
            objective = {
                "freeze_id": "OBJ-AGG",
                "status": "frozen",
                "frozen_before_audit": True,
                "frozen_at_utc": "2026-07-22T00:00:00Z",
                "source_campaign": cfg["campaign_id"],
                "unresolved": [],
                "models": {"qwen25_7b": {
                    name: {"lr": 1e-6, "steps": 10} for name in all_objectives
                }},
            }
            objective_path = root / "objective.yaml"
            objective_path.write_text(yaml.safe_dump(objective), encoding="utf-8")
            config_path = root / "campaign.yaml"
            config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            alpha = {
                "freeze_id": "ALPHA-AGG",
                "status": "frozen",
                "frozen_before_alpha_audit": True,
                "frozen_at_utc": "2026-07-22T01:00:00Z",
                "source_campaign": cfg["alpha_protection"]["campaign_id"],
                "source_phase": "development",
                "selection_rule": "minimax_cvar05_subject_to_every_run_reach_and_utility",
                "normalization": cfg["alpha_protection"]["normalization"],
                "orientation": cfg["alpha_protection"]["orientation"],
                "objective_freeze_sha256": alpha_campaign._sha256(objective_path),
                "campaign_config_sha256": alpha_campaign._sha256(config_path),
                "development_artifacts": [{"path": "dev.json", "sha256": "test"}],
                "models": {"qwen25_7b": {
                    parent: 0.5 for parent in cfg["alpha_protection"]["parents"]
                }},
                "unresolved": [],
            }
            (root / "alpha.yaml").write_text(yaml.safe_dump(alpha), encoding="utf-8")

            audit_root = root / "audit"
            output_parents = {"graddiff", "npo"}
            for author in cfg["alpha_protection"]["audit"]["authors"]:
                request = f"tofu-a{author}"
                for seed in cfg["alpha_protection"]["audit"]["seeds"]:
                    run = audit_root / "qwen25_7b" / request / f"seed-{seed}"
                    run.mkdir(parents=True)
                    rows = []
                    for parent in cfg["alpha_protection"]["parents"]:
                        channel = "loss_gradient" if parent in output_parents else "representation"
                        prior = 0.0 if channel == "loss_gradient" else 1.0
                        for value in cfg["alpha_protection"]["alpha_grid"]:
                            alpha_value = float(value)
                            rows.append({
                                "campaign_phase": "audit",
                                "model_id": "qwen25_7b",
                                "request": request,
                                "seed": seed,
                                "parent": parent,
                                "channel": channel,
                                "selector": alpha_label(alpha_value),
                                "selector_type": "mixture",
                                "alpha": alpha_value,
                                "declared_prior": alpha_value == prior,
                                "deployed": alpha_value == 0.5,
                                "reached": True,
                                "forget_recall": 0.05,
                                "utility_retention": 0.95,
                                "cvar05_dnll": 0.5 + abs(alpha_value - 0.5),
                                "mean_dnll": 0.25 + abs(alpha_value - 0.5),
                            })
                        for selector, selector_type, cvar in (
                            ("none", "none", 2.0),
                            ("random", "random", 1.5),
                            ("exact_grad_norm", "exact_gradient_ceiling", 1.25),
                        ):
                            rows.append({
                                "campaign_phase": "audit",
                                "model_id": "qwen25_7b",
                                "request": request,
                                "seed": seed,
                                "parent": parent,
                                "channel": channel,
                                "selector": selector,
                                "selector_type": selector_type,
                                "alpha": None,
                                "declared_prior": False,
                                "deployed": False,
                                "reached": True,
                                "forget_recall": 0.05,
                                "utility_retention": 0.95,
                                "cvar05_dnll": cvar,
                                "mean_dnll": cvar / 2,
                            })
                    payload = {
                        "manifest": {
                            "campaign_phase": "audit",
                            "campaign_id": cfg["alpha_protection"]["campaign_id"],
                            "model_id": "qwen25_7b",
                            "request": request,
                            "seed": seed,
                            "alpha_freeze_id": alpha["freeze_id"],
                        },
                        "results": rows,
                    }
                    (run / "results.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )

            out = root / "aggregate"
            subprocess.run([
                sys.executable,
                str(ROOT / "experiments/channel_matrix/aggregate_alpha_protection.py"),
                "--config", str(config_path),
                "--root", str(audit_root),
                "--out", str(out),
                "--n-boot", "20",
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            summary = json.loads(
                (out / "alpha_protection_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["n_cells"], 6)
            none = next(row for row in summary["paired_contrasts"]
                        if row["parent"] == "graddiff" and row["comparator"] == "none")
            self.assertEqual(none["n_paired_eligible"], 6)
            self.assertLess(
                none["mean_cvar_difference_deployed_minus_comparator"], 0
            )


if __name__ == "__main__":
    unittest.main()
