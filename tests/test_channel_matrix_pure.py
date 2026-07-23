"""Dependency-light campaign contract tests (runnable without torch/transformers)."""
from __future__ import annotations

import importlib.util
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


campaign = _module("channel_campaign", "experiments/channel_matrix/run_campaign.py")
table = _module("channel_table", "experiments/channel_matrix/make_main_table.py")


class ChannelCampaignContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = ROOT / "configs/channel_matrix/7b_tofu.yaml"
        cls.config = yaml.safe_load(cls.config_path.read_text(encoding="utf-8"))
        cls.models = campaign._enabled_models(cls.config, set())

    def test_candidate_pools_and_objective_rosters_are_disjoint_and_exact(self):
        campaign._validate_campaign(self.config)
        development = campaign._expand_int_ranges(
            self.config["common"]["candidate_author_pools"]["calibration"]
        )
        audit = {
            author: campaign._expand_int_ranges(raw)
            for author, raw in self.config["common"]["candidate_author_pools"]["audit"].items()
        }
        self.assertEqual(len(development), 30)
        self.assertTrue(all(len(pool) == 30 for pool in audit.values()))
        seen = set(development)
        for pool in audit.values():
            self.assertFalse(seen & pool)
            seen |= pool
        self.assertFalse(
            set(self.config["audit"]["objectives"])
            & set(self.config["audit"]["stress_objectives"])
        )

    def test_calibration_is_complete_and_reuses_request_level_sft_cache(self):
        commands = list(campaign.calibration_commands(
            self.config, self.models, ROOT / "runs/channel_matrix_7b"
        ))
        n_settings = sum(
            len(settings)
            for settings in self.config["calibration"]["objective_grid"].values()
        )
        expected = (
            len(self.models)
            * len(self.config["calibration"]["authors"])
            * len(self.config["calibration"]["seeds"])
            * n_settings
        )
        self.assertEqual(len(commands), expected)
        caches = {}
        for out, command in commands:
            self.assertEqual(command[command.index("--predictors") + 1], "")
            self.assertIn("--candidate-authors", command)
            author_seed = tuple(out.parts[-5:-2])
            cache = command[command.index("--sft-cache") + 1]
            caches.setdefault(author_seed, set()).add(cache)
        self.assertTrue(all(len(paths) == 1 for paths in caches.values()))

        selected = {self.config["calibration"]["authors"][0]}
        shard = list(campaign.calibration_commands(
            self.config,
            self.models,
            ROOT / "runs/channel_matrix_7b",
            selected_authors=selected,
        ))
        self.assertEqual(len(shard), expected // len(self.config["calibration"]["authors"]))
        self.assertTrue(all(
            f"tofu-a{next(iter(selected))}" in str(out) for out, _ in shard
        ))

    def test_author_shard_must_be_a_nonempty_roster_subset(self):
        self.assertEqual(
            campaign._filter_authors([198, 199], {199}, "calibration"),
            [199],
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            campaign._filter_authors([198, 199], {181}, "calibration")
        with self.assertRaisesRegex(ValueError, "empty"):
            campaign._filter_authors([198, 199], set(), "calibration")

    def test_fidelity_uses_only_frozen_development_cell(self):
        commands = list(campaign.fidelity_commands(
            self.config, self.models, ROOT / "runs/channel_matrix_7b"
        ))
        self.assertEqual(len(commands), len(self.models))
        _, _, command = commands[0]
        self.assertEqual(command[command.index("--dtype") + 1], "float32")
        self.assertEqual(command[command.index("--dirs") + 1], "64")
        self.assertEqual(command[command.index("--etas") + 1], "0.003")
        self.assertIn("--enforce-gate", command)
        self.assertEqual(command[command.index("--dataset") + 1], "tofu")

    def test_fidelity_command_carries_campaign_dataset(self):
        cfg = yaml.safe_load(
            (ROOT / "configs/channel_matrix/rwku_7b.yaml").read_text(encoding="utf-8")
        )
        models = campaign._enabled_models(cfg, set())
        _, _, command = next(iter(campaign.fidelity_commands(
            cfg, models, ROOT / "runs/channel_matrix_rwku7b"
        )))
        self.assertEqual(command[command.index("--dataset") + 1], "rwku")
        # The rwku fidelity request must carry the frozen remote pool: the
        # runner refuses --dataset rwku without --candidate-authors.
        self.assertEqual(command[command.index("--candidate-authors") + 1], "100-129")

    def test_draft_objective_freeze_blocks_audit(self):
        # The repo's live freeze became status=frozen on 2026-07-23, so the
        # draft-blocking contract is exercised against a synthetic draft copy.
        cfg = copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze = yaml.safe_load(
                (self.config_path.parent / cfg["audit"]["objective_freeze"])
                .read_text(encoding="utf-8")
            )
            freeze["status"] = "draft"
            freeze["frozen_before_audit"] = False
            (root / "freeze.yaml").write_text(yaml.safe_dump(freeze), encoding="utf-8")
            cfg["audit"]["objective_freeze"] = "freeze.yaml"
            with self.assertRaises(RuntimeError):
                campaign._load_freeze(root / "config.yaml", cfg, self.models)
        # And the live frozen file must keep loading cleanly for the audit wave.
        campaign._load_freeze(self.config_path, self.config, self.models)

    def test_frozen_audit_command_contains_core_and_stress_before_open(self):
        cfg = copy.deepcopy(self.config)
        all_objectives = cfg["audit"]["objectives"] + cfg["audit"]["stress_objectives"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze = {
                "freeze_id": "TEST-FREEZE",
                "status": "frozen",
                "frozen_before_audit": True,
                "frozen_at_utc": "2026-07-22T00:00:00Z",
                "source_campaign": cfg["campaign_id"],
                "unresolved": [],
                "models": {
                    "qwen25_7b": {
                        objective: {"lr": 1e-6, "steps": 10}
                        for objective in all_objectives
                    }
                },
            }
            (root / "freeze.yaml").write_text(
                yaml.safe_dump(freeze), encoding="utf-8"
            )
            certificate = {
                "schema": "fd-fidelity-certificate-v1",
                "passed": True,
                "model": self.models[0]["path"],
                "dtype": cfg["common"]["dtype"],
                "candidate_authors": sorted(campaign._expand_int_ranges(
                    cfg["common"]["candidate_author_pools"]["calibration"]
                )),
                "n_candidates": cfg["fidelity"]["n_candidates"],
                "block_last_n": cfg["common"]["block_last_n"],
                "R": cfg["common"]["probe_dirs"],
                "eta": cfg["common"]["probe_norm_eta"],
                "probe_seed": cfg["common"]["probe_seed"],
            }
            cert_path = root / "fidelity.json"
            cert_path.write_text(json.dumps(certificate), encoding="utf-8")
            cfg["audit"]["objective_freeze"] = "freeze.yaml"
            cfg["audit"]["fidelity_certificates"]["qwen25_7b"] = str(cert_path)
            config_path = root / "campaign.yaml"
            config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

            commands = list(campaign.audit_commands(
                config_path, cfg, self.models, ROOT / "runs/test-audit"
            ))
            self.assertEqual(
                len(commands),
                len(cfg["audit"]["authors"]) * len(cfg["audit"]["seeds"]),
            )
            _, command, metadata = commands[0]
            self.assertEqual(command[command.index("--dtype") + 1], "float32")
            self.assertEqual(
                command[command.index("--generators") + 1],
                ",".join(all_objectives),
            )
            self.assertIn("--require-all-predictors", command)
            self.assertEqual(metadata["core_objectives"], cfg["audit"]["objectives"])
            self.assertEqual(metadata["stress_objectives"], cfg["audit"]["stress_objectives"])

            selected = {cfg["audit"]["authors"][1]}
            shard = list(campaign.audit_commands(
                config_path,
                cfg,
                self.models,
                ROOT / "runs/test-audit",
                selected_authors=selected,
            ))
            self.assertEqual(len(shard), len(cfg["audit"]["seeds"]))
            self.assertTrue(all(
                f"tofu-a{next(iter(selected))}" in str(out) for out, _, _ in shard
            ))

    def test_table_marks_adaptation_failure_and_collapse(self):
        summary = {
            "objective_status": {
                "repnoise": {"failed_runs": 1, "collapsed_runs": 1},
            }
        }
        label = table._objective_label("repnoise", summary)
        self.assertIn("*", label)
        self.assertIn(r"\dagger", label)
        self.assertIn(r"\ddagger", label)

    def test_sealed_aggregate_and_main_table_end_to_end(self):
        core = self.config["audit"]["objectives"]
        stress = self.config["audit"]["stress_objectives"]
        objectives = core + stress
        predictors = self.config["audit"]["predictors"]
        rule = self.config["calibration"]["selection"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_root = root / "audit"
            for request in ("tofu-a181", "tofu-a186"):
                request_author = request.removeprefix("tofu-a")
                audit_authors = sorted(campaign._expand_int_ranges(
                    self.config["common"]["candidate_author_pools"]["audit"][request_author]
                ))
                for seed in (2025, 2026):
                    run = audit_root / "qwen25_7b" / request / f"seed-{seed}"
                    seal_dir = run / "seals" / request
                    seal_dir.mkdir(parents=True)
                    ids = [f"{request}-c{index:03d}" for index in range(30)]
                    ledger = []
                    for predictor_index, predictor in enumerate(predictors):
                        scores = {
                            key: float(index if predictor_index < 2 else 29 - index)
                            for index, key in enumerate(ids)
                        }
                        body = json.dumps(scores, sort_keys=True)
                        (seal_dir / f"{predictor}.json").write_text(body, encoding="utf-8")
                        ledger.extend([
                            {
                                "status": "sealed", "request": request,
                                "scorer": predictor,
                                "sha": hashlib.sha256(body.encode()).hexdigest(),
                            },
                            {"status": "opened", "request": request, "scorer": predictor},
                        ])
                    (run / "seal_ledger.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in ledger), encoding="utf-8"
                    )
                    for objective in objectives:
                        trajectory = run / f"traj_{objective}"
                        trajectory.mkdir(parents=True)
                        if objective in {"rmu", "repnoise", "circuit_breakers"}:
                            damage = {key: (29 - index) / 30 for index, key in enumerate(ids)}
                        elif objective == "ga":
                            damage = {key: 3.0 + index / 30 for index, key in enumerate(ids)}
                        else:
                            damage = {key: index / 30 for index, key in enumerate(ids)}
                        recall = 0.5 if objective == "idkdpo" else 0.05
                        payload = {
                            "objective": objective,
                            "request": request,
                            "nll0": {key: 1.0 for key in ids},
                            "snapshots": [{
                                "step": 10,
                                "forget_recall": recall,
                                "extra": {},
                                "nll": {key: 1.0 + value for key, value in damage.items()},
                            }],
                        }
                        (trajectory / "damage.json").write_text(
                            json.dumps(payload), encoding="utf-8"
                        )
                        (trajectory / "DONE").touch()
                    manifest = {
                        "request": request,
                        "model_id": "qwen25_7b",
                        "seed": seed,
                        "predictors": predictors,
                        "objectives": objectives,
                        "core_objectives": core,
                        "stress_objectives": stress,
                        "campaign_id": "test-campaign",
                        "campaign_config_sha256": "config",
                        "objective_freeze_id": "freeze",
                        "objective_freeze_sha256": "freeze-sha",
                        "fidelity_certificate_sha256": "fidelity-sha",
                        "dtype": "float32",
                        "trainable_scope": "probe_block",
                        "candidate_authors": audit_authors,
                        "probe_seed": 0,
                        "objective_acceptance_rule": rule,
                        "probe_config": {
                            "block_last_n": 8, "eta": 3e-4, "norm_eta": 3e-3,
                            "n_dirs": 64, "seed": 0, "loss": "seq_mean_answer_nll",
                        },
                        "implementation_variants": {"test": True},
                        "sentence_encoder": {
                            "model": "all-MiniLM-L6-v2", "package_version": "test"
                        },
                        "code_commit": "commit",
                        "code_dirty": False,
                    }
                    (run / "run_manifest.json").write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )

            aggregate_out = root / "aggregate"
            subprocess.run([
                sys.executable,
                str(ROOT / "experiments/channel_matrix/aggregate.py"),
                "--root", str(audit_root),
                "--out", str(aggregate_out),
                "--n-boot", "10",
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            tex = root / "table.tex"
            stress_tex = root / "stress.tex"
            subprocess.run([
                sys.executable,
                str(ROOT / "experiments/channel_matrix/make_main_table.py"),
                "--report", str(aggregate_out / "pooled_channel_report.csv"),
                "--summary", str(aggregate_out / "pooled_channel_report.json"),
                "--out", str(tex),
                "--stress-out", str(stress_tex),
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            summary = json.loads(
                (aggregate_out / "pooled_channel_report.json").read_text(encoding="utf-8")
            )
            rendered = tex.read_text(encoding="utf-8")
            rendered_stress = stress_tex.read_text(encoding="utf-8")
            self.assertEqual(summary["n_runs"], 4)
            self.assertEqual(summary["stress_objectives"], stress)
            self.assertEqual(summary["objective_status"]["idkdpo"]["failed_runs"], 4)
            self.assertIn("Bwd-free", rendered)
            self.assertNotIn("IdkDPO", rendered)
            self.assertIn("IdkDPO", rendered_stress)
            if shutil.which("pdflatex"):
                wrapper = root / "wrapper.tex"
                wrapper.write_text(
                    "\\documentclass{article}\n"
                    "\\usepackage{booktabs}\n"
                    "\\begin{document}\n"
                    "\\input{table.tex}\n"
                    "\\input{stress.tex}\n"
                    "\\end{document}\n",
                    encoding="utf-8",
                )
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "wrapper.tex"],
                    cwd=root, check=True, capture_output=True, text=True,
                )


if __name__ == "__main__":
    unittest.main()
