"""Read-only campaign oracle: which queue actions are allowed RIGHT NOW.

Autonomous agents (OpenCode, Hermes, Claude, ...) must call this before
enqueueing anything.  It mirrors the freeze gates that ``enqueue_table12.sh``
and ``run_campaign.py`` enforce, so an agent that follows its output never
trips a sealed-phase guard.  It never mutates queue state, configs, or runs.

    python experiments/cluster/next_actions.py          # human-readable table
    python experiments/cluster/next_actions.py --json   # machine-readable

Exit code is always 0 when the report itself could be produced; a blocked
campaign is data, not an error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]

# setting label -> (campaign config, queue root, first-wave phases)
CAMPAIGNS = {
    "tofu_qwen25_7b": ("configs/channel_matrix/7b_tofu.yaml", "runs/cluster_queue/wave2"),
    "tofu_qwen25_14b": ("configs/channel_matrix/14b_tofu.yaml", "runs/cluster_queue/wave1_14b"),
    "tofu_llama31_8b": ("configs/channel_matrix/llama8b_tofu.yaml", "runs/cluster_queue/wave_llama"),
    "wmdp_bio_mmlu_qwen25_7b": ("configs/channel_matrix/wmdp_7b.yaml", "runs/cluster_queue/wave_wmdp"),
    "wmdp_bio_mmlu_qwen25_14b": ("configs/channel_matrix/wmdp_14b.yaml", "runs/cluster_queue/wave_wmdp14b"),
    "rwku_qwen25_7b": ("configs/channel_matrix/rwku_7b.yaml", "runs/cluster_queue/wave_rwku"),
}

FREEZE_DIR = "configs/channel_matrix"


def _load_yaml(path: Path) -> dict | None:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return value if isinstance(value, dict) else None


def _find_key(mapping: object, key: str) -> object | None:
    """First value of ``key`` anywhere in a nested structure (config layout drifts)."""
    if isinstance(mapping, dict):
        if key in mapping:
            return mapping[key]
        for value in mapping.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(mapping, list):
        for value in mapping:
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def freeze_state(root: Path, config: dict, key: str) -> tuple[str, str | None]:
    """Return (state, path) with state in {frozen, draft, missing, unnamed}."""
    name = _find_key(config, key)
    if not isinstance(name, str) or not name.strip():
        return "unnamed", None
    path = root / FREEZE_DIR / name
    freeze = _load_yaml(path)
    if freeze is None:
        return "missing", str(path.relative_to(root))
    frozen = freeze.get("status") == "frozen" and freeze.get("frozen_before_audit") is True
    return ("frozen" if frozen else "draft"), str(path.relative_to(root))


def queue_counts(root: Path, queue: str) -> dict[str, int] | None:
    queue_dir = root / queue
    if not queue_dir.is_dir():
        return None
    counts = {}
    for state in ("pending", "claimed", "done", "failed"):
        state_dir = queue_dir / state
        counts[state] = (
            sum(1 for p in state_dir.glob("*.json") if not p.name.endswith(".meta.json"))
            if state_dir.is_dir()
            else 0
        )
    return counts


def worktree_clean(root: Path) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() == ""


def campaign_report(root: Path, setting: str, config_rel: str, queue: str) -> dict:
    report: dict = {
        "setting": setting,
        "config": config_rel,
        "queue": queue,
        "allowed_now": [],
        "blocked": [],
        "human_next": [],
    }
    config_path = root / config_rel
    config = _load_yaml(config_path)
    if config is None:
        report["blocked"].append("campaign config missing or unreadable -- authoring a "
                                 "new config is a HUMAN step, do not fabricate one")
        return report

    objective_state, objective_path = freeze_state(root, config, "objective_freeze")
    alpha_state, alpha_path = freeze_state(root, config, "alpha_freeze")
    report["objective_freeze"] = {"state": objective_state, "path": objective_path}
    report["alpha_freeze"] = {"state": alpha_state, "path": alpha_path}
    report["queue_counts"] = queue_counts(root, queue)

    model_path = _find_key(config, "path")
    if isinstance(model_path, str) and model_path.strip():
        provisioned = (Path(model_path) / "config.json").is_file()
        report["model_path"] = model_path
        report["model_provisioned"] = provisioned
        if not provisioned:
            report["blocked"].append(
                f"model not provisioned at {model_path} (see provision_*.sh) -- "
                "everything else waits on this"
            )
            return report

    if objective_state != "frozen":
        # Pre-freeze lane: grids are the sanctioned place for wide sweeps.
        report["allowed_now"] += ["fidelity", "calibration"]
        report["blocked"].append(
            f"audit blocked: objective freeze is {objective_state} "
            f"({objective_path}) -- drain calibration, then HUMAN runs "
            "select-freeze and commits the freeze"
        )
        report["human_next"].append("select-freeze -> commit objective_freeze")
        return report

    report["allowed_now"].append("audit")
    if alpha_state == "frozen":
        report["allowed_now"].append("alpha-audit")
    else:
        report["allowed_now"].append("alpha-development")
        report["blocked"].append(
            f"alpha-audit blocked: alpha freeze is {alpha_state} ({alpha_path}) -- "
            "drain alpha-development, then HUMAN runs select-alpha-freeze and "
            "commits the freeze"
        )
        report["human_next"].append("select-alpha-freeze -> commit alpha_freeze")
    return report


def build_report(root: Path) -> dict:
    clean = worktree_clean(root)
    campaigns = [
        campaign_report(root, setting, config_rel, queue)
        for setting, (config_rel, queue) in CAMPAIGNS.items()
    ]
    if clean is False:
        for campaign in campaigns:
            campaign["blocked"].append(
                "worktree dirty: audit-family enqueue refuses until the tree is "
                "committed/clean (do NOT git push from the cluster; hand results "
                "to the session instead)"
            )
    return {
        "worktree_clean": clean,
        "note": (
            "allowed_now lists queue phases this agent may enqueue via "
            "enqueue_table12.sh / make_units.py; anything in blocked or "
            "human_next is out of agent scope by preregistration design"
        ),
        "campaigns": campaigns,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--root", default=str(ROOT), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    report = build_report(Path(args.root))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(f"worktree_clean: {report['worktree_clean']}")
    for campaign in report["campaigns"]:
        print(f"\n== {campaign['setting']} ({campaign['queue']}) ==")
        print(f"  allowed_now : {', '.join(campaign['allowed_now']) or '(nothing)'}")
        for line in campaign["blocked"]:
            print(f"  blocked     : {line}")
        for line in campaign["human_next"]:
            print(f"  human_next  : {line}")
        counts = campaign.get("queue_counts")
        if counts:
            print(f"  queue       : {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
