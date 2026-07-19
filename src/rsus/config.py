"""Config loading with content hashing and run-dir provenance."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def git_sha(repo_root: str | Path) -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        return "unknown"


def new_run_dir(root: str | Path, cfg: dict, repo_root: str | Path = ".") -> Path:
    h = config_hash(cfg)
    run = Path(root) / h
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    meta = {
        "config_hash": h,
        "git_sha": git_sha(repo_root),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(run / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return run
