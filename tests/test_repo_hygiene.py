"""Regression tests for repo hygiene (predecessor repo once ignored its own
source tree via a bad .gitignore pattern)."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUST_BE_TRACKED = [
    "src/rsus/blocks.py",
    "src/rsus/probe/finite_diff.py",
    "tests/conftest.py",
    "prereg/constants.yaml",
    "pyproject.toml",
]


def test_critical_paths_not_ignored():
    for rel in MUST_BE_TRACKED:
        assert (ROOT / rel).exists(), rel
        r = subprocess.run(
            ["git", "check-ignore", "-q", rel], cwd=ROOT, capture_output=True
        )
        assert r.returncode != 0, f"{rel} is gitignored"


def test_runs_and_caches_are_ignored():
    for rel in ["runs/x", "checkpoints/x", "__pycache__/x"]:
        r = subprocess.run(
            ["git", "check-ignore", "-q", rel], cwd=ROOT, capture_output=True
        )
        assert r.returncode == 0, f"{rel} should be gitignored"
