"""Probe candidate unlearning benchmarks and dump their real schemas.

The paper contract lists WMDP-bio/MMLU, MUSE-News, MUSE-Books, RWKU, and
PISTOL, but ``src/rsus/data/registry.py`` is intentionally fail-closed: an
adapter may only be registered once it constructs a real
``rsus.data.base.Request``. Writing an adapter against a guessed schema wastes
GPU time on the cluster, so this CPU-only probe downloads nothing heavyweight
(builder metadata + one streamed example per split) and dumps what each
dataset actually looks like to ``runs/dataset_probes/<key>.json``.

Run on any cluster node (HF Hub reachable there):

    python experiments/paper/dataset_probe.py            # all candidates
    python experiments/paper/dataset_probe.py rwku pistol
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The cluster keeps every benchmark pre-downloaded in the shared HF cache;
# interactive shells often lack the variable, so default to it when present.
_SHARED_HF_HOME = "/group-volume/data/hf_home"
if os.path.isdir(_SHARED_HF_HOME):
    os.environ.setdefault("HF_HOME", _SHARED_HF_HOME)

# Multiple candidate hub ids per benchmark: the first one that resolves wins,
# unresolved ids are recorded in the dump so the schema report is honest.
CANDIDATES: dict[str, list[str]] = {
    "rwku": ["jinzhuoran/RWKU"],
    "pistol": ["xinchiqiu/PISTOL", "pistol-dataset/PISTOL"],
    "muse_news": ["muse-bench/MUSE-News"],
    "muse_books": ["muse-bench/MUSE-Books"],
    "wmdp": ["cais/wmdp"],
    "mmlu": ["cais/mmlu"],
}

MAX_CONFIGS = 24
MAX_TEXT = 400


def truncate_value(value, max_text: int = MAX_TEXT):
    """Shrink an example for the dump: long strings clipped, containers recursed."""
    if isinstance(value, str):
        return value if len(value) <= max_text else value[:max_text] + f"...<+{len(value) - max_text} chars>"
    if isinstance(value, dict):
        return {k: truncate_value(v, max_text) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        head = [truncate_value(v, max_text) for v in value[:3]]
        if len(value) > 3:
            head.append(f"...<+{len(value) - 3} items>")
        return head
    return value


def probe_repo(repo: str) -> dict:
    import datasets

    report: dict = {"repo": repo, "configs": {}}
    config_names = datasets.get_dataset_config_names(repo)
    report["config_names"] = config_names
    for config in config_names[:MAX_CONFIGS]:
        entry: dict = {}
        try:
            builder = datasets.load_dataset_builder(repo, config)
            info = builder.info
            entry["splits"] = {
                name: getattr(split, "num_examples", None)
                for name, split in (info.splits or {}).items()
            }
            entry["features"] = {k: str(v) for k, v in (info.features or {}).items()}
            samples = {}
            for split in list(entry["splits"] or {"train": None})[:2]:
                try:
                    stream = datasets.load_dataset(repo, config, split=split, streaming=True)
                    samples[split] = truncate_value(next(iter(stream)))
                except Exception as exc:  # noqa: BLE001 - probe records, never crashes
                    samples[split] = f"<stream failed: {type(exc).__name__}: {exc}>"
            entry["sample"] = samples
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["configs"][config] = entry
    if len(config_names) > MAX_CONFIGS:
        report["configs_truncated"] = len(config_names) - MAX_CONFIGS
    return report


def main() -> None:
    keys = sys.argv[1:] or list(CANDIDATES)
    unknown = [k for k in keys if k not in CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown benchmark key(s) {unknown}; choose from {list(CANDIDATES)}")
    out_dir = ROOT / "runs" / "dataset_probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        result: dict = {"benchmark": key, "tried": [], "resolved": None}
        for repo in CANDIDATES[key]:
            try:
                result["report"] = probe_repo(repo)
                result["resolved"] = repo
                break
            except Exception as exc:  # noqa: BLE001
                result["tried"].append({"repo": repo, "error": f"{type(exc).__name__}: {exc}"})
        out = out_dir / f"{key}.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        status = result["resolved"] or "UNRESOLVED"
        configs = list(result.get("report", {}).get("configs", {})) if result["resolved"] else []
        print(f"{key}: {status}  configs={configs[:8]}{'...' if len(configs) > 8 else ''}")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
