"""Probe candidate unlearning benchmarks and dump their real schemas.

The paper contract lists WMDP-bio/MMLU, MUSE-News, MUSE-Books, RWKU, and
PISTOL, but ``src/rsus/data/registry.py`` is intentionally fail-closed: an
adapter may only be registered once it constructs a real
``rsus.data.base.Request``. Writing an adapter against a guessed schema wastes
GPU time, so this CPU-only probe dumps what each dataset actually looks like
to ``runs/dataset_probes/<key>.json``.

Cache-first and offline by default: everything already under the shared
``HF_HOME`` cache is read directly from disk (config list from the cache
layout, features/splits from ``dataset_info.json``, one sample row from the
arrow file) with no network at all — direct Hugging Face access is unreliable
from the cluster (2026-07-23). Only ``--online`` reaches the hub, honoring
``HF_ENDPOINT`` so a mirror (e.g. https://hf-mirror.com) can be used:

    python experiments/paper/dataset_probe.py rwku          # cache only
    HF_ENDPOINT=https://hf-mirror.com \
      python experiments/paper/dataset_probe.py --online muse_news
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_SHARED_HF_HOME = "/group-volume/data/hf_home"
if os.path.isdir(_SHARED_HF_HOME):
    os.environ.setdefault("HF_HOME", _SHARED_HF_HOME)

CANDIDATES: dict[str, list[str]] = {
    "rwku": ["jinzhuoran/RWKU"],
    "pistol": ["xinchiqiu/PISTOL", "pistol-dataset/PISTOL"],
    "muse_news": ["muse-bench/MUSE-News"],
    "muse_books": ["muse-bench/MUSE-Books"],
    "wmdp": ["cais/wmdp"],
    "mmlu": ["cais/mmlu"],
    "tofu": ["locuslab/TOFU"],
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


def cache_dataset_dir(repo: str, hf_home: Path | None = None) -> Path:
    home = hf_home or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "datasets" / repo.replace("/", "___").lower()


def find_cached_configs(repo: str, hf_home: Path | None = None) -> dict[str, Path]:
    """Map config name -> directory holding dataset_info.json for cached configs."""
    base = cache_dataset_dir(repo, hf_home)
    if not base.is_dir():
        return {}
    configs: dict[str, Path] = {}
    for config_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        infos = sorted(config_dir.glob("*/*/dataset_info.json"))
        if infos:
            configs[config_dir.name] = infos[-1].parent  # newest fingerprint
    return configs


def parse_dataset_info(info_dir: Path) -> dict:
    info = json.loads((info_dir / "dataset_info.json").read_text(encoding="utf-8"))
    features = info.get("features", {})
    splits = info.get("splits", {})
    return {
        "splits": {name: spec.get("num_examples") for name, spec in splits.items()},
        "features": {name: json.dumps(spec, sort_keys=True) for name, spec in features.items()},
    }


def sample_from_arrow(info_dir: Path) -> dict:
    import datasets

    samples: dict = {}
    for arrow in sorted(info_dir.glob("*.arrow")):
        split = arrow.stem.rsplit("-", 1)[-1]
        try:
            ds = datasets.Dataset.from_file(str(arrow))
            samples[split] = truncate_value(ds[0]) if len(ds) else "<empty>"
        except Exception as exc:  # noqa: BLE001 - probe records, never crashes
            samples[split] = f"<arrow read failed: {type(exc).__name__}: {exc}>"
    return samples


def probe_cached_repo(repo: str) -> dict | None:
    configs = find_cached_configs(repo)
    if not configs:
        return None
    report: dict = {"repo": repo, "source": "local_cache",
                    "config_names": list(configs), "configs": {}}
    for config, info_dir in list(configs.items())[:MAX_CONFIGS]:
        entry = parse_dataset_info(info_dir)
        entry["sample"] = sample_from_arrow(info_dir)
        report["configs"][config] = entry
    return report


def probe_hub_repo(repo: str) -> dict:
    import datasets

    report: dict = {"repo": repo, "source": "hub", "configs": {}}
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
                except Exception as exc:  # noqa: BLE001
                    samples[split] = f"<stream failed: {type(exc).__name__}: {exc}>"
            entry["sample"] = samples
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["configs"][config] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("keys", nargs="*", help=f"benchmarks to probe (default: all of {list(CANDIDATES)})")
    parser.add_argument("--online", action="store_true",
                        help="allow hub access for repos missing from the local cache "
                             "(set HF_ENDPOINT to use a mirror)")
    args = parser.parse_args()

    keys = args.keys or list(CANDIDATES)
    unknown = [k for k in keys if k not in CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown benchmark key(s) {unknown}; choose from {list(CANDIDATES)}")

    if not args.online:
        # Hard offline: the shared cache is the source of truth and the cluster
        # cannot reliably reach huggingface.co.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    out_dir = ROOT / "runs" / "dataset_probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        result: dict = {"benchmark": key, "tried": [], "resolved": None}
        for repo in CANDIDATES[key]:
            cached = probe_cached_repo(repo)
            if cached is not None:
                result["report"] = cached
                result["resolved"] = f"{repo} (cache)"
                break
            if not args.online:
                result["tried"].append({"repo": repo, "error": "not in local cache (rerun with --online for hub)"})
                continue
            try:
                result["report"] = probe_hub_repo(repo)
                result["resolved"] = repo
                break
            except Exception as exc:  # noqa: BLE001
                result["tried"].append({"repo": repo, "error": f"{type(exc).__name__}: {exc}"})
        out = out_dir / f"{key}.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        status = result["resolved"] or "UNRESOLVED"
        configs = list(result.get("report", {}).get("configs", {})) if result["resolved"] else []
        print(f"{key}: {status}  configs={configs[:10]}{'...' if len(configs) > 10 else ''}")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
