"""Generate seed-replicate queue units from a sealed gate run's manifest.

Table 1 confidence intervals need seed replicates that are *identical* to the
original run except for ``--seed``.  Reconstructing the command by hand risks
silent drift (defaults changed since the original run), so this tool rebuilds
the exact CLI from the original run's ``run_manifest.json`` (which stores
``cli: vars(args)``) and overrides only the seed and the run tag.

Usage (on the cluster, from the repo root):

    python experiments/cluster/make_replicate_units.py \
      --source-run runs/gate_Qwen2.5-1.5B-Instruct_chanbal2 \
      --seeds 2026-2033 --enqueue --queue runs/cluster_queue/t2_gate15b

Every unit gets ``max_attempts: 1``: gate.py run tags are append-only, so a
second attempt with the same tag exits 1 by design and a failure needs a
human look (see the runbook triage section).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "experiments" / "gate_1p5b" / "gate.py"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from workqueue import Unit, WorkQueue  # noqa: E402

# Overridden or intentionally dropped when rebuilding the replicate command.
OVERRIDDEN = {"seed", "run_tag", "out_dir", "smoke"}


def parse_gate_flags(source: str) -> dict[str, dict]:
    """Map argparse dest -> {flag, store_true} by reading gate.py's source.

    Parsing the source instead of importing keeps this runnable on nodes
    without torch, and drift is caught: a manifest key with no matching
    current flag is a hard error below.
    """
    flags: dict[str, dict] = {}
    for match in re.finditer(
        r"add_argument\(\s*\"(--[a-z0-9-]+)\"(.*?)\)", source, re.DOTALL
    ):
        flag, rest = match.group(1), match.group(2)
        dest = flag.lstrip("-").replace("-", "_")
        flags[dest] = {"flag": flag, "store_true": 'action="store_true"' in rest}
    return flags


def replicate_command(cli: dict, flags: dict[str, dict], seed: int, run_tag: str,
                      python: str) -> list[str]:
    unknown = [k for k in cli if k not in flags and k not in OVERRIDDEN]
    if unknown:
        raise ValueError(
            f"manifest CLI keys with no matching gate.py flag: {sorted(unknown)}; "
            "gate.py has drifted since the source run — replicate by hand after "
            "reviewing the drift, or update this tool's mapping"
        )
    cmd = [python, "-u", str(GATE.relative_to(ROOT))]
    for dest, spec in flags.items():
        if dest in OVERRIDDEN or dest not in cli:
            continue
        value = cli[dest]
        if spec["store_true"]:
            if value:
                cmd.append(spec["flag"])
            continue
        if value is None:
            continue
        cmd.extend([spec["flag"], str(value)])
    cmd.extend(["--seed", str(seed), "--run-tag", run_tag])
    return cmd


def expand_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"empty or duplicated seed list: {raw!r}")
    return seeds


def build_units(manifest: dict, seeds: list[int], tag_base: str,
                python: str, gate_source: str) -> list[Unit]:
    cli = manifest["cli"]
    flags = parse_gate_flags(gate_source)
    original_seed = int(cli["seed"])
    units = []
    for seed in seeds:
        if seed == original_seed:
            raise ValueError(
                f"seed {seed} equals the source run's seed; replicates must differ"
            )
        run_tag = f"{tag_base}-s{seed}"
        units.append(Unit(
            unit_id=f"gate__{run_tag}",
            cmd=replicate_command(cli, flags, seed, run_tag, python),
            gpus=1,
            max_attempts=1,
        ))
    return units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True,
                        help="run directory containing run_manifest.json")
    parser.add_argument("--seeds", required=True, help="comma list/ranges, e.g. 2026-2033")
    parser.add_argument("--tag-base", default="",
                        help="run-tag prefix (default: source run's run_tag or dir name)")
    parser.add_argument("--out", help="write units JSONL here")
    parser.add_argument("--enqueue", action="store_true")
    parser.add_argument("--queue")
    args = parser.parse_args()

    source = Path(args.source_run)
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    tag_base = args.tag_base or manifest["cli"].get("run_tag") or source.name
    units = build_units(
        manifest,
        expand_seeds(args.seeds),
        tag_base,
        "python",  # resolved by the worker's activated venv
        GATE.read_text(encoding="utf-8"),
    )

    print(f"source: {source} (model={manifest.get('model_id')}, "
          f"seed={manifest['cli']['seed']}, objectives={manifest.get('objectives')})")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "".join(json.dumps(u.to_payload(), sort_keys=True) + "\n" for u in units),
            encoding="utf-8",
        )
        print(f"wrote {len(units)} unit(s) to {out}")
    if args.enqueue:
        if not args.queue:
            parser.error("--enqueue requires --queue")
        added = WorkQueue(Path(args.queue)).enqueue(units)
        print(f"enqueued {len(added)} unit(s) into {args.queue}")
    if not args.out and not args.enqueue:
        for unit in units:
            print(json.dumps(unit.to_payload(), sort_keys=True))


if __name__ == "__main__":
    main()
