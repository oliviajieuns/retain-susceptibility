"""Generate queue units for the sealed channel-matrix / alpha campaigns.

Sharding follows the finest grain the frozen runners already support, so a
unit never shares a run directory with another unit:

- ``fidelity``           one unit per model (single frozen cell)
- ``calibration``        one unit per model x calibration author (--only-authors)
- ``audit``              one unit per model x audit author (--only-authors)
- ``alpha-development``  one unit per model x author x seed (--worker)
- ``alpha-audit``        one unit per model x author x seed (--worker)

Freeze boundaries are not encoded here on purpose: select-freeze and
select-alpha-freeze remain human review steps, so enqueue one phase, wait for
it to drain, run the selector, commit the freeze, then enqueue the next phase.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from workqueue import Unit, WorkQueue  # noqa: E402

PHASES = ("fidelity", "calibration", "audit", "alpha-development", "alpha-audit")


def _enabled_models(cfg: dict, selected: set[str]) -> list[str]:
    models = [m["id"] for m in cfg["models"] if m.get("enabled", True)]
    if selected:
        unknown = selected - set(models)
        if unknown:
            raise ValueError(f"--model-id not enabled in config: {sorted(unknown)}")
        models = [m for m in models if m in selected]
    if not models:
        raise ValueError("no enabled model selected")
    return models


def build_units(cfg: dict, config_rel: str, phase: str, models: list[str],
                max_attempts: int, unit_suffix: str = "") -> list[Unit]:
    python = sys.executable or "python"
    units: list[Unit] = []

    def add(unit_id: str, cmd: list[str]) -> None:
        if unit_suffix:
            unit_id = f"{unit_id}__{unit_suffix}"
        units.append(Unit(unit_id=unit_id, cmd=cmd, gpus=1, max_attempts=max_attempts))

    if phase == "fidelity":
        for model in models:
            add(f"fid__{model}", [
                python, "-u", "experiments/channel_matrix/run_campaign.py",
                "--config", config_rel, "--phase", "fidelity",
                "--resume", "--model-id", model,
            ])
    elif phase in ("calibration", "audit"):
        roster = cfg[phase]["authors"]
        for model in models:
            for author in roster:
                add(f"{phase[:3]}__{model}__a{author}", [
                    python, "-u", "experiments/channel_matrix/run_campaign.py",
                    "--config", config_rel, "--phase", phase,
                    "--resume", "--model-id", model,
                    "--only-authors", str(author),
                ])
    elif phase in ("alpha-development", "alpha-audit"):
        alpha_phase = "development" if phase == "alpha-development" else "audit"
        block = cfg["alpha_protection"][alpha_phase]
        for model in models:
            for author in block["authors"]:
                for seed in block["seeds"]:
                    add(f"alpha-{alpha_phase[:3]}__{model}__a{author}__s{seed}", [
                        python, "-u", "experiments/channel_matrix/alpha_protection.py",
                        "--config", config_rel, "--phase", alpha_phase,
                        "--worker", "--resume", "--model-id", model,
                        "--author", str(author), "--seed", str(seed),
                    ])
    else:
        raise ValueError(f"unknown phase {phase!r}")
    return units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/channel_matrix/7b_tofu.yaml")
    parser.add_argument("--phase", action="append", required=True, choices=PHASES,
                        help="repeatable; phases are emitted in the given order")
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--unit-suffix", default="",
                        help="append __<suffix> to every unit id (e.g. r2 for a "
                             "grid-extension re-enqueue; the queue is append-only per id)")
    parser.add_argument("--out", help="write units JSONL here instead of/in addition to --enqueue")
    parser.add_argument("--enqueue", action="store_true", help="enqueue directly into --queue")
    parser.add_argument("--queue", help="queue root (required with --enqueue)")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() \
        else Path(args.config)
    config_rel = args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = _enabled_models(cfg, set(args.model_id))

    units: list[Unit] = []
    for phase in args.phase:
        units.extend(build_units(cfg, config_rel, phase, models, args.max_attempts,
                                 unit_suffix=args.unit_suffix))

    ids = [u.unit_id for u in units]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate unit ids generated; refusing to emit")

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
