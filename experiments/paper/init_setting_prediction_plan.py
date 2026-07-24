"""Build a prediction-only raw plan for one setting from its frozen weight.

The full init_raw_plan.py requires a master selection_freeze covering every
setting and parent plus the five appendix artifact contracts.  When only one
setting's prediction weight is frozen (e.g. the 7B core parents), this emits
the minimal immutable denominator aggregate_raw needs to build the ledger's
prediction rows: one (setting, parent, request, seed) unit per audit cell,
with the frozen prediction alpha and an unresolved protection selection.

The plan is still the denominator — only the named parents appear, and the
downstream contract keeps every other predeclared row as a missing denominator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.evidence.raw import raw_plan_from_mapping  # noqa: E402
from rsus.evidence.schemas import EvidenceValidationError  # noqa: E402


def _request_id(dataset: str, author: int) -> str:
    key = str(dataset).lower()
    if key in ("tofu", "locuslab/tofu"):
        return f"tofu-a{author}"
    if key in ("rwku", "jinzhuoran/rwku"):
        return f"rwku-t{author:03d}"
    if key in ("wmdp_bio_mmlu", "wmdp-bio/mmlu", "wmdp-bio"):
        return f"wmdp-r{author:03d}"
    raise EvidenceValidationError(f"no request-id convention for dataset {dataset!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-config", required=True)
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--prediction-alpha-freeze", required=True)
    parser.add_argument("--parents", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--top-q", type=float, default=0.10)
    parser.add_argument("--cvar-q", type=float, default=0.95)
    parser.add_argument(
        "--draws",
        nargs="+",
        default=["rand-000", "rand-001", "rand-002", "rand-003", "rand-004"],
        help="frozen repeated-random draw ids (unused by prediction, required "
             "by the plan schema)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = yaml.safe_load(Path(args.campaign_config).read_text(encoding="utf-8"))
        freeze = yaml.safe_load(
            Path(args.prediction_alpha_freeze).read_text(encoding="utf-8")
        )
        if freeze.get("status") != "frozen":
            raise EvidenceValidationError(
                "prediction alpha freeze is not frozen; review and freeze first"
            )
        table = freeze.get("prediction_alpha") or {}
        dataset = str(cfg.get("dataset", "tofu"))
        audit = cfg["audit"]
        authors = list(audit["authors"])
        seeds = list(audit["seeds"])

        units = []
        for parent in args.parents:
            if parent not in table:
                raise EvidenceValidationError(
                    f"parent {parent!r} has no frozen prediction alpha"
                )
            alpha = float(table[parent])
            for author in authors:
                for seed in seeds:
                    units.append({
                        "setting": args.setting_id,
                        "parent": parent,
                        "request": _request_id(dataset, author),
                        "seed": str(seed),
                        "prediction_selection": {
                            "valid": True, "fallback": False, "alpha": alpha,
                        },
                        # Protection weight is unresolved for these rows; the
                        # prediction-only ledger never reads it.
                        "protection_selection": {
                            "valid": False, "fallback": False, "alpha": None,
                        },
                        "repeated_random_draws": list(args.draws),
                    })

        plan = {
            "schema_version": 1,
            "campaign_id": cfg.get("campaign_id"),
            "selection_freeze_id": freeze.get("freeze_id"),
            "native_margins": {},
            "bootstrap": {
                "replicates": args.replicates,
                "seed": args.bootstrap_seed,
                "alpha": args.alpha,
                "top_q": args.top_q,
                "cvar_q": args.cvar_q,
            },
            "units": units,
        }
        raw_plan_from_mapping(plan)  # validate before writing
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out}: {len(units)} units "
              f"({len(args.parents)} parents x {len(authors)} requests x {len(seeds)} seeds)")
        return 0
    except (EvidenceValidationError, KeyError, OSError, yaml.YAMLError) as error:
        print(f"plan build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
