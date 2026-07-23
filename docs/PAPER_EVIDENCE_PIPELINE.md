# Paper evidence pipeline

`experiments/paper/build_evidence.py` is the only paper-facing result gate.
Experiment runners may write many diagnostic artifacts, but the paper consumes
only a normalized ledger checked against `configs/paper/evidence.yaml`.

The config is the denominator. Every configured setting--parent row appears in
the readiness report even if it was never attempted. A row cannot pass while a
planned trajectory is incomplete, a profile is invalid, a development weight
is unresolved/fallback, common support is missing, or a protection arm is
infeasible. Conditional estimates may still be inspected, but they never
license prose.

## Run

From the repository root:

```powershell
python experiments/paper/build_evidence.py
python experiments/paper/build_evidence.py --paper-root ../paper --require-ready
```

The first command always writes `results/paper/evidence_readiness.json` and is
useful while jobs are running. The second additionally verifies every completed
artifact's SHA-256 and atomically replaces
`<paper-root>/sections/generated/results_macros.tex`. It returns exit status 2
until all registered tables have complete data. Invalid schemas, hashes, or
unregistered rows return exit status 1 and do not touch the paper macro file.

The generated file owns exactly these commands:

- `\TailHeadline`
- `\PredictionHeadline`
- `\FidelityHeadline`
- `\ProtectionHeadline`
- `\TransferHeadline`

An incomplete evidence block remains a visible `\resph{...}` placeholder.

## Normalized row contract

The ledger has `schema_version: 1`, a `rows` list, and an `artifacts` mapping.
Each row is keyed by the predeclared `setting` and `parent`. Its funnel records:

```text
profiles_valid <= profiles_planned
trajectories_reached <= trajectories_completed
  <= trajectories_attempted <= trajectories_planned
prediction_common <= reached_with_valid_profile <= trajectories_reached
protection_common <= protection_feasible_all_arms
  <= reached_with_valid_profile
```

`completed: true` additionally requires all planned trajectories to have been
attempted and completed. Prediction supplies paired joint-minus-`S0` and
joint-minus-`S1` estimates, one-sided lower bounds, and corrected p-values.
Protection supplies mixture-minus-comparator estimates, one-sided upper bounds,
and corrected p-values for all four comparators (`no_repair`,
`repeated_random`, `s0`, `s1`) crossed with `mean` and `cvar95`. The exact-norm
reference may be recorded under `exact_norm`, but it is intentionally outside
the eight-way intersection--union test. Both claim blocks must explicitly set
`paired: true`; omitting it fails closed rather than assuming pairing.

Completed non-row artifacts require `path`, `sha256`, and may provide a
validated `headline_tex`. Relative paths are resolved against this repository.

## Table registry

The registry covers two main-paper tables and five appendix evidence blocks
across four physical appendix tables (budget and specificity share one float):

| Registry ID | Paper label | Required evidence |
|---|---|---|
| `main_core_evidence` | `tab:core-evidence` | primary prediction + protection rows |
| `main_robustness` | `tab:robustness` | all predeclared setting--parent rows |
| `appendix_scope_contract` | `tab:datasets` | frozen campaign manifest |
| `appendix_tail_prediction` | `tab:tail-structure` | primary prediction + tail artifact |
| `appendix_lse_fidelity_cost` | `tab:bwfree` | LSE fidelity/time/memory artifact |
| `appendix_protection_budget` | `tab:budget-sweep` | primary protection + budget sweep |
| `appendix_boundaries` | `tab:specificity` | all rows + negative controls |

Readiness and claim success are separate. A fully observed null or failed IUT
makes a table data-ready while correctly leaving the scientific claim failed.
The setting-level rule is also explicit: at least one output-readout and one
representation-readout parent must each pass both claims after a Bonferroni
correction within its predeclared parent group; the other parent rows remain
reported denominators. The multi-setting statement then requires
the primary setting, at least one of two model-transfer settings, and at least
two of three dataset replications. Stress settings can never rescue the rule.

## Candidate-level raw aggregation

`experiments/paper/aggregate_raw.py` is the shared CPU path from all dataset
adapters to the normalized ledger. Dataset runners write JSONL records; one
immutable JSON plan fixes every `(setting, parent, request, seed)` denominator
and repeated-random draw roster before results are inspected.

Create that plan only after development selections are frozen:

```powershell
# First run writes a deliberately invalid draft; fill every selected parent.
python experiments/paper/init_raw_plan.py `
  --write-selection-template configs/paper/selection_freeze.yaml

# Set status: frozen, a final non-PENDING freeze_id, and
# frozen_before_target: true, then freeze the executable denominator.
python experiments/paper/init_raw_plan.py `
  --selection-freeze configs/paper/selection_freeze.yaml `
  --out results/paper/raw_plan.json
```

`init_raw_plan.py` rejects unresolved target rosters, unprovisioned models,
missing parent implementations, duplicate seeds/draws, a draft freeze ID, and
fidelity settings that the cost runner cannot execute. A fallback selection
can be frozen explicitly but remains claim-ineligible. For development of one
ready setting, repeat `--setting SETTING_ID`; this is a partial execution plan,
not evidence for omitted settings.

The fidelity artifact keys include model ID and frozen source path, precision, the exact block regex,
the SHA-256 of every frozen runner argument, profiler, direction count, and
repeat. Therefore the cost runner must pass the
campaign model key explicitly (its path-basename default is intentionally not
accepted). For the current Qwen2.5-7B cell:

```powershell
python experiments/cost/bench.py `
  --model /group-volume/models/Qwen2.5-7B-Instruct `
  --model-id Qwen2.5-7B --device cuda --dtype float32 `
  --author 198 --candidate-authors 0-29 --n-candidates 128 `
  --candidate-seed 314159 --block-last-n 8 --norm-eta 0.003 `
  --dirs 16,32,64 --batch-size 4 --repeats 3 --seed 0 --k 0 `
  --min-rho 0.8 --min-overlap 0.7 --min-split-half 0.7 `
  --min-perturbation-survival 0.9 `
  --out runs/paper/lse_qwen25_7b.jsonl
```

Use the identical execution block for Qwen2.5-1.5B, changing only `--model`,
`--model-id Qwen2.5-1.5B`, and the output filename. The raw key validator also
checks precision, the runtime-emitted block regex, and the protocol hash, so a
wrong dtype, radius, threshold, roster seed, or `--block-last-n` cannot fill
the planned cell.

```powershell
python experiments/paper/aggregate_raw.py `
  --plan results/paper/raw_plan.json `
  --prediction-raw runs/tofu/prediction.jsonl `
  --protection-raw runs/tofu/protection.jsonl `
  --artifact-raw lse_fidelity_cost=runs/paper/lse.jsonl `
  --artifact-raw protection_budget_sweep=runs/paper/budget.jsonl `
  --artifact-raw specificity_negative_controls=runs/paper/specificity.jsonl `
  --artifact-raw tail_structure=runs/paper/prediction_supplement.jsonl `
  --out results/paper/evidence_ledger.json
```

An absent shard is allowed, but its unit remains planned and makes its row or
artifact incomplete. An extra unit, duplicate candidate, changed frozen
selection, undeclared random draw, duplicate measurement key, or raw artifact
without a plan contract is an error. The command prints the consumed plan's
SHA-256.

### Immutable plan and candidate records

The core plan has `schema_version: 1`, bootstrap fields (`replicates`, `seed`,
`alpha`, `top_q`, `cvar_q`), and a `units` list. Each unit contains its four
keys, frozen prediction/protection selections, and exact
`repeated_random_draws`. Selections cannot vary within a setting--parent row.

Prediction JSONL has one row per candidate and unit: the four unit keys,
`candidate_id`, semantic `group`, `s0`, `s1`, `joint`, `damage`,
`profile_valid`, `reached`, `trajectory_completed`, and the frozen selection.
The aggregator forms all correlations on identical candidates, averages seeds
within requests and requests equally, then bootstraps requests, seeds, and
semantic groups. Missing support is not repaired by intersection.

Protection JSONL has one row per candidate, arm, and unit, with `damage`, the
frozen selection, and four explicit slacks: `direct_forget_margin`,
`paraphrase_forget_margin`, `extraction_generation_margin`, and
`utility_margin`. `feasible` must equal their conjunction; a missing metric or
an inconsistent flag is rejected. Every row also carries the same
`parent_checkpoint_id` and `parent_checkpoint_first_reaching: true`, proving
that all arms branch from one first criterion-reaching parent state. Ordinary
arms omit `draw_id`; `repeated_random` supplies every planned `draw_id` and
`draw_complete`. The five arms must have exact candidate/group support.
Feasibility and common support remain separate funnels: fully feasible arms
with mismatched candidates cannot enter a paired effect. The eight
mean/CVaR95 effects are paired. Inside each bootstrap replicate, random draw
IDs are themselves resampled with replacement before averaging; a missing or
incomplete draw makes the unit infeasible and non-common.

### Runner-to-ledger boundary

The historical channel-matrix JSON (`results.json`) is not the candidate JSONL
schema above and is never consumed directly by the paper gate. In particular,
`aggregate_alpha_protection.py` is a legacy CVaR-only diagnostic. A paper GPU
runner/exporter must map the frozen mixture to `joint`, the unrepaired entry to
`no_repair`, every frozen random draw separately to `repeated_random`, and the
two endpoints to `s0`/`s1`; duplicate an endpoint as `joint` when a frozen
weight is exactly 0 or 1. It must derive each of the four slacks from the exact
reported checkpoint and preserve the parent block hash as
`parent_checkpoint_id`.

The strict bridge is `experiments/paper/export_alpha_protection.py`:

```powershell
python experiments/paper/export_alpha_protection.py `
  --plan results/paper/raw_plan.json `
  --runner-root runs/paper/tofu_qwen25_7b/alpha_protection/audit `
  --setting tofu_qwen25_7b `
  --out runs/paper/tofu_qwen25_7b/protection.jsonl
```

It validates the paper model source, every planned request/seed/parent,
protection alpha, random-draw roster, exact candidate/group support, four
slacks, and shared first-reaching checkpoint hash, then passes its own output
through the authoritative raw parser before writing atomically. The old 7B
YAML's 181/186/191 audit roster differs from the immutable paper target roster
and is rejected as missing/extra cells rather than relabeled. A remaining GPU
execution blocker is therefore the launcher configuration itself: generate a
paper-plan-matching target run (including `D_prot`-only weight selection)
before this exporter can succeed on real artifacts.

### Schema-backed appendix artifacts

Artifacts are generated beside the ledger and marked complete only after
their exact planned keys validate. Incomplete diagnostic JSON is written with
`completed: false`, so it cannot license a table or headline.

- `campaign_manifest` freezes every scope and feasibility table column.
- `tail_structure` derives damage concentration, semantic-group lift,
  hierarchical intervals, and permutation p-values from candidate rows. Its
  contracted `reference_roster` and `construction_checks` blocks are also
  mandatory, so Panels B/C cannot remain unbacked while Panel A is complete.
- `lse_fidelity_cost` requires fidelity/overlap, split-half and perturbation
  checks, synchronized time, peak memory, integrity, and backward-access cells.
- `protection_budget_sweep` requires worst effect/UCB, bottleneck,
  eligibility/pass, margins, accepted updates, common support, and random-draw
  completeness for every budget--parent cell.
- `specificity_negative_controls` requires the three correlations, top-q
  lift, displacement matching, and common support for every motion cell.

Measurement contracts declare `key_fields`, `group_by`, exact `planned` keys,
and typed metrics with deterministic aggregation. Output JSON retains every
cell's `source_keys`, providing cell-level provenance rather than only an
arbitrary file hash.

## Physical table source map

| Physical table/block | Cell or decision | Producer/source |
|---|---|---|
| Main `tab:core-evidence` | prediction effects/LBs | all-common candidate prediction rows |
| Main `tab:core-evidence` | protection effects/UCBs, margins, funnels | exact five-arm candidate protection rows |
| Main `tab:robustness` | coverage and least-favorable bounds | normalized ledger plus claim decisions; missing rows stay denominators |
| Appendix contract | scope and feasibility | immutable campaign manifest |
| Appendix tail/reference/construction | tail Panel A | candidate `damage` and semantic `group` |
| Appendix tail/reference/construction | Panels B/C | contracted supplementary raw keys |
| Appendix fidelity/cost | all cells | contracted LSE measurements and retained source keys |
| Appendix protection/specificity | budget Panel A | contracted budget--parent measurements |
| Appendix protection/specificity | specificity Panel B | contracted motion--conditioning measurements |

The measurement artifacts never manufacture upstream values. GPU runners must
export contracted unit-level measurements, including each budget--parent
bootstrap UCB, before those blocks become ready. Core and tail statistics are
computed directly here; fidelity instrumentation, accepted-update logs,
budget summaries, and motion controls remain outputs of their named runners.

## 2026-07-23 addendum — Table 1/2 body generation and the run exporter

The pipeline now renders the two main tables' bodies, not just the five
headline macros:

1. `experiments/paper/export_channel_matrix_raw.py` walks sealed gate audit
   cells (`<output_root>/audit/...`) and alpha-protection audit cells and
   emits candidate-level `prediction.jsonl`/`protection.jsonl` in the raw
   schema, opening sealed audit scores only through `rsus.sealing` (DONE
   markers required). It can also summarize a frozen fd-fidelity certificate
   into the `fidelity_inputs` JSON consumed by the table renderer.
2. `aggregate_raw.py` additionally computes, per row: a one-sided bound for
   the absolute joint rank correlation, the above-chance tail lift on
   tail-eligible cells (with eligible/total counts), the gain over the frozen
   simple control (`control` column in prediction records), absolute
   joint/no-repair mean+CVaR, per-comparator native-metric non-inferiority
   effects (`native_metric` in protection records; margins via the plan's
   `native_margins`), and mean accepted/rolled-back update diagnostics.
3. `decisions.py` enforces the paper's full IUTs: RQ1 = joint + both endpoint
   gains + tail lift with >=0.80 tail coverage; RQ3 = eight damage UCBs plus
   four native non-inferiority LBs plus feasibility/margins.
4. `build_evidence.py --paper-root` writes
   `sections/generated/table_core_evidence.tex` and
   `sections/generated/table_robustness.tex` (labels `tab:core-evidence`,
   `tab:robustness`) alongside the macros; every incomplete cell stays
   `\tblph`. RQ2 cells compose the frozen fidelity floors (`tau_rho=0.80`,
   `tau_K=0.70`) with the `g_H`/`g_ctl` bounds and only report pass when all
   bounds exist and clear their floors.

Campaign wave -> table mapping and the pre-run outcome forecast live in
`docs/plan_table12_campaign.md`.
