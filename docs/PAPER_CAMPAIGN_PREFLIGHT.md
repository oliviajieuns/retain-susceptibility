# Paper campaign preflight

Run the preflight before submitting **any** paper-campaign GPU job:

```powershell
python experiments/paper/preflight.py
```

Exit code `0` is the authorization boundary. Exit code `2` means the
contracts are readable but at least one setting or stage is unresolved; exit
code `1` means a malformed contract. The JSON report is written to
`results/paper/campaign_preflight.json` by default. This check does not load a
dataset or model and is safe on a login/CPU node.

The checked inputs are:

- `configs/paper/evidence.yaml`: the eight setting denominators used by the
  paper;
- `configs/paper/campaign.yaml`: adapters, exact rosters, model sources,
  frozen 8-setting/4-stage contract, precision, executors, and parent
  availability;
- `src/rsus/data/registry.py`: the adapters that actually exist in code.

## Why this is fail-closed

The historical gate runner imports TOFU directly. Reusing it with a new label
would therefore produce TOFU results masquerading as another benchmark. The
registry has no default adapter: an unknown key raises
`AdapterNotFoundError`, and preflight marks every affected stage not ready.
Do not add a metadata-only registry entry to silence this error. Register an
adapter only when its factory constructs a real `rsus.data.base.Request`.

Each paper setting is checked at four stages:

| Stage | Frozen roster | Required adapter capability |
|---|---|---|
| calibration | `D_cal` | `calibration` |
| prediction | `D_pred` | `prediction` |
| protection | `D_prot` | `protection` |
| target evaluation | `target` | `target_evaluation` |

An adapter capability is necessary but not an execution claim. Each stage
also names a real Python entrypoint. Preflight inspects its literal
`PAPER_STAGE_CONTRACT` marker and requires campaign-config dispatch, adapter
registry dispatch, exact-roster consumption, and stage-specific raw output.
Prediction must emit candidate-level prediction records; protection must emit
candidate-by-arm records, including repeated-random draw IDs and feasibility
margins. The legacy summary CSV/JSON aggregators do not satisfy this contract.
Target evaluation additionally requires an exact 8-setting-by-7-parent
prediction/protection selection freeze whose source campaign matches and whose
`frozen_before_target` flag is true. The repository intentionally has no such
file until all target-free development stages finish.

All four rosters must be nonempty, explicitly enumerated lists with no `TBD`,
duplicates, ranges, or pairwise overlap. The report preserves every ID and a
SHA-256 of each ordered roster, and the adapter validates that every ID belongs
to its real request domain. `D_cal`, `D_pred`, and `D_prot` are disjoint
target-free request sets; the target roster is separate and cannot enter
selection. The current TOFU split freezes all 20 forget10 author requests as
`2/4/4/10` across these four sets. The larger target roster keeps the
confirmatory request-level endpoint from resting on only two requests while
retaining independent development and selector-selection pools.

For every model, preflight reports the exact source, dtype, provisioned flag,
and a seven-entry availability map for GradDiff, NPO, SimNPO, GRU, RMU,
RepNoise, and Circuit Breakers. The map is intersected with the objective
registry imported from code, so a YAML `true` cannot stand in for a missing
implementation. Qwen2.5-7B and Llama-3.1-8B are currently
declared `float32`: `paper_contract.confirmatory_dtype` makes this a checked
constraint, rather than merely one of several accepted dtypes. The frozen
loss-shake radius is applied to the actual
parameter tensor, and a multi-million-dimensional unit shake can disappear
below a bfloat16 coordinate ULP. Bfloat16 may become confirmatory only after a
separate fp32-shadow implementation and fidelity certificate exist. A missing
or `auto` dtype, an unprovisioned model, a local model path absent on the host,
or even one unavailable parent blocks all stages for that setting.

## Honest current blockers

`configs/paper/campaign.yaml` intentionally remains non-ready until the
following work is real:

| Dataset/model | Current status |
|---|---|
| TOFU + Qwen2.5-7B | schema-ready; legacy runner does not consume the paper roster or emit candidate-level raw records |
| TOFU + Qwen2.5-1.5B | schema-ready; same executor blocker |
| TOFU + Llama-3.1-8B | fp32 declared; model not yet provisioned |
| WMDP-bio/MMLU | adapter and all four rosters unresolved |
| MUSE-News | adapter and all four rosters unresolved |
| MUSE-Books | adapter and all four rosters unresolved |
| RWKU | adapter and all four rosters unresolved |
| PISTOL | adapter and all four rosters unresolved |

These are blockers, not loaders. No synthetic/fake implementation is provided
for the unsupported benchmarks. Consequently the repository contract currently
reports zero executor-ready stages; this is intentional and more accurate than
calling a setting runnable because its dataset factory exists.

## Adding a real adapter

1. Define the dataset's `Example` construction and deletion-request semantics.
   Candidate `group` values must encode the unit used for disjoint folds.
2. Implement a factory that returns one frozen `Request`; validate that its
   forget set, candidate universe, and native audit IDs are consistent.
3. Add dataset-specific unit tests, including stable IDs and manifest hashes.
4. Register a `DatasetAdapter` with only the stages the implementation truly
   supports. Never reuse the TOFU factory.
5. Replace each dataset's four distinct `TBD_*` lists in `campaign.yaml` with
   exact IDs and set a matching `roster_unit`.
6. Run preflight again. Only an all-green report may feed GPU launch scripts.

A paper-stage executor must additionally expose a literal
`PAPER_STAGE_CONTRACT` mapping with schema version 1, its supported stage IDs,
the three input-contract flags, and the appropriate candidate-level raw-output
flag. Adding the marker without implementing and testing those behaviors is a
contract violation.

The controlled substrate is also registered because it already returns the
same `Request` contract. It lacks `target_evaluation`, so it cannot silently
stand in for a paper benchmark either.
