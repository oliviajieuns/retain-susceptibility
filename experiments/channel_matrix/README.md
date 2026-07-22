# 7B/8B channel-matrix campaign

The complete roster rationale, exact loss definitions, leakage controls, and
paper-writing contract are in [DESIGN.md](DESIGN.md).

This campaign turns the single-request 1.5B diagnostic into an uncertainty-
bearing mechanism table.  It does **not** tune for a large correlation.  Parent
hyperparameters are calibrated on development-only authors using forget reach
and ordinary utility, frozen, and only then evaluated against sealed audit
rankings.

Development candidates and all three request-specific audit candidate pools
are mutually disjoint; each audit run still exposes exactly 300 candidates
only after all core and stress trajectories finish.

## Scientific roster

- Output/loss-gradient: GradDiff, NPO+GD, SimNPO+GD, GRU.
- Representation: RMU, block-controlled RepNoise-style noising, and
  block-controlled Representation Rerouting/Circuit Breakers.
- GA and IdkDPO remain appendix controls: the current GA run is a collapse
  regime and IdkDPO did not reach the criterion.

NPO uses summed sequence NLL (the published sequence log-probability ratio),
whereas SimNPO uses length-normalized NLL.  Every parent and every gradient
probe is restricted to the same declared last-eight-MLP-down block.  This is
both the one-H100 fp32 memory strategy and the causal support-matching control.

The two starred representation parents are controlled adaptations, not exact
reproductions of the released safety systems.  Original RepNoise uses
multi-kernel MMD over post-MLP activations across all layers plus auxiliary
layer-wise ascent.  Original Circuit Breakers uses LoRA, multiple target layers,
and a coefficient schedule.  Do not remove this disclosure from a table that
uses the current implementations.

## Cluster commands

```bash
source /group-volume/jieuns.shin/venvs/exp/bin/activate
cd /group-volume/jieuns.shin/retain-susceptibility
git pull
python -m pip install -e '.[dev,campaign]'
python -m pytest -q
nvidia-smi
```

Before the offline audit, pre-cache both TOFU and the configured
`sentence-transformers/all-MiniLM-L6-v2` encoder. The audit process sets the
Hugging Face libraries to offline mode and fails rather than resolving a new
remote revision.

Inspect the complete calibration launch without using a GPU:

```bash
python experiments/channel_matrix/run_campaign.py \
  --config configs/channel_matrix/7b_tofu.yaml \
  --phase calibration --dry-run
```

Run the numerical fidelity gate at the already frozen operating point. This
uses only the disjoint development candidate pool and produces the certificate
that the audit launcher requires:

```bash
python experiments/channel_matrix/run_campaign.py \
  --config configs/channel_matrix/7b_tofu.yaml \
  --phase fidelity --resume
```

Run development calibration. `--resume` skips complete cells; partial artifacts
are preserved and cause a loud stop rather than an overwrite:

```bash
python experiments/channel_matrix/run_campaign.py \
  --config configs/channel_matrix/7b_tofu.yaml \
  --phase calibration --resume \
  > channel7b_calibration_$(hostname).out 2>&1
```

Calibration reuses one validated fp32 SFT snapshot per
`(model, deletion request, seed)`. Objective grids never share optimizer state;
only the identical pre-unlearning starting weights are cached.

Apply the predeclared reach/utility rule.  This script never reads predictor
seals or correlations:

```bash
python experiments/channel_matrix/select_freeze.py \
  --config configs/channel_matrix/7b_tofu.yaml \
  --root runs/channel_matrix_7b/calibration \
  --out runs/channel_matrix_7b/objective_freeze.recommended.yaml
```

Review unresolved arms.  If all are resolved, copy the chosen values into
`objective_freeze.yaml`, assign a dated `freeze_id` and `frozen_at_utc`, clear
`unresolved`, set `status: frozen` and `frozen_before_audit: true`, and commit
that file **before** starting audit.
The launcher rejects a draft freeze.

```bash
python experiments/channel_matrix/run_campaign.py \
  --config configs/channel_matrix/7b_tofu.yaml \
  --phase audit --resume \
  > channel7b_audit_$(hostname).out 2>&1
```

Aggregate with model/request/seed/candidate hierarchical bootstrap, then render
the proposed main table:

```bash
python experiments/channel_matrix/aggregate.py \
  --root runs/channel_matrix_7b/audit \
  --out runs/channel_matrix_7b/aggregate --n-boot 2000

python experiments/channel_matrix/make_main_table.py \
  --report runs/channel_matrix_7b/aggregate/pooled_channel_report.csv \
  --summary runs/channel_matrix_7b/aggregate/pooled_channel_report.json \
  --out docs/tables/table1_channel_matrix_7b.tex \
  --stress-out docs/tables/table1_stress_7b.tex
```

## Second architecture

The current cluster inventory documents Qwen2.5-7B-Instruct but not a second
7B/8B family.  After provisioning `Llama-3.1-8B-Instruct` at the path in the
YAML, set its `enabled` field to `true`, calibrate it independently, and add its
model-specific frozen settings.  The same code also supports a local
Mistral-7B path by adding a model entry; the registered MLP block selector is
compatible with Qwen/Llama/Mistral decoder naming and fails loudly otherwise.

## Table and dataset policy

The main matrix should pool within-request correlations over models, requests,
and seeds, with a hierarchical CI in every cell.  Model-specific matrices and
all cautionary/control rows belong in the appendix.  The headline remains the
roster-level difference-in-differences, plus every output/representation pair;
the largest individual correlation is not the endpoint.

Do not pool TOFU, WMDP, and MUSE candidates into the same correlation cell:
their candidate universes and damage base rates are not commensurate.  Keep
TOFU as the controlled multi-request main matrix, then add a compact
dataset-by-dataset interaction panel for WMDP-bio (representation-native),
MUSE-News (real-corpus/output-native), and RWKU.  This directly tests whether
the interaction survives base-rate changes without turning the table into a
dataset leaderboard.
