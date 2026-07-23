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
git fetch origin codex/channel-matrix-7b
git switch codex/channel-matrix-7b
git pull --ff-only origin codex/channel-matrix-7b
python -m pytest -q
nvidia-smi
```

The official shared environment already contains the campaign dependencies;
do not create a node-local environment.  The convenience entry point below
performs model-path, dependency, CUDA, and git-state checks:

```bash
GPU=0 MODEL_ID=qwen25_7b \
  bash experiments/channel_matrix/h100_campaign.sh preflight
```

Keep every log below the git-ignored `runs/` directory.  A log written in the
repository root makes the worktree dirty and therefore blocks sealed audit.

Before the offline audit, pre-cache both TOFU and the configured
`sentence-transformers/all-MiniLM-L6-v2` encoder. The audit process sets the
Hugging Face libraries to offline mode and fails rather than resolving a new
remote revision.

```bash
bash experiments/channel_matrix/h100_campaign.sh prefetch
```

Inspect the complete calibration launch without using a GPU:

```bash
GPU=0 MODEL_ID=qwen25_7b \
  bash experiments/channel_matrix/h100_campaign.sh dry-calibration
```

Run the numerical fidelity gate at the already frozen operating point. This
uses only the disjoint development candidate pool and produces the certificate
that the audit launcher requires:

```bash
mkdir -p runs/logs
GPU=0 MODEL_ID=qwen25_7b nohup \
  bash experiments/channel_matrix/h100_campaign.sh fidelity \
  > "runs/logs/channel7b_fidelity_$(hostname).out" 2>&1 &
```

Run development calibration. `--resume` skips complete cells; partial artifacts
are preserved and cause a loud stop rather than an overwrite:

```bash
GPU=0 MODEL_ID=qwen25_7b nohup \
  bash experiments/channel_matrix/h100_campaign.sh calibration \
  > "runs/logs/channel7b_calibration_$(hostname).out" 2>&1 &
```

On a two-H100 node, shard by deletion request. `AUTHORS` is an execution-only
filter that must be a subset of the frozen phase roster; it does not change
candidate pools, seeds, objectives, or any seal. The two shards below have
different output directories and different request-level SFT caches:

```bash
mkdir -p runs/logs
GPU=0 AUTHORS=198 MODEL_ID=qwen25_7b nohup \
  bash experiments/channel_matrix/h100_campaign.sh calibration \
  > "runs/logs/channel7b_cal_a198_gpu0_$(hostname).out" 2>&1 &

GPU=1 AUTHORS=199 MODEL_ID=qwen25_7b nohup \
  bash experiments/channel_matrix/h100_campaign.sh calibration \
  > "runs/logs/channel7b_cal_a199_gpu1_$(hostname).out" 2>&1 &
```

Never run an unsharded copy of the same phase concurrently with these jobs.
The launcher rejects authors outside the phase roster. Monitor without opening
or modifying any result artifact:

```bash
jobs -l
tail -f runs/logs/channel7b_cal_a198_gpu0_"$(hostname)".out
watch -n 5 nvidia-smi
```

Calibration reuses one validated fp32 SFT snapshot per
`(model, deletion request, seed)`. Objective grids never share optimizer state;
only the identical pre-unlearning starting weights are cached.

Apply the predeclared reach/utility rule.  This script never reads predictor
seals or correlations:

```bash
bash experiments/channel_matrix/h100_campaign.sh select-freeze
```

Review unresolved arms.  If all are resolved, copy the chosen values into
`objective_freeze.yaml`, assign a dated `freeze_id` and `frozen_at_utc`, clear
`unresolved`, set `status: frozen` and `frozen_before_audit: true`, and commit
that file **before** starting audit.
The launcher rejects a draft freeze.

The cluster cannot push.  Return the generated recommendation for review and
remote commit, then pull that frozen commit back onto the cluster.  Do not edit
the freeze and launch audit from an uncommitted worktree.

```bash
GPU=0 MODEL_ID=qwen25_7b nohup \
  bash experiments/channel_matrix/h100_campaign.sh audit \
  > "runs/logs/channel7b_audit_$(hostname).out" 2>&1 &
```

The audit can use the same request sharding after both freezes are committed.
For the three-author roster, a deterministic two-GPU allocation is
`GPU=0 AUTHORS=181,191` and `GPU=1 AUTHORS=186`. This only schedules sealed
cells; it never changes the roster or permits alpha selection from audit data.

Aggregate with model/request/seed/candidate hierarchical bootstrap, then render
the proposed main table:

```bash
bash experiments/channel_matrix/h100_campaign.sh aggregate
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

## Adaptive channel-mixture protection (prospective 7B extension)

The protection extension uses

```text
s_alpha(x) = (1-alpha) rank01_discovery(fd_norm(x))
             + alpha rank01_discovery(knn_feature(x)).
```

Thus `alpha=0` is the output/gradient endpoint and `alpha=1` is the hidden-
representation endpoint. Both rank transforms, the Top-30 protect pool, and
the matched remote pool are computed only inside the discovery fold. Audit
scores and audit damage cannot affect allocation.

This extension starts only after `objective_freeze.yaml` is frozen. It runs
the five-point alpha grid for all seven primary parents (GradDiff, NPO,
SimNPO, GRU, RMU, RepNoise, and Circuit Breakers) on development
authors 198/199. For each model and parent, the selector minimizes worst-run
development CVaR subject to every run reaching forget recall at most 0.10 and
ordinary utility retention at least 0.90. Ties prefer mean CVaR, then the
distance to the readout-independent midpoint 0.5, then the smaller alpha. No
feasible alpha means unresolved; there is no audit
fallback.

The utility probe is fixed to the first four QA of authors 150--179 (`n=120`),
which is disjoint from every deletion request and damage-candidate pool. A
parent trajectory is executed once per cell and its reached trainable block is
reused for all repair selectors, so every selector starts from bit-identical
parent weights without rerunning parent optimization.

After objective calibration and its committed freeze:

```bash
GPU=0 MODEL_ID=qwen25_7b \
  bash experiments/channel_matrix/h100_campaign.sh dry-alpha-development

GPU=0 MODEL_ID=qwen25_7b nohup \
  bash experiments/channel_matrix/h100_campaign.sh alpha-development \
  > "runs/logs/channel7b_alpha_dev_$(hostname).out" 2>&1 &
```

The corresponding two-GPU development launch is:

```bash
GPU=0 AUTHORS=198 MODEL_ID=qwen25_7b nohup \
  bash experiments/channel_matrix/h100_campaign.sh alpha-development \
  > "runs/logs/channel7b_alpha_dev_a198_gpu0_$(hostname).out" 2>&1 &

GPU=1 AUTHORS=199 MODEL_ID=qwen25_7b nohup \
  bash experiments/channel_matrix/h100_campaign.sh alpha-development \
  > "runs/logs/channel7b_alpha_dev_a199_gpu1_$(hostname).out" 2>&1 &
```

Create a development-only recommendation:

```bash
bash experiments/channel_matrix/h100_campaign.sh select-alpha-freeze
```

Review `runs/channel_matrix_7b/alpha_protection_freeze.recommended.yaml`.
Copy the selected values and diagnostics into
`configs/channel_matrix/alpha_protection_freeze.yaml`, assign a dated ID and
UTC timestamp, set `status: frozen` and
`frozen_before_alpha_audit: true`, clear `unresolved`, commit and push, then
pull that exact clean commit on the H100 host. The selector rejects audit
artifacts even if they are accidentally placed below its input root.

Only after that commit:

```bash
GPU=0 MODEL_ID=qwen25_7b \
  bash experiments/channel_matrix/h100_campaign.sh dry-alpha-audit

GPU=0 MODEL_ID=qwen25_7b nohup \
  bash experiments/channel_matrix/h100_campaign.sh alpha-audit \
  > "runs/logs/channel7b_alpha_audit_$(hostname).out" 2>&1 &

# Optional debugging only; this writes a superseded conditional CVaR summary.
bash experiments/channel_matrix/h100_campaign.sh legacy-alpha-diagnostic
```

The audit runs the already frozen alpha as the deployable method. The other
grid points are retained as a descriptive response curve, never an audit
oracle. Confirmatory comparators are no protection, five independently frozen
random allocations, and both mixture endpoints. The candidate-backward exact
energy reference replaces only the loss-shake component at the same frozen
alpha and remains outside the IUT. Every repair arm starts from the identical
first-reaching parent checkpoint and reports only its last saved checkpoint
satisfying direct, paraphrase, greedy autoregressive generation, and utility
constraints. Aggregate contrasts require exact candidate support and retain
explicit reach/feasibility counts.

`aggregate_alpha_protection.py` and the launcher command above are deliberately
legacy diagnostics: they do not produce the paper's mean+CVaR eight-way IUT.
Claim-bearing output must be converted by
`experiments/paper/export_alpha_protection.py` into the candidate-level
five-arm schema fixed by `results/paper/raw_plan.json`, then consumed by
`experiments/paper/aggregate_raw.py` and
`experiments/paper/build_evidence.py`; see
`docs/PAPER_EVIDENCE_PIPELINE.md`.

The roster in `configs/channel_matrix/7b_tofu.yaml` is also an older 7B
diagnostic roster. It must not be relabeled as a paper target run: the paper
contract freezes distinct `D_cal`, `D_pred`, `D_prot`, and target rosters in
`configs/paper/campaign.yaml`. In particular, this runner still chooses its
protection weight on authors 198/199 and audits 181/186/191, whereas the paper
contract uses `D_prot` 184--187 and target 188--197. A paper launch therefore
requires a runner configuration generated from the immutable paper plan, not
reuse of this YAML.

Implementation note: this extension also fixes the remote-band quantile to use
discovery scores only. Earlier `xprot` artifacts were produced before this
fix and must not be silently pooled with the new campaign; if they remain in a
paper table, label their implementation version or rerun them.
