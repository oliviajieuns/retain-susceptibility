# AGENTS.md — autonomous-agent operating contract for this repository

Audience: autonomous coding/ops agents (OpenCode, Hermes, Claude, Codex, ...)
running ON the H100 cluster to fill the paper's Table 1/2. Read this whole
file before your first command. Korean human-facing detail lives in
`CLAUDE.md`, `docs/CLUSTER_FLEET_RUNBOOK.md`, and `docs/plan_2026-07-23_fleet.md`;
this file is the binding subset for agents.

## 0. Why you cannot "just remove" the seals (read this once, believe it)

This paper's claim is **prospective** prediction and **predeclared** decision
value. The seal/freeze machinery is not infrastructure friction — it IS the
scientific claim:

- RQ1 claims a profile sealed BEFORE unlearning predicts damage AFTER. If any
  parameter, weight, or checkpoint is chosen after seeing audit damage, the
  word "prospective" becomes false and Table 1 becomes unpublishable.
- Table 2's denominators ("every planned row stays in the denominator") are
  the anti-cherry-picking guarantee. Re-running variants until one looks good
  and reporting that one is exactly what the design exists to prevent.
- The runners enforce this mechanically: seals are append-only, audit runners
  refuse dirty worktrees and reused run-tags, `aggregate_raw` refuses
  unplanned units. **A guard firing is a signal, never a bug. Do not patch,
  weaken, bypass, or "fix" a guard to make a run proceed.**

Your hundreds of experiments belong in the pre-freeze lanes and across
settings (§3), where wide autonomous sweeps are exactly what the design wants.

## 1. Session bootstrap (every session, verbatim)

```bash
source /group-volume/jieuns.shin/venvs/exp/bin/activate
cd /group-volume/jieuns.shin/retain-susceptibility
export HF_HOME=/group-volume/data/hf_home
git pull --ff-only          # NEVER during a sealed phase with workers running
python -m pytest -q         # expect all green before touching queues
python experiments/cluster/next_actions.py   # what you may do right now
```

Environment facts (do not rediscover, do not "fix"):

- GPUs: H100 80GB, 8 per node. fp32 7B uses a whole GPU — one unit per GPU,
  never co-schedule. Check `nvidia-smi` before any manual launch.
- Models are LOCAL under `/group-volume/models/` — always pass local paths.
  HF Hub is blocked/unstable: keep `HF_HUB_OFFLINE=1` semantics (runners set
  their own offline flags; never add online calls).
- GitHub: pull only. **Push from the cluster is impossible (egress-blocked)
  and committing on the cluster is forbidden** — deliver artifacts back to
  the human/session via the handoff in §6.
- The official venv is `/group-volume/jieuns.shin/venvs/exp`. Do not create
  or recommend another environment, and do not `pip install` into it.

## 2. Hard rules (violating any of these invalidates the campaign)

NEVER, under any framing or "temporary" justification:

1. Edit, delete, or regenerate anything under `prereg/`, any
   `configs/channel_matrix/*freeze*.yaml` whose `status: frozen`, or
   `configs/paper/{campaign,evidence}.yaml` frozen blocks. Amendments are a
   HUMAN step (dated block, committed by the session before target runs).
2. Modify `src/rsus/sealing.py`, seal ledgers, `DONE` markers, run manifests,
   or anything inside an existing `runs/**` output directory (move partial
   dirs to `runs/forensics/` during triage — never delete, never reuse).
3. Re-run, re-tune, or extend a TARGET/audit cell after its outcome exists —
   including "one more seed", "slightly better lr", or re-enqueueing a done
   unit with new parameters. Frozen means frozen.
4. Select or tune ANY value (alpha, lr, steps, thresholds, pool size) using
   sealed-audit outcomes. Development folds only (D_cal/D_pred/D_prot,
   authors a198/a199 for TOFU).
5. Flip a freeze file to `status: frozen` yourself, or enqueue a phase that
   `next_actions.py` did not list under `allowed_now`.
6. `git push`, `git commit` on the cluster, force-`git pull` while sealed
   workers run, or hand-edit queue JSON (attempts, max_attempts, cmd).
7. Weaken a failing guard/test to green (worktree-dirty refusal, run-tag
   refusal, partial-dir refusal, bootstrap rejection cap, fidelity floors).

ALWAYS:

- New run = NEW run-tag / unit id (append-only everywhere). Deliberate
  re-runs use `make_units.py --unit-suffix rN`, nothing else.
- Logs carry the hostname automatically; never redirect two runs to one file.
- Custom `gate.py`-family units: `"max_attempts": 1` (run-tags cannot retry).
- Before enqueueing audit-family phases the worktree must be clean.

## 3. Where your autonomy IS wanted (the sanctioned sweep lanes)

Everything below is target-free by construction. Scale these as wide as the
fleet allows — this is where "hundreds of runs" legitimately live:

| Lane | What you may vary freely | Runner / phase |
|---|---|---|
| Objective calibration grids | lr, steps, per-objective hyperparams (e.g. npo beta), grid extensions via `--unit-suffix r2` | `--phase calibration` per model×dataset config |
| Loss-shake fidelity | R (directions), eta, repeats, block depth checks | `--phase fidelity`, `experiments/diag/fd_fidelity.py` |
| Alpha development | alpha grid on DEV requests/seeds only | `--phase alpha-development` |
| Stage-2 capacity response | eta2, steps, guard params on DEV folds | dev-fold runs of `alpha_protection.py` / `gate.py` |
| New settings (breadth) | whole W1→W4 chains for 14B, Llama-8B, WMDP, RWKU — each its own queue | `enqueue_table12.sh {audit-14b,wmdp,wmdp-14b,llama,rwku-audit}` |
| Mechanism replication | 1.5B seed replicates via manifest-exact tool | `make_replicate_units.py --source-run ... --seeds ...` |
| CPU analysis | export→aggregate→build_evidence, readiness checks, plots | §6 (login node, no GPU) |

The selection criterion inside calibration is predeclared (forget recall
<= 0.10 reached AND utility floors held) — an agent may propose the operating
point per objective, but the freeze commit is human.

Idle-GPU rule: fill idle GPUs ONLY from these lanes (other configs' units,
replicates, fidelity). Never fill them by widening a sealed phase.

## 4. The per-setting state machine (what "next" means)

```
W1  fidelity + calibration      (agent, queue)      <- wide sweeps OK
W1.5 select-freeze -> commit objective_freeze       <- HUMAN GATE
W2  audit                       (agent, queue)      <- frozen configs only
W3  alpha-development           (agent, queue)      <- dev folds, parallel w/ W2
W3.5 select-alpha-freeze -> commit alpha_freeze     <- HUMAN GATE
W4  alpha-audit                 (agent, queue)      <- frozen alpha only
W5  CPU evidence chain          (agent, login node) <- §6
```

Query the machine, don't infer it:

```bash
python experiments/cluster/next_actions.py --json
```

It reports, per setting: model provisioned, objective/alpha freeze state,
queue counts, `allowed_now` phases, and which human gate blocks the rest.
Enqueue only what appears in `allowed_now`.

## 5. Queue operations (the only way you run GPU work)

```bash
# status overview of all Table 1/2 waves
bash experiments/cluster/enqueue_table12.sh status

# enqueue a wave (idempotent; duplicates are refused and reported)
bash experiments/cluster/enqueue_table12.sh audit-7b     # -> wave2 (Table 1)
bash experiments/cluster/enqueue_table12.sh wmdp         # -> wave_wmdp
# ... see the script header for the full subcommand list

# start workers on THIS node (8 workers, one per GPU; returns immediately)
bash experiments/cluster/launch_node.sh runs/cluster_queue/wave2

# monitor / recover
python experiments/cluster/workqueue.py status --brief --queue runs/cluster_queue/wave2
python experiments/cluster/workqueue.py requeue-stale --queue <Q>   # only after
#   verifying on the owning host that the worker process is truly dead
python experiments/cluster/workqueue.py retry-failed  --queue <Q>   # only after
#   root-causing the failure and moving partial run dirs to runs/forensics/
python experiments/cluster/fleet_status.py                          # fleet view
```

Failure triage, in order: read the failed unit's log path from `status`;
classify (OOM / node death / partial-dir refusal / guard refusal); partial-dir
refusal means move the partial run dir to `runs/forensics/<name>.<epoch>` then
`retry-failed`; guard refusal means STOP and report to the human — it is
usually rule §2 protecting you.

## 6. CPU evidence chain and result handoff (no GPU needed)

After audit/alpha waves drain (run on a login node or any node, CPU):

```bash
# 1) immutable plan (requires the HUMAN-committed selection freeze)
python experiments/paper/init_raw_plan.py \
  --selection-freeze configs/paper/selection_freeze.yaml \
  --setting tofu_qwen25_7b --out results/paper/raw_plan.json
# 2) sealed run outputs -> candidate-level shards (+ fidelity summary)
python experiments/paper/export_channel_matrix_raw.py \
  --campaign-config configs/channel_matrix/7b_tofu.yaml \
  --setting-id tofu_qwen25_7b \
  --prediction-alpha-freeze configs/channel_matrix/prediction_alpha_freeze_7b.yaml \
  --control-predictor knn_embed \
  --fidelity-certificate runs/channel_matrix_7b/fidelity/qwen25_7b.json \
  --out-dir results/paper/raw/tofu_qwen25_7b
# 3) normalized ledger
python experiments/paper/aggregate_raw.py --plan results/paper/raw_plan.json \
  --prediction-raw results/paper/raw/tofu_qwen25_7b/prediction.jsonl \
  --protection-raw results/paper/raw/tofu_qwen25_7b/protection.jsonl \
  --out results/paper/evidence_ledger.json
# 4) readiness + tables (point --paper-root at a scratch copy on the cluster)
python experiments/paper/build_evidence.py
```

Handoff (cluster cannot push): paste back to the human/session, verbatim —
`results/paper/evidence_ledger.json`, `evidence_readiness.json`, the two
generated `.tex` tables, fidelity summaries, and the per-wave
`workqueue.py status` output. The session recreates and commits them.
Placeholders (`\tblph`) remaining in tables while `--require-ready` is unmet
is the designed honest state — never hand-fill a number.

## 7. Reporting discipline

- Report failures as failures (exit codes, log paths). A non-reaching parent,
  an infeasible arm, or a NOT LICENSED multi-setting verdict is a valid
  scientific result — record it, do not retry it into silence.
- Keep a per-session run journal (what was enqueued, drained, triaged) in
  chat output; do not write ad-hoc notes into the repo.
- When in doubt between "act" and "ask the human": every freeze, every config
  authored from scratch, every amendment, every deletion — ask.
