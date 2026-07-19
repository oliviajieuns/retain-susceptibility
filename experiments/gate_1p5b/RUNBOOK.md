# 1.5B TOFU Gate — Cluster Runbook

Purpose: cheapest real-data forecast of Tables 1–2 before committing the
full budget. One forget10 author, Qwen2.5-1.5B-Instruct, ~1–3 H100-hours.

## Prerequisites (once)

1. Clone (private repo — use your GitHub SSH key on the cluster):
   `git clone git@github.com:oliviajieuns/retain-susceptibility.git`
2. Environment (Python ≥3.10):
   ```
   python -m venv .venv && source .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install transformers datasets pyyaml pytest sentence-transformers
   ```
3. `export HF_TOKEN=...` (higher rate limits; TOFU and Qwen are public).
4. Sanity: `python -m pytest` → all green (CPU, ~1 min);
   `python experiments/gate_1p5b/gate.py --smoke` → completes, writes
   `runs/gate_smoke/`.

## Gate run

```
nohup python experiments/gate_1p5b/gate.py \
  --device cuda --dtype float32 \
  --universe-authors 30 --pool-size 32 --seed 2025 \
  > gate.out 2>&1 &
```

Detached (`nohup`/`tmux`) — do not run in a session that dies on logout.
`--dtype float32` is deliberate for the probe (bf16's loss noise floor
swamps the finite difference at eta=3e-4; the bf16 probe arm is a later
sensitivity, not the gate).

Knobs to watch/adjust in `gate.out`:
- **SFT**: loss should come down toward ~0.8 (`--sft-steps`, `--sft-lr`).
  Do NOT overtrain toward 0 — saturated candidates have vanishing
  gradients and kill the probe signal (CPU-pilot finding).
- **Generators**: terminal `recall` should land near/below ~0.2 and
  `mean_audit_dmg` should be clearly positive but small (≲0.5 nats).
  If damage is huge, halve `--gen-lr`: destructive budgets wash out the
  local signal and favor similarity (CPU-pilot finding).
- **Stage 1** (`ours`): must pass the gate within `--s1-max-steps`;
  raise if `stage-1 gate not reached`.
- **Floor sanity**: the logged `floor m=` must be moderate (roughly 3-6
  nats for TOFU answers under the pre-SFT Qwen reference). A value near
  ~12 nats means the reference model is effectively chance-level (this is
  exactly the vocabulary-scale pathology the paper describes; the smoke
  run reproduces it because its reference is random-init) — check that
  `--model` loaded correctly before burning GPU time on Stage 1.

## Optional companions (same session, if time allows)

- Table 2 runs the full roster by default
  (`ga,graddiff,npo,simnpo,idkdpo,rmu,gru,s2s,npo_transplant,ours`);
  trim with `--t2-roster npo,npo_transplant,s2s,ours` if GPU time is short
  — those four are the confirmatory contrasts. Paraphrase recall is
  recorded automatically at every snapshot (`para_recall` in table2.json).
- Numerical-stability sweep (appendix table, ~15 min):
  `python experiments/stability/sweep.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda`
- Profiling-cost bench (Table 4 needs 7B/14B, but a 1.5B row validates):
  `python experiments/cost/bench.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --repeats 5`

## Reading the result

- `runs/gate_*/table1.json` — success signal: fd rho clearly above
  knn_* rows on NPO/GradDiff columns (RMU expected weaker), AUROC ≳0.7.
- `runs/gate_*/table2.json` — success signal: `ours` and/or
  `npo_transplant` below plain `npo` on audit mean/CVaR dNLL at
  reach=true. This is the least-supported link; if it fails here, check
  stage-1 damage vs repair gain before concluding.
- Copy back: `runs/gate_*/{table1.json,table2.json,gate.log}` +
  `seal_ledger.jsonl` (the ordering proof).

## Open choices encoded as TODO

- Native metadata audit rule for TOFU is not frozen yet; the gate audits
  the untouched random fold + full distribution instead (adapter carries
  an empty native set until the rule is preregistered).
- delta budgets (D4) use placeholder defaults (1e-2 / 1e-1).
