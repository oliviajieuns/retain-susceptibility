# retain-susceptibility — Code Design

Spec of record: the paper on Overleaf ("When Does LLM Unlearning Fail?
Probing Update-Conditioned Retain Susceptibility"), currently the
probe-to-protection revision of 2026-07-19 (single causal chain:
gradient alignment → susceptibility profile → realized-damage prediction →
profile-guided protection). Where this document and the paper disagree, the
paper wins; where the paper is silent, `prereg/constants.yaml` wins.

Predecessor: `unlearning-entanglement-repli` (branch `llm-port`). This repo
starts clean because the Stage-2 method changed shape (sorted-W2 blended
objective → one-sided identity-paired anchor with constrained enforcement)
and the experiment program changed ground truth (RQ1 = realized damage at
saved checkpoints of pre-fixed third-party generators, sealed audit folds).
Portable pieces are listed in §9.

---

## 1. What the code must produce

The paper's main tables, in order of priority:

| Table | Content | Core code path |
|-------|---------|----------------|
| T1 `tab:discovery` | Predictors vs realized damage d(x)=ℓ(x;θ_T)−ℓ(x;θ_0) at the terminal-budget checkpoint of NPO+retain / RMU / GradDiff, untouched audit fold. Columns: per-optimizer Spearman ρ, AUROC (top-K damage membership), Overlap@K, Tail ρ | seal scores → `generators/*` (damage recording) → `analysis/prediction` |
| T2 `tab:main` | Protection at first criterion-reaching checkpoint: Reach, Step, paraphrase recall, native mean ΔNLL, native CVaR.95, utility. Rows: GA/GradDiff/NPO+retain/SimNPO/IdkDPO/RMU/GRU/Cheng-S2S/NPO+transplant/ours | `generators/*` + `stage1+stage2` → `evalx/protection` |
| T3 `tab:ablations` | Matched-parent contrasts: FD vs JVP; partition susceptibility vs repr-sim vs random; NPO transplant vs NPO+retain; full vs no-projection vs no-guard | config matrix over existing flags → `analysis/ablation` |
| T4 `tab:cost` | Profiling cost: FD / exact JVP / chunked vmap grad-dot / streaming backward × {wall, peak mem, throughput} at 7B & 14B | `CostRecord` telemetry + `experiments/cost` bench |

Supporting (appendix): tab:numerical-stability (η/precision/block grids —
scorer registry + spec sweeps), tab:guard-recovery (projection×guard
factorial + symmetric/sorted/seq-only arms — `stage2` flags),
tab:resource-parity (per-method disclosure — runner telemetry),
tab:transfer / tab:full-transfer-results (same pipelines on other
data/model adapters).

Fixed definitions shared by all of the above (appendix 'Result Variables'):
damage d_t(x)=ℓ(x;θ_t)−ℓ(x;θ_0) (positive harmful); CVaR.95 = mean of the
largest max(1,⌈.05|A|⌉) values; score-CVaR uses the same upper fraction of
signed scores; seeds average within request before target/request-level
inference; hierarchical bootstrap over targets then requests, plus
leave-one-target-out.

## 2. Design principles

1. **Paper-code contract.** Every equation in §3–§4 of the paper maps to one
   named function with a docstring citing its label (`eq:fdscore`,
   `eq:pools`, `eq:stage1`, `eq:guard`, `eq:wpgd`, `eq:massbound`). No
   second implementation of any of them anywhere in the tree.
2. **CPU-testable everything.** Every module runs on the laptop (no CUDA)
   with a tiny random-init causal LM. GPU is a scale knob, not a
   correctness knob. The unit battery is the merge gate.
3. **Preregistration as code.** Frozen constants live in
   `prereg/constants.yaml` (content-hashed); sealed audit-fold scores live
   in `seals/` with an append-only `seal_ledger.jsonl`. Analysis code
   refuses to read a seal without a ledger entry that postdates the
   trajectory-completion marker.
4. **Cost telemetry is first-class.** Every scorer and stage returns a
   `CostRecord` (wall, peak mem, tokens, fwd/bwd counts). RQ6 tables are
   generated, never hand-filled.
5. **Determinism + manifests.** Same config hash + seed ⇒ identical pool
   manifests (SHA256) and scores. No wall-clock or unseeded randomness in
   library code.
6. **Anonymizable.** No author-identifying strings in code, configs, or
   test fixtures. Repo stays private until camera-ready.

## 3. Repo layout

```
retain-susceptibility/
  DESIGN.md
  pyproject.toml            # package: rsus; torch, transformers, datasets, pyyaml, pytest
  prereg/
    constants.yaml          # eta grid, tau quantiles, floor rule, deltas, eps, k, ckpt grid, seeds
    seal_ledger.jsonl       # append-only unseal log
  src/rsus/
    blocks.py               # BlockSpec: declared block B, select/flatten/perturb params
    losses.py               # prompt-masked per-seq mean NLL + per-token NLL w/ index map
    costs.py                # CostRecord + meters (wall, cuda max-mem, token/pass counters)
    config.py               # YAML load + content hash + run-dir provenance
    runlog.py               # jsonl event logging
    probe/
      base.py               # ProbeSpec, ScoreProfile, scorer registry
      finite_diff.py        # eq:fdscore — 1 bwd(Df) + 2 batched fwd sweeps at theta±eta*ghat
      jvp.py                # exact batched forward-mode comparator
      graddot.py            # chunked-vmap grad-dot + streaming per-candidate backward
      baselines.py          # kNN feature vote, sentence-embed kNN, lexical kNN,
                            # grad-norm, last-layer closed form, random dir/rank
    partition.py            # P = top-K eligible positive scores (frozen fallback rule),
                            # R0 = template-matched near-zero remote stream,
                            # discovery/audit folds, manifests
    refcache.py             # Stage-1 exit-gate cache: per-seq vector + per-(example,token-pos)
                            # vector + index map; no reference model copy afterwards
    stage1.py               # eq:stage1 — aug-Lagrangian ascent w/ clip_c, EMA dual ascent,
                            # calibrated floor m, exit gate
    guards.py               # eq:guard — one_sided(eps) primary; symmetric, sorted_profile,
                            # seq/tok levels; budget check
    stage2.py               # eq:wpgd + constrained enforcement — dual ascent on budgets,
                            # refresh-step acceptance (rollback + eta2 shrink), basis projection
    generators/             # third-party objectives (RQ1 ground truth + RQ2 baselines)
      __init__.py           # registry + parity accounting hooks
      ga.py graddiff.py npo.py simnpo.py idkdpo.py rmu.py gru.py
      retain_ft.py npo_adjw.py
      s2s.py                # faithful re-implementation of the split-aware two-stage baseline
    evalx/
      contract.py           # five-condition clean window over saved checkpoints
      audits.py             # paraphrase NLL, relearn (RTT), quantization, MIA (Min-K%++)
      metrics.py            # recall, retention ratios, CDF/CVaR summaries over C(q)
    data/
      base.py               # Request, CandidateUniverse, Fold abstractions
      tofu.py               # headline: 190-author SFT, 20 single-author forget10 requests
      muse.py rwku.py knowundo.py wmdp.py
      substrate.py          # controlled substrate w/ ground-truth adjacency by construction
    runner.py               # one (request, method, config) → trajectory + artifacts
  experiments/
    smoke/                  # CPU tiny-model profiles for every RQ path
    rq1_prediction/ rq2_headline/ rq3_guard/ rq4_panels/ rq6_cost/
  analysis/                 # pandas + hierarchical bootstrap; reads run artifacts only
  tests/                    # CPU unit battery (merge gate)
  third_party/              # pinned-SHA cross-check code only (never imported by rsus)
  scripts/                  # detached launchers, env setup, seal/unseal CLI
```

## 4. Core abstractions

```python
# data/base.py
@dataclass(frozen=True)
class Request:
    request_id: str
    forget: list[Example]              # |Df| = 20 QA pairs (headline)
    universe: CandidateUniverse        # frozen before any score/outcome
    native_audit_ids: frozenset[str]   # benchmark-native confirmatory pool
    folds: FoldMap                     # discovery vs untouched audit, author granularity
    meta: RequestMeta                  # size/recall/NLL/answer-length matching fields

# probe/base.py
@dataclass(frozen=True)
class ProbeSpec:
    block: BlockSpec                   # late-layer MLP down-projections (preregistered)
    eta: float                         # probe step (grid preregistered)
    loss: str = "seq_mean_answer_nll"
    seed: int = 0

@dataclass(frozen=True)
class ScoreProfile:
    request_id: str
    scores: dict[str, float]           # candidate_id -> s(x)
    spec: ProbeSpec
    cost: CostRecord

class Scorer(Protocol):
    name: str
    def score(self, model, request: Request, spec: ProbeSpec) -> ScoreProfile: ...
```

Registry names (fixed; used in configs, tables, and tests):
`fd` (primary), `jvp`, `vmap_graddot`, `streaming_backward`,
`knn_feature`, `knn_embed`, `knn_lexical`, `grad_norm`, `last_layer`,
`random_dir`, `random_rank`, `fd_constrained` (sensitivity arm).

```python
# guards.py
class GuardKind(Enum): ONE_SIDED, SYMMETRIC, SORTED_PROFILE

def guard_penalty(losses, refs, kind, eps=0.0) -> Tensor:
    """eq:guard. ONE_SIDED: mean relu(refs - eps - losses)**2 (primary).
    SYMMETRIC: mean (losses - refs)**2 (ablation; upper-bounds sorted W2^2).
    SORTED_PROFILE: mean (sort(losses) - sort(refs))**2 (ablation)."""

# refcache.py — pairing is identity-critical
@dataclass(frozen=True)
class RefCache:
    seq_refs: Tensor                   # [|Df|] aligned to request.forget order
    tok_refs: Tensor                   # [T_total] flattened answer tokens
    tok_index: list[tuple[str,int]]    # (example_id, answer_pos) per flat slot — the ONLY
                                       # legal alignment between Stage-1 cache and Stage-2 eval
    floor_m: float
    manifest_sha: str
```

## 5. Stage 2 (the new algorithm)

Constrained repair: `min L_adj  s.t.  D↓_seq ≤ δ²_seq, D↓_tok ≤ δ²_tok`.

```
state: theta, v=0, lam_seq=lam_tok=0, eta2, snapshot=last_accepted
loop t = 0,1,2,...:
  if t % k == 0:                                  # refresh
    u_seq, u_tok = full forget-set losses (whole Df; no minibatch at |Df|=20)
    D_seq, D_tok = guard_penalty(u, refs, ONE_SIDED, eps)
    lam_seq += rho_g * max(0, D_seq - delta_seq^2)   # dual ascent on measured violation
    lam_tok += rho_g * max(0, D_tok - delta_tok^2)   # (EMA-smoothed like Stage 1)
    if D_seq > delta_seq^2 or D_tok > delta_tok^2:
      restore(last_accepted); eta2 *= shrink; continue   # acceptance rule: reject refresh
    last_accepted = snapshot(theta, v, lam)              # bound premise holds by construction
    basis = orthonormal-ish {grad_B(-L_F), grad_B(-L_rem) [, grad_B(-L_u)]}
  v = mu_v * v + grad_B(L_adj + lam_seq*D_seq + lam_tok*D_tok)
  ghat = v - sum_i zeta_i b_i        # eq:wpgd: ridge Gram solve, per-tensor-trace ridge,
                                     # cond-guard fallback to single-basis
  theta -= eta2 * ghat;  v = ghat
```

Division of labor (mirrors the paper): projection = proactive, mean-level;
anchor = reactive, identity-paired (zero gradient inside the slack band);
enforcement = multiplier ascent + acceptance. Telemetry each refresh:
`D_seq, D_tok, lam_*, accepted?, ridge residual norm, basis cosines,
min answer-token NLL sentinel, exact-match recovery sentinel`.

Ablation flags (RQ3): `guard=off|one_sided|symmetric|sorted`,
`levels=seq|seq+tok`, `projection=on|off` — all combinations runnable.

## 6. Sealing & preregistration mechanics

- `prereg/constants.yaml` is written once per freeze; its SHA256 goes into
  every run dir. Changing it requires a new dated freeze block (old blocks
  are never deleted).
- Audit-fold scores: `partition.py` writes them to
  `seals/<request>/<scorer>.parquet` + SHA in `seal_ledger.jsonl` with
  `status=sealed`. RQ1 analysis calls `scripts/unseal.py`, which refuses
  unless `runs/<generator>/DONE` markers exist for all pre-fixed
  third-party trajectories; unsealing appends `status=opened` + timestamp.
  Honest-by-construction and auditable in the artifact.
- Discovery-fold scores are unrestricted (optimizer may see them).

## 7. Testing strategy (CPU battery, merge gate)

Tiny fixtures: random-init 2-layer causal LM (~1M params, Qwen-style
config), synthetic requests, substrate generator. Invariants:

1. `fd` vs analytic backward grad-dot: Spearman ρ = 1.0 and atol match on
   tiny model (eq:fdscore truncation check across the η grid).
2. `jvp` == autograd directional derivative exactly.
3. `vmap_graddot` == `streaming_backward` == `jvp` rankings.
4. Model restored bit-exact after ±η perturbation (fd is side-effect-free).
5. Guard algebra: one-sided is zero at/above slack band and differentiable
   below; symmetric ≥ sorted-profile on random vectors (rearrangement);
   Chebyshev bound of eq:massbound holds empirically on random drifts.
6. Stage 2 acceptance: injected budget violation ⇒ rollback + η₂ shrink;
   accepted refresh ⇒ measured D ≤ δ².
7. Projection: at refresh with ε=0, realized step ⟂ basis; collinear basis
   triggers single-basis fallback; ridge residual reported.
8. Stage 1: λ ascends under violated constraint; clip_c kills gradient
   above c but keeps forward value; exit gate fires iff all seqs ≥ m and
   remote recall ≥ 0.9.
9. RefCache token index map: round-trips (example, position) alignment
   under shuffled batch order.
10. Toy e2e: memorize → Stage 1 to floor → Stage 2 repairs adjacent loss
    without breaching either budget at any accepted refresh.
11. Partition determinism: same seed ⇒ identical manifest SHAs; tie-break
    stability.
12. Seal ledger: analysis import path raises on sealed-unopened access.
13. Cost meters: fwd/bwd counts match hand counts on tiny run.
14. Gitignore regression test: `src/`, `tests/`, `prereg/` are tracked
    (lesson from predecessor repo).

## 8. Orchestration & run layout

- `runner.py`: one (request, method, config) → `runs/<hash>/` with config,
  git SHA, checkpoint grid artifacts, event log, cost records. Checkpoints
  are the ONLY evaluation substrate (no interpolation).
- Long runs: detached launch pattern (`scripts/launch_detached.ps1` /
  `nohup` on cluster) — harness-tracked background tasks die with the
  session (predecessor lesson).
- `experiments/smoke/`: every RQ path on tiny model + substrate, CPU,
  < 5 min total; run in CI-of-one before any cluster submission.
- Cluster profile deferred until scheduler is known (open decision D3).

## 9. Porting map from `unlearning-entanglement-repli`

| Old | Disposition |
|-----|-------------|
| `src/lm/losses.py` | Port; add per-token NLL with index map |
| `src/lm/ref_cache.py` | Port; extend to token level + manifest SHA |
| `src/lm/adjacency.py` | Split: `one_step_interference` → `probe/finite_diff.py`; kNN vote → `probe/baselines.py`; pool construction → `partition.py` (add folds/sealing) |
| `src/lm/method_lm.py` `lagrange_stage1_lm` | Port to `stage1.py` (EMA dual ascent, floor, exit gate) with light cleanup |
| `src/lm/method_lm.py` `wpgd_stage2_lm` | **Do not port.** Sorted-W2 + α-blend is the retired design; `stage2.py` is written fresh against §5, with the old file kept in `third_party/` for the sorted ablation cross-check |
| `src/lm/config.py`, `run_logging.py` | Port |
| `src/lm/data/tofu.py` | Port; add fold map + native-audit ids |
| `src/lm/eval_lm.py`, `runner.py` | Rewrite (contract-based evaluation, checkpoint grid) |
| ToxiGen/classification code | Leave behind (fidelity audit already done; results archived) |

## 10. Milestones

- **N0** Skeleton: pyproject, blocks/losses/costs/config, tiny fixtures,
  battery bootstrapped, gitignore test. (laptop)
- **N1** Probe: fd + jvp + vmap/streaming comparators + baselines,
  invariants 1–4, cost meters. (laptop)
- **N2** Partition + sealing + substrate generator, invariants 11–12.
  (laptop)
- **N3** Stage 1 port + guards + Stage 2 fresh, invariants 5–10, toy e2e.
  (laptop)
- **N4** Table-driven pipelines (laptop, tiny model + substrate):
  - **N4a (T1)**: `generators/` trajectory runner recording per-candidate
    NLL at θ0 and every saved checkpoint (terminal-budget primary, DONE
    markers for sealing); objectives ga, graddiff, npo(+retain), rmu;
    `analysis/prediction.py` (Spearman/AUROC/Overlap@K/Tail ρ,
    seed-averaging, macro+range, hierarchical bootstrap, LOTO); smoke run
    emits a filled miniature Table 1 CSV.
  - **N4b (T2)**: criterion/reach evaluation (`evalx/protection.py`),
    first-reaching checkpoint, native mean/CVaR damage, utility probe;
    ours wrapped as a method; NPO+susceptibility transplant; smoke Table 2
    CSV. Paraphrase recall lands with the TOFU adapter.
  - **N4c (T3)**: ablation config matrix (partition source swap,
    projection/guard off) + `analysis/ablation.py` matched-parent deltas.
  - **N4d (T4)**: `experiments/cost/` bench harness over the four
    profiling implementations (median [IQR] over repeats) — numbers real
    only at N5, harness validated on CPU.
  - **N4e**: TOFU adapter (native metadata audit, paraphrase sets, folds);
    remaining T2 baselines (simnpo, idkdpo, gru, s2s).
- **N5** First GPU session: 1.5B centering pilot, real pool manifests,
  T1 generator trajectories, T4 timings at 7B/14B. (cluster)

## 11. Open decisions (need Olivia)

- **D1** Repo host + name: private GitHub? `retain-susceptibility` is the
  working name; must stay anonymizable for double-blind.
- **D2** Generators: thin in-repo re-implementations (full parity
  accounting, unit-checked) vs OpenUnlearning as a pinned dependency.
  Design assumes in-repo with `third_party/` pins for cross-checks.
- **D3** Cluster scheduler (SLURM? bare SSH?) — launcher stubs until known.
- **D4** δ_seq/δ_tok and slack ε defaults for the prereg freeze (paper
  leaves them to `tab:prereg-constants`).
- **D5** MIA implementation source for Min-K%++.
