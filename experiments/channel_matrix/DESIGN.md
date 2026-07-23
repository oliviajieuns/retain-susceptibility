# Main-table 7B/8B channel-matrix design

> **Mechanism precursor, not the current claim contract.** The historical
> difference-in-differences endpoint below is diagnostic only. The current
> paper requires paired joint-over-both-endpoint prediction gains and the
> eight-way protection IUT defined in `configs/paper/evidence.yaml`.

Status: implementation and prospective protocol, 2026-07-22. No 7B/8B audit
result is asserted in this document. Empty results must remain empty until the
frozen audit has completed.

## What this extension is meant to establish

The unit of mechanism is the **unlearning objective**, not the dataset or the
backbone. An objective is assigned to a channel before any result is read,
according to what its removal term consumes:

- output/loss-gradient channel: token likelihoods or preferences;
- representation channel: internal hidden states.

The endpoint is therefore the channel interaction

```text
Delta = mean_OC[rho(fd_norm, damage) - rho(kNN_hidden, damage)]
      - mean_RC[rho(fd_norm, damage) - rho(kNN_hidden, damage)].
```

`Delta > 0` is the prospective claim. A large isolated correlation, a method
leaderboard, or a favorable cell selected after audit is not the endpoint.

## Core objective roster

The core matrix uses four output objectives and three representation
objectives. This is a chronological and conceptual coverage choice, not a
selection based on the current TOFU correlations.

| Channel | Objective | Generation and reason for inclusion | Removal signal read by the loss | Status in this repository |
|---|---|---|---|---|
| Output | GradDiff | Canonical retain-regularized baseline used by TOFU-era LLM unlearning | mean answer NLL | Controlled common-block implementation |
| Output | NPO+GD | COLM 2024; canonical preference-style alternative to unstable GA | **summed** sequence NLL ratio to the frozen reference | Paper-faithful loss, common-block update |
| Output | SimNPO+GD | NeurIPS 2025; reference-free and length-normalized NPO generation | **mean** answer NLL in the sigmoid objective | Paper-faithful loss, common-block update |
| Output | GRU | ICML 2025; gradient-geometry generation | output-loss unlearning gradient, projected only when it conflicts with retain gradient | Minimum-deviation projection, common-block update |
| Representation | RMU | ICML 2024/WMDP; canonical representation-misdirection objective | hidden-state distance to a fixed control vector | Faithful loss geometry at the declared hidden layer and block |
| Representation | RepNoise-style | NeurIPS 2024; stochastic representation destruction | hidden-state distance to fresh Gaussian targets | **Adaptation**: not the original all-layer multi-kernel MMD recipe |
| Representation | RR/CB-style | NeurIPS 2024 Circuit Breakers; representation rerouting | positive cosine to the frozen harmful representation | **Adaptation**: core rerouting loss without original LoRA/schedule |

Primary sources:

- [TOFU / GradDiff baselines](https://arxiv.org/abs/2401.06121)
- [Negative Preference Optimization](https://arxiv.org/abs/2404.05868)
- [SimNPO](https://proceedings.neurips.cc/paper_files/paper/2025/hash/02443e4e008231e0af0855f3cc70ed17-Abstract-Conference.html)
- [Gradient Rectified Unlearning](https://proceedings.mlr.press/v267/wang25de.html)
- [WMDP / RMU](https://proceedings.mlr.press/v235/li24bc.html)
- [RepNoise](https://proceedings.neurips.cc/paper_files/paper/2024/hash/172be8b0b88fc2b4aee74237d43f8c04-Abstract-Conference.html)
- [Circuit Breakers](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97ca7168c2c333df5ea61ece3b3276e1-Abstract-Conference.html)

### Stress controls, not core columns

- **GA** is the canonical instability control. The existing 1.5B run reached
  the forget criterion only with model-wide collapse, so GA must be shown as a
  collapse-marked appendix control rather than allowed to dominate the core
  claim.
- **IdkDPO** is a preference/refusal stress control. The existing run did not
  reach the frozen criterion. It remains visible with a failure marker.
- No fourth representation objective is added merely to make the column counts
  symmetric. Recent feature-suppression systems such as
  [CRISP](https://aclanthology.org/2026.acl-long.82/) introduce an SAE-specific
  substrate, while representation-guided methods such as
  [ReGLU](https://aclanthology.org/2026.findings-acl.717/) can be hybrid under
  the paper's rule because their actual forget-suppression term may still read
  output likelihood. They are appropriate appendix extensions, not automatic
  core representation columns.

## Loss definitions that must not be conflated

Let `ell_mean(x)` be answer-token mean NLL and `ell_sum(x)` be the summed
answer-token NLL.

```text
GradDiff: -lambda_f E_f[ell_mean] + lambda_r E_r[ell_mean]

NPO: -(2/beta) E_f log sigmoid(beta(ell_sum(theta)-ell_sum(theta_0)))
     + lambda_r E_r[ell_mean]

SimNPO: -(2/beta) E_f log sigmoid(beta ell_mean(theta)-gamma)
        + lambda_r E_r[ell_mean]

GRU: g_u = grad[-E_f ell_mean]
     g_r = grad[ E_r ell_mean]
     g = g_u - min(<g_u,g_r>/||g_r||^2, 0) g_r
```

The NPO/SimNPO sum-versus-mean distinction is intentional. Candidate damage is
always reported as change in `ell_mean`; changing the parent loss must not
silently change the damage estimand.

## Models and update support

Core model families:

1. `Qwen2.5-7B-Instruct` (currently provisioned on the documented H100 host).
2. `Llama-3.1-8B-Instruct` (configuration included; enable only after the local
   checkpoint is provisioned and independently calibrated).

An optional third robustness model is Mistral-7B-v0.3. It is not required for
the core claim and should not replace a missing failed core run after results
are visible.

The one-H100 fp32 protocol updates the last eight MLP `down_proj` matrices for
SFT and every parent objective. Exact gradient norm and randomized sensitivity
read that identical block. This common support is a causal-control and memory
decision; it means the experiment compares objective loss geometry on a shared
substrate, not each paper's best method-specific training recipe.

## Data, development, and audit separation

TOFU remains the controlled main matrix because it supplies homogeneous
per-candidate retained QA damage. The extension uses:

- development deletion requests: authors 198 and 199;
- sealed audit deletion requests: authors 181, 186, and 191;
- development retained candidates: the 30 frozen even authors from 0--58;
- audit retained candidates: a different frozen 30-author pool for each audit
  request, interleaved over authors 60--149 (a181 gets `60,63,...,147`, a186
  gets `61,64,...,148`, and a191 gets `62,65,...,149`).

All four retained-author pools are mutually disjoint. Each contains 600 QA.
Each request-specific audit pool is split at author granularity into 15
discovery and 15 audit authors for each seed, leaving exactly `n=300` sealed
audit candidates per run. Parent retain minibatches use discovery candidates
only. Request-specific pools also prevent shared retained examples from making
the multi-request confidence interval look artificially precise.

Do not pool TOFU, WMDP, MUSE, and RWKU candidates into a single correlation.
Use a separate dataset-by-dataset interaction panel for WMDP-bio, MUSE-News,
and RWKU. MUSE-Books and PISTOL remain stress tests. The cross-dataset question
is whether `Delta > 0` survives changed base rates.

## Prospective execution discipline

1. Commit code and the campaign configuration.
2. Run the frozen `R=64`, `eta=3e-3`, fp32 fidelity cell on the development
   candidate pool. Audit refuses a missing or failed model-specific certificate.
3. Run every predeclared objective grid cell on both development requests.
   No susceptibility predictor is computed in calibration.
4. `select_freeze.py` requires the complete grid and selects only by forget
   reach and ordinary retained damage. It never reads a score seal or a
   predictor correlation.
5. Review the recommendation, fill every model/objective setting, assign a
   dated freeze ID and timestamp, clear `unresolved`, mark it frozen, and commit.
6. Start audit only from a clean worktree. For each run, the code SFTs the
   model, verifies full-set memorization, computes all predictors, seals audit
   scores, completes **every** frozen parent trajectory, and only then opens
   the scores.
7. Aggregate only a complete balanced model x request x seed design. Confidence
   intervals resample model, request, seed, and candidate levels.

The launcher records hashes of the campaign YAML, objective freeze, fidelity
certificate, candidate universe, forget set, and code commit in run manifests.
Partial sealed runs are preserved and never overwritten.

## Main-table reporting contract

- Core: five measured predictor rows by seven parent columns, with a
  hierarchical 95% CI in every Spearman cell.
- Caption: models, requests, seeds, number of runs, parameter block, dtype,
  audit `n`, and the roster interaction with CI.
- Appendix: per-model matrices; GA and IdkDPO stress controls; random ranking,
  answer length and initial NLL controls; all objective settings and status.
- A dagger is generated if any run misses the frozen forget criterion. A double
  dagger is generated if any run crosses the frozen collapse threshold.
- RepNoise-style and RR/CB-style retain an adaptation marker in every table.
- Failed, diffuse, negative, or mismatched results remain in the table. No
  objective, model, request, or seed may be removed after audit inspection.

## Safe paper wording before results exist

Allowed:

> We prospectively evaluate whether channel matching persists at 7B/8B scale
> across two model families, three deletion requests, and two seeds.

> The declared channel supplies a prior over the appropriate susceptibility
> probe; the primary endpoint is the resulting difference-in-differences.

Not allowed before completion:

- “The interaction generalizes to 7B models.”
- “All output objectives favor randomized sensitivity.”
- “Representation proximity consistently wins.”
- Any numerical cell, interval, or claimed model count not produced by the
  aggregate pipeline.
