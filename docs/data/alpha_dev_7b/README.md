# 7B alpha-protection development verdict (2026-07-24) — RQ3 unresolved

Selector output (`select_alpha_freeze.py`, dev requests a198/a199 × seed 2025,
artifact sha256s inside the yaml): **both parents UNRESOLVED — no feasible
alpha exists, so no alpha-audit runs** (predeclared rule: no best-effort
fallback may be audited).

Measured failure boundaries, per parent:

- **graddiff** (lr 8e-7 × 240): every arm including no-repair fails the
  ordinary-utility floor — utility retention 0.30–0.46 vs the 0.90 contract,
  worst CVaR05 9.1–21.2. The frozen operating point satisfies the
  retained-candidate damage box used at calibration while destroying the
  disjoint ordinary-utility QA set; a 120-step guarded repair cannot recover
  a 3x utility collapse. Structural infeasibility.
- **rmu** (lr 3.2e-5 × 240): a198 passes every arm (para 0.088–0.094,
  utility 0.912–0.926, CVaR ~0.5). a199 fails **paraphrase forgetting on
  every arm including no-repair** (para_recall 0.120–0.129 vs 0.10) — a
  parent-checkpoint property repair cannot touch. Request-difficulty
  heterogeneity (a199 consistently harder) replicates the 1.5B observation
  at 7B.

## rmu reach recalibration (recal-v2, 2026-07-24): still UNRESOLVED, opposite floor

The 3.2e-5 verdict left reach marginal, so a pre-audit amendment raised the
rmu learning rate to **4.8e-5 × 240** (`configs/channel_matrix/7b_tofu_recal2.yaml`,
isolated `runs/channel_matrix_7b_recal2`) and re-selected alpha on the same
development requests a198/a199 × seed 2025. Selector verdict:
`resolved: false`, `alpha: null` (`unresolved: [qwen25_7b/rmu]`).

The stronger rate **fixes reach but collapses utility** — every alpha
0.0/0.25/0.5/0.75/1.0 is `feasible: false` with the same signature (minimax
across a198/a199):

| alpha | max_forget_recall (≤0.10) | min_utility_retention (≥0.90) | worst CVaR05 |
|---|---|---|---|
| 0.00 | 0.099 ✅ | 0.889 ❌ | 0.816 |
| 0.25 | 0.099 ✅ | 0.885 ❌ | 0.829 |
| 0.50 | 0.097 ✅ | 0.882 ❌ | 0.835 |
| 0.75 | 0.096 ✅ | 0.881 ❌ | 0.840 |
| 1.00 | 0.093 ✅ | 0.879 ❌ | 0.836 |

The binding constraint is now **utility retention (0.88 < 0.90), not
paraphrase** — the unlearning itself (visible already on no-repair) violates
the ordinary-utility contract. So rmu has **no feasible operating point at
7B**: the two calibrations fail on *opposite* floors —

- **lr 3.2e-5** reaches on a198 but leaves a199 paraphrase-forgetting
  persistent (0.12 > 0.10);
- **lr 4.8e-5** reaches robustly on both (forget_recall 0.093–0.099) but
  craters utility (0.88 < 0.90 on every alpha).

This is a measured **reach ↔ utility (capacity) tension** for the
representation-readout parent, not a tuning miss: raising pressure to clear
one floor pushes the other below its threshold.

Provenance (recal-v2): `objective_freeze_id: FREEZE-2026-07-24-7B-RMU-REACH-RECAL-V2`,
`objective_freeze_sha256: d4f85b54c837b1d6452dec38b1ff029dd556ff86be7811a43770bf765a079554`,
`campaign_config_sha256: a5e86c9fbc4d83bf34085b5c5dd2084c75a3c1d389d92bc470a7e717952087be`;
development results.json sha256 a198
`c3f183a19e55999f9d580f6df04ce92df677f652bb10343dc518eef1acb318d4`,
a199 `4822af09ce233ba8485bb9872a977700a5cae2afa17a9b1d8170b8e776d8d624`.

Paper consequences: `tab:core-evidence` Panel B reports these as measured
infeasibility boundaries (RQ3 E/P = n/--); the RQ1/RQ2 chain proceeds
independently on the completed sealed audits. Section 5 candidate
observation: "reach + candidate-damage bounds do not imply the four-audit
final contract at 7B — output readout fails by utility collapse, and
representation readout has no feasible operating point at all: at low
pressure paraphrase forgetting persists on the harder request, at high
pressure ordinary utility collapses (a reach ↔ utility tension)."
