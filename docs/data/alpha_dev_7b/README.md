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

Paper consequences: `tab:core-evidence` Panel B reports these as measured
infeasibility boundaries (RQ3 E/P = n/--); the RQ1/RQ2 chain proceeds
independently on the completed sealed audits. Section 5 candidate
observation: "reach + candidate-damage bounds do not imply the four-audit
final contract at 7B — output readout fails by utility collapse,
representation readout by paraphrase persistence on the harder request."
