# AMENDMENT-2026-07-23-cvar-bound

**Scope.** Development-selection utility bound `cvar05_dnll_max` for the
7B-scale channel-matrix campaigns
(`kdd27-channel-matrix-7b-v1`, `kdd27-channel-matrix-rwku7b-v1`):
**8.0 → 16.0**. The mean bound (2.0), the forgetting criterion
(recall ≤ 0.10), and every other selection rule are unchanged. The 14B
campaign (`kdd27-channel-matrix-14b-v1`) retains the original 8.0 bound,
under which passing settings exist (graddiff 0.038/0.46/1.69,
gru 0.029/0.95/2.43).

**Why the original value is wrong.** The 8.0 bound was transplanted from the
1.5B substrate/toy experiments before any 7B-scale development run existed.
Three settings of measured development evidence now show the achievable
graddiff frontier at reach sits above it while every other criterion is met:

| Setting | best in-reach setting | reach_max | mean ΔNLL | CVaR05 |
|---|---|---|---|---|
| TOFU × Qwen2.5-7B | graddiff lr 5e-7 × 360 | 0.005 | 1.31 | **11.86** |
| TOFU × Qwen2.5-7B | graddiff lr 7.5e-7 × 240 | 0.006 | 1.44 | **12.24** |
| RWKU × Qwen2.5-7B | graddiff lr 1e-6 × 240 | 0.095 | 2.03 | **15.48** |

The gentle-lr/long-schedule direction thins the tail monotonically
(17.3 → 13.3 → 12.2 → 11.9 on TOFU-7B) and plateaus near 12, i.e. the bound
excludes the entire reachable operating region at this scale, not a
poorly-chosen corner of it. 16.0 (2× the original) sits above the measured
frontier with margin and remains far below the collapse regimes the bound is
meant to exclude (gru/repnoise at reach: CVaR 20–44).

**Timing and outcome-independence.** Adopted 2026-07-23, before any audit
run, any probe/predictor score, or any sealed outcome exists for either
affected campaign. Selection continues to read only development forgetting
reach and retained-utility aggregates; no predictor correlation is inspected.
Both bounds' outcomes will be reported: settings that additionally satisfy
the original 8.0 bound are flagged as such wherever they exist (all frozen
14B settings; any 7B-scale representation-family setting that passes with
CVaR ≤ 8).

**Decision record.** Proposed by the analysis session from the three-setting
frontier measurement; approved by the experiment owner on 2026-07-23 (D-3)
in preference to core-roster reduction or further grid search against a
measured plateau.
