# AMENDMENT-2026-07-23-primary-setting

Date: 2026-07-23 (adopted before any sealed-audit or alpha-audit outcome
existed for any campaign)

## Change

The paper's primary setting moves from `tofu_qwen25_1p5b` to
`tofu_qwen25_7b` in `configs/paper/evidence.yaml`:

- `settings.tofu_qwen25_1p5b.role`: `primary` → `scale_boundary`
- `settings.tofu_qwen25_7b.role`: `model_scale` → `primary`
- `multi_setting_rule.primary_required`: `[tofu_qwen25_1p5b]` → `[tofu_qwen25_7b]`
- `multi_setting_rule.groups.model_transfer.settings`:
  `[tofu_qwen25_7b, tofu_llama31_8b]` → `[tofu_qwen25_14b, tofu_llama31_8b]`
  (a new predeclared `tofu_qwen25_14b` setting joins the roster)
- `tables.main_core_evidence.settings`: `[tofu_qwen25_1p5b]` → `[tofu_qwen25_7b]`
- Appendix tail/budget tables follow the primary setting.
- `multi_setting_rule.id`: `..._v1` → `..._v2`

## Basis (development evidence only)

Three-to-four rounds of development-only calibration (25+ settings at 1.5B,
260+ cells total; `docs/data/calibration_2026-07-23/`) measured that **no
parent objective jointly satisfies the frozen forgetting criterion
(recall ≤ 0.10) and the utility bounds (mean ΔNLL ≤ 2.0, CVaR05 ≤ 8.0) at
Qwen2.5-1.5B** under the probe-block contract. The decisive frontier:
graddiff reaches only with a destroyed tail (CVaR 8.8–39.1) or stays out of
reach within bounds; rmu saturates at recall ≈ 0.29 with near-zero damage.
The 7B campaign resolved a full output+representation chain (graddiff
lr 8e-7 × 240, rmu lr 3.2e-5 × 240; `FREEZE-2026-07-23-7B-DEV3ROUNDS`).

## What this amendment does NOT do

- It does not remove the 1.5B rows: they remain predeclared denominators in
  `tab:robustness` and are reported as a measured scale boundary
  (reach-and-preserve jointly infeasible at 1.5B block capacity).
- It does not change any decision threshold, IUT membership, bootstrap rule,
  comparator roster, or parent roster.
- It does not touch the frozen 7B calibration selections, which were made by
  forget-reach + utility rules only, never by prediction or audit outcomes.

## Ordering guarantee

At adoption time the 7B/14B/RWKU audit queues had been enqueued but no audit
unit's sealed scores had been opened and no alpha-audit had run; the sealed
evidence that decides RQ1–RQ3 remains untouched by this change.
