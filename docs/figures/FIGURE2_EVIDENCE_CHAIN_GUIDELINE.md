# Figure 2 — "Evidence chain for the sealed evaluation framework": redraw guideline

> Design decided 2026-07-24 by a multi-agent adversarial pass (narrative
> extraction → three independent concepts → hostile art-director/reviewer
> judgement). This file is the single source of truth for rebuilding
> `figures/fig6_evidence_chain.tex` (label `fig:channel-main`, the paper's
> "Figure 2"). The generator is `experiments/paper/plot_evidence_chain.py`.

## 0. What decides everything (the two hard constraints)

1. **Figure 1 (`fig5_experiment_anatomy`) already owns the schematic** — the
   5-stage pipeline, the SEAL box, the forgetting gate, the RQ1/2/3 mapping,
   with **no data**. Figure 2 must therefore be the **evidence-carrying twin**:
   the same claim structure, now filled with **real measured bounds** and at
   least one **honest boundary**. Any design whose primary device is a
   flow/timeline/wall re-draws Figure 1 and is rejected.
2. **The paper's identity is fail-closed + anti-overclaiming.** Every claim is
   a *one-sided 95% bound* that had to clear a *predeclared floor*; the honest
   "no" (non-reach, low-precision collapse, RQ3 infeasibility) is load-bearing.
   A green-checkmark hero betrays the paper. Section 5 literally calls this
   figure the **"claim ladder."**

## 1. Chosen design — "Claim ladder with a measured core"

Three stacked horizontal bands, **bottom → top = RQ2 → RQ1 → RQ3** (fidelity is
the floor you stand on; prediction is load-bearing; protection is the payoff).
Each band speaks one grammar: **an achieved 95% bound (●, with a whisker) versus
its predeclared floor (│ tick)**. The middle (RQ1) band opens that grammar into
a small honest scatter so the paper's load-bearing but *modest* result is
**shown, not asserted as a pass-tick**.

Rejected alternatives and why (keep for the record):
- **"Sealed timeline"** (spine + seal wall + gate + two RQ tiles): it is Figure 1
  in a data costume; the "green column passes through unchanged" *asserts* the
  seal rather than measuring anything. Salvaged only its caption line.
- **"Prediction scatter hero"** (giant ρ≈0.21 scatter): enlarges the weakest
  number into visual overclaiming and amputates 2/3 of the evidence chain.
  Its scatter survives — shrunk — as the RQ1 rung.

### Burn-in message (caption, one line)
> *Every claim is a bound that had to clear a floor set before any outcome
> existed — and the chain stops the moment one doesn't.*

## 2. Canvas & layout

- `figure*`, `\resizebox{\textwidth}{!}{...}` over a ~19 cm × ~9.5 cm tikz canvas
  (match `fig6`'s absolute-cm axis placement so it drops into the existing
  scaffold).
- Header strip (full width, ~0.9 cm): text
  `PREDECLARED FLOORS — sealed at θ0, before any outcome existed.` with **exactly
  one** small seal glyph at the left (restraint — never per-floor; skeuomorphic
  wax/lockbox kitsch is what reviewers sneer at).
- Bands, bottom→top: RQ2 (~2.6 cm) · RQ1 (~3.4 cm, the tallest — it's the core)
  · RQ3 (~2.6 cm). A shared vertical **ink line at x=0 / floor axis** ties the
  bands into one "seal line."

## 3. Palette (Okabe–Ito, CVD-safe; matches colors already declared in fig6)

| Role | Hex | Rule |
|---|---|---|
| `q_G` loss-shake energy / RQ2 fidelity | `#0072B2` evGrad | cool = computed at θ0 |
| `q_H` proximity / **revealed damage points** | `#D55E00` evProx | |
| `S_α` joint profile / **a bound that cleared its floor** | `#009E73` evMix | green = "cleared a floor," *never* a checkmark |
| controls, failed side of a floor, non-reach, snapped/not-licensed | `#767676` evCtrl | |
| forgetting gate / process | `#CC79A7` | |
| **seal / floor ticks / x=0 line** | `#1A1A1A` ink | |
| realized outcome (amber, if used for damage axis) | `#E69F00` | revealed-after-gate |

Colour–time coupling rule: **cool categorical hues only ever label things that
exist at θ0; the revealed outcome is amber.** That coupling is what makes the
seal legible without drawing a wall. Run `dataviz` `validate_palette.js` on the
final categorical set before shipping.

## 4. Band specs

### RQ2 band (bottom) — reuse fig6 Panel B idiom
- Horizontal `xbar`, y-dir reversed. Rows: `ρ_AC`, `f_K`, `ρ_AB`, `ρ_BC`,
  `frac. changed`, `eff/η`.
- Bar = point estimate (evGrad); `│` ink floor tick per row (ρ_AC→0.80,
  f_K→0.70); whisker to the one-sided 95% LB.
- **Honest "no" #1:** the bf16 row (frac. changed 0.0019, eff 0.079) in evCtrl
  gray with chip *"low-precision perturbation collapse = invalid, not a
  favorable zero."*
- Right badge: `RQ2 cleared` rendered as `LB − floor > 0`, never a checkmark.

### RQ1 band (middle) — the measured core (shrunk scatter)
- Scatter: x = sealed joint rank `S_α` (frozen θ0), y = revealed audit damage
  `d_t†` (nats), points evProx. evMix monotone trend + shaded one-sided 95% LB
  band. Corner label `ρ = 0.21 [LB > 0]` **at true size, no enlargement.**
- **Two faint ghost trends** (q_G-only, q_H-only endpoints) + a flat evCtrl
  control line, so `g_G>0, g_H>0, g_ctl>0` read in one glance — this is the
  whole reason the rung is a scatter not a tick.
- **Honest "no" #2:** a non-reach chip *"k parents non-reaching → excluded from
  the denominator, never scored."*
- Right margin: a mini bound-vs-floor pair for `min(g_G,g_H) [LB]` and
  `L_tail [LB]; coverage ≥ 80%`, so the middle rung still speaks the ladder
  grammar on its summary side.

### RQ3 band (top) — bound grammar + the structural snap
- Small caterpillar of the **8 damage UCBs** (4 comparators {no-repair,
  repeated-random, S0, S1} × {mean, CVaR.95}): point + whisker to the 95%
  *upper* bound; floor = the x=0 ink line; passing arms have UCB<0 (evMix).
  Below: the **4 native-metric non-inferiority LBs**, floor at the frozen
  δ_nat margin.
- **Honest "no" #3 (load-bearing):** show the frozen-operating-point case where
  one arm's UCB crosses to the wrong side of 0. Render that rung **snapped** —
  a clean break-gap with light hatch (subtle, *not* cartoon splinters) — and set
  the right badge to gray **`COMPOSITE CHAIN: NOT LICENSED here`**, chip
  *"infeasible at frozen (K_p, budget) op-point."* Word it as the *composite
  chain* license: RQ1/RQ2 rungs stay intact and individually passed — RQ3
  infeasibility does **not** invalidate RQ1 (they are separate estimands).

## 5. Honesty budget (must all be present)

1. bf16 collapse grayed in RQ2.
2. non-reach chip in RQ1 (denominator funnel visible: planned ≥ completed ≥
   profile-valid ≥ gate-reaching ≥ feasible ≥ passing).
3. the snapped RQ3 rung + "NOT LICENSED" badge.

Three escalating "no"s make the fail-closed identity unmissable in 5 s.

## 6. Data to generate (`plot_evidence_chain.py`, pure-CPU numpy)

The 7B aggregate macros are still placeholders, so the figure ships on a
**conceptual small-sample** calibrated to the real anchors (allowed — the user
authorized fresh small-scale data; this is a concept figure, and every number is
reproducible from the seeded script and labelled "illustrative, calibrated to
the 7B operating point" in the caption until the sealed audit aggregates).

- **RQ2:** reuse the values already in fig6 — fp32 ρ_AB .95 / ρ_BC .97 /
  ρ_AC .92 / frac .997 / eff 1.0; bf16 collapse .0019 / .079; f_K 0.77 vs 0.70,
  ρ_AC 0.92 vs 0.80. When the rescored certificate lands
  (`results/paper/fidelity_summaries/tofu_qwen25_7b.json`: f_rho 0.9247,
  f_k 0.769, LBs 0.884 / 0.615) swap those in.
- **RQ1:** n≈120 audit candidates from a monotone copula with
  Spearman(rank, damage) ≈ 0.21; q_G-only ≈ 0.14, q_H-only ≈ 0.16 (joint > both
  → small positive g_G, g_H), control ≈ 0. Hierarchical bootstrap 2000×
  (resample requests → seeds → groups) for the LB band, ρ LB>0, and
  L_tail LB>0 at coverage ≥ 80%. When the real ledger exists, read
  `results/paper/evidence_ledger.json` instead.
- **RQ3:** ~40 paired requests × 4 comparators; 8 ΔNLL UCBs (passing case all
  <0) + 4 native LBs (>0) via paired bootstrap; plus a second frozen-op-point
  draw where one arm's UCB>0 to drive the snapped rung — grounded in the real
  7B verdict (`docs/data/alpha_dev_7b/`: RQ3 infeasible at the frozen operating
  points).

Every generated series is seeded and dumped to a sibling JSON so the figure is
reproducible and swappable for sealed numbers the moment they aggregate.

## 7. Production notes

- ~80% is the already-proven fig6 Panel-B `xbar + │-floor` idiom (low risk). New
  work: one embedded pgfplots scatter at absolute coords (fig6 already places
  multiple axes this way) and the subtle snapped-rung (a gap + light hatch —
  resist the cartoon). Overall risk 3/5.
- The generator emits `--tikz` (pgfplots, drops into the paper) and `--png`
  (matplotlib, for quick visual QA). Run the dataviz layout eyeball pass
  (label collisions, overflow, dark-mode steps) before committing the `.tex`.
- Keep `\Description{...}` accurate for accessibility; identity is never
  color-alone (every series is also direct-labelled or legended).
