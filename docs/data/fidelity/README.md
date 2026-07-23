# fd_fidelity_7b_bf16.json

Measured 2026-07-23 on run261224-ul2 (queue unit `t3-fdfid-7b-bf16`,
`experiments/diag/fd_fidelity.py`, gate deliberately not enforced so the
failing numbers are recorded). Backs the Sec. 5 claim that bf16 fails the
loss-shake fidelity gate on 7B: the frozen radius eta=3e-3 lands below the
bf16 coordinate ULP, so only 0.19% of block coordinates actually change and
the effective perturbation norm is 7.9% of the requested radius
(thresholds: >=90% for both); rho_AC 0.707 and rho_BC 0.668 also miss their
0.80 gates. Transcribed from the cluster run output (cluster cannot push).
