# Stage 1 — Latent ODE: drop `p`, infer it from a window of `q`

We observe only the angle `q(t)` (on a regular dense grid). Three pieces:

- **Encoder**: small MLP over a window of `k_enc=8` consecutive `q` values → `z₀ ∈ ℝ²`.
- **ODE**: same MLP-as-vector-field as Stage 0, `dz/dt = f_θ(z)`. Autonomous.
- **Decoder**: **fixed** as `q = z[0]`. No parameters — makes `z[0]` interpretable as the predicted angle, frees `z[1]` to be whatever hidden coord the encoder finds useful.

Training: encode `z₀` from the first 8 obs, integrate forward for 32 steps total (24-step prediction horizon), MSE on `q` over the whole window.

## Run

```bash
cd stages/01_latent_drop_p
uv run python train.py                                    # defaults
uv run python train.py 'data.amplitudes=[1.0]'            # single-amp ablation
```

Outputs in `logs/<date>/<time>/`:
- `q_vs_time.png` — predicted `q(t)` vs truth, per test amplitude (Stage 0's `angle_vs_time` analog).
- `latent_orbits.png` — phase portrait in `(z[0], z[1])` with true `(q, p)` overlaid.
- `latent_vs_p.png` — scatter `z[1]` vs true `p`: is the hidden coord a clean function of momentum?
- `latent_basis_fit.png` — post-hoc linear fit `z[1] = a·q + b·p` per amplitude (and global) with R².

## Headline findings

**It works qualitatively.** With only `q` observed, the latent ODE recovers
oscillation at roughly the right amplitude *and* roughly the right period for
each test radius — including the unseen interior amplitude A=0.6. The encoder
successfully extracts enough info from 8 consecutive `q` values to seed a
meaningful integration.

**`z[1]` is a rotated basis, not `p` itself.** A post-hoc linear fit
`z[1] = a·q + b·p` confirms this directly:

| A | a (·q) | b (·p) | R² |
|---|---|---|---|
| 0.3 | -0.64 | +0.61 | 0.866 |
| 0.6 | -0.64 | +0.68 | 0.912 |
| 1.0 | -0.61 | +0.82 | 0.947 |
| 1.5 | -0.51 | +1.12 | **0.996** |

Global: `z[1] ≈ -0.56·q + 0.97·p`, R² = 0.95.

Two clean takeaways:

1. **The basis is approximately amplitude-invariant.** `a` is roughly constant
   (-0.51 to -0.64) across all four amplitudes. The encoder picked roughly the
   same recipe everywhere — not a per-amplitude one. (We *did not* explicitly
   tell the encoder what amplitude it was looking at; it has to infer it from
   the window of `q` values, and it apparently uses the same rotated basis.)
2. **`b` scales with amplitude, R² is best at large radii.** At A=1.5 the
   linear fit is essentially perfect; at A=0.3 there's clear nonlinear
   residual (the scatter forms a loop, not a line — visible in
   `latent_basis_fit.png`). Likely cause: at small amplitudes the encoder has
   less signal to extract `p` from (smaller velocities → smaller finite
   differences), so `z[1]` is dominated more by encoder bias and less by `p`.

**~3 orders of magnitude worse than Stage 0.** Rollout MSE ~1e-2 vs ~1e-5 in
Stage 0 (3-amp). Phase drift accumulates over the 3-period eval horizon;
amplitude reconstruction is essentially right but timing drifts.

| Amplitude | Stage 1 q-rollout MSE | Stage 0 rollout MSE |
|---|---|---|
| A=0.3 (train) | 1.1e-02 | 2.8e-05 |
| **A=0.6 (UNSEEN)** | **3.6e-02** | **1.3e-04** |
| A=1.0 (train) | 4.7e-02 | 4.8e-05 |
| A=1.5 (train) | 1.1e-02 | 1.0e-05 |

**Training instability at large amplitudes.** Best model was captured at
step 200; late training improved small amps but blew up A=1.5 back to ~4e-1.
Probably the encoder for A=1.5 (largest range of `q` values) needs more
expressive capacity or different scaling.

## What I think is going on

The encoder has 8 samples spanning 0.7 s — enough to estimate `dq/dt` at the
window's center via finite differences, but not enough to disambiguate
"slow-near-the-top" vs "fast-near-the-bottom" beyond a leading-order
approximation. The latent ODE then propagates that approximate state for
3 periods, and the small initial error accumulates into a visible phase drift.

The headline lesson is more conceptual than numerical: **the model can
recover physics from partial observation**, just at lower fidelity than
full-state. That sets up Stage 2 (add noise) and Stage 3 (variational
encoder), where we'll get to ask whether the model knows how uncertain it is.
