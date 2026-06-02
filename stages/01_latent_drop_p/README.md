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

## Headline findings

**It works qualitatively.** With only `q` observed, the latent ODE recovers
oscillation at roughly the right amplitude *and* roughly the right period for
each test radius — including the unseen interior amplitude A=0.6. The encoder
successfully extracts enough info from 8 consecutive `q` values to seed a
meaningful integration.

**`z[1]` is not `p`.** It's a linear-ish combination of `q` and `p`. The
`latent_vs_p` scatter forms a *loop*, not a line: if `z[1]` were purely a
function of `p` the points would collapse to a curve. The model chose some
rotated coordinate system; the fixed decoder only constrains `z[0] = q`, so
`z[1]` is free, and "rotated p" is a perfectly valid choice from the
optimizer's point of view. Visible directly in `latent_orbits.png` as the
shear between the gray (true) ellipse and the colored (latent) one.

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
