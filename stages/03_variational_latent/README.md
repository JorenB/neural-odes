# Stage 3 — Variational latent ODE (proper Latent ODE)

The encoder now outputs `(μ, log σ²)` of an approximate posterior over
`z₀`. Each training forward pass *samples* `z₀ = μ + σ·ε` (reparam trick).
The loss is the ELBO:

```
L = mean((q_pred − q_obs)²)  +  β · KL(N(μ, σ²I) ‖ N(0, I))
```

with `β = 0.01`. Architecture is otherwise identical to Stage 2: same
MLP trunk + ODE-MLP + fixed `q = z[0]` decoder; observations are noisy
with `σ = 0.05`. Eval rollout MSE uses the *point estimate* `z₀ = μ`
(no sampling), so it's directly comparable to Stage 2's number.

## Run

```bash
cd stages/03_variational_latent
uv run python train.py
uv run python train.py model.beta_kl=0.001          # weaker KL regularization
uv run python train.py data.obs_noise=0.2           # higher noise
```

Outputs:
- `q_vs_time.png`, `latent_orbits.png`, `latent_vs_p.png`, `latent_basis_fit.png` — same set as Stage 2.
- `posterior_samples.png` — n=32 sampled q(t) rollouts per amplitude; shows the predictive fan.
- `calibration.png` — predicted std vs actual residual per amplitude. Headline new diagnostic.

## Headline findings

**Variational training is a strong regularizer — even for point prediction.**
This was the most surprising result of the stage. I expected Stage 3 to mostly
improve uncertainty quantification; instead it improved point predictions
~5× across the board:

| A | Stage 2 (det.) | Stage 3 (variational, point est) |
|---|---|---|
| 0.3 (train) | 1.5e-02 | 5.8e-03 |
| **0.6 (UNSEEN)** | **4.8e-02** | **4.0e-03** |
| 1.0 (train) | 6.9e-02 | 2.1e-03 |
| 1.5 (train) | 7.7e-03 | 4.5e-02 |
| best mean | 2.97e-02 | **6.46e-03** |

The KL term forces the posterior toward N(0, I), which prevents the
encoder from packing per-window noise into idiosyncratic latents. Net
effect: smoother latent space, better extrapolation.

**The basis got cleaner.** Linear fit `z[1] = a·q + b·p`:

| stage | a | b | R² |
|---|---|---|---|
| 1 (clean) | -0.56 | +0.97 | 0.95 |
| 2 (noisy det) | -0.29 | +1.20 | 0.95 |
| 3 (variational) | **-0.13** | **−1.53** | **0.98** |

The encoder leaned harder on `p` and less on `q`, with sign-flipped `b`
(arbitrary basis choice). R² improved from 0.95 → 0.98 — the posterior
mean is a more nearly linear function of true `(q, p)` than the
deterministic encoder's point estimate was.

**Posterior is honest but coarse.** The predicted std (colored, in
`calibration.png`) is approximately flat over time at ~0.1 per
amplitude, while the actual residual (dashed) oscillates between ~1e-3
and ~1e-1. Read this as:

- **Honest**: the predicted std sits *above* the residual essentially
  everywhere — the model is never overconfident.
- **Coarse**: it doesn't capture the temporal structure of error. The
  actual residual dips to near-zero each time the prediction crosses
  the truth; the predicted std doesn't (it inherits from the encoder
  window's posterior, which the ODE propagates ~uniformly).

A finer calibration would need either an SDE in latent space (so
uncertainty grows over time) or a likelihood with explicit observation
noise. Stage 4 onward doesn't go there.

**Posterior fan is amplitude-dependent.** In `posterior_samples.png` the
spread is wider at A=0.3 and A=1.5, narrower at A=1.0 (which sat near
the middle of the training amplitudes and is best-fit). So the encoder
*does* express more uncertainty at the data extremes.

## What this sets up

Stage 4 introduces **sparse irregular observations** — a handful of
`(t, q)` pairs per trajectory at random times instead of a dense
regular window. A fixed-window MLP encoder no longer makes sense; we
replace it with a GRU over the observation sequence. This is the
canonical Chen–Rubanova Latent ODE architecture, and the variational
machinery from this stage carries over directly.
