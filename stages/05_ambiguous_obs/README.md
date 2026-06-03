# Stage 5 — Ambiguous observation map (`o = q²`)

The single change vs Stage 4 is the **observation function**: instead of
observing `q + noise`, the encoder receives `o = q² + noise`. This is a
globally two-to-one map — for any positive `o`, true `q` could have been
`+√o` or `−√o` — so the posterior over `z[0] = q(t₀)` is genuinely
**bimodal**.

Architecture is otherwise identical to Stage 4 (GRU encoder, variational
posterior, latent ODE, fixed decoder `q = z[0]`). The reconstruction loss
now compares `o_pred = q_pred²` to `o_obs`, since `q` is no longer
directly observable.

## Run

```bash
cd stages/05_ambiguous_obs
uv run python train.py
```

Outputs: same six figures as Stage 4, plus per-step logging of two metrics:

- **q-MSE**: sign-sensitive — compares `q_pred(t)` to true `q(t)`.
- **o-MSE**: sign-agnostic — compares `q_pred(t)²` to true `q(t)²`. This is the right "did you reconstruct the observable?" metric.

## Headline finding: the diagonal Gaussian fails predictably

Best model selected by o-MSE (sign-agnostic). Final metrics:

| A | o-MSE | q-MSE | ratio q/o |
|---|---|---|---|
| 0.3 (train) | 2.7e-3 | 4.2e-2 | 16× |
| 0.6 (UNSEEN) | 5.2e-2 | 4.3e-1 | 8× |
| 1.0 (train) | 1.1e-1 | 8.2e-1 | 7× |
| 1.5 (train) | 5.7e-1 | 1.7e+0 | 3× |

q-MSE is **uniformly an order of magnitude worse than o-MSE**. The model is
roughly tracking the observable (the squared angle) but failing badly on
the underlying angle itself — exactly the sign-ambiguity failure we
predicted.

## Two distinct failure modes, side by side

`posterior_samples.png` reveals what's actually happening:

- **Low amplitudes (A=0.3, A=0.6) → mode averaging.** The 32 sampled
  posterior rollouts spread widely (e.g., −0.75 to +0.4 at A=0.3) with
  sample mean near 0. The Gaussian's mean settled halfway between the two
  modes and σ inflated to span them. Honest about uncertainty, useless for
  prediction.

- **High amplitudes (A=1.0, A=1.5) → mode collapse.** The samples cluster
  tightly around a single value (around -0.6 at A=1.0, -1.0 at A=1.5).
  The encoder committed to one of the two branches (in this run, the
  negative one, presumably from initialization bias) and the variance
  head collapsed. No posterior uncertainty at all about which mode.

So the model exhibits **both pathologies** in the same run, in different
regimes of the input. The diagonal Gaussian can do one or the other but
not represent the actual bimodal posterior. The optimizer picks whichever
mode-handling tactic gives the lower ELBO per amplitude.

## The other diagnostic figures

Most of them look wrecked, which is correct given the model has given up
on tracking the dynamics:

- `q_vs_time.png` — predictions are **flat near a constant** at each
  amplitude (no oscillation). True q oscillates between ±A. The model
  reproduces neither the amplitude nor the period of q.
- `latent_basis_fit.png` — R² **negative across all amplitudes**.
  `z[1]` is no longer a clean linear function of `(q, p)` — the encoder
  has lost any meaningful basis structure.
- `latent_vs_p.png`, `latent_orbits.png` — show scattered, structureless
  data, consistent with the encoder having given up on representing the
  dynamics in latent space.
- `calibration.png` — predicted std is much larger at low amplitudes
  (mode averaging) than high (mode collapse), as expected.

## Why the model goes flat instead of oscillating

A subtle point worth noting: even at the "mode-collapsed" amplitudes
(A=1.0, A=1.5), the predictions don't oscillate — they're roughly flat
near a fixed value. Why?

With `q²` observations, the encoder also can't recover the sign of
`p = dq/dt`. (It can recover `|p|` from the rate of change of `q²`, but
not the direction.) So even after committing to one `q` branch, the
encoder has no idea whether the pendulum is swinging *into* that
position or *out of* it. The cleanest single-state z₀ is therefore
something like `(q*, p ≈ 0)` — a momentum-free state at the chosen
q. The ODE then propagates that from rest, and since the encoder
chose a low-energy point, the resulting rollout barely moves.

That's why predictions are flat: the model converged to "release from
rest at the most likely q I can infer," which has near-zero velocity by
construction.

## What this tells us about the framework

This is the cleanest possible demonstration that **variational inference
is bounded by the expressiveness of the posterior family**. A single
diagonal Gaussian — the simplest variational choice — fundamentally
cannot represent bimodal uncertainty.

Fixes that *would* work, none of which we're implementing here:

- **Mixture of Gaussians posterior**: outputs `(μ_k, σ_k, π_k)` for each
  of K components. K=2 would handle the q² ambiguity exactly.
- **Normalizing flow posterior**: replace the Gaussian with an expressive
  parameterized family.
- **Particle filter / importance sampling**: maintain a set of weighted
  samples instead of a parametric posterior.

Real Latent ODE applications often live in regimes where unimodal
posteriors are fine (sensor noise is small relative to the dynamics'
distinguishability). When they're not, the failure mode looks exactly
like Stage 5.

## What this sets up

Stage 5 is the last "vector input" stage. Next is the **perception
ladder**: replace the scalar observation with a 1D heatmap (Stage 6a),
then a 2D image (6b), then a cartoon pendulum (6c, 6d). The encoder
swaps from a GRU over scalars to a CNN over images, but the variational
+ ODE backbone stays the same.
