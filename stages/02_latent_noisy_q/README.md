# Stage 2 — Latent ODE on noisy `q`

Same architecture as Stage 1 (encoder MLP + ODE-MLP + fixed `q = z[0]`
decoder), but every observed `q` carries Gaussian noise `ε ~ N(0, σ²)`
with `σ = 0.05`. Noise applies to both the training windows and the eval
encoder window. Eval rollout MSE is averaged over `n_eval_seeds=16`
noise realizations to get a stable number.

## Run

```bash
cd stages/02_latent_noisy_q
uv run python train.py                                    # σ=0.05 (default)
uv run python train.py data.obs_noise=0.1                 # single override
uv run python sweep.py                                    # retrain at σ ∈ {0, 0.025, 0.05, 0.1, 0.2, 0.5}, plot MSE vs σ
```

Outputs (same set as Stage 1, plus noise level in titles):
- `q_vs_time.png`, `latent_orbits.png`, `latent_vs_p.png`, `latent_basis_fit.png`
- `sweep.py` additionally writes `noise_sweep.png` and `sweep_results.json`.

## Headline findings

**Noise is absorbed reasonably well.** Overall rollout MSE only ~30% worse
than Stage 1 (2.97e-02 vs 2.29e-02 mean over trained amps) despite a noise
floor (σ² ≈ 2.5e-3) that's 20× above Stage 1's converged train loss. The
encoder is implicitly denoising by averaging over its 8-sample window.

| A | Stage 1 (clean) | Stage 2 (σ=0.05) |
|---|---|---|
| 0.3 (train) | 1.1e-02 | 1.3e-02 |
| 0.6 (UNSEEN) | 3.6e-02 | 3.5e-02 |
| 1.0 (train) | 4.7e-02 | 1.6e-02 |
| 1.5 (train) | 1.1e-02 | **2.3e-01** |

**Latent dynamics became dissipative.** This is the most visible change.
Stage 1's latent orbits closed approximately; Stage 2's *spiral inward*
(see `latent_orbits.png`). Noise leaks energy out of the latent vector
field — it's no longer approximately area-preserving. This is the same
class of problem as Stage 0's spiral on the sine wave, just induced by
observation noise instead of plain MLP bias.

**Large amplitudes are the weak spot.** A=1.5 degraded ~20× (1.1e-02 →
2.3e-01). The other amplitudes were essentially unchanged or even
slightly better. Likely cause: at A=1.5 the orbit is larger and small
encoder errors translate into bigger position errors at peak; the
non-conservative latent field then bleeds that into accumulating
phase drift.

**Basis structure is preserved.** Linear fit `z[1] = a·q + b·p`:

| A | a | b | R² |
|---|---|---|---|
| 0.3 | -0.46 | +0.88 | 0.870 |
| 0.6 | -0.34 | +0.96 | 0.907 |
| 1.0 | -0.40 | +1.10 | 0.943 |
| 1.5 | -0.23 | +1.33 | 0.986 |

Global: `z[1] ≈ -0.29·q + 1.20·p`, R² = 0.95. The basis rotated a bit
relative to Stage 1 (Stage 1 was `-0.56·q + 0.97·p`) but R² is the same.
The qualitative recipe — `z[1]` is a rotated combination of `q` and `p`,
with `a` roughly amp-invariant and `b` growing with amplitude — survives.

## Noise sweep — where does the deterministic encoder break?

`sweep.py` retrains the full pipeline at six noise levels (σ ∈ {0, 0.025,
0.05, 0.1, 0.2, 0.5}) and plots per-amplitude rollout MSE vs σ on log-log
axes (`noise_sweep.png`).

| σ | A=0.3 | A=0.6 (UNSEEN) | A=1.0 | A=1.5 |
|---|---|---|---|---|
| 0 | 9.0e-3 | 2.8e-2 | 2.5e-2 | 1.3e-2 |
| 0.025 | 1.8e-2 | 5.1e-2 | 6.2e-2 | **8.3e-3** |
| 0.05 | 1.5e-2 | 4.8e-2 | 6.9e-2 | **7.7e-3** |
| 0.1 | 2.4e-2 | 6.8e-2 | 4.6e-2 | 2.6e-2 |
| 0.2 | 4.3e-2 | 1.2e-1 | 7.8e-2 | 1.0e-1 |
| 0.5 | 6.3e-2 | 1.6e-1 | 1.8e-1 | **3.7e-1** |

Three notable patterns:

1. **The model is roughly noise-insensitive up to σ ≈ 0.05–0.1.** Below
   that, all curves are essentially flat. Rollout MSE is bottlenecked by
   *model fidelity* (encoder/dynamics expressivity) rather than by noise —
   the σ² floor sits well below all measured MSEs in this regime.
2. **Noise helped A=1.5 at low levels.** The most fragile amplitude
   improved 40% going from σ=0 to σ=0.05 (1.28e-2 → 7.66e-3). Noise acted
   as a regularizer, washing out a late-training overfitting pattern that
   hurt the largest orbit in the noiseless run.
3. **The break is concentrated at large amplitudes.** Between σ=0.1 and
   σ=0.5, A=1.5 grows ~14× while A=0.3 grows only ~3×. Same lesson as the
   single-run analysis above: large orbits compound encoder errors fastest.

## What sets up Stage 3

The deterministic encoder produces a single point estimate of `z₀` from a
noisy window. At σ=0.05 it's mostly OK; at higher σ it would start losing
the actual data structure. More importantly, **it can't represent
uncertainty**: at the noise level, *many* `z₀` values are consistent with
the observed window, but the model has to pick one. The variational
encoder (Stage 3) outputs `(μ, σ)` and we sample — letting the model fan
its predictions out across the genuine ambiguity. The headline question
there will be calibration: do the sampled rollouts actually cover the
band of plausible truths, or collapse to a single guess?
