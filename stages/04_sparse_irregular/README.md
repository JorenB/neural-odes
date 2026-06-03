# Stage 4 — Sparse irregular observations + GRU encoder

The fixed-window MLP encoder from Stages 1–3 is replaced with a **GRU
that processes (t, q) pairs in reverse chronological order** (latest
first, earliest last). Per training sample, the encoder now sees
`N=10` observations at **uniformly random irregular times** in a
6-second window, instead of 8 dense regular observations in a 0.7-second
window. The ODE, decoder, KL term, and overall ELBO loss are unchanged
from Stage 3.

This is the canonical Chen-Rubanova Latent ODE setup, at least
architecturally — we still anchor z₀ at local time 0 (start of window)
and the GRU does *discrete* recurrent updates rather than running a
sub-ODE between observations (ODE-RNN would be the more principled
variant).

## Run

```bash
cd stages/04_sparse_irregular
uv run python train.py
uv run python train.py data.n_obs=5                 # sparser
uv run python train.py data.window_horizon=12.0     # wider window
```

Outputs: same six figures as Stage 3 (q_vs_time, latent_orbits,
latent_vs_p, latent_basis_fit, posterior_samples, calibration).

## Headline findings

**Sparse irregular observations work — better than Stage 3 on point
predictions.**

| A | Stage 3 (dense regular, 8 obs in 0.7 s) | Stage 4 (sparse irreg, 10 obs in 6 s) |
|---|---|---|
| 0.3 (train) | 5.8e-03 | **1.4e-03** |
| **0.6 (UNSEEN)** | 4.0e-03 | **2.4e-03** |
| 1.0 (train) | 2.1e-03 | 2.3e-03 |
| 1.5 (train) | 4.5e-02 | **6.2e-03** |
| best mean | 6.5e-03 | **3.3e-03** |

A=1.5 improved 7× — the amp that has been the weakest spot since Stage 2.
This isn't really a fair head-to-head with Stage 3 (different observation
horizons), but the result tells us the GRU encoder doesn't *cost* us
anything; it can deliver competitive or better point predictions on a
much sparser, more realistic input regime.

**The basis got even cleaner.** Linear fit `z[1] = a·q + b·p`:

| stage | global a | global b | global R² |
|---|---|---|---|
| 1 (clean, MLP) | -0.56 | +0.97 | 0.95 |
| 2 (noisy, MLP) | -0.29 | +1.20 | 0.95 |
| 3 (variational, MLP) | -0.13 | -1.53 | 0.98 |
| **4 (variational, GRU)** | **-0.02** | **-1.39** | **0.99** |

`a` is essentially zero — the GRU + variational combination picked a
basis where `z[1]` is *almost purely a (scaled, sign-flipped) `p`*, with
negligible `q` mixing. R² rose to 0.99. The probabilistic + sequential
structure both seem to pull the encoder toward a coordinate frame
aligned with the system's natural axes.

**The fix that mattered.** First-pass eval was catastrophically bad
(best MSE ~5e-1) because of a subtle bug: in training the encoder saw
windows starting at a *random* phase along the orbit, but eval scored
against the trajectory starting from `(q=A, p=0)`. The model was
correctly encoding the random-phase window it saw, but we were comparing
to a different trajectory. Fixing eval to use `phase0 = 0` (the encoder
window aligned with the eval-truth trajectory's start) made the model
work as expected. Worth noting because the *training* recon was fine
the whole time — the bug only manifested at eval, which is easy to miss.

**Posterior calibration: same shape as Stage 3.** Predicted std stays
roughly flat over time at ~0.1, residual oscillates 1e-3 to 1e-1, std
sits above residual everywhere (honest but coarse). Sample fan in
`posterior_samples.png` is wider at A=0.3 and A=0.6 (where the encoder
has less signal-to-noise) and tighter at A=1.0/1.5.

## What I think this tells us

The GRU encoder isn't doing anything obviously magical — it has fewer
*total* observations than Stage 3 (10 vs 8 is roughly the same, and
density-wise the spacings are much sparser). But it has much **wider
temporal coverage** (6 s vs 0.7 s), which is genuinely more
information about the dynamics. Seeing the pendulum at scattered
moments across most of one period gives the encoder a much stronger
"this is the trajectory" signal than seeing 8 nearly-identical readings
in 0.7 seconds, even though the latter has finer local derivative info.

So the lesson is partly architectural (GRU lets us use irregular
sparse data at all) and partly informational (sparse-but-wide beats
dense-but-narrow, in this regime). For real-world data the
architectural piece is the one that matters; the informational
observation is a bonus.

## What this sets up

Stage 5 keeps the GRU encoder and adds an **ambiguous observation
map**: instead of `o = q + noise`, we observe `o = sin(q)` (folds the
orbit, degenerate near q=±π/2) or `o = q²` (loses sign entirely). The
question becomes: when the observation function is many-to-one, does
the variational posterior correctly represent the genuine ambiguity in
`z₀`, or does it collapse to one mode?
