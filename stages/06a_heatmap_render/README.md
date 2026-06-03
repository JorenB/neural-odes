# Stage 6a — First rung of the perception ladder: 1D heatmap observation

The encoder no longer sees a scalar `q`. Each observation is a length-64
Gaussian bump rendered at position `q`:

```
o[i] = exp(-(positions[i] - q)² / (2 σ²))    for i = 0..63
```

with `positions` evenly spanning `[-2.5, +2.5]` in q-space and bump
width `σ_pix = 0.15`. The encoder GRU's input vector per step is
`(t_i, o_i)` of dim `1 + 64 = 65`. Everything else (variational
posterior, KL, ODE, fixed `q = z[0]` decoder, sparse irregular
sampling, 10 obs per window over 6 s) carries over from Stage 4.

This stage is the cheapest first test of the "high-dim nonlinear
observation → low-dim ODE latent" pipeline. We avoid CNN cost by
keeping the observation 1D and letting the GRU handle the input
directly. Stage 6b is where the CNN enters.

## Run

```bash
cd stages/06a_heatmap_render
uv run python train.py
```

Outputs: same six figures as Stage 4, plus a new one:

- `encoder_input_example.png` — visualizes a typical encoder window: per
  amplitude, a 10×64 image showing 10 noisy heatmap observations stacked
  vertically, with red dots marking the true `q` at each observation
  time. Sanity check that the input looks the way we think it should.

## Headline findings

**Works as well as Stage 4 once the loss is properly scaled.**

| A | Stage 4 (scalar q) | Stage 6a (heatmap obs) |
|---|---|---|
| 0.3 (train) | 1.4e-3 | 3.7e-3 |
| 0.6 (UNSEEN) | 2.4e-3 | 1.3e-2 |
| 1.0 (train) | 2.3e-3 | 1.2e-3 |
| 1.5 (train) | 6.2e-3 | 2.3e-3 |
| best mean | 3.3e-3 | 1.7e-3 |

Slight degradation on A=0.6 (the held-out amplitude), slight improvement
elsewhere. Net: roughly the same regime as Stage 4. The architecture
*does* compose end-to-end with a 64-dim nonlinear observation. 

**Linear basis stays clean.** Linear fit `z[1] = a·q + b·p`:

| | global a | global b | global R² |
|---|---|---|---|
| Stage 4 (scalar q) | -0.02 | -1.39 | 0.99 |
| **Stage 6a (heatmap)** | **+0.03** | **+1.90** | **0.98** |

`a` essentially zero, |b| ~ 2, R² ~ 0.98. The encoder picked a basis
where `z[1]` is again essentially a scaled `p` (this time with positive
sign by random init). Identical *qualitative* structure to Stage 4 —
the basis interpretation survives going to high-dimensional observations.

## The loss-scaling pitfall (worth flagging)

First training run was a **catastrophic failure**: posterior collapse,
predictions converged to a flat line at q ≈ -0.2 regardless of input.
Quick diagnosis: KL dropped to ~0.1 (vs Stage 3-4's ~4), meaning the
encoder's posterior had drifted to the prior and was ignoring its input.

Root cause was the recon-loss scale. The original `mean((o_pred -
o_target)²)` averaged over `(time × batch × 64 pixels)`. Most of the
64 pixels are zero on both sides (the bumps are narrow), so per-pixel
MSE was artificially low — the gradient was dominated by the many
"match-the-zeros" pixels and barely cared about the few
"match-the-bump" pixels. Net: weak recon signal, KL won by default,
the encoder gave up.

Fix: **sum** the squared error over pixels (then mean over time and
batch). One-line change:

```python
recon = ((o_pred - o_target) ** 2).sum(dim=-1).mean()
```

This is also the standard ELBO formulation (the Gaussian log-likelihood
sums over observation dimensions, not averages). After the fix, KL
recovered to ~8 (well above the prior), the encoder produced informative
posteriors, and the model converged to results comparable to Stage 4.

Worth remembering as a general principle: **when the observation
dimension grows, the recon loss should scale with it**, otherwise it
gets washed out by the KL term and the variational posterior collapses.
For Stage 6b/c/d (32×32 images = 1024 pixels) this will matter even more.

## What this sets up

The pipeline now works on high-dim nonlinear observations. Stage 6b
swaps the 1D heatmap for a 2D image (32×32, the breathing circle) and
the GRU's per-step input MLP for a small CNN. The variational +
heatmap-loss-scaling lessons from this stage carry directly over.
