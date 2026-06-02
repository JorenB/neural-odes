# Roadmap — Latent neural ODEs, from vectors to pixels

A staged progression toward the headline goal:

> A neural ODE that learns pendulum physics — including amplitude-dependent
> period — from cartoon video alone, with no access to the underlying state.

## Organizing principle

Each "logical stage" is a self-contained directory under `stages/`. The
discipline is that you can `cd` into any stage's folder, read the script,
and understand what that stage does *without* reading any other stage. We
deliberately duplicate small helpers across stages rather than factor them
into a `common/` module, so each stage stays a complete pedagogical artifact
you can look back on later.

```
stages/
  00_baseline_pendulum/
    train.py
    config.yaml
    README.md
    logs/             # gitignored, per-run figures land here
  01_latent_drop_p/
    ...
```

Each stage's README captures: what changes vs. the previous stage, the
question it answers, the headline diagnostic figure, and what we found.

## Stages

| # | Title | Observation | Encoder | True latent dim | Headline question |
|---|---|---|---|---|---|
| 0 | Baseline pendulum | `(q, p)` direct | — | 2 | Can the MLP-ODE learn nonlinear dynamics? Period-vs-amplitude from data? |
| 1 | Drop `p` | `q(t)` dense, clean | small MLP over a window of `q`s | 2 | Can the encoder back out `p` from a temporal context? |
| 2 | Noisy `q` | `q(t)` + Gaussian noise | same as 1 | 2 | Does the deterministic encoder gracefully absorb noise, or break? |
| 3 | Variational latent ODE | `q(t)` + noise | same as 2, but outputs `(μ, σ)` | 2 | Quantified uncertainty in `z₀`; does sampling cover the true ambiguity? |
| 4 | Sparse irregular obs | a handful of `(t, q)` pairs at random times | GRU/RNN over sequence | 2 | The original Chen–Rubanova setup; can the model handle real-data sampling? |
| 5 | Ambiguous obs map | `o = sin(q)` or `o = q²` | variational, sequence-aware | 2 | When `h` is many-to-one, does the model correctly represent the symmetry? |
| 6a | 1D heatmap render | length-64 Gaussian bump at position `q` | small MLP | 2 | Nonlinear high-dim obs without CNN cost — debug the perception piece. |
| 6b | Breathing circle (image) | 32×32 image, radius oscillates | CNN | 1 | First end-to-end CNN + ODE + CNN. Does the pipeline compose at all? |
| 6c | Cartoon pendulum, single amp | 32×32 image, rod-with-bob | CNN over frame window | 2 | Recover 2D dynamics from images; encoder must use temporal context for `p`. |
| 6d | Cartoon pendulum, multi-amp | same, multiple amplitudes | CNN over frame window | 2 | The climax: does period-vs-amplitude generalization survive the pixel regime? |

## Per-stage detail

### Stage 1 — drop `p`

We keep the trained pendulum dynamics from Stage 0 but only observe `q(t)`
on the same regular dense grid. The model now has three pieces:

1. **Encoder** `g_φ: q[t₀ : t₀+k] → z₀ ∈ ℝ²` — small MLP that looks at a
   short window of recent `q` values.
2. **ODE** `f_θ: z → dz/dt` — same MLP as before.
3. **Decoder** `h_ψ: z → q` — linear projection (or fixed: `q = z[0]`).

Train end-to-end: encode `z₀`, integrate, decode back to predicted `q(t)`,
compare to true `q(t)`.

**Headline diagnostic:** plot the *latent* trajectory `z(t)` against the true
`(q, p)` trajectory (up to a linear change of variables, since the network
can use any 2D coordinates it likes). Does the model recover something
isomorphic to phase space?

**Expected outcome:** modest degradation vs Stage 0 — the encoder needs to
back out velocity from a temporal window, which costs accuracy. Worth
measuring: how does rollout MSE depend on the encoder window length?

### Stage 2 — noisy `q`

Add `σ ~ 0.05` Gaussian noise to observed `q(t)`. Same architecture. The
deterministic encoder now has to denoise on top of inferring `p`.

**Headline diagnostic:** rollout MSE as a function of noise level. At what
σ does the model break, and *how* does it break — gracefully averaging, or
fitting noise?

**Why we care:** sets up the case for going variational. If high-σ
deterministic training collapses to a single mean guess and ignores
ambiguity in `z₀`, the variational version (Stage 3) should help.

### Stage 3 — variational latent ODE (proper Latent ODE)

Encoder now outputs `(μ_φ, σ_φ)`. Sample `z₀ ~ N(μ_φ, σ_φ²)` via the
reparameterization trick. Loss = reconstruction + KL to a prior
`N(0, I)`. Now we have a **generative model**: given an observation
window we can sample many trajectories consistent with it.

**Headline diagnostic:** at moderate noise, plot 50 sampled rollouts from
the posterior — do they fan out into the genuine band of consistent
trajectories, or collapse to one curve? Calibration check: does sample
spread track the actual residual?

### Stage 4 — sparse irregular observations

Drop the "dense `q(t)`" assumption. Each trajectory comes with ~10 random
`(t, q)` pairs. A fixed-window MLP encoder no longer makes sense; replace
with a **GRU run backwards over the observation sequence** (the
Chen–Rubanova trick — also lets the GRU handle irregular spacings via
explicit `Δt` input).

This is the canonical Latent ODE architecture from the 2019 paper.

**Headline diagnostic:** how does reconstruction quality scale with the
number of observations per trajectory? Where's the floor below which the
model can't recover dynamics?

### Stage 5 — ambiguous observation map

Two sub-experiments worth trying:

- `o = sin(q)`: folds the orbit; near `q = ±π/2` the map is degenerate.
- `o = q²`: loses the sign of `q` entirely.

**Headline diagnostic:** for `o = q²`, the posterior over `z₀` should be
*bimodal* (q vs −q). Does the variational encoder represent this, or does
it collapse to one mode? (Likely collapses — a single Gaussian can't
represent bimodality. Discussion piece: what would fix this? Mixture
posterior? Normalizing flow? Or just live with collapse and discuss it.)

### Stage 6a — 1D heatmap render

Render `q` as a Gaussian bump of width σ in a length-64 array,
`o[i] = exp(-(i/64 · 2π − q)² / (2σ²))`. Now observation ∈ ℝ⁶⁴, nonlinear
in `q`, but small enough to debug fast.

**Why this rung exists:** it's the cheapest test of the
"high-dim nonlinear obs + ODE" combination before paying CNN training
cost. Encoder is a small MLP `ℝ⁶⁴ → ℝ²`.

### Stage 6b — breathing circle

Render a circle on a 32×32 canvas with radius `r(t) = r₀ + A·cos(t)`.
True latent dim is **1**. CNN encoder + small ODE + CNN decoder.

**Headline diagnostic:** plot the learned latent `z(t)` — it should
trace a 1D sinusoid (up to scaling). Confirms the CNN + ODE pipeline
composes end-to-end with negligible physics in the way.

### Stage 6c — cartoon pendulum, single amplitude

32×32 image of a rod-with-bob at angle `q(t)`. Same CNN encoder/decoder
as 6b but now true latent dim is 2 — and crucially, **angular velocity
is not visible in a single frame**. The encoder must look at a window of
consecutive frames to recover `p`.

**Headline diagnostic:** train at single amplitude, compare rollout
trajectories in latent space against the true `(q, p)` orbit (linear
mapping to be found post-hoc).

### Stage 6d — cartoon pendulum, multi-amplitude

Same render, but training data covers multiple amplitudes
(e.g. [0.3, 1.0, 1.5]). The capstone question:

> Does the network, with **no access to q or p**, learn that the period
> depends on amplitude — purely from cartoon video?

**Headline diagnostic:** the Stage-0 `angle_vs_time.png` equivalent in
the pixel regime. Decode rollouts to image sequences, recover q(t) from
the predicted frames, and overlay against truth at each amplitude.
Unseen interior amplitudes should still hit the right period.

If 6d works, we've climbed the whole ladder. If it fails, it's because
either (i) the encoder can't extract amplitude reliably, or (ii) the
latent dynamics overfit one rotation rate — both are informative
failures.

## Notes

- **Compute.** Stages 1–5 should run in seconds-to-minutes on CPU. Stages
  6b–d (CNNs on small images) may push toward minutes per run; still
  CPU-tractable but a GPU helps.
- **Branching.** No strong opinion yet — we can stay on `pendulum` and let
  `stages/` be the organizing axis, or branch off (`latent`, `pixels`) when
  the work meaningfully forks. Decide as we go.
- **Stage README contract.** When a stage finishes, its README captures
  one paragraph on what it does, one paragraph on findings, and a pointer
  to the best run's figure(s).
