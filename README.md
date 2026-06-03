# Neural ODEs — experiments

A sandbox for building intuition with neural ODEs, organized as a staged
curriculum from full-state observation to learning physics from cartoon
video. See **[ROADMAP.md](ROADMAP.md)** for the full plan.

## Setup (uv)

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Layout

Each logical stage is a self-contained directory under `stages/`. You can
`cd` into a stage, run it, and read it without touching any other stage.
Per-stage READMEs capture what changed and what we learned.

```
stages/
  00_baseline_pendulum/    full-state pendulum, MLP vector-field ODE
  01_latent_drop_p/        ... (next up)
  ...
```

Runs land in `<stage>/logs/<date>/<time>/` (gitignored). Hydra writes the
resolved config and a per-run log there alongside the figures.

## Run

```bash
cd stages/00_baseline_pendulum
uv run python train.py                                    # defaults
uv run python train.py 'data.amplitudes=[0.3,1.0,1.5]'    # CLI overrides
```

## Branches

- `main` — the original harmonic-oscillator sandbox (`train_sine.py`). Kept
  as a historical reference.
- `hamiltonian` — same sine system, ODE function replaced by an HNN
  (energy as a scalar field, vector field as `J·∇H`).
- `pendulum` — the curriculum lives here: nonlinear pendulum baseline
  (Stage 0) and everything that follows.

## What we've learned so far

### Sine-wave branch (`main`, `hamiltonian`)

- A sine wave needs a **2D** phase state `[x, v]`; a 1D autonomous ODE can't oscillate.
- Long-horizon training collapses to the mean → train on **short windows**.
- **Irregular timestamps** are handled natively and even help (augmentation over `dt`).
- The model learns the **local law** (vector field); the solver turns it into trajectories.
- Trained on one amplitude, unseen radii orbit at the **wrong frequency** (visible in
  `position_vs_time.png`, hidden in the phase portrait). Training on multiple radii fixes
  **interior** interpolation but not exterior extrapolation.
- An **HNN** inductive bias (learn scalar H, derive field as J·∇H) hard-enforces energy
  conservation, fixing the outward spiral — but H is only pinned near data, so it doesn't
  rescue outward extrapolation either.

### Pendulum branch — executive summary

See per-stage READMEs under `stages/` for details.

- **Stage 0** — baseline neural ODE on the full-state pendulum. Three trained amplitudes are enough for the model to internalize the nonlinear period-vs-amplitude relationship; interior amplitudes interpolate well.
- **Stage 1** — drop `p`, infer it from a window of `q` observations via a small MLP encoder. Works at ~3 orders of magnitude worse MSE than Stage 0. `z[1]` is approximately a rotated `p`, not literally `p`.
- **Stage 2** — add noise to `q`. Encoder absorbs it reasonably well up to σ ≈ 0.1, then degrades. Noise even *helps* the most fragile amplitude at low levels (acts as regularization).
- **Stage 3** — variational encoder (proper Latent ODE). Surprise: variational training improved *point* predictions ~5× — KL regularization smooths the latent space, not just adds uncertainty quantification. Posterior calibration is "honest but coarse."
- **Stage 4** — sparse irregular observations with a GRU encoder. Works *better* than Stage 3 — sparse-but-wide observation coverage beats dense-but-narrow. Latent basis fit cleanest yet (R² ≈ 0.99).
- **Stage 5** — ambiguous observation map (`o = q²`). Clean demonstration that diagonal-Gaussian posteriors can't represent bimodal evidence: mode averaging at low amplitudes, mode collapse at high, flat predictions in both regimes.

**Threads running through the curriculum:**
- Linear-fit R² of `z[1]` against `(q, p)` improved monotonically across stages — adding regularization (KL) and sequential structure (GRU) both pulled the encoder toward physically natural latent coordinates.
- Variational machinery does more than uncertainty: KL acts as a structural regularizer that improves point predictions and generalization.
- Sample-efficient learning of nonlinear dynamics works when (a) the observation map is invertible, (b) the encoder can recover the state's full dimensionality, and (c) the posterior family can express the actual uncertainty. Stage 5 breaks (c); the perception ladder ahead is where (a) and (b) get genuinely difficult.
