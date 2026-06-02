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

### Pendulum branch — Stage 0

- Same infra carries over verbatim once `true_state` is replaced by a per-amplitude
  numerical LUT — the autonomous shared-local-grid batching trick still applies.
- Trained on a single amplitude A=1.0, the network learns to rotate *everything* at
  A=1.0's frequency. Unseen amplitudes all oscillate at the wrong period — hidden in the
  phase portrait, obvious in `angle_vs_time.png`.
- Training on three amplitudes `[0.3, 1.0, 1.5]` is enough for the network to internalize
  the nonlinear period-vs-amplitude relationship. The unseen A=0.6 interpolates with
  rollout MSE ~1e-4 — three orders of magnitude better than the single-amp run.
- So "interpolation in amplitude works" generalizes from the sine setup: even when the
  thing being interpolated is a genuinely nonlinear function (period of the pendulum),
  three well-chosen training points pin it down across the interval.
