# Neural ODEs — experiments

A sandbox for building intuition with neural ODEs. Branches:

- `main` — harmonic oscillator (sine wave). The barebones starting point.
- `hamiltonian` — same system, but the ODE function is an HNN (energy as a
  scalar field, vector field derived symplectically).
- `pendulum` — nonlinear pendulum (`dq/dt = p`, `dp/dt = -sin q`). Period
  depends on amplitude — the new question is whether the network picks that up.

## Setup (uv)

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Run

```bash
uv run python train_pendulum.py                                    # defaults (config.yaml)
uv run python train_pendulum.py data.amplitudes=[0.6,1.2]          # CLI overrides
```

Each run writes a timestamped dir under `logs/` (Hydra) containing:
- `pendulum_neural_ode.png` — phase portrait (orbit geometry)
- `angle_vs_time.png` — q(t) vs t (exposes amplitude-dependent period)
- `law_evolution.png` — learned vector field over training
- `.hydra/` (resolved config + overrides) and `train_pendulum.log`

## What we've learned so far

### Sine-wave branch
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

### Pendulum branch
- Same infra carries over verbatim once `true_state` is replaced by a per-amplitude
  numerical LUT — the autonomous shared-local-grid batching trick still applies.
- Trained on a single amplitude A=1.0, the network learns to rotate *everything* at
  A=1.0's frequency. Unseen amplitudes all oscillate at the wrong period (small ones
  too fast, large ones too slow vs truth) — hidden in the phase portrait, obvious in
  `angle_vs_time.png`.
- Training on three amplitudes `[0.3, 1.0, 1.5]` is enough for the network to internalize
  the **nonlinear period-vs-amplitude relationship**. The unseen A=0.6 interpolates with
  rollout MSE ~1e-4 — three orders of magnitude better than the single-amp run.
- So "interpolation in amplitude works" generalizes from the sine setup: even when the
  thing being interpolated is a genuinely nonlinear function (period of the pendulum),
  three well-chosen training points are enough to pin it down across the interval.
