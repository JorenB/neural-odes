# Neural ODEs — experiments

Learning the dynamics of a 1D oscillating particle with a neural ODE, as a
sandbox for building intuition.

## Setup (uv)

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Run

```bash
uv run python train_sine.py                          # defaults (config.yaml)
uv run python train_sine.py data.amplitudes=[1.0,2.0]   # CLI overrides
```

Each run writes a timestamped dir under `logs/` (Hydra) containing:
- `sine_neural_ode.png` — phase portrait (orbit geometry)
- `position_vs_time.png` — x(t) vs t (exposes timing/frequency error)
- `law_evolution.png` — learned vector field over training
- `.hydra/` (resolved config + overrides) and `train_sine.log`

## What we've learned so far

- A sine wave needs a **2D** phase state `[x, v]`; a 1D autonomous ODE can't oscillate.
- Long-horizon training collapses to the mean → train on **short windows**.
- **Irregular timestamps** are handled natively and even help (augmentation over `dt`).
- The model learns the **local law** (vector field); the solver turns it into trajectories.
- It does **not** generalize off-data: trained on one amplitude, unseen radii orbit at the
  **wrong frequency** (visible in `position_vs_time.png`, hidden in the phase portrait).
  Training on multiple radii fixes the **interior** (interpolation) but not the exterior.
