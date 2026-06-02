# Stage 0 — Baseline pendulum (full observation)

Neural ODE on the simple pendulum (`dq/dt = p`, `dp/dt = -sin q`) with both
`q` and `p` directly observed. MLP vector-field, irregular short-window
training, multi-amplitude data.

## Run

```bash
cd stages/00_baseline_pendulum
uv run python train.py                                    # defaults
uv run python train.py 'data.amplitudes=[0.3,1.0,1.5]'    # multi-amp
```

Outputs land in `logs/<date>/<time>/`:
- `pendulum_neural_ode.png` — phase portrait
- `angle_vs_time.png` — q(t), exposes period error
- `law_evolution.png` — learned vs true vector field over training

## Headline finding

Single amplitude (A=1.0): all unseen amplitudes orbit at the wrong period —
the network learned "use A=1.0's rotation rate everywhere." Three amplitudes
[0.3, 1.0, 1.5] is enough to pin the nonlinear period-vs-amplitude curve
across the interval; interior interpolation (A=0.6) drops to ~1e-4 rollout
MSE. Exterior extrapolation untested in this stage.
