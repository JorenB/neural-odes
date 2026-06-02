"""
Neural ODE for the simple (gravity) pendulum.

The story
---------
A pendulum (unit mass, length, gravity) obeys
    dq/dt = p,    dp/dt = -sin(q),
where q is angle from rest and p is angular momentum. Unlike the harmonic
oscillator we started from, this system is NONLINEAR -- the period depends on
the amplitude q_max. That's the headline feature for this experiment: can the
neural ODE pick up amplitude-dependent timing from data?

The model is still a flow in the 2D phase space z = [q, p]:
    dz/dt = f_theta(z),
and we train it so that integrating from z(0) reproduces the observed
trajectory. The training infrastructure is unchanged from the sine setup
(short irregular windows, shared local grid, best-model tracking, snapshots).

What's different from train_sine.py
-----------------------------------
- No closed-form solution. We integrate the TRUE pendulum equations once at
  high resolution per amplitude to build a lookup table, then interpolate to
  serve targets at arbitrary times. `PendulumTruth.state(A, t)` is the analog
  of the old `A * true_state(t)`.
- `t_max` is now an absolute duration (periods aren't 2*pi anymore).
- True field in `law_evolution.png` is `[p, -sin(q)]` (was `[v, -x]`).
- Plot labels: x->q, v->p, "circles"->"orbits". Orbits are no longer circles;
  amplitude scales q_max and p_max separately.

Config & runs
-------------
All hyperparameters live in config.yaml. Hydra creates a fresh timestamped
directory under logs/ for every run and we save all figures there. Override
anything from the CLI, e.g.:
    uv run python train_pendulum.py data.amplitudes=[0.6,1.2] train.steps=4000
"""

import copy
import logging
import math
import os

import hydra
import matplotlib
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torchdiffeq import odeint

matplotlib.use("Agg")                       # save to file, no display needed
import matplotlib.pyplot as plt

# Hydra routes this logger to both the console and <run_dir>/train_pendulum.log.
log = logging.getLogger(__name__)


def true_pendulum_field(t, z):
    """Ground-truth vector field: dz/dt = [p, -sin(q)]. Autonomous: ignores t."""
    q, p = z[..., 0:1], z[..., 1:2]
    return torch.cat([p, -torch.sin(q)], dim=-1)


class PendulumTruth:
    """Precomputed high-res trajectories for each requested amplitude.

    No closed form -> we integrate the true field once over [0, t_max] per
    amplitude, then linearly interpolate to evaluate at arbitrary times.
    Used both to make training targets and to draw ground truth in figures.
    Initial condition for amplitude A is (q=A, p=0): "released from rest at
    angle A," which sets H = 1 - cos(A) and parameterizes the orbit by q_max.
    """

    # Round amplitudes to this many decimals when keying the LUT, so that
    # 0.3 (Python float) and 0.3 round-tripped through torch.float32 hash the
    # same way.
    _KEY_PRECISION = 5

    @classmethod
    def _key(cls, A):
        return round(float(A), cls._KEY_PRECISION)

    def __init__(self, amplitudes, t_max, n_lut, method):
        self.t_max = float(t_max)
        self.t_lut = torch.linspace(0.0, self.t_max, n_lut)
        self.trajs = {}
        for A in amplitudes:
            z0 = torch.tensor([float(A), 0.0])
            with torch.no_grad():
                traj = odeint(true_pendulum_field, z0, self.t_lut, method=method)
            self.trajs[self._key(A)] = traj                          # (n_lut, 2)

    def state(self, A_batch, times):
        """Look up true state.

        A_batch: (B,) float tensor of amplitudes (each must match a LUT key).
        times:   (T,) shared local grid OR (T, B) per-item absolute times.
        Returns: (T, B, 2), matching odeint's (time, batch, state) layout.
        """
        B = A_batch.shape[0]
        T = times.shape[0]
        t_TB = times[:, None].expand(T, B) if times.ndim == 1 else times

        t_clamped = t_TB.clamp(0.0, float(self.t_lut[-1]) - 1e-7)
        idx = torch.bucketize(t_clamped, self.t_lut) - 1
        idx = idx.clamp(0, len(self.t_lut) - 2)
        t0 = self.t_lut[idx]; t1 = self.t_lut[idx + 1]
        w = ((t_clamped - t0) / (t1 - t0)).unsqueeze(-1)             # (T, B, 1)

        out = torch.empty(T, B, 2)
        for b in range(B):                                           # B is small
            traj = self.trajs[self._key(A_batch[b].item())]
            i = idx[:, b]
            out[:, b] = torch.lerp(traj[i], traj[i + 1], w[:, b])
        return out


class ODEFunc(nn.Module):
    """Maps a state z -> its time derivative dz/dt. Autonomous: ignores t."""

    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),                # smooth nonlinearity -> smooth vector field
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, z):
        return self.net(z)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    out_dir = HydraConfig.get().runtime.output_dir   # this run's logs/ subdir
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    log.info("output dir: %s", out_dir)
    torch.manual_seed(cfg.seed)

    # --- 1. Ground-truth data ---------------------------------------------
    # Build LUTs for every amplitude we'll ever query (train OR test). This is
    # the pendulum analog of the closed-form sin/cos: one numerical integration
    # per orbit, cached, then interpolated at arbitrary times below.
    t_max = float(cfg.data.t_max)
    train_amps = list(cfg.data.amplitudes)
    test_amps = list(cfg.data.test_amplitudes)
    all_amps = sorted(set(train_amps) | set(test_amps))
    train_amps_t = torch.tensor(train_amps)
    dt_mean = t_max / cfg.data.n_obs                  # nominal sampling density
    method = cfg.train.solver
    truth = PendulumTruth(all_amps, t_max, cfg.data.n_lut, method)

    # --- 2. Model + optimizer ---------------------------------------------
    func = ODEFunc(hidden=cfg.model.hidden)
    optimizer = torch.optim.Adam(func.parameters(), lr=cfg.train.lr)
    scheduler = None
    if cfg.train.cosine_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.steps, eta_min=cfg.train.lr * 0.01)

    # Phase-space probe grid. Pendulum orbits are NOT circles: at amplitude A,
    # q_max = A but p_max = 2 sin(A/2). Size the grid to cover both.
    q_lim = 1.2 * max(test_amps)
    p_lim = 1.2 * max(2.0 * math.sin(A / 2.0) for A in test_amps)
    lim = max(q_lim, p_lim)
    gq, gp = torch.meshgrid(torch.linspace(-lim, lim, 19),
                            torch.linspace(-lim, lim, 19), indexing="xy")
    grid = torch.stack([gq.reshape(-1), gp.reshape(-1)], dim=1)

    # Dense, regular time grid used only to draw smooth rollout curves at eval.
    t_plot = torch.linspace(0.0, t_max, 300)
    test_amps_t = torch.tensor(test_amps)

    def rollouts():
        # Roll out from (q=A, p=0) for each test amplitude (some unseen).
        return [odeint(func, torch.tensor([A, 0.0]), t_plot, method=method)
                for A in test_amps]

    @torch.no_grad()
    def eval_rollout_mse():
        # Stable eval metric: full-rollout MSE vs the true orbit, per amplitude.
        truth_TB2 = truth.state(test_amps_t, t_plot)                 # (T, B, 2)
        return {A: torch.mean((roll - truth_TB2[:, i]) ** 2).item()
                for i, (A, roll) in enumerate(zip(test_amps, rollouts()))}

    snap_steps = set(cfg.train.snap_steps)
    snapshots = []          # (step, arrows, list-of-rollouts) per snapshot

    def take_snapshot(step):
        with torch.no_grad():
            arrows = func(0.0, grid)
            rolls = [r.clone() for r in rollouts()]
        snapshots.append((step, arrows.clone(), rolls))

    # --- 3. Training loop -------------------------------------------------
    # Same shared-local-grid trick as the sine setup: dynamics are autonomous,
    # so we sample one irregular local grid per step and start each batch item
    # from its own random (amplitude, phase). One odeint call covers the batch.
    bw = cfg.train.batch_window
    B = cfg.train.batch_size
    # Cap window phase so phase0 + window stays inside the LUT range.
    phase_max = max(0.0, t_max - 2.0 * dt_mean * (bw - 1))

    def sample_batch():
        A_idx = torch.randint(0, len(train_amps_t), (B,))
        A = train_amps_t[A_idx]                                       # (B,)
        phase0 = torch.rand(B) * phase_max                            # (B,)
        gaps = torch.rand(bw - 1) * (2 * dt_mean)
        local_t = torch.cat([torch.zeros(1), torch.cumsum(gaps, 0)])  # (bw,)
        times = phase0[None, :] + local_t[:, None]                    # (bw, B)
        z0 = truth.state(A, phase0.unsqueeze(0))[0]                   # (B, 2)
        targets = truth.state(A, times)                               # (bw, B, 2)
        noise = cfg.data.obs_noise
        return (z0 + noise * torch.randn_like(z0),
                local_t,
                targets + noise * torch.randn_like(targets))

    if 0 in snap_steps:
        take_snapshot(0)                                # capture initialization
    best_err = float("inf")                             # best ON-DATA rollout MSE
    best_state = copy.deepcopy(func.state_dict())
    for step in range(1, cfg.train.steps + 1):
        optimizer.zero_grad()
        z0_b, local_t, targets = sample_batch()
        pred = odeint(func, z0_b, local_t, method=method)   # (bw, B, 2), one call
        loss = torch.mean((pred - targets) ** 2)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step in snap_steps:
            take_snapshot(step)
        if step % cfg.train.log_every == 0 or step == 1:
            errs = eval_rollout_mse()
            seen_err = sum(errs[A] for A in train_amps) / len(train_amps)
            if seen_err < best_err:
                best_err = seen_err
                best_state = copy.deepcopy(func.state_dict())
            err_str = "  ".join(f"A={A}:{errs[A]:.2e}" for A in test_amps)
            log.info("step %4d  lr %.4f  train_loss %.3e  rollout_mse[ %s ]  best %.3e",
                     step, optimizer.param_groups[0]["lr"], loss.item(), err_str, best_err)

    func.load_state_dict(best_state)                    # use best model for figures
    log.info("restored best model (on-data rollout mse = %.3e)", best_err)

    # --- 4. Result: phase portrait over several amplitudes ----------------
    # True orbits (gray) vs neural ODE rollouts. Amplitudes NOT in train_amps
    # test whether the law generalizes off the data.
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(test_amps) - 1)) for i in range(len(test_amps))]
    with torch.no_grad():
        final_rolls = rollouts()
        truth_TB2 = truth.state(test_amps_t, t_plot)

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (A, roll, c) in enumerate(zip(test_amps, final_rolls, colors)):
        true_orbit = truth_TB2[:, i]
        ax.plot(true_orbit[:, 0], true_orbit[:, 1], color="lightgray", lw=3)
        seen = "train" if A in train_amps else "UNSEEN"
        ax.plot(roll[:, 0], roll[:, 1], "--", color=c, lw=1.8,
                label=f"A={A} ({seen})")
    ax.set_xlabel("q"); ax.set_ylabel("p"); ax.set_aspect("equal")
    ax.set_title("Rollouts vs true orbits (gray)\nsolid gray = truth, dashed = neural ODE")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "pendulum_neural_ode.png")
    fig.savefig(p1, dpi=120)
    log.info("saved plot -> %s", p1)

    # --- 4b. Angle vs time: exposes the timing/frequency error ------------
    # The phase portrait shows orbit GEOMETRY; the headline pendulum effect
    # (period depending on amplitude) lives in TIMING. q(t) is where to look.
    nt = len(test_amps)
    fig3, axes3 = plt.subplots(1, nt, figsize=(3.6 * nt, 3.0), squeeze=False)
    for i, (ax, A, roll, c) in enumerate(zip(axes3[0], test_amps, final_rolls, colors)):
        true_q = truth_TB2[:, i, 0]
        ax.plot(t_plot, true_q, color="lightgray", lw=3, label="true q(t)")
        ax.plot(t_plot, roll[:, 0], "--", color=c, lw=1.4, label="neural ODE")
        seen = "train" if A in train_amps else "UNSEEN"
        ax.set_title(f"A={A} ({seen})")
        ax.set_xlabel("t")
    axes3[0][0].set_ylabel("q"); axes3[0][0].legend(fontsize=7, loc="upper right")
    fig3.suptitle("Angle vs time: amplitude-dependent period (invisible in phase space)")
    fig3.tight_layout()
    p3 = os.path.join(out_dir, "angle_vs_time.png")
    fig3.savefig(p3, dpi=120)
    log.info("saved plot -> %s", p3)

    # --- 5. How the "guessed law" evolves during training -----------------
    # blue arrows = learned field f_theta; gray arrows = true law [p, -sin(q)];
    # colored solid = rollouts from each test amplitude at that step.
    true_arrows = torch.stack([grid[:, 1], -torch.sin(grid[:, 0])], dim=1)
    n = len(snapshots)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig2, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows),
                              squeeze=False)
    flat = [a for row in axes for a in row]
    for ax, (step, arrows, rolls) in zip(flat, snapshots):
        ax.quiver(grid[:, 0], grid[:, 1], true_arrows[:, 0], true_arrows[:, 1],
                  color="lightgray", alpha=0.7, width=0.005)
        ax.quiver(grid[:, 0], grid[:, 1], arrows[:, 0], arrows[:, 1],
                  color="tab:blue", alpha=0.9, width=0.005)
        for A, roll, c in zip(test_amps, rolls, colors):
            ax.plot(roll[:, 0], roll[:, 1], color=c, lw=1.6)
        ax.set_title(f"step {step}")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_xlabel("q"); ax.set_ylabel("p")
    for ax in flat[n:]:                                 # hide unused panels
        ax.axis("off")
    fig2.suptitle("Learned field (blue) vs true pendulum field (gray); rollouts from test amplitudes")
    fig2.tight_layout()
    p2 = os.path.join(out_dir, "law_evolution.png")
    fig2.savefig(p2, dpi=120)
    log.info("saved plot -> %s", p2)


if __name__ == "__main__":
    main()
