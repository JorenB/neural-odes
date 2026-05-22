"""
Barebones neural ODE: learn the dynamics of a 1D oscillating particle.

The story
---------
We observe a particle whose position oscillates as x(t) = sin(t).
We model it as a flow in a 2D phase space z = [x, v] (position, velocity),
because a 1D autonomous ODE cannot oscillate (it would need two different
velocities at the same position). We let a neural net learn the vector field
    dz/dt = f_theta(z)
and train it so that integrating from z(0) reproduces the observed trajectory.

The true (unknown-to-the-model) physics is the harmonic oscillator
    d/dt [x, v] = [v, -x]
whose solution with z(0) = [0, 1] is exactly [sin(t), cos(t)].
The network never sees this equation; it must discover the vector field.

Config & runs
-------------
All hyperparameters live in config.yaml. Hydra creates a fresh timestamped
directory under logs/ for every run and we save all figures there. Override
anything from the CLI, e.g.:
    uv run python train_sine.py data.obs_noise=0.05 train.steps=4000
"""

import copy
import logging
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

# Hydra routes this logger to both the console and <run_dir>/train_sine.log.
log = logging.getLogger(__name__)


def true_state(times):
    """Closed-form harmonic-oscillator state at arbitrary `times`.
    Returns states with a trailing dim of size 2: [sin(t), cos(t)]."""
    return torch.stack([torch.sin(times), torch.cos(times)], dim=-1)


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
        # odeint calls f(t, z); we ignore t (the dynamics don't depend on it).
        return self.net(z)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    out_dir = HydraConfig.get().runtime.output_dir   # this run's logs/ subdir
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    log.info("output dir: %s", out_dir)
    torch.manual_seed(cfg.seed)

    # --- 1. Ground-truth data ---------------------------------------------
    # One oscillator per amplitude. Amplitude A scales the state linearly:
    # state = A * [sin(t), cos(t)], i.e. a circle of radius A in phase space.
    # The closed form lets us sample short trajectory segments on the fly (see
    # sample_batch below) at arbitrary, IRREGULAR times -- no fixed grid.
    t_max = 2.0 * torch.pi * cfg.data.periods
    train_amps = list(cfg.data.amplitudes)
    train_amps_t = torch.tensor(train_amps)
    dt_mean = t_max / cfg.data.n_obs        # nominal sampling density (mean gap)
    method = cfg.train.solver

    # --- 2. Model + optimizer ---------------------------------------------
    func = ODEFunc(hidden=cfg.model.hidden)
    optimizer = torch.optim.Adam(func.parameters(), lr=cfg.train.lr)
    scheduler = None
    if cfg.train.cosine_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.steps, eta_min=cfg.train.lr * 0.01)

    # Phase-space probe grid, sized to cover the largest test radius.
    lim = 1.2 * max(cfg.data.test_amplitudes)
    gx, gv = torch.meshgrid(torch.linspace(-lim, lim, 19),
                            torch.linspace(-lim, lim, 19), indexing="xy")
    grid = torch.stack([gx.reshape(-1), gv.reshape(-1)], dim=1)

    # Dense, regular time grid used only to draw smooth rollout curves at eval.
    t_plot = torch.linspace(0.0, t_max, 300)
    test_amps = list(cfg.data.test_amplitudes)

    def rollouts():
        # Roll out from [0, A] for each test amplitude (some unseen in training).
        return [odeint(func, torch.tensor([0.0, A]), t_plot, method=method)
                for A in test_amps]

    def eval_rollout_mse():
        # Stable eval metric: full-rollout MSE vs the true circle, per radius.
        # Unlike the windowed train loss, this is low-variance and comparable
        # across steps -- the right thing to MONITOR and to pick a best model by.
        with torch.no_grad():
            errs = {}
            for A, roll in zip(test_amps, rollouts()):
                errs[A] = torch.mean((roll - A * true_state(t_plot)) ** 2).item()
        return errs

    snap_steps = set(cfg.train.snap_steps)
    snapshots = []          # (step, arrows, list-of-rollouts) per snapshot

    def take_snapshot(step):
        with torch.no_grad():
            arrows = func(0.0, grid)
            rolls = [r.clone() for r in rollouts()]
        snapshots.append((step, arrows.clone(), rolls))

    # --- 3. Training loop -------------------------------------------------
    # Train on SHORT segments. Because the dynamics are AUTONOMOUS (time-
    # translation invariant), we can share ONE irregular local time grid across
    # the whole batch and just start each segment from a different random phase
    # and amplitude. That collapses the batch into a SINGLE odeint call (fast),
    # while still training on irregular spacings (a fresh random grid each step).
    bw = cfg.train.batch_window
    B = cfg.train.batch_size

    def sample_batch():
        A = train_amps_t[torch.randint(0, len(train_amps_t), (B,))]   # (B,)
        phase0 = torch.rand(B) * t_max                                # (B,) start times
        gaps = torch.rand(bw - 1) * (2 * dt_mean)                     # irregular gaps
        local_t = torch.cat([torch.zeros(1), torch.cumsum(gaps, 0)])  # (bw,), starts at 0
        times = phase0[None, :] + local_t[:, None]                    # (bw, B) absolute
        z0 = A[:, None] * true_state(phase0)                          # (B, 2)
        targets = A[None, :, None] * true_state(times)                # (bw, B, 2)
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
            # Track best by mean rollout MSE over the TRAINED radii only.
            seen_err = sum(errs[A] for A in train_amps) / len(train_amps)
            if seen_err < best_err:
                best_err = seen_err
                best_state = copy.deepcopy(func.state_dict())
            err_str = "  ".join(f"A={A}:{errs[A]:.2e}" for A in test_amps)
            log.info("step %4d  lr %.4f  train_loss %.3e  rollout_mse[ %s ]  best %.3e",
                     step, optimizer.param_groups[0]["lr"], loss.item(), err_str, best_err)

    func.load_state_dict(best_state)                    # use best model for figures
    log.info("restored best model (on-data rollout mse = %.3e)", best_err)

    # --- 4. Result: phase portrait over several radii ---------------------
    # True circles (gray) vs the neural ODE rolled out from each test radius.
    # Radii NOT in train_amps test whether the law generalizes off the data.
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(test_amps) - 1)) for i in range(len(test_amps))]
    with torch.no_grad():
        final_rolls = rollouts()

    fig, ax = plt.subplots(figsize=(6, 6))
    for A, roll, c in zip(test_amps, final_rolls, colors):
        true_circle = A * true_state(t_plot)
        ax.plot(true_circle[:, 0], true_circle[:, 1], color="lightgray", lw=3)
        seen = "train" if A in train_amps else "UNSEEN"
        ax.plot(roll[:, 0], roll[:, 1], "--", color=c, lw=1.8,
                label=f"A={A} ({seen})")
    ax.set_xlabel("x"); ax.set_ylabel("v"); ax.set_aspect("equal")
    ax.set_title("Rollouts vs true circles (gray)\nsolid gray = truth, dashed = neural ODE")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "sine_neural_ode.png")
    fig.savefig(p1, dpi=120)
    log.info("saved plot -> %s", p1)

    # --- 4b. Position vs time: exposes the timing/frequency error ---------
    # The phase portrait only shows orbit GEOMETRY (right-sized circles). The
    # real off-data error is in TIMING: unseen radii orbit at the wrong speed,
    # so the predicted x(t) drifts out of phase with the truth as t grows.
    nt = len(test_amps)
    fig3, axes3 = plt.subplots(1, nt, figsize=(3.6 * nt, 3.0), squeeze=False)
    for ax, A, roll, c in zip(axes3[0], test_amps, final_rolls, colors):
        true_x = (A * true_state(t_plot))[:, 0]
        ax.plot(t_plot, true_x, color="lightgray", lw=3, label="true x(t)")
        ax.plot(t_plot, roll[:, 0], "--", color=c, lw=1.4, label="neural ODE")
        seen = "train" if A in train_amps else "UNSEEN"
        ax.set_title(f"A={A} ({seen})")
        ax.set_xlabel("t")
    axes3[0][0].set_ylabel("x"); axes3[0][0].legend(fontsize=7, loc="upper right")
    fig3.suptitle("Position vs time: phase drift on UNSEEN radii (invisible in phase space)")
    fig3.tight_layout()
    p3 = os.path.join(out_dir, "position_vs_time.png")
    fig3.savefig(p3, dpi=120)
    log.info("saved plot -> %s", p3)

    # --- 5. How the "guessed law" evolves during training -----------------
    # blue arrows = learned field f_theta; gray arrows = true law [v, -x];
    # colored dashed = rollouts from each test radius at that step.
    true_arrows = torch.stack([grid[:, 1], -grid[:, 0]], dim=1)   # [v, -x]
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
        ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("v")
    for ax in flat[n:]:                                 # hide unused panels
        ax.axis("off")
    fig2.suptitle("Learned field (blue) vs true rotation (gray); rollouts from test radii")
    fig2.tight_layout()
    p2 = os.path.join(out_dir, "law_evolution.png")
    fig2.savefig(p2, dpi=120)
    log.info("saved plot -> %s", p2)


if __name__ == "__main__":
    main()
