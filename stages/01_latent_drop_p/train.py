"""
Stage 1 -- Latent ODE: drop p, infer it from a window of q observations.

The story
---------
We observe only the angle q(t) of the pendulum (on a regular dense grid). The
model now has three pieces:

  Encoder  g_phi : (q[t0], q[t0+dt], ..., q[t0+(k-1)dt]) --> z0 in R^2
           A small MLP over a window of k_enc consecutive q values.

  ODE      f_theta : z --> dz/dt
           Same MLP-as-vector-field as Stage 0. Autonomous.

  Decoder  h : z --> q
           FIXED. h(z) = z[0]. No parameters.

Training: encode z0 from the first k_enc obs in a window, integrate forward,
decode q at every time step, MSE against the true q observations over the
WHOLE window (encoder window included).

Why a fixed decoder?
--------------------
With h(z) = z[0], the first latent coordinate IS the model's predicted angle,
by construction. That leaves z[1] as the hidden coordinate the encoder must
use to make the rotation dynamics work. The natural hope: z[1] correlates
with the true angular momentum p, because that's the information physics
*requires* to extrapolate q forward. The headline diagnostic (latent_vs_p)
plots z[1] against true p directly.

Data: same multi-amplitude pendulum LUT as Stage 0. We use REGULAR time
spacing here so the encoder doesn't need to know per-sample dt; irregular
sampling comes back in Stage 4 with a sequence-aware encoder.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


def true_pendulum_field(t, z):
    """Ground-truth vector field: dz/dt = [p, -sin(q)]. Autonomous."""
    q, p = z[..., 0:1], z[..., 1:2]
    return torch.cat([p, -torch.sin(q)], dim=-1)


class PendulumTruth:
    """Per-amplitude high-res (q, p) trajectories. Same as Stage 0."""

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
        """A_batch: (B,) amps. times: (T,) shared or (T, B). Returns (T, B, 2)."""
        B = A_batch.shape[0]
        T = times.shape[0]
        t_TB = times[:, None].expand(T, B) if times.ndim == 1 else times
        t_clamped = t_TB.clamp(0.0, float(self.t_lut[-1]) - 1e-7)
        idx = torch.bucketize(t_clamped, self.t_lut) - 1
        idx = idx.clamp(0, len(self.t_lut) - 2)
        t0 = self.t_lut[idx]; t1 = self.t_lut[idx + 1]
        w = ((t_clamped - t0) / (t1 - t0)).unsqueeze(-1)             # (T, B, 1)
        out = torch.empty(T, B, 2)
        for b in range(B):
            traj = self.trajs[self._key(A_batch[b].item())]
            i = idx[:, b]
            out[:, b] = torch.lerp(traj[i], traj[i + 1], w[:, b])
        return out


class Encoder(nn.Module):
    """MLP from a window of k_enc consecutive q observations to latent z0."""

    def __init__(self, k_enc, latent_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k_enc, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, q_window):
        # q_window: (B, k_enc) -> z0: (B, latent_dim)
        return self.net(q_window)


class ODEFunc(nn.Module):
    """Latent vector field dz/dt = f_theta(z). Autonomous."""

    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, z):
        return self.net(z)


def decode_q(z):
    """Fixed decoder: q = z[0]. No parameters; makes z[0] interpretable as angle."""
    return z[..., 0]


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    out_dir = HydraConfig.get().runtime.output_dir
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    log.info("output dir: %s", out_dir)
    torch.manual_seed(cfg.seed)

    # --- 1. Data ----------------------------------------------------------
    t_max = float(cfg.data.t_max)
    dt = float(cfg.data.dt)
    train_amps = list(cfg.data.amplitudes)
    test_amps = list(cfg.data.test_amplitudes)
    all_amps = sorted(set(train_amps) | set(test_amps))
    train_amps_t = torch.tensor(train_amps)
    test_amps_t = torch.tensor(test_amps)
    method = cfg.train.solver
    truth = PendulumTruth(all_amps, t_max, cfg.data.n_lut, method)

    k_enc = int(cfg.data.k_enc)
    T = int(cfg.data.window_len)
    assert T > k_enc, "window_len must exceed k_enc"
    local_t = torch.arange(T, dtype=torch.float32) * dt              # (T,) regular

    # --- 2. Model + optimizer ---------------------------------------------
    encoder = Encoder(k_enc=k_enc, latent_dim=cfg.model.latent_dim,
                      hidden=cfg.model.hidden)
    func = ODEFunc(dim=cfg.model.latent_dim, hidden=cfg.model.hidden)
    params = list(encoder.parameters()) + list(func.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.train.lr)
    scheduler = None
    if cfg.train.cosine_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.steps, eta_min=cfg.train.lr * 0.01)

    # phase0 must leave room for a full window inside the LUT range.
    phase_max = max(0.0, t_max - dt * (T - 1))
    B = cfg.train.batch_size

    def sample_batch():
        # Sample random (amplitude, starting phase). Return (T, B) tensor of q
        # observations on a regular dt grid.
        A_idx = torch.randint(0, len(train_amps_t), (B,))
        A = train_amps_t[A_idx]                                      # (B,)
        phase0 = torch.rand(B) * phase_max                           # (B,)
        times = phase0[None, :] + local_t[:, None]                   # (T, B)
        q_obs = truth.state(A, times)[..., 0]                        # (T, B), q only
        noise = cfg.data.obs_noise
        return q_obs + noise * torch.randn_like(q_obs)

    def rollout_from(A_t, t_grid):
        """Encode initial window for each A, integrate over t_grid.
        Returns (q_pred: (T_grid, B), z: (T_grid, B, latent_dim))."""
        t_enc = (torch.arange(k_enc, dtype=torch.float32) * dt)[:, None].expand(
            k_enc, len(A_t))                                          # (k_enc, B)
        q_enc = truth.state(A_t, t_enc)[..., 0].T                    # (B, k_enc)
        z0 = encoder(q_enc)                                          # (B, latent)
        z = odeint(func, z0, t_grid, method=method)                  # (T_grid, B, latent)
        return decode_q(z), z

    @torch.no_grad()
    def eval_rollout_mse():
        t_plot = torch.linspace(0.0, t_max, cfg.data.n_plot)
        q_pred, _ = rollout_from(test_amps_t, t_plot)
        truth_TB = truth.state(test_amps_t, t_plot)[..., 0]          # (T_plot, B)
        return {A: torch.mean((q_pred[:, i] - truth_TB[:, i]) ** 2).item()
                for i, A in enumerate(test_amps)}

    # --- 3. Training loop -------------------------------------------------
    best_err = float("inf")
    best_state = (copy.deepcopy(encoder.state_dict()),
                  copy.deepcopy(func.state_dict()))
    for step in range(1, cfg.train.steps + 1):
        optimizer.zero_grad()
        q_obs = sample_batch()                                       # (T, B)
        z0 = encoder(q_obs[:k_enc].T)                                # (B, latent)
        z = odeint(func, z0, local_t, method=method)                 # (T, B, latent)
        q_pred = decode_q(z)                                         # (T, B)
        loss = torch.mean((q_pred - q_obs) ** 2)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % cfg.train.log_every == 0 or step == 1:
            errs = eval_rollout_mse()
            seen_err = sum(errs[A] for A in train_amps) / len(train_amps)
            if seen_err < best_err:
                best_err = seen_err
                best_state = (copy.deepcopy(encoder.state_dict()),
                              copy.deepcopy(func.state_dict()))
            err_str = "  ".join(f"A={A}:{errs[A]:.2e}" for A in test_amps)
            log.info("step %4d  lr %.4f  train_loss %.3e  q_rollout_mse[ %s ]  best %.3e",
                     step, optimizer.param_groups[0]["lr"], loss.item(), err_str, best_err)

    encoder.load_state_dict(best_state[0])
    func.load_state_dict(best_state[1])
    log.info("restored best model (on-data q-rollout mse = %.3e)", best_err)

    # --- 4. Figures -------------------------------------------------------
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(test_amps) - 1)) for i in range(len(test_amps))]
    t_plot = torch.linspace(0.0, t_max, cfg.data.n_plot)
    with torch.no_grad():
        q_pred, z_traj = rollout_from(test_amps_t, t_plot)           # (T, B), (T, B, latent)
        truth_TB2 = truth.state(test_amps_t, t_plot)                 # (T, B, 2)

    nt = len(test_amps)

    # 4a. q vs t per amplitude (Stage 0 analog -- the headline reconstruction)
    fig, axes = plt.subplots(1, nt, figsize=(3.6 * nt, 3.0), squeeze=False)
    for i, (ax, A, c) in enumerate(zip(axes[0], test_amps, colors)):
        true_q = truth_TB2[:, i, 0]
        ax.plot(t_plot, true_q, color="lightgray", lw=3, label="true q(t)")
        ax.plot(t_plot, q_pred[:, i], "--", color=c, lw=1.4, label="latent ODE")
        seen = "train" if A in train_amps else "UNSEEN"
        ax.set_title(f"A={A} ({seen})")
        ax.set_xlabel("t")
    axes[0][0].set_ylabel("q"); axes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Predicted q(t) from drop-p latent ODE (only q observed)")
    fig.tight_layout()
    p = os.path.join(out_dir, "q_vs_time.png")
    fig.savefig(p, dpi=120); log.info("saved plot -> %s", p)

    # 4b. Latent phase portrait with true (q, p) overlay
    fig, axes = plt.subplots(1, nt, figsize=(3.6 * nt, 3.6), squeeze=False)
    for i, (ax, A, c) in enumerate(zip(axes[0], test_amps, colors)):
        true_qp = truth_TB2[:, i]
        ax.plot(true_qp[:, 0], true_qp[:, 1], color="lightgray", lw=3,
                label="true (q, p)")
        z_i = z_traj[:, i]
        ax.plot(z_i[:, 0], z_i[:, 1], "--", color=c, lw=1.4,
                label="latent (z[0], z[1])")
        seen = "train" if A in train_amps else "UNSEEN"
        ax.set_title(f"A={A} ({seen})")
        ax.set_xlabel("z[0]  (= predicted q)")
    axes[0][0].set_ylabel("z[1]  (hidden)")
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Latent phase portrait vs true (q, p) -- did z[1] track p?")
    fig.tight_layout()
    p = os.path.join(out_dir, "latent_orbits.png")
    fig.savefig(p, dpi=120); log.info("saved plot -> %s", p)

    # 4c. Scatter z[1] vs true p -- the cleanest "did it learn momentum?" test
    fig, axes = plt.subplots(1, nt, figsize=(3.6 * nt, 3.0), squeeze=False)
    for i, (ax, A, c) in enumerate(zip(axes[0], test_amps, colors)):
        true_p = truth_TB2[:, i, 1]
        z1 = z_traj[:, i, 1]
        ax.scatter(true_p, z1, color=c, s=8, alpha=0.6)
        ax.axhline(0, color="lightgray", lw=0.5)
        ax.axvline(0, color="lightgray", lw=0.5)
        seen = "train" if A in train_amps else "UNSEEN"
        ax.set_title(f"A={A} ({seen})")
        ax.set_xlabel("true p")
    axes[0][0].set_ylabel("z[1]")
    fig.suptitle("z[1] vs true p: is the hidden coord a clean function of momentum?")
    fig.tight_layout()
    p = os.path.join(out_dir, "latent_vs_p.png")
    fig.savefig(p, dpi=120); log.info("saved plot -> %s", p)

    # 4d. Post-hoc linear fit z[1] = a*q + b*p ----------------------------
    # If the encoder learned a single rotated basis, (a, b) is constant
    # across amplitudes. If it learned an amplitude-specific recipe,
    # (a, b) varies. R^2 says how well a *linear* fit explains z[1] at all.
    q_all = truth_TB2[..., 0]                                    # (T, B)
    p_all = truth_TB2[..., 1]
    z1_all = z_traj[..., 1]                                      # (T, B)

    def lin_fit(q, p, z1):
        X = torch.stack([q, p], dim=-1)                          # (..., 2)
        y = z1.unsqueeze(-1)
        ab, *_ = torch.linalg.lstsq(X, y)
        a, b = ab[0, 0].item(), ab[1, 0].item()
        ss_res = ((y - X @ ab) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return a, b, r2

    a_g, b_g, r2_g = lin_fit(q_all.flatten(), p_all.flatten(), z1_all.flatten())
    log.info("global linear fit  z[1] = %+.3f*q  %+.3f*p   R^2 = %.4f",
             a_g, b_g, r2_g)
    log.info("per-amplitude linear fits z[1] = a*q + b*p:")
    per_amp = []
    for i, A in enumerate(test_amps):
        a, b, r2 = lin_fit(q_all[:, i], p_all[:, i], z1_all[:, i])
        per_amp.append((A, a, b, r2))
        log.info("  A=%-4s  a=%+.3f  b=%+.3f  R^2=%.4f", str(A), a, b, r2)

    # Plot: scatter of actual z[1] vs fitted a*q + b*p, with diagonal.
    fig, axes = plt.subplots(1, nt, figsize=(3.6 * nt, 3.0), squeeze=False)
    for i, (ax, (A, a, b, r2), c) in enumerate(zip(axes[0], per_amp, colors)):
        actual = z1_all[:, i]
        fitted = a * q_all[:, i] + b * p_all[:, i]
        ax.scatter(actual, fitted, color=c, s=8, alpha=0.6)
        lo = min(actual.min().item(), fitted.min().item())
        hi = max(actual.max().item(), fitted.max().item())
        ax.plot([lo, hi], [lo, hi], color="lightgray", lw=1, zorder=0)
        seen = "train" if A in train_amps else "UNSEEN"
        ax.set_title(f"A={A} ({seen})\na={a:+.2f}  b={b:+.2f}  R²={r2:.3f}",
                     fontsize=9)
        ax.set_xlabel("z[1] actual")
    axes[0][0].set_ylabel("a·q + b·p (fit)")
    fig.suptitle(f"Linear fit z[1] = a·q + b·p per amplitude    "
                 f"(global: a={a_g:+.2f}  b={b_g:+.2f}  R²={r2_g:.3f})")
    fig.tight_layout()
    p = os.path.join(out_dir, "latent_basis_fit.png")
    fig.savefig(p, dpi=120); log.info("saved plot -> %s", p)


if __name__ == "__main__":
    main()
