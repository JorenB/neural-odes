"""
Stage 6a -- First rung of the perception ladder: 1D heatmap observation.

What changes vs Stage 4
-----------------------
Instead of observing q as a single scalar (plus noise), we now observe q
as a Gaussian bump rendered in a length-heatmap_n (=64) 1D array:

    o[i] = exp(-(positions[i] - q)^2 / (2 * sigma^2))   for i = 0..N-1

where `positions` spans [-heatmap_range, +heatmap_range] in q-space. The
observation is high-dimensional (64 numbers) and a NONLINEAR function of q,
but we deliberately avoid the cost of a CNN by keeping the input 1D and
letting the GRU handle it directly. This stage tests "does the high-dim-obs
+ ODE pipeline compose at all?" before we pay CNN cost in Stage 6b.

Architecture
------------
  Encoder  GRU(input_size = 1 + heatmap_n, hidden=64) + two heads
           Per step: (t_i, heatmap_i) flattened into a single (1+N,) vector.
  Decoder  Fixed: q_pred = z[0]. To compute loss we re-render the predicted
           heatmap and compare it to the observed heatmap.
  ODE      same as Stage 4 (autonomous MLP, 2D latent).

Loss = mean((heatmap_pred - heatmap_target)^2)  +  beta * KL

Headline question
-----------------
Does the architecture compose end-to-end with a 64-dim nonlinear
observation? We expect rollout MSE on q to be somewhat worse than Stage 4
because the encoder has to do "extract q from heatmap" AND "back-project
to t=0" jointly. The qualitative picture (interpretable latent, clean
linear fit) should survive.
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


def render_heatmap(q, positions, sigma):
    """Render q (any shape) as a Gaussian bump in a heatmap of length len(positions).
    Output shape: q.shape + (len(positions),)."""
    diff = q.unsqueeze(-1) - positions                                # broadcast
    return torch.exp(-0.5 * (diff / sigma) ** 2)


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


class VariationalEncoderGRU(nn.Module):
    """GRU-based variational encoder for irregular sparse heatmap observations.

    Each observation is a (1 + heatmap_n)-vector: time + heatmap. The GRU
    consumes the observations in REVERSE chronological order so the final
    hidden state is most influenced by observations near the anchor time
    t=0. Two heads project the final hidden state to posterior (mu, logvar)."""

    def __init__(self, obs_dim, hidden_size=64, latent_dim=2):
        super().__init__()
        # Each GRU step receives the time concatenated with the heatmap.
        self.gru = nn.GRU(input_size=1 + obs_dim, hidden_size=hidden_size,
                          batch_first=True)
        self.head_mu = nn.Linear(hidden_size, latent_dim)
        self.head_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, times, values):
        """times: (B, N) sorted ascending. values: (B, N, obs_dim) -- the heatmap
        observations. Returns (mu, logvar), each (B, latent_dim)."""
        times_rev = times.flip(dims=[1])                              # (B, N)
        values_rev = values.flip(dims=[1])                            # (B, N, obs_dim)
        x = torch.cat([times_rev.unsqueeze(-1), values_rev], dim=-1)  # (B, N, 1+obs_dim)
        _, h_final = self.gru(x)                                      # (1, B, hidden)
        h = h_final.squeeze(0)                                        # (B, hidden)
        return self.head_mu(h), self.head_logvar(h)

    def sample(self, times, values):
        mu, logvar = self.forward(times, values)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps, mu, logvar


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
    train_amps = list(cfg.data.amplitudes)
    test_amps = list(cfg.data.test_amplitudes)
    all_amps = sorted(set(train_amps) | set(test_amps))
    train_amps_t = torch.tensor(train_amps)
    test_amps_t = torch.tensor(test_amps)
    method = cfg.train.solver
    truth = PendulumTruth(all_amps, t_max, cfg.data.n_lut, method)

    horizon = float(cfg.data.window_horizon)
    n_obs = int(cfg.data.n_obs)
    T = int(cfg.data.target_T)
    target_dt = float(cfg.data.target_dt)
    local_t = torch.arange(T, dtype=torch.float32) * target_dt       # (T,) dense regular target grid

    # Heatmap rendering grid (the model's "observation pixel positions").
    heatmap_n = int(cfg.data.heatmap_n)
    heatmap_range = float(cfg.data.heatmap_range)
    heatmap_sigma = float(cfg.data.heatmap_sigma)
    positions = torch.linspace(-heatmap_range, heatmap_range, heatmap_n)  # (N,)

    # --- 2. Model + optimizer ---------------------------------------------
    encoder = VariationalEncoderGRU(obs_dim=heatmap_n,
                                    hidden_size=cfg.model.hidden,
                                    latent_dim=cfg.model.latent_dim)
    beta_kl = float(cfg.model.beta_kl)
    func = ODEFunc(dim=cfg.model.latent_dim, hidden=cfg.model.hidden)
    params = list(encoder.parameters()) + list(func.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.train.lr)
    scheduler = None
    if cfg.train.cosine_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.steps, eta_min=cfg.train.lr * 0.01)

    # phase0 must leave a full horizon-wide window inside the LUT range.
    phase_max = max(0.0, t_max - horizon)
    B = cfg.train.batch_size

    def sample_obs_window_at(A_t, phase0):
        """Sample (B, n_obs) random local observation times in [0, horizon].
        For each, render the true q as a Gaussian bump on the heatmap grid and
        add per-pixel noise. Returns (obs_local: (B, N), o_obs: (B, N, heatmap_n))."""
        Bn = len(A_t)
        obs_local = torch.rand(Bn, n_obs) * horizon                   # (B, N)
        obs_local, _ = obs_local.sort(dim=1)
        obs_abs = phase0[:, None] + obs_local                         # (B, N)
        q_true = truth.state(A_t, obs_abs.T)[..., 0].T                # (B, N)
        o_obs = render_heatmap(q_true, positions, heatmap_sigma)      # (B, N, heatmap_n)
        o_obs = o_obs + cfg.data.obs_noise * torch.randn_like(o_obs)
        return obs_local, o_obs

    def sample_batch():
        """Per-step batch: random amplitudes/phases/observation windows,
        plus a shared dense target grid where targets are rendered heatmaps."""
        A_idx = torch.randint(0, len(train_amps_t), (B,))
        A = train_amps_t[A_idx]                                       # (B,)
        phase0 = torch.rand(B) * phase_max                            # (B,)
        obs_local, o_obs = sample_obs_window_at(A, phase0)            # (B, N), (B, N, N_pix)
        target_abs = phase0[None, :] + local_t[:, None]               # (T, B)
        q_target_true = truth.state(A, target_abs)[..., 0]            # (T, B)
        o_target = render_heatmap(q_target_true, positions, heatmap_sigma)  # (T, B, N_pix)
        o_target = o_target + cfg.data.obs_noise * torch.randn_like(o_target)
        return obs_local, o_obs, o_target

    def rollout_from(A_t, t_grid, sample=False):
        """Eval helper: phase0=0 (encoder sees the trajectory from its own
        initial condition). Returns (q_pred (T, B), z (T, B, latent))."""
        Bn = len(A_t)
        phase0 = torch.zeros(Bn)
        obs_local, o_obs = sample_obs_window_at(A_t, phase0)
        mu, logvar = encoder(obs_local, o_obs)
        if sample:
            z0 = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        else:
            z0 = mu
        z = odeint(func, z0, t_grid, method=method)
        return decode_q(z), z

    def posterior_samples(A_t, t_grid, n_samples):
        Bn = len(A_t)
        phase0 = torch.zeros(Bn)
        obs_local, o_obs = sample_obs_window_at(A_t, phase0)
        mu, logvar = encoder(obs_local, o_obs)
        std = (0.5 * logvar).exp()
        samples = []
        for _ in range(n_samples):
            z0 = mu + std * torch.randn_like(mu)
            z = odeint(func, z0, t_grid, method=method)
            samples.append(decode_q(z))
        return torch.stack(samples, dim=0)

    @torch.no_grad()
    def eval_rollout_mse():
        """Returns (q_mse_dict, o_mse_dict). q-MSE compares predicted q to true q.
        o-MSE compares rendered q_pred heatmap to true q heatmap (the actual
        observable -- includes any sign confusion the encoder might have)."""
        t_plot = torch.linspace(0.0, horizon, cfg.data.n_plot)
        q_true = truth.state(test_amps_t, t_plot)[..., 0]             # (T, B)
        o_true = render_heatmap(q_true, positions, heatmap_sigma)     # (T, B, N_pix)
        n_seeds = int(cfg.data.n_eval_seeds)
        acc_q = torch.zeros(len(test_amps))
        acc_o = torch.zeros(len(test_amps))
        for _ in range(n_seeds):
            q_pred, _ = rollout_from(test_amps_t, t_plot, sample=False)
            o_pred = render_heatmap(q_pred, positions, heatmap_sigma)
            acc_q += ((q_pred - q_true) ** 2).mean(dim=0)
            acc_o += ((o_pred - o_true) ** 2).mean(dim=(0, -1))       # mean over time AND pixels
        acc_q /= n_seeds; acc_o /= n_seeds
        return ({A: acc_q[i].item() for i, A in enumerate(test_amps)},
                {A: acc_o[i].item() for i, A in enumerate(test_amps)})

    # --- 3. Training loop -------------------------------------------------
    best_err = float("inf")
    best_state = (copy.deepcopy(encoder.state_dict()),
                  copy.deepcopy(func.state_dict()))
    for step in range(1, cfg.train.steps + 1):
        optimizer.zero_grad()
        obs_local, o_obs, o_target = sample_batch()                  # (B,N), (B,N,N_pix), (T,B,N_pix)
        z0, mu, logvar = encoder.sample(obs_local, o_obs)
        z = odeint(func, z0, local_t, method=method)                 # (T, B, latent)
        q_pred = decode_q(z)                                         # (T, B)
        o_pred = render_heatmap(q_pred, positions, heatmap_sigma)    # (T, B, N_pix)
        # SUM over pixels (otherwise mean-over-64-mostly-zero-pixels dilutes
        # the recon signal so much that the posterior collapses to the prior).
        # Mean over time and batch as usual.
        recon = ((o_pred - o_target) ** 2).sum(dim=-1).mean()
        kl = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar).sum(dim=-1).mean()
        loss = recon + beta_kl * kl
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % cfg.train.log_every == 0 or step == 1:
            q_errs, o_errs = eval_rollout_mse()
            # Best model picked by q-MSE -- the sign-sensitive metric we
            # ultimately care about.
            seen_err = sum(q_errs[A] for A in train_amps) / len(train_amps)
            if seen_err < best_err:
                best_err = seen_err
                best_state = (copy.deepcopy(encoder.state_dict()),
                              copy.deepcopy(func.state_dict()))
            q_str = "  ".join(f"A={A}:{q_errs[A]:.2e}" for A in test_amps)
            o_str = "  ".join(f"A={A}:{o_errs[A]:.2e}" for A in test_amps)
            log.info("step %4d  lr %.4f  recon %.3e  kl %.3e  "
                     "q_mse[ %s ]  o_mse[ %s ]  best_q %.3e",
                     step, optimizer.param_groups[0]["lr"],
                     recon.item(), kl.item(), q_str, o_str, best_err)

    encoder.load_state_dict(best_state[0])
    func.load_state_dict(best_state[1])
    log.info("restored best model (on-data q-rollout mse = %.3e)", best_err)

    # --- 4. Figures -------------------------------------------------------
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(test_amps) - 1)) for i in range(len(test_amps))]
    t_plot = torch.linspace(0.0, horizon, cfg.data.n_plot)
    with torch.no_grad():
        q_pred, z_traj = rollout_from(test_amps_t, t_plot)           # (T, B), (T, B, latent)
        truth_TB2 = truth.state(test_amps_t, t_plot)                 # (T, B, 2)

    nt = len(test_amps)

    # 4-pre. Example encoder input: rendered heatmap window for one test amp.
    # Sanity-check that the model is "seeing" what we think it sees.
    with torch.no_grad():
        ex_obs_local, ex_o_obs = sample_obs_window_at(
            test_amps_t, torch.zeros(len(test_amps)))                # (B, N), (B, N, N_pix)
    fig, axes = plt.subplots(1, nt, figsize=(3.0 * nt, 3.5), squeeze=False)
    for i, (ax, A) in enumerate(zip(axes[0], test_amps)):
        im = ax.imshow(ex_o_obs[i].numpy(), aspect="auto", origin="lower",
                       extent=[-heatmap_range, heatmap_range, 0, n_obs],
                       cmap="viridis", interpolation="nearest")
        # Overlay the true q at each observation time so the eye can check.
        t_enc_local = ex_obs_local[i].numpy()
        with torch.no_grad():
            true_q_enc = truth.state(
                test_amps_t[i:i+1], ex_obs_local[i:i+1].T)[..., 0].squeeze(1)
        ax.scatter(true_q_enc.numpy(), torch.arange(n_obs).numpy() + 0.5,
                   color="red", s=12, label="true q")
        ax.set_title(f"A={A}  (encoder input)")
        ax.set_xlabel("q (pixel position)")
        if i == 0:
            ax.set_ylabel("observation index (0 = earliest)")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Encoder input: heatmap of q at {n_obs} irregular times  "
                 f"(N_pix={heatmap_n}, σ_pix={heatmap_sigma})")
    fig.tight_layout()
    p = os.path.join(out_dir, "encoder_input_example.png")
    fig.savefig(p, dpi=120); log.info("saved plot -> %s", p)

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
    fig.suptitle(f"Predicted q(t) from drop-p latent ODE  (heatmap obs N_pix={heatmap_n}, σ_pix={heatmap_sigma}, GRU, N_obs={n_obs} over {horizon}s, noise={cfg.data.obs_noise}, β={beta_kl})")
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
    fig.suptitle(f"Latent phase portrait vs true (q, p)  (heatmap obs N_pix={heatmap_n}, σ_pix={heatmap_sigma}, GRU, N_obs={n_obs} over {horizon}s, noise={cfg.data.obs_noise}, β={beta_kl})")
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
    fig.suptitle(f"z[1] vs true p  (heatmap obs N_pix={heatmap_n}, σ_pix={heatmap_sigma}, GRU, N_obs={n_obs} over {horizon}s, noise={cfg.data.obs_noise}, β={beta_kl})")
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

    # 4e. Posterior samples: predictive fan from q(z0|obs) ----------------
    # For each test amp, draw n_posterior_samples z0 from the posterior and
    # integrate. Faint lines = individual samples; solid = sample mean.
    # Tells us: is the posterior wide enough to cover truth? Or collapsed?
    n_samp = int(cfg.data.n_posterior_samples)
    with torch.no_grad():
        q_samples = posterior_samples(test_amps_t, t_plot, n_samp)   # (S, T, B)
    q_samples_np = q_samples.numpy()
    q_mean = q_samples.mean(dim=0)                                   # (T, B)
    q_std = q_samples.std(dim=0)                                     # (T, B)

    fig, axes = plt.subplots(1, nt, figsize=(3.6 * nt, 3.0), squeeze=False)
    for i, (ax, A, c) in enumerate(zip(axes[0], test_amps, colors)):
        true_q = truth_TB2[:, i, 0]
        ax.plot(t_plot, true_q, color="lightgray", lw=3, label="true q(t)")
        for s in range(n_samp):
            ax.plot(t_plot, q_samples_np[s, :, i], color=c, lw=0.4, alpha=0.25)
        ax.plot(t_plot, q_mean[:, i], color=c, lw=1.6, label="sample mean")
        seen = "train" if A in train_amps else "UNSEEN"
        ax.set_title(f"A={A} ({seen})")
        ax.set_xlabel("t")
    axes[0][0].set_ylabel("q"); axes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Posterior predictive samples  "
                 f"({n_samp} draws, σ={cfg.data.obs_noise}, β={beta_kl})")
    fig.tight_layout()
    p = os.path.join(out_dir, "posterior_samples.png")
    fig.savefig(p, dpi=120); log.info("saved plot -> %s", p)

    # 4f. Calibration: predicted std vs actual residual --------------------
    # If the posterior is well-calibrated, the empirical std of sampled q(t)
    # should be of the same order as |sample mean - truth|. If predicted std
    # << residual, posterior is overconfident; if >> residual, underconfident.
    residual = (q_mean - truth_TB2[..., 0]).abs()                    # (T, B)
    fig, axes = plt.subplots(1, nt, figsize=(3.6 * nt, 3.0), squeeze=False)
    for i, (ax, A, c) in enumerate(zip(axes[0], test_amps, colors)):
        ax.plot(t_plot, q_std[:, i], color=c, lw=1.6, label="predicted std")
        ax.plot(t_plot, residual[:, i], color="black", lw=1.0, ls="--",
                label="|mean − truth|")
        seen = "train" if A in train_amps else "UNSEEN"
        ax.set_title(f"A={A} ({seen})")
        ax.set_xlabel("t")
        ax.set_yscale("log")
    axes[0][0].set_ylabel("q error / spread"); axes[0][0].legend(fontsize=7)
    fig.suptitle(f"Calibration: posterior std vs actual residual  "
                 f"(σ={cfg.data.obs_noise}, β={beta_kl})")
    fig.tight_layout()
    p = os.path.join(out_dir, "calibration.png")
    fig.savefig(p, dpi=120); log.info("saved plot -> %s", p)


if __name__ == "__main__":
    main()
