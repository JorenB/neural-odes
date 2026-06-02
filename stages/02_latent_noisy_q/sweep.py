"""
Stage 2 sweep -- rollout MSE as a function of observation noise σ.

Retrains the latent ODE from scratch at each σ in `sigmas` (so we measure
how the *trained* model degrades, not just how robust a single fixed model
is). Plots per-amplitude rollout MSE vs σ to show where the deterministic
encoder breaks down.

Run from this directory:
    uv run python sweep.py

~3 minutes total: 6 σ values * ~30 s per 2000-step training run.
"""

import copy
import json
import logging
import os
import sys

import hydra
import matplotlib
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torchdiffeq import odeint

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same-stage import: reuse Stage 2's classes so we stay in sync with train.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import PendulumTruth, Encoder, ODEFunc, decode_q

log = logging.getLogger(__name__)


def train_and_eval(cfg: DictConfig, sigma: float):
    """Train + best-model select at noise sigma. Return per-amp eval MSE dict."""
    torch.manual_seed(cfg.seed)

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
    local_t = torch.arange(T, dtype=torch.float32) * dt

    encoder = Encoder(k_enc=k_enc, latent_dim=cfg.model.latent_dim,
                      hidden=cfg.model.hidden)
    func = ODEFunc(dim=cfg.model.latent_dim, hidden=cfg.model.hidden)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(func.parameters()), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.steps, eta_min=cfg.train.lr * 0.01)

    phase_max = max(0.0, t_max - dt * (T - 1))
    B = cfg.train.batch_size

    def sample_batch():
        A_idx = torch.randint(0, len(train_amps_t), (B,))
        A = train_amps_t[A_idx]
        phase0 = torch.rand(B) * phase_max
        times = phase0[None, :] + local_t[:, None]
        q_obs = truth.state(A, times)[..., 0]
        return q_obs + sigma * torch.randn_like(q_obs)

    def rollout(A_t, t_grid):
        t_enc = (torch.arange(k_enc, dtype=torch.float32) * dt)[:, None].expand(
            k_enc, len(A_t))
        q_enc = truth.state(A_t, t_enc)[..., 0].T
        q_enc = q_enc + sigma * torch.randn_like(q_enc)
        z0 = encoder(q_enc)
        z = odeint(func, z0, t_grid, method=method)
        return decode_q(z)

    @torch.no_grad()
    def eval_mse():
        t_plot = torch.linspace(0.0, t_max, cfg.data.n_plot)
        truth_TB = truth.state(test_amps_t, t_plot)[..., 0]
        acc = torch.zeros(len(test_amps))
        for _ in range(cfg.data.n_eval_seeds):
            q_pred = rollout(test_amps_t, t_plot)
            acc += ((q_pred - truth_TB) ** 2).mean(dim=0)
        acc /= cfg.data.n_eval_seeds
        return {A: acc[i].item() for i, A in enumerate(test_amps)}

    best_err = float("inf")
    best_state = (copy.deepcopy(encoder.state_dict()),
                  copy.deepcopy(func.state_dict()))
    for step in range(1, cfg.train.steps + 1):
        optimizer.zero_grad()
        q_obs = sample_batch()
        z0 = encoder(q_obs[:k_enc].T)
        z = odeint(func, z0, local_t, method=method)
        loss = torch.mean((decode_q(z) - q_obs) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % cfg.train.log_every == 0:
            errs = eval_mse()
            seen_err = sum(errs[A] for A in train_amps) / len(train_amps)
            if seen_err < best_err:
                best_err = seen_err
                best_state = (copy.deepcopy(encoder.state_dict()),
                              copy.deepcopy(func.state_dict()))

    encoder.load_state_dict(best_state[0])
    func.load_state_dict(best_state[1])
    return eval_mse()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    out_dir = HydraConfig.get().runtime.output_dir
    log.info("noise sweep output dir: %s", out_dir)

    sigmas = [0.0, 0.025, 0.05, 0.1, 0.2, 0.5]
    test_amps = list(cfg.data.test_amplitudes)
    train_amps = list(cfg.data.amplitudes)

    results = {}                                        # sigma -> {A: mse}
    for sigma in sigmas:
        log.info("=== training at sigma=%.3f ===", sigma)
        results[sigma] = train_and_eval(cfg, sigma)
        log.info("sigma=%.3f  MSE: %s", sigma,
                 "  ".join(f"A={A}:{results[sigma][A]:.2e}" for A in test_amps))

    with open(os.path.join(out_dir, "sweep_results.json"), "w") as f:
        json.dump({f"{s:.3f}": v for s, v in results.items()}, f, indent=2)

    # --- Plot MSE vs σ per amplitude --------------------------------------
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(test_amps) - 1)) for i in range(len(test_amps))]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, A in enumerate(test_amps):
        ys = [results[s][A] for s in sigmas]
        seen = "train" if A in train_amps else "UNSEEN"
        ax.plot(sigmas, ys, marker="o", color=colors[i],
                label=f"A={A} ({seen})", lw=2)
    # Reference: σ² noise floor on q (lower bound on what reconstruction can achieve)
    ax.plot(sigmas, [s**2 for s in sigmas], "k--", lw=1, alpha=0.5,
            label="σ² (irreducible noise floor)")
    ax.set_xlabel("observation noise σ")
    ax.set_ylabel("rollout MSE on q  (averaged over noise seeds)")
    ax.set_yscale("log")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_title("Stage 2 noise sweep: when does the deterministic encoder break?\n"
                 "(retrained from scratch at each σ; best-model snapshot)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "noise_sweep.png")
    fig.savefig(p, dpi=120)
    log.info("saved %s", p)


if __name__ == "__main__":
    main()
