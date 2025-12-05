#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.conf_base import Config

def run_model_ii(
    organ: str,
    n_mc: int = 20_000,
    n_traj: int = 5_000,
    t_max: int = 100_000,
    n_common_points: int = 10_000,
    seed: int = 123,
    outdir: str = "results/model_ii",
    N: float | int = 8e9
) -> dict:
    
    organ = organ.lower()
    if organ not in ["brain", "heart"]:
        raise ValueError("Model II only supports 'brain' or 'heart'")

    full_outdir = os.path.join(outdir, organ)
    os.makedirs(full_outdir, exist_ok=True)
    print(f"Running Model II for {organ}...")

    cfg = Config(organ=organ, N=N, time_max=t_max, time_points=n_common_points, mc_samples=n_mc)
    params = cfg.get_params()

    mu_mean = params["mu_mean"]
    mu_std = params["mu_std"]
    x0_mean = params["x0_mean"]
    x0_std = params["x0_std"]
    x_crit = params["x_c"]
    lambda_bg = cfg.lambda_bg

    t_vals = np.linspace(0, t_max, n_common_points)
    rng = np.random.default_rng(seed)

    def to_lognorm(mean, std):
        if mean <= 0 or std <= 0:
            return np.log(mean), 1e-12
        cv2 = (std / mean) ** 2
        sigma = np.sqrt(np.log(1 + cv2))
        mu_ln = np.log(mean) - 0.5 * sigma ** 2
        return mu_ln, sigma

    mu_ln, mu_sigma = to_lognorm(mu_mean, mu_std)
    K_ln, K_sigma = to_lognorm(x0_mean, x0_std)

    mu_samples_full = rng.lognormal(mu_ln, mu_sigma, n_mc)
    K_samples_full = rng.lognormal(K_ln, K_sigma, n_mc)

    X_full = K_samples_full[:, None] * np.exp(-mu_samples_full[:, None] * t_vals[None, :])
    S_organ = np.mean(X_full > x_crit, axis=0)
    S_bg = np.exp(-lambda_bg * t_vals)
    S_total = S_organ * S_bg

    if n_traj < n_mc:
        idx = rng.choice(n_mc, size=n_traj, replace=False)
        K_traj = K_samples_full[idx]
        mu_traj = mu_samples_full[idx]
    else:
        K_traj = K_samples_full
        mu_traj = mu_samples_full

    X_traj = K_traj[:, None] * np.exp(-mu_traj[:, None] * t_vals[None, :])

    pd.DataFrame({
        "time": t_vals,
        "S_organ(t)": S_organ,
        "S_background(t)": S_bg,
        "S_combined(t)": S_total
    }).to_csv(os.path.join(full_outdir, "survival_mean.csv"), index=False)

    S_total_samples = (X_traj > x_crit).astype(float) * S_bg
    perc_surv = [2.5, 25, 50, 75, 97.5]
    df_bands = pd.DataFrame({"time": t_vals})
    for p in perc_surv:
        df_bands[f"S_total_p{p}"] = np.percentile(S_total_samples, p, axis=0)
    df_bands.to_csv(os.path.join(full_outdir, "survival_percentiles.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(t_vals, S_total, color="blue", label="Mean S(t)")
    plt.fill_between(t_vals, df_bands["S_total_p2.5"], df_bands["S_total_p97.5"], alpha=0.2)
    plt.xlabel("Time"); plt.ylabel("Survival"); plt.title(f"Model II — {organ}")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(full_outdir, "survival_plot.png"), dpi=200)
    plt.close()

    traj_perc = [2.5] + list(range(5, 100, 5)) + [97.5]
    df_traj = pd.DataFrame({"time": t_vals})
    for p in traj_perc:
        df_traj[f"X_p{p}"] = np.percentile(X_traj, p, axis=0)
    df_traj.to_csv(os.path.join(full_outdir, "trajectories_percentiles.csv"), index=False)

    plt.figure(figsize=(8, 5))
    median_X = np.percentile(X_traj, 50, axis=0)
    low_X = np.percentile(X_traj, 2.5, axis=0)
    high_X = np.percentile(X_traj, 97.5, axis=0)

    norm_median = median_X / x0_mean
    norm_low = low_X / x0_mean
    norm_high = high_X / x0_mean
    norm_crit = x_crit / x0_mean

    plt.plot(t_vals, norm_median, color="purple", linewidth=2, label="Median X(t)")
    plt.fill_between(
        t_vals, 
        norm_low, 
        norm_high, 
        color="purple",
        alpha=0.3,
        label="95% CI"
    )
    plt.axhline(norm_crit, color="red", linestyle="--", linewidth=1.5, label="Critical threshold")

    plt.xlabel("Time")
    plt.ylabel("X(t) / K_mean")
    plt.title(f"Model II — {organ.capitalize()}")
    plt.grid(True, alpha=0.3)
    plt.xlim(0,1000)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(full_outdir, "trajectories_plot.png"), dpi=200)
    plt.close()

    print(f"Model II for {organ} done. Output: {full_outdir}")
    return {"t": t_vals, "S_total": S_total, "X_traj": X_traj}