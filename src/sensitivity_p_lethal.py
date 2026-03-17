import os
import numpy as np
from scipy.stats import lognorm as _lognorm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.utils.conf_base import Config
from typing import Dict


def _to_lognorm(mean: float, std: float):
    """Convert mean/std parameterisation to lognormal (mu_ln, sigma)."""
    if mean <= 0 or std <= 0:
        return np.log(abs(mean) + 1e-300), 1e-12
    cv2 = (std / mean) ** 2
    sigma = np.sqrt(np.log(1 + cv2))
    mu_ln = np.log(mean) - 0.5 * sigma ** 2
    return mu_ln, sigma


def _mu_from_p(snv_mean: float, snv_se: float,
               indels_mean: float, indels_se: float,
               p_snv: np.ndarray, p_indels: np.ndarray):
    """
    Compute exact mu_mean and mu_std arrays for given p_SNV and p_indels arrays.

    Args:
        snv_mean: mean SNV rate
        snv_se: standard error of SNV rate
        indels_mean: mean indels rate
        indels_se: standard error of indels rate
        p_snv: array of sampled p_SNV values
        p_indels: array of sampled p_indels values

    Returns:
        Tuple of (mu_mean_arr, mu_std_arr), both shape (n_samples,)
    """
    mu_mean = snv_mean * p_snv + indels_mean * p_indels
    mu_std  = np.sqrt((snv_se * p_snv) ** 2 + (indels_se * p_indels) ** 2)
    return mu_mean, mu_std


def run_sensitivity_p_lethal(
        organ: str,
        organ_s: str | None = None,
        n_samples: int = 50,
        p_min: float = 1e-7,
        p_max: float = 1e-3,
        p_lpc_min: float | None = None,
        p_lpc_max: float | None = None,
        n_mc: int = 5_000,
        n_workers: int = 4,
        t_max: int = 100_000,
        n_common_points: int = 5_000,
        seed: int = 42,
        outdir: str = "results/sensitivity_p_lethal",
        N: float | int = 8e9,
) -> Dict:
    """
    Sensitivity analysis by independently sampling p_SNV and p_indels
    from a log-Uniform distribution.

    For each sample, p_SNV and p_indels are drawn independently from
    LogUniform(p_min, p_max), then mu_mean and mu_std are computed exactly:

        mu_mean = SNV_mean * p_SNV + indels_mean * p_indels
        mu_std  = sqrt((SNV_se * p_SNV)^2 + (indels_se * p_indels)^2)

    For Model IIIB (liver+LPC), p_SNV_LPC and p_indels_LPC are sampled
    independently from LogUniform(p_lpc_min, p_lpc_max). If p_lpc_min /
    p_lpc_max are not provided, the organ interval [p_min, p_max] is reused.

    Args:
        organ: organ to simulate (brain, heart, liver, lungs)
        organ_s: 'LPC' for Model IIIB (liver only)
        n_samples: number of (p_SNV, p_indels) pairs to draw
        p_min: lower bound of the organ log-uniform sampling interval
        p_max: upper bound of the organ log-uniform sampling interval
        p_lpc_min: lower bound for LPC sampling (defaults to p_min if None)
        p_lpc_max: upper bound for LPC sampling (defaults to p_max if None)
        n_mc: Monte Carlo runs per sample
        n_workers: parallel workers (Model III only)
        t_max: maximum simulation time
        n_common_points: time-grid resolution
        seed: base random seed
        outdir: root output directory
        N: population size / threshold (1/N)

    Returns:
        dict with keys: t_vals, p_SNV_samples, p_indels_samples,
        mu_mean_arr, S_matrix, median_survival, bands
    """
    organ = organ.lower()
    label = f"{organ}_{organ_s}" if organ_s else organ
    full_outdir = os.path.join(outdir, label)
    os.makedirs(full_outdir, exist_ok=True)
    print(f"Sensitivity analysis (p_SNV, p_indels) for {label} — {n_samples} samples ...")

    cfg = Config(
        organ=organ, organ_s=organ_s, N=N,
        time_max=t_max, time_points=n_common_points, mc_samples=n_mc
    )
    params    = cfg.get_params()
    conf_vals = cfg.values[organ]

    snv_mean    = conf_vals["SNV"]["mean"]
    indels_mean = conf_vals["indels"]["mean"]
    snv_se      = cfg._convert_ci_se(organ, "SNV")
    indels_se   = cfg._convert_ci_se(organ, "indels")

    p_snv_base    = conf_vals["p"]["SNV"]
    p_indels_base = conf_vals["p"]["indels"]

    rng          = np.random.default_rng(seed)
    log_samples  = rng.uniform(np.log(p_min), np.log(p_max), size=(n_samples, 2))
    p_snv_arr    = np.exp(log_samples[:, 0])
    p_indels_arr = np.exp(log_samples[:, 1])

    mu_mean_arr, mu_std_arr = _mu_from_p(
        snv_mean, snv_se, indels_mean, indels_se, p_snv_arr, p_indels_arr
    )

    if organ_s:
        conf_lpc        = cfg.values[organ_s]
        snv_lpc_mean    = conf_lpc["SNV"]["mean"]
        indels_lpc_mean = conf_lpc["indels"]["mean"]
        snv_lpc_se      = cfg._convert_ci_se(organ_s, "SNV")
        indels_lpc_se   = cfg._convert_ci_se(organ_s, "indels")

        lpc_lo = np.log(p_lpc_min if p_lpc_min is not None else p_min)
        lpc_hi = np.log(p_lpc_max if p_lpc_max is not None else p_max)

        log_lpc          = rng.uniform(lpc_lo, lpc_hi, size=(n_samples, 2))
        p_snv_lpc_arr    = np.exp(log_lpc[:, 0])
        p_indels_lpc_arr = np.exp(log_lpc[:, 1])

        mu_s_mean_arr, mu_s_std_arr = _mu_from_p(
            snv_lpc_mean, snv_lpc_se, indels_lpc_mean, indels_lpc_se,
            p_snv_lpc_arr, p_indels_lpc_arr
        )
    else:
        p_snv_lpc_arr = p_indels_lpc_arr = mu_s_mean_arr = mu_s_std_arr = None

    t_vals   = np.linspace(0, t_max, n_common_points)
    S_bg     = np.exp(-cfg.lambda_bg * t_vals)
    S_matrix = np.zeros((n_samples, n_common_points))

    if cfg.model == "II":
        _sensitivity_model_ii(
            params, mu_mean_arr, mu_std_arr, t_vals, S_bg, n_mc, seed, S_matrix
        )
    else:
        _sensitivity_model_iii(
            organ, organ_s, params,
            mu_mean_arr, mu_std_arr, mu_s_mean_arr, mu_s_std_arr,
            t_vals, S_bg, t_max, n_mc, n_workers, seed, S_matrix
        )

    median_survival = _compute_median_survival(S_matrix, t_vals)

    survival_thresholds = [0.5, 0.01, 0.001, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    df_thresh = _compute_threshold_times(S_matrix, t_vals, survival_thresholds)

    sweep_dict = {
        "p_SNV":           p_snv_arr,
        "p_indels":        p_indels_arr,
        "mu_mean":         mu_mean_arr,
        "median_survival": median_survival,
    }
    if organ_s:
        sweep_dict["p_SNV_LPC"]    = p_snv_lpc_arr
        sweep_dict["p_indels_LPC"] = p_indels_lpc_arr
        sweep_dict["mu_s_mean"]    = mu_s_mean_arr

    df_sweep = (
        pd.DataFrame(sweep_dict)
        .join(df_thresh)
        .sort_values("mu_mean")
        .reset_index(drop=True)
    )
    df_sweep.to_csv(os.path.join(full_outdir, "p_lethal_sweep.csv"), index=False)

    percentiles = [2.5, 25, 50, 75, 97.5]
    df_bands = pd.DataFrame({"time": t_vals})
    for p in percentiles:
        df_bands[f"S_p{p}"] = np.percentile(S_matrix, p, axis=0)
    df_bands.to_csv(os.path.join(full_outdir, "survival_sensitivity_bands.csv"), index=False)

    df_full = pd.DataFrame(S_matrix, columns=[f"t{j}" for j in range(n_common_points)])
    df_full.insert(0, "p_indels", p_indels_arr)
    df_full.insert(0, "p_SNV",    p_snv_arr)
    df_full.to_csv(os.path.join(full_outdir, "S_matrix.csv"), index=False)

    _plot_survival_bands(t_vals, df_bands, label, full_outdir)
    _plot_survival_by_mu(t_vals, S_matrix, mu_mean_arr, label, full_outdir)
    _plot_median_scatter(
        df_sweep, p_snv_base, p_indels_base, label, full_outdir
    )

    print(f"Done. Results in: {full_outdir}")
    return {
        "t_vals":           t_vals,
        "p_SNV_samples":    p_snv_arr,
        "p_indels_samples": p_indels_arr,
        "mu_mean_arr":      mu_mean_arr,
        "S_matrix":         S_matrix,
        "median_survival":  median_survival,
        "bands":            df_bands,
    }

def _sensitivity_model_ii(
        params, mu_mean_arr, mu_std_arr, t_vals, S_bg, n_mc, seed, S_matrix
):
    x_crit  = params["x_c"]
    x0_mean = params["x0_mean"]
    x0_std  = params["x0_std"]
    K_ln, K_sigma = _to_lognorm(x0_mean, x0_std)

    for i, (mu_mean_i, mu_std_i) in enumerate(zip(mu_mean_arr, mu_std_arr)):
        mu_ln_i, mu_sigma_i = _to_lognorm(mu_mean_i, mu_std_i)

        rng_i   = np.random.default_rng(seed + i + 1)
        mu_samp = rng_i.lognormal(mu_ln_i, mu_sigma_i, n_mc)

        thresholds = x_crit * np.exp(np.outer(mu_samp, t_vals))
        np.clip(thresholds, None, 1e300, out=thresholds)
        S_organ_per_mu = 1.0 - _lognorm.cdf(thresholds, s=K_sigma, scale=np.exp(K_ln))
        S_matrix[i] = np.mean(S_organ_per_mu, axis=0) * S_bg

        if (i + 1) % max(1, len(mu_mean_arr) // 5) == 0:
            print(f"  sample {i + 1}/{len(mu_mean_arr)}")


def _sensitivity_model_iii(
        organ, organ_s, params,
        mu_mean_arr, mu_std_arr, mu_s_mean_arr, mu_s_std_arr,
        t_vals, S_bg, t_max, n_mc, n_workers, seed, S_matrix
):
    from src.utils.ode_sim import monte_carlo_parallel
    from src.utils.rhs_factory import rhs_base
    from src.utils.kaplan_meier_utils import kaplan_meier

    x_crit       = params["x_c"]
    initial_cond = params["initial"]
    param_fixed  = params["fixed"]
    rhs_func     = rhs_base(organ=organ, organ_s=organ_s)

    for i, (mu_mean_i, mu_std_i) in enumerate(zip(mu_mean_arr, mu_std_arr)):
        param_specs_i       = dict(params["sampled"])
        param_specs_i["mu"] = (mu_mean_i, mu_std_i)

        if organ_s and mu_s_mean_arr is not None:
            param_specs_i["mu_s"] = (float(mu_s_mean_arr[i]), float(mu_s_std_arr[i]))

        death_times, _ = monte_carlo_parallel(
            n_runs=n_mc,
            initial_cond=initial_cond,
            param_fixed=param_fixed,
            param_specs=param_specs_i,
            x_crit=x_crit,
            t_max=t_max,
            n_workers=n_workers,
            save_traces=0,
            seed=seed + i + 1,
            rhs_factory=rhs_func,
            organ_s=organ_s,
        )

        censored    = np.isnan(death_times)
        event_times = np.where(censored, t_max, death_times)
        km_t, km_s, _, _, _, _ = kaplan_meier(event_times, censored)
        km_interp   = np.interp(t_vals, km_t, km_s, left=1.0, right=float(km_s[-1]))
        S_matrix[i] = km_interp * S_bg

        if (i + 1) % max(1, len(mu_mean_arr) // 5) == 0:
            print(f"  sample {i + 1}/{len(mu_mean_arr)}")

def _compute_median_survival(S_matrix: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    medians = np.empty(S_matrix.shape[0])
    for i, s in enumerate(S_matrix):
        idx = np.where(s <= 0.5)[0]
        medians[i] = float(t_vals[idx[0]]) if len(idx) > 0 else np.nan
    return medians


def _compute_threshold_times(S_matrix: np.ndarray, t_vals: np.ndarray,
                              thresholds: list) -> pd.DataFrame:
    """
    For each row of S_matrix, find the first time S(t) drops to or below
    each threshold value.

    Args:
        S_matrix: survival matrix, shape (n_samples, n_time)
        t_vals: time grid, shape (n_time,)
        thresholds: list of survival thresholds (e.g. [0.5, 0.1, 0.01])

    Returns:
        DataFrame with columns t_S{threshold} for each threshold
    """
    rows = []
    for s in S_matrix:
        row = {}
        for thr in thresholds:
            idx = np.where(s <= thr)[0]
            row[f"t_S{thr}"] = float(t_vals[idx[0]]) if len(idx) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def _plot_survival_bands(t_vals, df_bands, label, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(t_vals, df_bands["S_p2.5"],  df_bands["S_p97.5"],
                    alpha=0.18, color="steelblue", label="95 % band")
    ax.fill_between(t_vals, df_bands["S_p25"],   df_bands["S_p75"],
                    alpha=0.35, color="steelblue", label="IQR band")
    ax.plot(t_vals, df_bands["S_p50"], color="steelblue", linewidth=2, label="Median S(t)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival S(t)")
    ax.set_title(f"p_lethal sensitivity — {label.capitalize()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "survival_sensitivity_bands.png"), dpi=200)
    plt.close(fig)


def _plot_survival_by_mu(t_vals, S_matrix, mu_mean_arr, label, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    sort_idx = np.argsort(mu_mean_arr)
    log_mu   = np.log10(mu_mean_arr)
    norm     = plt.Normalize(log_mu.min(), log_mu.max())
    cmap     = cm.viridis

    for i in sort_idx:
        ax.plot(t_vals, S_matrix[i],
                color=cmap(norm(log_mu[i])), alpha=0.55, linewidth=0.7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("log₁₀(μ_mean)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival S(t)")
    ax.set_title(f"S(t) by μ_mean — {label.capitalize()}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "survival_by_mu.png"), dpi=200)
    plt.close(fig)


def _plot_median_scatter(df_sweep, p_snv_base, p_indels_base, label, outdir):
    df_plot = df_sweep.dropna(subset=["median_survival"])
    fig, ax = plt.subplots(figsize=(7, 6))

    sc = ax.scatter(
        df_plot["p_SNV"], df_plot["p_indels"],
        c=df_plot["median_survival"], cmap="plasma",
        s=40, alpha=0.85, zorder=3
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Median Survival Time")

    ax.scatter(p_snv_base, p_indels_base,
               marker="*", s=220, color="black", zorder=5,
               label=f"baseline ({p_snv_base:.2e}, {p_indels_base:.2e})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p_SNV (log scale)")
    ax.set_ylabel("p_indels (log scale)")
    ax.set_title(f"Median Survival — (p_SNV, p_indels) — {label.capitalize()}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "median_survival_scatter.png"), dpi=200)
    plt.close(fig)
