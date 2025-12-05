import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm

from src.utils.conf_base import Config

_precomputed_mus = {}

def get_mc_mus(mu_mean: float, mu_std: float, mc_samples: int, seed: int):
    """Generate lognormal mu samples."""
    key = (mu_mean, mu_std, mc_samples)
    if key in _precomputed_mus:
        return _precomputed_mus[key]

    rng = np.random.default_rng(seed)
    if mu_std <= 0 or mu_mean <= 0:
        mus = np.full(mc_samples, mu_mean)
    else:
        cv = mu_std / mu_mean
        sigma_ln = np.sqrt(np.log(1 + cv**2))
        mu_ln = np.log(mu_mean) - 0.5 * sigma_ln**2
        mus = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=mc_samples)

    _precomputed_mus[key] = mus
    return mus

def make_hazard_functions(conf: dict, seed: int):
    mu_ln_K, sigma_ln_K = conf["mu_lognormal"], conf["sigma_lognormal"]
    x_c = conf["x_c"]
    lambda_bg = conf["lambda_bg"]

    mu_samples = get_mc_mus(
            mu_mean=conf["mu_mean"],
            mu_std=conf["mu_std"],
            mc_samples=conf["mc_samples"],
            seed=seed
        )

    def organ_hazard(t):
        if t <= 0:
            return 0.0

        thresholds = x_c * np.exp(mu_samples * t)
        thresholds = np.clip(thresholds, 1e-300, 1e300)

        pdf_vals = lognorm.pdf(thresholds, s=sigma_ln_K, scale=np.exp(mu_ln_K))
        cdf_vals = lognorm.cdf(thresholds, s=sigma_ln_K, scale=np.exp(mu_ln_K))

        weights = x_c * mu_samples * np.exp(mu_samples * t)
        numerator = np.mean(pdf_vals * weights)
        denominator = np.mean(1.0 - cdf_vals)

        if denominator <= 0:
            return np.inf
        return numerator / denominator

    def total_hazard(t):
        return organ_hazard(t) + lambda_bg

    return organ_hazard, total_hazard


def compute_hazard(organ: str,
                    n_mc: int = 10_000,
                    t_max: float = 10_000.0,
                    time_points: int = 10_000,
                    sparse: bool = False,
                    outdir: str = "/results",
                    seed: int = 12345
                ):

    if organ not in ["brain", "heart"]:
        raise ValueError("Hazard model only supports 'brain' or 'heart' (Model II).")
    
    config = Config(organ=organ, time_max=t_max, time_points=time_points, mc_samples=n_mc)
    conf = config.get_params()

    organ_hazard_func, total_hazard_func = make_hazard_functions(conf, seed)

    if sparse:
        t_early = np.linspace(0, 100, 1000)
        t_late = np.linspace(100, t_max, 500)
        t_vals = np.concatenate([t_early, t_late[1:]])
    else:
        t_vals = np.linspace(0, t_max, time_points)

    h_organ_vals = np.empty_like(t_vals)
    h_total_vals = np.empty_like(t_vals)

    for i, t in enumerate(t_vals):
        h_organ_vals[i] = organ_hazard_func(t)
        h_total_vals[i] = total_hazard_func(t)

    os.makedirs(outdir, exist_ok=True)

    df = pd.DataFrame({
        "age_years": t_vals,
        "organ_hazard_rate": h_organ_vals,
        "total_hazard_rate": h_total_vals
    })

    csv_path = os.path.join(outdir, f"{organ}_Model_II_hazard.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    color = 'tab:red'
    plt.ylabel('Hazard Rate (per year)', color=color)
    plt.plot(t_vals, h_organ_vals, color=color, linestyle='--', label="Total Hazard")
    plt.tick_params(axis='y', labelcolor=color)

    plt.title(f"Survival and Hazard for {organ.capitalize()}")
    plot_path = os.path.join(outdir, f"{organ}_survival_hazard.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    plt.show()
