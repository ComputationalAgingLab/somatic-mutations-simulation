import os
from typing import List, Dict, Union, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import pandas as pd

plt.style.use("bmh")

class Config:
    def __init__(self, 
                 organ: str, 
                 mu_distribution: str = "lognormal", 
                 mc_samples: int = 10_000):
        
        self.organ = organ.lower()
        self.mu_distribution = mu_distribution
        self.mc_samples = mc_samples

        if self.organ not in ["liver", "lung", "brain", "heart"]:
            raise ValueError("This organ is not available for the calculation.")

        if self.organ in ["brain", "heart"]:
            print("Exponential decay model is being simulated.")
        else:
            print("Model with self-renewal is being simulated.")

    def get_vals(self) -> Dict:
        values = {
            "brain": {
                "SNV": {"mean": 17.489, "CI_low": 16.112, "CI_high": 18.865},
                "indels": {"mean": 6.926, "CI_low": 5.484, "CI_high": 8.367},
                "p": {"SNV": 12.34e-5, "indels": 3.87e-5},
                "K": {"mean": 3.5e9, "se": 7e8},
                "x_crit": 0.6,
            },
            "heart": {
                "SNV": {"mean": 36.369, "CI_low": 19.519, "CI_high": 53.218},
                "indels": {"mean": 14.403, "CI_low": 6.644, "CI_high": 23.604},
                "p": {"SNV": 5.67e-5, "indels": 3.87e-5},
                "K": {"mean": 3.2e9, "se": 7.5e8},
                "x_crit": 0.55,
            },
            'liver':{
                'SNV':{'mean':52.783, 'CI_low':36.647, 'CI_high':68.92},
                'indels':{'mean':1.158, 'CI_low':0.721, 'CI_high':1.595},
                'p':{'SNV':2.37e-5, 'indels':8.74e-7},
                'r':{'mean':7.46, 'CI_low':4.58, 'CI_high':11.76},
                'K':{'mean':2.26e11, 'se':6.38e10},
                'x_crit': 0.2,
                'H': 90
            },
            'lung':{
                'SNV':{'mean':28.519, 'CI_low':20.725, 'CI_high':36.312},
                'indels':{'mean':2.573, 'CI_low':1.182, 'CI_high':3.963},
                'p':{'SNV':7.34e-5, 'indels':5.93e-9},
                'r_b':{'mean':89.7, 'CI_low':73.5, 'CI_high':142.3},
                'K':{'mean':1.03e9, 'se':2.5944e8},
                'x_crit': 0.23,
                'H': 170
            }
        }

        conf = values[self.organ]

        mu_mean = conf["SNV"]["mean"] * conf["p"]["SNV"] + conf["indels"]["mean"] * conf["p"]["indels"]
        snv_se = (conf["SNV"]["CI_high"] - conf["SNV"]["CI_low"]) / (2 * 1.96)
        indel_se = (conf["indels"]["CI_high"] - conf["indels"]["CI_low"]) / (2 * 1.96)
        mu_std = np.sqrt((snv_se * conf["p"]["SNV"]) ** 2 + (indel_se * conf["p"]["indels"]) ** 2)

        x0_mean = conf["K"]["mean"]
        x0_std = conf["K"].get("se", 0.1 * x0_mean)
        x_c = conf["x_crit"] * x0_mean

        cv = x0_std / x0_mean
        if cv > 0:
            sigma_lognormal = np.sqrt(np.log(1 + cv**2))
            mu_lognormal = np.log(x0_mean) - 0.5 * sigma_lognormal**2
        else:
            sigma_lognormal = 1e-6
            mu_lognormal = np.log(x0_mean)

        return {
            "organ": self.organ,
            "mu_mean": mu_mean,
            "mu_std": mu_std,
            "x0_mean": x0_mean,
            "x0_std": x0_std,
            "x_c": x_c,
            "mu_normal": x0_mean,
            "sigma_normal": x0_std,
            "mu_lognormal": mu_lognormal,
            "sigma_lognormal": sigma_lognormal,
            "lambda_bg": 0.0016133681587490935,
            "N": 8e9,
            "time_max": 20_000,
            "time_points": 20_000,
            "debug": False,
            "mu_distribution": self.mu_distribution,
            "mc_samples": self.mc_samples,
            "gh_nodes": 80,
        }

_precomputed_mus = {}

def get_mc_mus(conf, distribution: str):
    key = (distribution, conf["mu_mean"], conf["mu_std"], conf["mc_samples"])
    if key in _precomputed_mus:
        return _precomputed_mus[key]

    mu, sigma = conf["mu_mean"], conf["mu_std"]
    rng = np.random.default_rng(12345)

    if distribution == "lognormal":
        if sigma <= 0 or mu <= 0:
            mus = np.array([mu])
        else:
            cv = sigma / mu
            sigma_ln = np.sqrt(np.log(1 + cv**2))
            mu_ln = np.log(mu) - 0.5 * sigma_ln**2
            mus = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=conf["mc_samples"])
    else:
        raise ValueError("Monte Carlo only for lognormal distribution")

    _precomputed_mus[key] = mus

    return mus


def compute_survival_vectorized(conf, 
                                organ: str, 
                                mu_dist: str, 
                                K_dist: str, 
                                percentiles: list) -> None:
    if K_dist != "lognormal":
        raise NotImplementedError("Only lognormal K is supported in vectorized version.")

    x_c = conf["x_c"]
    mu_ln_K = conf["mu_lognormal"]
    sigma_ln_K = conf["sigma_lognormal"]
    lambda_bg = conf["lambda_bg"]
    time_max = conf["time_max"]
    time_points = conf["time_points"]

    t_vals = np.linspace(0, time_max, time_points)

    mu_samples = get_mc_mus(conf, mu_dist)

    thresholds = x_c * np.exp(np.outer(mu_samples, t_vals))
    thresholds = np.minimum(thresholds, 1e300)

    S_organ = 1.0 - lognorm.cdf(thresholds, s=sigma_ln_K, scale=np.exp(mu_ln_K))

    S_bg = np.exp(-lambda_bg * t_vals)

    S_total = S_organ * S_bg

    S_percentiles = {}
    for q in percentiles:
        S_percentiles[q] = np.percentile(S_total, 100 * q, axis=0)

    outdir = f"survival_Model_II_{organ}"
    os.makedirs(outdir, exist_ok=True)

    df_mean = pd.DataFrame({
        "time": t_vals,
        "S_combined(t)": np.mean(S_total, axis=0)
    })
    df_mean.to_csv(os.path.join(outdir, f"{organ}_mu-{mu_dist}_K-{K_dist}_mean.csv"), index=False)

    df_bands = pd.DataFrame({"time": t_vals})
    for q in percentiles:
        df_bands[f"S_q{q}"] = S_percentiles[q]
    df_bands.to_csv(os.path.join(outdir, f"{organ}_mu-{mu_dist}_K-{K_dist}_bands.csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, S_percentiles[0.5], label="Median S(t)", color="blue", linewidth=2)
    plt.fill_between(t_vals, S_percentiles[0.25], S_percentiles[0.75],
                     color="blue", alpha=0.3, label="50% CI")
    plt.fill_between(t_vals, S_percentiles[0.025], S_percentiles[0.975],
                     color="blue", alpha=0.15, label="95% CI")
    plt.xlabel("Time")
    plt.ylabel("S(t)")
    plt.title(f"{organ.capitalize()} — µ:{mu_dist}, K:{K_dist} — S(t) uncertainty bands")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{organ}_mu-{mu_dist}_K-{K_dist}_St_uncertainty.png"), dpi=200)
    plt.close()


def run_vectorized(organ: str, mc_samples: int = 20_000) -> None:

    percentiles = [0.975, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65,
                   0.6, 0.55, 0.5, 0.45, 0.4,
                   0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.025]

    config = Config(organ=organ, mu_distribution="lognormal", mc_samples=mc_samples)
    conf = config.get_vals()

    print(f"Running vectorized survival for {organ}...")
    compute_survival_vectorized(conf, organ, "lognormal", "lognormal", percentiles)
    print(f"Done. Results in survival_vectorized_{organ}/")

if __name__ == "__main__":
    run_vectorized("heart", mc_samples=10_000)
