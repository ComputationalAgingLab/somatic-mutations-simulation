import os
from typing import Tuple, List, Union, Any, Dict

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

def get_mc_mus(conf, distribution: str = "lognormal"):
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

def make_hazard_functions(conf: dict):
    mu_ln_K, sigma_ln_K = conf["mu_lognormal"], conf["sigma_lognormal"]
    x_c = conf["x_c"]
    lambda_bg = conf["lambda_bg"]

    mu_samples = get_mc_mus(conf, "lognormal")

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


def run_simulation_and_save(organ: str, 
                            mc_samples: int = 20_000, 
                            max_age: float = 100.0, 
                            n_time_points: int = 200,
                            sparse: bool = True) -> None:
    
    config = Config(organ=organ, mu_distribution="lognormal", mc_samples=mc_samples)
    conf = config.get_vals()

    organ_hazard_func, total_hazard_func = make_hazard_functions(conf)

    if sparse:
        t_early = np.linspace(0, 100, 1000)
        t_late = np.linspace(100, max_age, 500)
        t_vals = np.concatenate([t_early, t_late[1:]])
    else:
        t_vals = np.linspace(0, max_age, n_time_points)

    h_organ_vals = np.empty_like(t_vals)
    h_total_vals = np.empty_like(t_vals)

    for i, t in enumerate(t_vals):
        h_organ_vals[i] = organ_hazard_func(t)
        h_total_vals[i] = total_hazard_func(t)

    os.makedirs("hazard_heart", exist_ok=True)

    df = pd.DataFrame({
        "age_years": t_vals,
        "organ_hazard_rate": h_organ_vals,
        "total_hazard_rate": h_total_vals
    })

    csv_path = os.path.join("hazard_heart", f"{organ}_Model_II_hazard.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    color = 'tab:red'
    plt.ylabel('Hazard Rate (per year)', color=color)
    plt.plot(t_vals, h_organ_vals, color=color, linestyle='--', label="Total Hazard")
    plt.tick_params(axis='y', labelcolor=color)

    plt.title(f"Survival and Hazard for {organ.capitalize()}")
    plot_path = os.path.join("hazard_heart", f"{organ}_survival_hazard.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_simulation_and_save(organ="heart", mc_samples=10_000, max_age=10_000.0, n_time_points=10_000)
