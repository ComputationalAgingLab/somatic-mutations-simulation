from typing import Dict, Tuple
import numpy as np

class Config:
    def __init__(self, 
                 organ: str = "liver",
                 organ_s: str | None = None,
                 lambda_bg : float = 0.0016133681587490935,
                 N : float | int = 8e9,
                 time_max: float | int = 20_000,
                 time_points: int = 20_000,
                 mc_samples: int = 10_000) -> Dict:
        """
        Basic config class for model initialization

        Args:
        * organ: organ to simulate.
        * organ_s: for model IIIB, choosing the LPC if needed.
        * lambda_bg: background hazard rate
        * N: for the threshold (1/N)
        * time_max: max time for simulation
        * time_points: num of time points to sample
        * mc_samples: Monte-Carlo runs
        """
        self.organ = organ.lower()
        self.organ_s = organ_s
        self.N = N
        self.lambda_bg = lambda_bg
        self.mc_samples = mc_samples
        self.time_points = time_points
        self.time_max = time_max

        self.values = {
            'liver':{
                'SNV':{'mean':52.783, 'CI_low':36.647, 'CI_high':68.92},
                'indels':{'mean':1.158, 'CI_low':0.721, 'CI_high':1.595},
                'p':{'SNV':2.37e-5, 'indels':8.74e-7},
                'r':{'mean':7.46, 'CI_low':4.58, 'CI_high':11.76},
                'K':{'mean':2.26e11, 'se':6.38e10},
                'x_crit': 0.2,
                'H': 90
            },
            'LPC':{
                'SNV':{'mean':33.726, 'CI_low':23.415, 'CI_high':44.036},
                'indels':{'mean':0.693, 'CI_low':0.432, 'CI_high':0.955},
                'p':{'SNV':2.37e-5, 'indels':3.17e-8},
                'K': {'mean':7.26e9, 'se': 4.8e9}
            },
            'lungs':{
                'SNV':{'mean':28.519, 'CI_low':20.725, 'CI_high':36.312},
                'indels':{'mean':2.573, 'CI_low':1.182, 'CI_high':3.963},
                'p':{'SNV':7.34e-5, 'indels':5.93e-9},
                'r':{'mean':89.7, 'CI_low':73.5, 'CI_high':142.3},
                'K':{'mean':1.03e9, 'se':2.5944e8},
                'x_crit': 0.23,
                'H': 170,
                'fbbfxx': 1.1,
            },
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
            }
        }

        if self.organ not in self.values:
            raise ValueError(f"Organ '{organ}' not available. Available: {list(self.values.keys())}")
        
        if self.organ in ["brain", "heart"]:
            self.model = "II"
            print("Exponential decay model is being simulated.")
        else:
            self.model = "III"
            print("Model with self-renewal is being simulated.")

    def _calculate_mean_mu(self, organ: str) -> float:
        conf = self.values[organ]

        mu_mean = (conf['SNV']['mean'] * conf['p']['SNV'] +
                     conf['indels']['mean'] * conf['p']['indels'])
        
        return mu_mean
    
    def _convert_ci_se(self, organ: str, param: str) -> float:
        conf = self.values[organ]

        se = (conf[param]["CI_high"] - conf[param]["CI_low"]) / (2 * 1.96)

        return se
    
    def _calculate_std_sum(self, organ, snv_se, indel_se) -> float:
        conf = self.values[organ]

        std = np.sqrt((snv_se * conf["p"]["SNV"]) ** 2 + (indel_se * conf["p"]["indels"]) ** 2)

        return std

    def _get_se_mean_mu(self, organ: str) -> Tuple[float, float]:
        mu_mean = self._calculate_mean_mu(organ=organ)

        snv_se = self._convert_ci_se(organ=organ, param="SNV")

        indel_se = self._convert_ci_se(organ=organ, param="indels")

        mu_std = self._calculate_std_sum(organ=organ, snv_se=snv_se, indel_se=indel_se)

        return mu_mean, mu_std
    
    def _get_se_r(self, organ: str) -> Tuple[float, float]:
        r_se = self._convert_ci_se(organ=organ, param="r")

        r_mean = self.values[organ]["r"]["mean"]

        return r_mean, r_se
    
    def _get_params_model_three(self) -> Dict:
        conf = self.values[self.organ]

        mu_mean, mu_std = self._get_se_mean_mu(organ=self.organ)

        r_mean, r_se = self._get_se_r(organ=self.organ)

        param_fixed = {
            "H": conf['H']
        }

        param_specs = {
            "r": (r_mean, r_se),
            "K": (conf['K']['mean'], conf['K']['se']),
            "mu": (mu_mean, mu_std)
        }

        initial_cond = {
            "P0": conf['H']
        }

        x_crit = conf['x_crit'] * conf['K']['mean']

        if self.organ_s:
            conf_s = self.values[self.organ_s]
            param_fixed["r_s"] = 0.112
            mu_s_mean, mu_s_std = self._get_se_mean_mu(organ=self.organ_s)
            param_specs["mu_s"] = (mu_s_mean, mu_s_std)
            param_specs["Q"] = (conf_s["K"]["mean"], conf_s["K"]["se"])

        return {
            "organ": self.organ,
            "fixed": param_fixed,
            "sampled": param_specs,
            "initial": initial_cond,
            "x_c": x_crit,
            "lambda_bg": self.lambda_bg,
            "N": self.N,
            "time_max": self.time_max,
            "time_points": self.time_points,
            "mc_samples": self.mc_samples,
        }
    
    def _get_params_model_two(self) -> Dict:
        conf = self.values[self.organ]

        mu_mean, mu_std = self._get_se_mean_mu(organ=self.organ)

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
            "mu_lognormal": mu_lognormal,
            "sigma_lognormal": sigma_lognormal,
            "lambda_bg": self.lambda_bg,
            "N": self.N,
            "time_max": self.time_max,
            "time_points": self.time_points,
            "mc_samples": self.mc_samples,
        }
    
    def get_params(self) -> Dict:
        """
        Method for getting parameters for the simulation

        ** Args **
        - Every arg is given in the __init__
        """
        if self.model == "II":
            return self._get_params_model_two()
        else:
            return self._get_params_model_three()