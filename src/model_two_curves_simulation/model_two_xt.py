import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict

class ExponentialDecaySimulator:
    """
    Simulates X(t) = K * exp(-μ * t) with lognormal-lognormal parameter sampling
    """
    def __init__(self, 
                 n_samples=1000, 
                 t_max=5000, 
                 n_time_points=500, 
                 organ: str = "brain"):
        
        self.n_samples = n_samples
        self.t_max = t_max
        self.n_time_points = n_time_points
        self.time_points = np.linspace(0, t_max, n_time_points)
        self.results = {}
        self.organ = organ

    def config(self) -> Dict:
        """
        Create a config of parameter values for the simulation
        """
        config_all = {"heart":
                {
                "SNV": {"mean": 36.369, "CI_low": 19.519, "CI_high": 53.218},
                "indels": {"mean": 14.403, "CI_low": 6.644, "CI_high": 23.604},
                "p": {"SNV": 5.67e-5, "indels": 3.87e-5},
                "K": {"mean": 3.2e9, "se": 7.5e8},
                "x_crit": 0.55
                },
            "brain":
                {
                "SNV": {"mean": 17.489, "CI_low": 16.112, "CI_high": 18.865},
                "indels": {"mean": 6.926, "CI_low": 5.484, "CI_high": 8.367},
                "p": {"SNV": 12.34e-5, "indels": 3.87e-5},
                "K": {"mean": 3.5e9, "se": 7e8},
                "x_crit": 0.6
                }
            }
        
        return config_all.get(self.organ, "brain")
    
    def generate_params(self):
        """
        Generate K and mu params (mean, std)
        """
        conf = self.config()

        alpha_mean = conf["SNV"]["mean"] * conf["p"]["SNV"] + conf["indels"]["mean"] * conf["p"]["indels"]
        snv_se = (conf["SNV"]["CI_high"] - conf["SNV"]["CI_low"]) / (2 * 1.96)
        indel_se = (conf["indels"]["CI_high"] - conf["indels"]["CI_low"]) / (2 * 1.96)
        alpha_std = np.sqrt((snv_se * conf["p"]["SNV"]) ** 2 + (indel_se * conf["p"]["indels"]) ** 2)

        k_params = {'mean': conf['K']['mean'], 'std': conf['K']['se']}
        mu_params = {'mean': alpha_mean, 'std': alpha_std}

        return k_params, mu_params

    def sample_parameters(self):
        """
        Sample K and mu parameters (both lognormal)
        """
        k_params, mu_params = self.generate_params()

        samples = {'K': [], 'mu': []}

        for _ in range(self.n_samples):
            k_mean_log = np.log(k_params['mean']) - 0.5 * np.log(1 + (k_params['std']/k_params['mean'])**2)
            k_std_log = np.sqrt(np.log(1 + (k_params['std']/k_params['mean'])**2))
            K = np.random.lognormal(k_mean_log, k_std_log)

            mu_mean_log = np.log(mu_params['mean']) - 0.5 * np.log(1 + (mu_params['std']/mu_params['mean'])**2)
            mu_std_log = np.sqrt(np.log(1 + (mu_params['std']/mu_params['mean'])**2))
            mu = np.random.lognormal(mu_mean_log, mu_std_log)

            samples['K'].append(K)
            samples['mu'].append(mu)

        return np.array(samples['K']), np.array(samples['mu'])

    def simulate_trajectories(self):
        """
        Calculate X(t) trajectories for all parameter samples
        """
        K_samples, mu_samples = self.sample_parameters()
        trajectories = []
        for K, mu in zip(K_samples, mu_samples):
            trajectory = K * np.exp(-mu * self.time_points)
            trajectories.append(trajectory)
        return np.array(trajectories)

    def calculate_statistics(self, trajectories, percentiles=None):
        """
        Calculate mean, std, min, max, and custom percentiles at each time point
        """
        if percentiles is None:
            percentiles = np.array([2.5] + [x for x in range(5, 100, 5)] + [97.5])/100

        perc_points = [p*100 if p <= 1 else p for p in percentiles]

        stats_data = []
        for i, t in enumerate(self.time_points):
            values = trajectories[:, i]
            perc_values = np.percentile(values, perc_points)

            row = {
                't': t,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }
            for p, val in zip(percentiles, perc_values):
                key = f"q{int(p*1000):04d}"
                row[key] = val

            stats_data.append(row)

        return pd.DataFrame(stats_data)

    def run_simulation(self):
        """
        Run simulation for lognormal-lognormal distribution
        """
        k_params, mu_params = self.generate_params()
        print(f"Running lognormal-lognormal simulation with {self.n_samples} samples...")
        print(f"K parameters: mean={k_params['mean']}, std={k_params['std']}")
        print(f"μ parameters: mean={mu_params['mean']}, std={mu_params['std']}")
        print("-" * 50)

        K_samples, mu_samples = self.sample_parameters()
        trajectories = self.simulate_trajectories()
        stats_df = self.calculate_statistics(trajectories)

        self.results['K lognormal - µ lognormal'] = {
            'stats': stats_df,
            'K_samples': K_samples,
            'mu_samples': mu_samples,
            'trajectories': trajectories
        }

        print("Simulation completed!")
        return self.results

    def plot_results(self, save_plots: bool = True, figsize=(8, 6), show: bool = False):
        """
        Plot trajectories with median and 95% CI
        """
        params = self.config()

        thr = params["x_crit"]

        K, _ = self.generate_params()

        K = K["mean"]

        if not self.results:
            print("No results to plot. Run simulation first.")
            return

        stats = self.results['K lognormal - µ lognormal']['stats']

        plt.figure(figsize=figsize)
        plt.fill_between(stats['t'], stats['q0025']/K, stats['q0975']/K,
                         alpha=0.2, color='blue', label='95% CI')
        plt.plot(stats['t'], stats['q0500']/K, color='blue',
                 linewidth=2, label='Median')

        plt.axhline(thr, c='gray', ls='--', label='Critical threshold')
        plt.xlabel('Time (Years)')
        plt.ylabel(r'X(t) / $K_{mean}$')
        plt.title('Exponential Decay (Lognormal-Lognormal)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_plots:
            plt.savefig('exponential_decay_lognormal_lognormal.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'exponential_decay_lognormal_lognormal.png'")

        if show:
            plt.show()

    def save_to_csv(self):
        """
        Save simulation results to CSV file
        """

        filename = f'Model_II_{self.organ}_Xt_percentiles_every_5.csv'

        if not self.results:
            print("No results to save. Run simulation first.")
            return

        stats = self.results['K lognormal - µ lognormal']['stats'].copy()
        stats['distribution'] = 'K lognormal - µ lognormal'

        cols = ['distribution', 't'] + [c for c in stats.columns if c not in ['distribution','t']]
        stats = stats[cols]

        stats.to_csv(filename, index=False)
        print(f"Results saved to '{filename}'")

        return stats

    def save_parameters_to_csv(self):
        """
        Save parameter samples to CSV file
        """

        filename=f'Model_II_{self.organ}_parameter_samples_every_5.csv'

        if not self.results:
            print("No results to save. Run simulation first.")
            return

        K_samples = self.results['K lognormal - µ lognormal']['K_samples']
        mu_samples = self.results['K lognormal - µ lognormal']['mu_samples']

        all_params = []
        for i, (K, mu) in enumerate(zip(K_samples, mu_samples)):
            all_params.append({
                'distribution': 'K lognormal - µ lognormal',
                'sample_id': i,
                'K': K,
                'mu': mu
            })

        params_df = pd.DataFrame(all_params)
        params_df.to_csv(filename, index=False)
        print(f"Parameter samples saved to '{filename}'")

        return params_df
    
def run_simulation(organ: str = "brain"):
    simulator = ExponentialDecaySimulator(
        n_samples=10_000,
        t_max=10_000,
        n_time_points=10_000-1,
        organ=organ
    )

    results = simulator.run_simulation()

    simulator.plot_results(save_plots=True)
    stats_df = simulator.save_to_csv()
    params_df = simulator.save_parameters_to_csv()

if __name__ == "__main__":
    run_simulation()
