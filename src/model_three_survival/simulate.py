import os

import numpy as np
import pandas as pd

from src.utils.conf_base import Config
from src.utils.ode_sim import mean_se_to_lognorm_params, sample_lognormal_from_mean_se, monte_carlo_parallel
from src.utils.rhs_factory import rhs_base
from src.utils.kaplan_meier_utils import kaplan_meier, save_km_plots, save_km_to_csv, save_trajectory_plots, find_time_for_survival

def run_pipeline_compare_fixed_model_b(organ_x="liver", organ_s=None, n_km=2000, n_traj=400, t_max=200_000,
                               n_workers=4, save_traces=100, seed=123, outdir="results",
                               n_common_points=10000):
    
    os.makedirs(outdir, exist_ok=True)
    cfg = Config(organ=organ_x, organ_s=organ_s)
    param_fixed, param_specs, initial_cond, x_crit = cfg.get_params()
    print(f"Config: organ_x={organ_x}, organ_s={organ_s}")
    print(f"x_crit = {x_crit:.3e}")

    results = {}
    ts_common = np.linspace(0, t_max, n_common_points)

    for dist in ("lognormal",):
        print(f"\n=== Running {dist} sampling for KM (n={n_km}) ===")
        death_times, _ = monte_carlo_parallel(n_km, initial_cond, param_fixed, param_specs,
                                              x_crit, t_max, n_workers, 0, dist, seed)
        cens = np.isnan(death_times)
        event_times = np.where(cens, t_max, death_times)
        km_t, km_s, km_l, km_u, median_est, na_est = kaplan_meier(event_times, cens)
        results[dist] = {
            "death_times": death_times,
            "km_times": km_t,
            "km_surv": km_s,
            "km_lower": km_l,
            "km_upper": km_u,
            "median": median_est,
            "Nelson-Aalen": na_est
        }

        km_csv = os.path.join(outdir, f"km_{organ_x}_{organ_s}_{dist}.csv")
        save_km_to_csv(km_csv, km_t, km_s, km_l, km_u, median_est)
        save_km_plots(outdir, f"{organ_x}_{organ_s}_{dist}", km_t, km_s, km_l, km_u)

        if km_t.size > 0:
            km_ci = np.column_stack((km_l, km_u))
            #save_combined_survival(outdir, f"{organ_x}_{organ_s}_{dist}", km_t, km_s, km_ci, hazard_rate)

        observed = death_times[~cens]
        if observed.size > 0:
            perc_summary = {
                p: float(np.percentile(observed, p))
                for p in (2.5, 10, 25, 50, 75, 90, 97.5)
            }
        else:
            perc_summary = {p: np.nan for p in (2.5, 10, 25, 50, 75, 90, 97.5)}
        results[dist]["event_time_percentiles"] = perc_summary
        pct_csv = os.path.join(outdir, f"death_time_percentiles_{organ_x}_{organ_s}_{dist}.csv")
        pd.DataFrame([perc_summary]).to_csv(pct_csv, index=False)
        print(f"[{dist}] death time percentiles: {perc_summary}")

        print(f"=== Running {dist} sampling for trajectories (n={n_traj}) ===")
        _, traces = monte_carlo_parallel(n_traj, initial_cond, param_fixed, param_specs,
                                         x_crit, t_max, n_workers, save_traces, dist, int(seed)+999)

        if traces:
            Xs, Ss, Ps = [], [], []
            for tvals, yvals in traces.values():
                if yvals is None:
                    continue
                X_interp = np.interp(ts_common, tvals, yvals[0, :])
                S_interp = np.interp(ts_common, tvals, yvals[1, :])
                P_interp = np.interp(ts_common, tvals, yvals[2, :])
                Xs.append(X_interp)
                Ss.append(S_interp)
                Ps.append(P_interp)

            Xmat = np.vstack(Xs)
            Smat = np.vstack(Ss)
            Pmat = np.vstack(Ps)

            meanX, stdX = Xmat.mean(axis=0), Xmat.std(axis=0)
            meanS, stdS = Smat.mean(axis=0), Smat.std(axis=0)
            meanP, stdP = Pmat.mean(axis=0), Pmat.std(axis=0)

            eps = 1e-12
            meanX = np.clip(meanX, eps, None)
            meanS = np.clip(meanS, eps, None)
            meanP = np.clip(meanP, eps, None)

            results[dist].update({
                "ts": ts_common,
                "Xmat": Xmat, "Smat": Smat, "Pmat": Pmat,
                "meanX": meanX, "stdX": stdX,
                "meanS": meanS, "stdS": stdS,
                "meanP": meanP, "stdP": stdP
            })

            df_traj = pd.DataFrame({
                "time": ts_common,
                "X_mean": meanX, "X_std": stdX,
                "X_min": meanX - stdX, "X_max": meanX + stdX,
                "S_mean": meanS, "S_std": stdS,
                "S_min": np.maximum(meanS - stdS, 0.0), "S_max": meanS + stdS,
                "P_mean": meanP, "P_std": stdP,
                "P_min": np.maximum(meanP - stdP, 0.0), "P_max": meanP + stdP
            })
            
            traj_csv_path = os.path.join(outdir, f"trajectories_{organ_x}_{organ_s}_{dist}.csv")
            df_traj.to_csv(traj_csv_path, index=False)
            print(f"Saved trajectories CSV (mean/std): {traj_csv_path}")

            percentiles = [2.5] + [x for x in range(5, 100, 5)] + [97.5]

            pct_dict_X = {}
            pct_dict_S = {}
            pct_dict_P = {}
            pct_df = pd.DataFrame({"time": ts_common})
            for p in percentiles:
                pct_dict_X[p] = np.percentile(Xmat, p, axis=0)
                pct_dict_S[p] = np.percentile(Smat, p, axis=0)
                pct_dict_P[p] = np.percentile(Pmat, p, axis=0)

                pct_df[f"X_p{p}"] = pct_dict_X[p]
                pct_df[f"S_p{p}"] = pct_dict_S[p]
                pct_df[f"P_p{p}"] = pct_dict_P[p]

            pct_csv_path = os.path.join(outdir, f"trajectories_percentiles_{organ_x}_{organ_s}_{dist}.csv")
            pct_df.to_csv(pct_csv_path, index=False)
            print(f"Saved trajectories percentile CSV: {pct_csv_path}")

            save_trajectory_plots(outdir, f"{organ_x}_{organ_s}_{dist}", ts_common, pct_dict_X, pct_dict_S, pct_dict_P)
        else:
            print(f"No traces saved for {dist}; skipping trajectory outputs.")

    return results, ts_common
