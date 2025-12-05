import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.conf_base import Config
from src.utils.ode_sim import monte_carlo_parallel, single_run_worker
from src.utils.rhs_factory import rhs_base
from src.utils.kaplan_meier_utils import (
    kaplan_meier, save_km_plots, save_km_to_csv, save_trajectory_plots
)

def run_model_iii(
    organ_x: str,
    organ_s: str | None = None,
    n_mc: int = 2000,
    n_traj: int = 400,
    t_max: int = 100_000,
    n_workers: int = 4,
    save_traces: int = 100,
    seed: int = 123,
    outdir: str = "results/model_iii",
    n_common_points: int = 10_000,
    N: float | int = 8e9
) -> dict:
    
    organ_x = organ_x.lower()
    if organ_x not in ["liver", "lungs"]:
        raise ValueError("Model III only supports 'liver' or 'lungs'")
    if organ_s and organ_x != "liver":
        raise ValueError("organ_s only allowed for liver")

    label = f"{organ_x}_{organ_s}" if organ_s else organ_x
    full_outdir = os.path.join(outdir, label)
    os.makedirs(full_outdir, exist_ok=True)
    print(f"Running Model III for {label}...")

    cfg = Config(organ=organ_x, organ_s=organ_s, N=N, time_max=t_max, time_points=n_common_points, mc_samples=n_mc)
    if cfg.model != "III":
        raise RuntimeError("Expected Model III")

    params = cfg.get_params()
    x_crit = params["x_c"]
    initial_cond = params["initial"]
    param_fixed = params["fixed"]
    param_specs = params["sampled"]
    rhs_func = rhs_base(organ=organ_x, organ_s=organ_s)

    ts_common = np.linspace(0, t_max, n_common_points)

    death_times, _ = monte_carlo_parallel(
        n_runs=n_mc,
        initial_cond=initial_cond,
        param_fixed=param_fixed,
        param_specs=param_specs,
        crit=x_crit,
        t_max=t_max,
        n_workers=n_workers,
        save_traces=0,
        seed=seed,
        rhs_factory=rhs_func,
        organ_s=organ_s
    )

    censored = np.isnan(death_times)
    event_times = np.where(censored, t_max, death_times)
    km_t, km_s, km_l, km_u, median_est, na_est = kaplan_meier(event_times, censored)

    save_km_to_csv(os.path.join(full_outdir, "km.csv"), km_t, km_s, km_l, km_u, median_est, na_est)
    save_km_plots(full_outdir, label, km_t, km_s, km_l, km_u)

    observed = death_times[~censored]
    perc_summary = {p: (float(np.percentile(observed, p)) if observed.size > 0 else np.nan)
                    for p in (2.5, 10, 25, 50, 75, 90, 97.5)}
    pd.DataFrame([perc_summary]).to_csv(os.path.join(full_outdir, "death_time_percentiles.csv"), index=False)

    _, traces = monte_carlo_parallel(
        n_runs=n_traj,
        initial_cond=initial_cond,
        param_fixed=param_fixed,
        param_specs=param_specs,
        crit=x_crit,
        t_max=t_max,
        n_workers=n_workers,
        save_traces=save_traces,
        seed=seed + 999,
        rhs_factory=rhs_func,
        organ_s=organ_s
    )

    if traces:
        first_y = next(iter(traces.values()))[1]
        n_states = first_y.shape[0]
        names = ["X", "P"] if n_states == 2 else ["X", "S", "P"]

        mats = {n: [] for n in names}
        for t_eval, y_eval in traces.values():
            if y_eval is None: continue
            for i, n in enumerate(names):
                interp = np.interp(ts_common, t_eval, y_eval[i, :])
                mats[n].append(np.clip(interp, 1e-12, None))

        for n in names:
            mats[n] = np.vstack(mats[n])

        percentiles = [2.5, 25, 50, 75, 97.5]
        df_pct = pd.DataFrame({"time": ts_common})
        pct_dicts = {}
        for n in names:
            pct_dict = {}
            for p in percentiles:
                val = np.percentile(mats[n], p, axis=0)
                pct_dict[p] = val
                df_pct[f"{n}_p{p}"] = val
            pct_dicts[n] = pct_dict

        df_pct.to_csv(os.path.join(full_outdir, "trajectories_percentiles.csv"), index=False)
        save_trajectory_plots(full_outdir, label, ts_common, pct_dicts["X"], pct_dicts.get("P", {}))
    else:
        print("No trajectory traces saved.")

    print(f"Model III for {label} done. Output: {full_outdir}")
    return {"km": (km_t, km_s), "death_times": death_times}