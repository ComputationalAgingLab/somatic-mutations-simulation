import math
import os
from typing import Dict, List, Union, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.style.use("bmh")

class Config:
    def __init__(self, organ="lungs"):
        self.organ = organ.lower()
        self.values = {
            'lungs':{
                'SNV':{'mean':28.519, 'CI_low':20.725, 'CI_high':36.312},
                'indels':{'mean':2.573, 'CI_low':1.182, 'CI_high':3.963},
                'p':{'SNV':7.34e-5, 'indels':5.93e-9},
                'r_b':{'mean':89.7, 'CI_low':73.5, 'CI_high':142.3},
                'K':{'mean':1.03e9, 'se':2.5944e8},
                'b_crit_frac': 0.23,
                'H': 170
            }
        }
        if self.organ not in self.values:
            raise ValueError(f"Organ '{organ}' not available. Available: {list(self.values.keys())}")

    def get_params(self):
        conf = self.values[self.organ]

        mu_b_mean = (conf['SNV']['mean'] * conf['p']['SNV'] +
                     conf['indels']['mean'] * conf['p']['indels'])

        snv_se = (conf['SNV']['CI_high'] - conf['SNV']['CI_low']) / (2 * 1.96)
        indel_se = (conf['indels']['CI_high'] - conf['indels']['CI_low']) / (2 * 1.96)
        mu_b_std = np.sqrt((snv_se * conf['p']['SNV'])**2 + (indel_se * conf['p']['indels'])**2)

        r_b_se = (conf['r_b']['CI_high'] - conf['r_b']['CI_low']) / (2 * 1.96)

        param_fixed = {
            "H": conf['H']
        }

        param_specs = {
            "r_b": (conf['r_b']['mean'], r_b_se),
            "K": (conf['K']['mean'], conf['K']['se']),
            "mu_b": (mu_b_mean, mu_b_std)
        }

        initial_cond = {
            "B0": conf['K']['mean'],
            "P0": conf['H']
        }

        b_crit = conf['b_crit_frac'] * conf['K']['mean']

        return param_fixed, param_specs, initial_cond, b_crit


def mean_se_to_lognorm_params(mean, se):
    if mean <= 0:
        raise ValueError("Mean must be > 0 for lognormal parameterization")
    if se <= 0:
        return math.log(mean), 1e-12
    cv2 = (se / mean) ** 2
    sigma_ln = math.sqrt(math.log(1.0 + cv2))
    mu_ln = math.log(mean) - 0.5 * sigma_ln ** 2
    return mu_ln, sigma_ln

def sample_lognormal_from_mean_se(mean, se, rng):
    mu_ln, sigma_ln = mean_se_to_lognorm_params(mean, se)
    return rng.lognormal(mean=mu_ln, sigma=sigma_ln)

def sample_normal_from_mean_se(mean, se, rng):
    if se <= 0:
        return mean
    return rng.normal(loc=mean, scale=se)


def rhs_factory(params):
    mu_b = params.get("mu_b", 0.0)
    r_b = params.get("r_b", 0.0)
    H = params.get("H", 1.0)
    K = params.get("K", 1.0)

    def rhs(t, y):
        B, P = y

        if B < 1e-10:
            return [0.0, 0.0]

        q_bb = -3.4446824936018943
        #### Switch (1.1) to anything you like to show f_bb/f_xx ####
        q_xx = q_bb + np.log(1.1)
        slope = 3.5619901404912166

        u_bb = slope * (1 - B/K) - q_bb
        u_xx = slope * (1 - B/K) - q_xx

        denom = 2 * math.exp(u_bb) + 1.0
        f_bb = math.exp(u_bb) / denom
        f_xx = math.exp(u_xx) / denom
        f_xb = 1.0 / denom

        total_div_fac = r_b * (P / H) * (1.0 - B / K)

        dBdt = - mu_b * B + total_div_fac * (f_bb - f_xx) * B
        dPdt = - (r_b * (1.0 / H) * (1.0 - B / K)) * (2.0 * f_bb + f_xb) * P

        return [dBdt, dPdt]

    return rhs


def make_event_bcrit(b_crit):
    def event(t, y):
        return y[0] - b_crit
    event.terminal = True
    event.direction = -1
    return event

def make_event_bzero():
    def event(t, y):
        eps = 1e-6
        return y[0] - eps
    event.terminal = True
    event.direction = -1
    return event

def single_run_worker(i, seed, initial_cond, param_fixed, param_specs, t_max,
                      save_trace_flag, mu_distribution, b_crit):
    rng = np.random.default_rng(seed)

    sampled = {}
    for name, (mean, se) in param_specs.items():
        if mu_distribution == "lognormal":
            sampled[name] = sample_lognormal_from_mean_se(mean, se, rng)
        else:
            sampled[name] = sample_normal_from_mean_se(mean, se, rng)

    params = dict(param_fixed)
    params.update(sampled)

    y0 = [
        sampled["K"],
        initial_cond["P0"]
    ]

    rhs = rhs_factory(params)
    event = make_event_bcrit(b_crit)
    event_zero = make_event_bzero()

    try:
        sol = solve_ivp(rhs, (0, t_max), y0, events=[event, event_zero],
                        dense_output=save_trace_flag, rtol=1e-6, atol=1e-12, method='Radau')
    except Exception:
        return (i, np.nan, None)

    death_time = np.nan
    trace = None
    if sol.status == 1 and sol.t_events:
        if sol.t_events[0].size > 0:
            death_time = float(sol.t_events[0][0])
        elif sol.t_events[1].size > 0:
            death_time = float(sol.t_events[1][0])

    if sol.y.shape[1] > 0 and sol.y[0, :].min() < 0:
        idx_neg = np.where(sol.y[0, :] < 0)[0]
        if len(idx_neg) > 0:
            death_time = np.nan
            print(f"Warning: Run {i} had negative X values despite protections")
    
    if save_trace_flag and sol.sol is not None:
        t_eval = np.linspace(0, t_max, 200)
        y_eval = sol.sol(t_eval)
        y_eval[0, :] = np.maximum(y_eval[0, :], 0.0)
        y_eval[1, :] = np.maximum(y_eval[1, :], 0.0)
        trace = (t_eval, y_eval)

    return (i, death_time, trace)


def monte_carlo_parallel(n_runs, initial_cond, param_fixed, param_specs, crit,
                         t_max, n_workers, save_traces, mu_distribution, seed):
    """
    Monte Carlo wrapper using ProcessPoolExecutor. Note: crit is used by worker (b_crit).
    """
    base_seed = int(seed) if seed is not None else np.random.randint(0, 2**31-1)
    args = []
    for i in range(n_runs):
        args.append((i, base_seed + i, initial_cond, param_fixed, param_specs,
                     t_max, (i < save_traces), mu_distribution, crit))

    death_times = np.full(n_runs, np.nan)
    traces = {}
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(single_run_worker, *a): a[0] for a in args}
        completed = 0
        for fut in as_completed(futures):
            idx, dt, trace = fut.result()
            death_times[idx] = dt
            if trace is not None:
                traces[idx] = trace
            completed += 1
            if n_runs >= 20 and completed % max(1, n_runs // 20) == 0:
                print(f"[{mu_distribution}] {completed}/{n_runs} done")
    return death_times, traces


def kaplan_meier(event_times, censored_mask):
    """
    Returns km times, survival, lower CI, upper CI, median.
    Ensures t=0, S=1 is included so S(t)<1 only after first event.
    """
    observed = event_times[~censored_mask]
    if observed.size == 0:
        return np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.nan

    uniq_times = np.sort(np.unique(np.sort(observed)))
    surv = 1.0
    var_sum = 0.0
    na = 0.0
    times, survs, lower, upper, na_est = [], [], [], [], []
    z = 1.96
    median_est = np.nan
    for t in uniq_times:
        d = np.sum((event_times == t) & (~censored_mask))
        at_risk = np.sum(event_times >= t)
        if at_risk <= 0:
            continue
        q = 1 - d / at_risk
        q_d = d / at_risk
        surv *= q
        na += q_d
        if at_risk - d > 0:
            var_sum += d / (at_risk * (at_risk - d))
        se = surv * math.sqrt(var_sum) if var_sum >= 0 else 0.0
        times.append(t)
        survs.append(surv)
        lower.append(max(0.0, surv - z * se))
        upper.append(min(1.0, surv + z * se))
        na_est.append(na)
        if np.isnan(median_est) and surv <= 0.5:
            median_est = t

    times_arr = np.concatenate(([0.0], np.array(times)))
    survs_arr = np.concatenate(([1.0], np.array(survs)))
    lower_arr = np.concatenate(([1.0], np.array(lower)))
    upper_arr = np.concatenate(([1.0], np.array(upper)))
    na_est_arr = np.concatenate(([0.0], np.array(na_est)))

    return times_arr, survs_arr, lower_arr, upper_arr, median_est, na_est_arr


def find_time_from_km(target_s, km_t, km_s):
    """
    Return earliest time t where km_s(t) <= target_s.
    km_t and km_s are arrays from kaplan_meier (with t=0 included).
    """
    if km_t.size == 0:
        return np.nan
    if target_s := target_s if False else None:
        pass
    if target_s is None:
        pass
    for t, s in zip(km_t, km_s):
        if s <= target_s:
            return t
    return np.nan

def find_time_for_survival(target_s, km_t, km_s):
    if km_t.size == 0:
        return np.nan
    if km_s[0] <= target_s:
        return 0.0
    for t, s in zip(km_t, km_s):
        if s <= target_s:
            return t
    return np.nan


def save_km_to_csv(filename, times, survs, lower, upper, median, na):
    df = pd.DataFrame({
        "time": times,
        "survival": survs,
        "lower_CI": lower,
        "upper_CI": upper,
        "Nelson-Aalen": na
    })
    df.to_csv(filename, index=False)
    medfile = filename.replace(".csv", "_median.csv")
    pd.DataFrame({"median": [median]}).to_csv(medfile, index=False)


def save_km_plots(outdir, label, km_t, km_s, km_l, km_u):
    if km_t.size == 0:
        return

    t_plot = np.concatenate(([0], np.repeat(km_t[1:], 2)))
    s_plot = np.concatenate(([1.0], np.repeat(km_s[1:], 2)))

    plt.figure(figsize=(8,5))
    plt.step(t_plot, s_plot, where="post", label=f"{label} KM")
    plt.fill_between(t_plot, np.concatenate(([1.0], np.repeat(km_l[1:], 2))),
                     np.concatenate(([1.0], np.repeat(km_u[1:], 2))),
                     step="post", alpha=0.2)
    plt.xlabel("Time"); plt.ylabel("Survival probability")
    plt.title(f"KM survival (linear) - {label}")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"KM_{label}_linear.png"))
    plt.close()

    eps = 1e-12
    s_plot_clamped = np.clip(s_plot, eps, 1.0)
    l_plot_clamped = np.clip(np.concatenate(([1.0], np.repeat(km_l[1:], 2))), eps, 1.0)
    u_plot_clamped = np.clip(np.concatenate(([1.0], np.repeat(km_u[1:], 2))), eps, 1.0)

    plt.figure(figsize=(8,5))
    plt.step(t_plot, s_plot_clamped, where="post", label=f"{label} KM")
    plt.fill_between(t_plot, l_plot_clamped, u_plot_clamped, step="post", alpha=0.2)
    plt.yscale("log")
    plt.xlabel("Time"); plt.ylabel("Survival probability (log scale)")
    plt.title(f"KM survival (log y) - {label}")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"KM_{label}_log.png"))
    plt.close()


def save_trajectory_plots(outdir, label, ts, pct_B, pct_P):
    """
    pct_B and pct_P are dicts keyed by percentile values (e.g. 2.5, 50, 97.5)
    """
    def _plot_with_band(ts, median, low95, high95, ylabel, title, fname_lin, fname_log):
        plt.figure(figsize=(8,4))
        plt.plot(ts, median, label=f"{label} median")
        plt.fill_between(ts, low95, high95, alpha=0.2)
        plt.xlabel("Time"); plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(fname_lin); plt.close()

        plt.figure(figsize=(8,4))
        plt.plot(ts, median, label=f"{label} median")
        plt.fill_between(ts, low95, high95, alpha=0.2)
        plt.yscale("log")
        plt.xlabel("Time"); plt.ylabel(ylabel + " (log)")
        plt.title(title + " (log)")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(fname_log); plt.close()

    if pct_B:
        med = pct_B[50]
        low95 = pct_B[2.5]
        high95 = pct_B[97.5]
        _plot_with_band(ts, med, np.maximum(low95, 1e-12), high95, "B", f"B trajectory bands - {label}",
                        os.path.join(outdir, f"B_{label}_linear.png"),
                        os.path.join(outdir, f"B_{label}_log.png"))
    if pct_P:
        med = pct_P[50]
        low95 = pct_P[2.5]
        high95 = pct_P[97.5]
        _plot_with_band(ts, med, np.maximum(low95, 1e-12), high95, "P", f"P trajectory bands - {label}",
                        os.path.join(outdir, f"P_{label}_linear.png"),
                        os.path.join(outdir, f"P_{label}_log.png"))


def run_pipeline_compare_fixed(organ="liver", n_km=2000, n_traj=400, t_max=200_000,
                               n_workers=4, save_traces=100, seed=123, outdir="results",
                               n_common_points=10000):
    os.makedirs(outdir, exist_ok=True)
    cfg = Config(organ)
    param_fixed, param_specs, initial_cond, b_crit = cfg.get_params()
    print(f"Config: organ={organ}; b_crit={b_crit:.3e}")

    results = {}
    ts_common = np.linspace(0, t_max, n_common_points)

    percentile_list = [2.5] + [x for x in range(5, 100, 5)] + [97.5]

    traj_percentiles = [2.5] + [x for x in range(5, 100, 5)] + [97.5]

    for dist in ("lognormal",):
        print(f"\n=== Running {dist} sampling for KM (n={n_km}) ===")
        death_times, _ = monte_carlo_parallel(n_km, initial_cond, param_fixed, param_specs,
                                              b_crit, t_max, n_workers, 0, dist, seed)
        cens = np.isnan(death_times)
        event_times = np.where(cens, t_max, death_times)
        km_t, km_s, km_l, km_u, median_est, na_est = kaplan_meier(event_times, cens)

        percentiles = {}
        if np.any(~cens):
            observed = death_times[~cens]
            for q in percentile_list:
                percentiles[f"p{q}"] = float(np.percentile(observed, q))
        else:
            for q in percentile_list:
                percentiles[f"p{q}"] = np.nan

        t_S_0_01 = find_time_for_survival(0.01, km_t, km_s)
        t_S_1_over_N = find_time_for_survival(1.0 / 8e9, km_t, km_s)

        results[dist] = {
            "death_times": death_times,
            "km_times": km_t,
            "km_surv": km_s,
            "km_lower": km_l,
            "km_upper": km_u,
            "median": median_est,
            "Nelson-Aalen": na_est,
            "percentiles": percentiles,
            "t_S_0.01": t_S_0_01,
            "t_S_1_over_N": t_S_1_over_N
        }

        km_csv = os.path.join(outdir, f"km_{organ}_{dist}.csv")
        save_km_to_csv(km_csv, km_t, km_s, km_l, km_u, median_est, na_est)
        save_km_plots(outdir, f"{organ}_{dist}", km_t, km_s, km_l, km_u)

        extra_rows = []
        for k, v in percentiles.items():
            extra_rows.append({"metric": k, "value": v})
        extra_rows.append({"metric": "S=0.01_time", "value": t_S_0_01})
        extra_rows.append({"metric": "S=1/N_time", "value": t_S_1_over_N})
        pd.DataFrame(extra_rows).to_csv(os.path.join(outdir, f"extra_stats_{organ}_{dist}.csv"), index=False)

        print(f"Saved KM and extra stats for {dist}. S=0.01 at {t_S_0_01}, S=1/N at {t_S_1_over_N}")

        print(f"=== Running {dist} sampling for trajectories (n={n_traj}) ===")
        _, traces = monte_carlo_parallel(n_traj, initial_cond, param_fixed, param_specs,
                                         b_crit, t_max, n_workers, save_traces, dist, int(seed)+999)

        if traces:
            Bs, Ps = [], []
            for tvals, yvals in traces.values():
                if yvals is None:
                    continue
                B_interp = np.interp(ts_common, tvals, yvals[0, :])
                P_interp = np.interp(ts_common, tvals, yvals[1, :])
                Bs.append(B_interp)
                Ps.append(P_interp)

            Bmat = np.vstack(Bs) if Bs else np.empty((0, len(ts_common)))
            Pmat = np.vstack(Ps) if Ps else np.empty((0, len(ts_common)))

            meanB = Bmat.mean(axis=0) if Bmat.size else np.full_like(ts_common, np.nan)
            stdB = Bmat.std(axis=0) if Bmat.size else np.full_like(ts_common, np.nan)
            meanP = Pmat.mean(axis=0) if Pmat.size else np.full_like(ts_common, np.nan)
            stdP = Pmat.std(axis=0) if Pmat.size else np.full_like(ts_common, np.nan)

            eps = 1e-12
            meanB = np.clip(meanB, eps, None)
            meanP = np.clip(meanP, eps, None)

            df_traj = pd.DataFrame({
                "time": ts_common,
                "B_mean": meanB, "B_std": stdB,
                "B_min": meanB - stdB, "B_max": meanB + stdB,
                "P_mean": meanP, "P_std": stdP,
                "P_min": np.maximum(meanP - stdP, 0.0), "P_max": meanP + stdP
            })
            traj_csv_path = os.path.join(outdir, f"trajectories_{organ}_{dist}.csv")
            df_traj.to_csv(traj_csv_path, index=False)
            print(f"Saved trajectories CSV: {traj_csv_path}")

            pct_dict_B = {}
            pct_dict_P = {}
            pct_df = pd.DataFrame({"time": ts_common})
            for p in traj_percentiles:
                pctB = np.percentile(Bmat, p, axis=0) if Bmat.size else np.full_like(ts_common, np.nan)
                pctP = np.percentile(Pmat, p, axis=0) if Pmat.size else np.full_like(ts_common, np.nan)
                pct_dict_B[p] = pctB
                pct_dict_P[p] = pctP
                pct_df[f"B_p{p}"] = pctB
                pct_df[f"P_p{p}"] = pctP

            pct_csv_path = os.path.join(outdir, f"trajectories_percentiles_{organ}_{dist}.csv")
            pct_df.to_csv(pct_csv_path, index=False)
            print(f"Saved trajectories percentile CSV: {pct_csv_path}")

            mapped_pct_B = {50: pct_dict_B[50], 2.5: pct_dict_B[2.5], 97.5: pct_dict_B[97.5]}
            mapped_pct_P = {50: pct_dict_P[50], 2.5: pct_dict_P[2.5], 97.5: pct_dict_P[97.5]}
            save_trajectory_plots(outdir, f"{organ}_{dist}", ts_common, mapped_pct_B, mapped_pct_P)
        else:
            print(f"No traces saved for {dist}; skipping trajectory outputs.")

    summary_rows = []
    for dist in ("lognormal",):
        entry = {"variant": dist}
        entry.update(results[dist]["percentiles"])
        entry["t_S_0.01"] = results[dist]["t_S_0.01"]
        entry["t_S_1_over_N"] = results[dist]["t_S_1_over_N"]
        summary_rows.append(entry)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(outdir, f"{organ}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved at {summary_path}")

    return results, ts_common


if __name__ == "__main__":
    outdir = "Model_IIIC_Lungs"
    results, ts = run_pipeline_compare_fixed(
        organ="lungs",
        n_km=1_000_000,
        n_traj=5_000,
        t_max=200_000,
        n_workers=6,
        save_traces=2_000,
        seed=123,
        outdir=outdir,
        hazard_rate=0.0016133681587490935,
        n_common_points=20_000
    )

    for dist in ("lognormal",):
        med = results[dist].get("median", np.nan)
        print(f"{dist} median survival time: {med}")

    print("All outputs saved to:", outdir)
