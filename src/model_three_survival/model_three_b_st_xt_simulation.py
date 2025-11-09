import math
import os
from typing import List, Dict, Union, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class Config:
    def __init__(self, 
                 organ_x="liver", 
                 organ_s="LPC"):
        
        self.organ_x = organ_x.lower()
        self.organ_s = organ_s
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
            }
        }
        if self.organ_x not in self.values:
            raise ValueError(f"Organ X '{organ_x}' not available. Available: {list(self.values.keys())}")
        if self.organ_s not in self.values:
            raise ValueError(f"Organ S '{organ_s}' not available. Available: {list(self.values.keys())}")

    def get_params(self) -> Union[Dict, Dict, Dict, Any]:

        liver_conf = self.values[self.organ_x]
        lpc_conf = self.values[self.organ_s]

        mu_x_mean = (liver_conf['SNV']['mean'] * liver_conf['p']['SNV'] +
                        liver_conf['indels']['mean'] * liver_conf['p']['indels'])

        snv_se_x = (liver_conf['SNV']['CI_high'] - liver_conf['SNV']['CI_low']) / (2 * 1.96)
        indel_se_x = (liver_conf['indels']['CI_high'] - liver_conf['indels']['CI_low']) / (2 * 1.96)

        mu_x_std = np.sqrt((snv_se_x * liver_conf['p']['SNV'])**2 +
                              (indel_se_x * liver_conf['p']['indels'])**2)

        mu_s_mean = (lpc_conf['SNV']['mean'] * lpc_conf['p']['SNV'] +
                        lpc_conf['indels']['mean'] * lpc_conf['p']['indels'])

        snv_se_s = (lpc_conf['SNV']['CI_high'] - lpc_conf['SNV']['CI_low']) / (2 * 1.96)
        indel_se_s = (lpc_conf['indels']['CI_high'] - lpc_conf['indels']['CI_low']) / (2 * 1.96)
        mu_s_std = np.sqrt((snv_se_s * lpc_conf['p']['SNV'])**2 +
                              (indel_se_s * lpc_conf['p']['indels'])**2)

        r_se = (liver_conf['r']['CI_high'] - liver_conf['r']['CI_low']) / (2 * 1.96)

        param_fixed = {
            "H": liver_conf['H'],
            "r_s": 0.112
        }

        param_specs = {
            "r": (liver_conf['r']['mean'], r_se),
            "K": (liver_conf['K']['mean'], liver_conf['K']['se']),
            "Q": (lpc_conf['K']['mean'], lpc_conf['K']['se']),
            "mu": (mu_x_mean, mu_x_std),
            "mu_s": (mu_s_mean, mu_s_std)
        }

        initial_cond = {
            "X0": liver_conf['K']['mean'],
            "S0": lpc_conf['K']['mean'],
            "P0": liver_conf['H']
        }

        x_crit = liver_conf['x_crit'] * liver_conf['K']['mean']

        return param_fixed, param_specs, initial_cond, x_crit

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
    mu = params.get("mu", 0.0)
    r = params.get("r", 0.0)
    H = params.get("H", 1.0)
    K = params.get("K", 1.0)
    Q = params.get("Q", 1.0)
    r_s = params.get("r_s", 0.0)
    mu_s = params.get("mu_s", 0.0)

    def rhs(t, y):
        X, S, P = y

        if X < 1e-10:
            return [0.0, 0.0]
        uss = 3.45 - 6.04*S/Q + 0.51*X/K
        uxx = 0.43 + 6.04*S/Q - 8.56*X/K

        f_ss = np.exp(uss)/(np.exp(uss) + np.exp(uxx) + 1)
        f_xx = np.exp(uxx)/(np.exp(uss) + np.exp(uxx) + 1)
        f_xs = 1/(np.exp(uss) + np.exp(uxx) + 1)

        g_s = (1 - X / K)**2 + (1 - S / Q)**2 - (1 - X / K)**2 * (1 - S / Q)**2

        dXdt = -mu * X + r * (P / H) * (1 - X / K) * X + r_s * g_s * S * (2 * f_xx + f_xs)
        dSdt = -mu_s * S + r_s * g_s * S * (f_ss - f_xx)
        dPdt = -2 * r * (1 / H) * (1 - X / K) * P + ((H - P) / X) * r_s * g_s * S * (2 * f_xx + f_xs)

        return [dXdt, dSdt, dPdt]

    return rhs

def make_event_xcrit(x_crit):
    def event(t, y):
        return y[0] - x_crit
    event.terminal = True
    event.direction = -1
    return event

def make_event_zero():
    def event(t, y):
        eps = 1e-6
        return y[0] - eps
    event.terminal = True
    event.direction = -1
    return event

def single_run_worker(i, seed, initial_cond, param_fixed, param_specs, t_max,
                      save_trace_flag, mu_distribution, x_crit):
    """
    Single MC run. Returns (i, death_time_or_nan, trace_or_None).
    Uses provided x_crit (absolute units).
    """
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
        sampled["Q"],   
        initial_cond["P0"]
    ]

    rhs = rhs_factory(params)
    event = make_event_xcrit(x_crit)
    event_zero = make_event_zero()

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


def monte_carlo_parallel(n_runs, initial_cond, param_fixed, param_specs, x_crit,
                         t_max, n_workers, save_traces, mu_distribution, seed):
    base_seed = int(seed) if seed is not None else np.random.randint(0, 2**31-1)
    args = []
    for i in range(n_runs):
        args.append((i, base_seed + i, initial_cond, param_fixed, param_specs,
                     t_max, (i < save_traces), mu_distribution, x_crit))

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
    observed = event_times[~censored_mask]
    if observed.size == 0:
        return np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.nan

    uniq_times = np.sort(np.unique(np.sort(observed)))
    surv = 1.0
    nels_aal = 0.0
    var_sum = 0.0
    times, survs, lower, upper, na_est = [], [], [], [], []
    z = abs(np.round(np.sqrt(2) * 1.0, 2))
    z = 1.96
    median_est = np.nan
    for t in uniq_times:
        d = np.sum((event_times == t) & (~censored_mask))
        at_risk = np.sum(event_times >= t)
        if at_risk <= 0:
            continue
        q = 1 - d / at_risk
        q_d = d / at_risk
        nels_aal += q_d
        surv *= q
        if at_risk - d > 0:
            var_sum += d / (at_risk * (at_risk - d))
        se = surv * math.sqrt(var_sum) if var_sum >= 0 else 0.0
        times.append(t)
        survs.append(surv)
        lower.append(max(0.0, surv - z * se))
        upper.append(min(1.0, surv + z * se))
        na_est.append(nels_aal)
        if np.isnan(median_est) and surv <= 0.5:
            median_est = t

    times_arr = np.concatenate(([0.0], np.array(times)))
    survs_arr = np.concatenate(([1.0], np.array(survs)))
    lower_arr = np.concatenate(([1.0], np.array(lower)))
    upper_arr = np.concatenate(([1.0], np.array(upper)))
    nels_aal_est = np.concatenate(([0.0], np.array(na_est)))

    return times_arr, survs_arr, lower_arr, upper_arr, median_est, nels_aal_est

def save_km_to_csv(filename, na, times, survs, lower, upper, median):
    df = pd.DataFrame({
        "time": times,
        "S_organ(t)": survs,
        "lower_CI": lower,
        "upper_CI": upper,
        "Nelson-Aalen": na
    })
    df.to_csv(filename, index=False)
    medfile = filename.replace(".csv", "_median.csv")
    pd.DataFrame({"median": [median]}).to_csv(medfile, index=False)

def find_threshold_time_from_mean(ts, meanS, threshold):
    if np.all(meanS > threshold):
        return np.nan
    idx = np.where(meanS <= threshold)[0][0]
    if idx == 0:
        return ts[0]
    t1, t0 = ts[idx], ts[idx-1]
    s1, s0 = meanS[idx], meanS[idx-1]
    if s0 == s1:
        return t1
    frac = (s0 - threshold) / (s0 - s1)
    return t0 + frac * (t1 - t0)

def save_S_results_csv(outpath, ts, meanS, stdS, thresholds_times_dict):
    df = pd.DataFrame({
        "time": ts,
        "meanS": meanS,
        "stdS": stdS,
        "mean_minus_std": meanS - stdS,
        "mean_plus_std": meanS + stdS
    })
    df.to_csv(outpath, index=False)
    thr_df = pd.DataFrame([
        {"threshold": thr, "time": thresholds_times_dict[thr]}
        for thr in sorted(thresholds_times_dict.keys(), reverse=True)
    ])
    thr_df.to_csv(outpath.replace(".csv", "_threshold_times.csv"), index=False)

def save_km_plots(outdir, label, km_t, km_s, km_l, km_u):
    if km_t.size == 0:
        return

    plt.figure(figsize=(8,5))
    plt.step(km_t, km_s, where="post", label=f"{label} KM")
    plt.fill_between(km_t, km_l, km_u, step="post", alpha=0.2)
    plt.xlabel("Time"); plt.ylabel("Survival probability")
    plt.title(f"KM survival (linear) - {label}")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"KM_{label}_linear.png"))
    plt.close()

    eps = 1e-12
    s_plot_clamped = np.clip(km_s, eps, 1.0)
    l_plot_clamped = np.clip(km_l, eps, 1.0)
    u_plot_clamped = np.clip(km_u, eps, 1.0)

    plt.figure(figsize=(8,5))
    plt.step(km_t, s_plot_clamped, where="post", label=f"{label} KM")
    plt.fill_between(km_t, l_plot_clamped, u_plot_clamped, step="post", alpha=0.2)
    plt.yscale("log")
    plt.xlabel("Time"); plt.ylabel("Survival probability (log scale)")
    plt.title(f"KM survival (log y) - {label}")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"KM_{label}_log.png"))
    plt.close()

def save_trajectory_plots(outdir, label, ts, pct_dict_X, pct_dict_S, pct_dict_P):
    """
    pct_dict_* expected to contain percentiles arrays keyed by percentile numbers,
    e.g. pct_dict_X[50] is median array for X.
    We'll plot median with shaded 95% CI (2.5 - 97.5).
    """
    def _plot_with_band(ts, median, low95, high95, ylabel, title, fname_lin, fname_log):
        plt.figure(figsize=(8,4))
        plt.plot(ts, median, label=f"{label} median")
        plt.fill_between(ts, low95, high95, alpha=0.2, label="95% CI")
        plt.xlabel("Time"); plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(fname_lin); plt.close()

        plt.figure(figsize=(8,4))
        plt.plot(ts, median, label=f"{label} median")
        plt.fill_between(ts, low95, high95, alpha=0.2, label="95% CI")
        plt.yscale("log")
        plt.xlabel("Time"); plt.ylabel(ylabel + " (log)")
        plt.title(title + " (log)")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(fname_log); plt.close()

    if pct_dict_X is not None:
        med = pct_dict_X.get(50)
        low95 = pct_dict_X.get(2.5)
        high95 = pct_dict_X.get(97.5)
        _plot_with_band(ts, med, np.maximum(low95, 1e-12), high95, "X", f"X trajectory bands - {label}",
                        os.path.join(outdir, f"X_{label}_linear.png"),
                        os.path.join(outdir, f"X_{label}_log.png"))
        
    if pct_dict_S is not None:
        med = pct_dict_S.get(50)
        low95 = pct_dict_S.get(2.5)
        high95 = pct_dict_S.get(97.5)
        _plot_with_band(ts, med, np.maximum(low95, 1e-12), high95, "S", f"S trajectory bands - {label}",
                        os.path.join(outdir, f"S_{label}_linear.png"),
                        os.path.join(outdir, f"S_{label}_log.png"))
        
    if pct_dict_P is not None:
        med = pct_dict_P.get(50)
        low95 = pct_dict_P.get(2.5)
        high95 = pct_dict_P.get(97.5)
        _plot_with_band(ts, med, np.maximum(low95, 1e-12), high95, "P", f"P trajectory bands - {label}",
                        os.path.join(outdir, f"P_{label}_linear.png"),
                        os.path.join(outdir, f"P_{label}_log.png"))

def save_combined_survival(outdir, label, km_t, km_s, km_s_ci, hazard_rate):
    if km_t.size == 0:
        return

    mech_t = km_t
    mech_s = km_s
    mech_ci_low, mech_ci_high = km_s_ci.T

    rand_s = np.exp(-hazard_rate * mech_t)

    comb_s = mech_s * rand_s
    comb_ci_low = mech_ci_low * rand_s
    comb_ci_high = mech_ci_high * rand_s

    df = pd.DataFrame({
        "time": mech_t,
        "mech": mech_s,
        "mech_ci_low": mech_ci_low,
        "mech_ci_high": mech_ci_high,
        "random": rand_s,
        "comb": comb_s,
        "comb_ci_low": comb_ci_low,
        "comb_ci_high": comb_ci_high
    })
    csv_path = os.path.join(outdir, f"survival_{label}_combined.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8,5))
    plt.step(mech_t, mech_s, where="post", label="Mechanistic")
    plt.fill_between(mech_t, mech_ci_low, mech_ci_high, step="post", alpha=0.2)
    plt.plot(mech_t, rand_s, "--", label=f"Random (exp, λ={hazard_rate:g})")
    plt.plot(mech_t, comb_s, "-", label="Combined")
    plt.fill_between(mech_t, comb_ci_low, comb_ci_high, alpha=0.2)

    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.title(f"Survival comparison - {label}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(outdir, f"survival_{label}_combined.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved combined survival plot+CSV: {plot_path}, {csv_path}")

def run_pipeline_compare_fixed_model_b(organ_x="liver", organ_s="LPC", n_km=2000, n_traj=400, t_max=200_000,
                               n_workers=4, save_traces=100, seed=123, outdir="results",
                               hazard_rate=1/8e9, n_common_points=10000):
    
    os.makedirs(outdir, exist_ok=True)
    cfg = Config(organ_x, organ_s)
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
            save_combined_survival(outdir, f"{organ_x}_{organ_s}_{dist}", km_t, km_s, km_ci, hazard_rate)

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

if __name__ == "__main__":
    outdir = "Model_IIIB_Liver_LPC"
    results, ts_common = run_pipeline_compare_fixed_model_b(
        organ_x="liver",
        organ_s="LPC",
        n_km=1_000_000,
        n_traj=5_000,
        t_max=100_000,
        n_workers=6,
        save_traces=2_500,
        seed=123,
        outdir=outdir,
        hazard_rate=0.0016133681587490935,
        n_common_points=10_000
        )

    for dist in ("lognormal",):
        med = results[dist].get("median", np.nan)
        print(f"{dist} median survival time (KM estimate): {med}")
        if "meanS" in results[dist] and results[dist]["meanS"] is not None:
            ts = results[dist]["ts"]
            meanS = results[dist]["meanS"]
            N = 8e9
            thr_dict = {
                0.5: find_threshold_time_from_mean(ts, meanS, 0.5),
                0.01: find_threshold_time_from_mean(ts, meanS, 0.01),
                1.0/N: find_threshold_time_from_mean(ts, meanS, 1.0/N)
            }
            print(f"{dist} S(t) threshold times: {thr_dict}")

    print("All outputs saved to:", outdir)
