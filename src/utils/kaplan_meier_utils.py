import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from .modeling import ffill_data, smooth_lambdas, save_na_estimates

def kaplan_meier(event_times, censored_mask):
    """
    Returns km times, Nelson-Aalen estimate; survival, lower CI, upper CI, median for S(t).
    Ensures t=0, S=1 is included so S(t)<1 and Nelson-Aalen > 0 only after first event.
    """
    observed = event_times[~censored_mask]
    if observed.size == 0:
        return (
            np.array([0.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.nan,
            np.array([0.0])
        )

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
        se = surv * np.sqrt(var_sum) if var_sum >= 0 else 0.0
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
    
    matches = np.where(km_s <= target_s)[0]
    return km_t[matches[0]] if matches.size > 0 else np.nan

def save_km_to_csv(filename, times, survs, lower, upper, median, na):
    df = pd.DataFrame({
        "time": times,
        "S_organ(t)": survs,
        "lower_CI": lower,
        "upper_CI": upper,
        "Nelson-Aalen": na
    })

    ffill_data(df=df, save_path=filename)

    na_file = filename.replace(".csv", "_na_hazard.csv")

    na_data = df[["time", "Nelson-Aalen"]]

    smooth_na = smooth_lambdas(df=na_data)

    save_na_estimates(regular=smooth_na, path=na_file)

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

def save_trajectory_plots(outdir, label, ts, pct_X, pct_P):
    """
    pct_X and pct_P are dicts keyed by percentile values (e.g. 2.5, 50, 97.5)
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

    if pct_X:
        med = pct_X[50]
        low95 = pct_X[2.5]
        high95 = pct_X[97.5]
        _plot_with_band(ts, med, np.maximum(low95, 1e-12), high95, "X", f"X trajectory bands - {label}",
                        os.path.join(outdir, f"X_{label}_linear.png"),
                        os.path.join(outdir, f"X_{label}_log.png"))
    if pct_P:
        med = pct_P[50]
        low95 = pct_P[2.5]
        high95 = pct_P[97.5]
        _plot_with_band(ts, med, np.maximum(low95, 1e-12), high95, "P", f"P trajectory bands - {label}",
                        os.path.join(outdir, f"P_{label}_linear.png"),
                        os.path.join(outdir, f"P_{label}_log.png"))