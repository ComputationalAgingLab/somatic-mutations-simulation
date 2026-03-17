import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline

def first_time_below(s: np.ndarray, t: np.ndarray, thresh: float) -> float | None:
    """
    Return the first time t where s(t) drops to or below thresh.

    Args:
        s: survival (or any monotone-decreasing) array
        t: corresponding time array, same length as s
        thresh: threshold value

    Returns:
        First time at or below thresh, or None if never reached
    """
    mask = s <= thresh
    if np.any(mask):
        return float(t[np.where(mask)[0][0]])
    return None


def compute_threshold_times(s: np.ndarray, t: np.ndarray,
                            thresholds: list) -> pd.DataFrame:
    """
    Compute the first crossing time of each threshold for a survival curve.

    Args:
        s: survival array
        t: time array
        thresholds: list of survival threshold values (e.g. [0.5, 0.1, 0.01])

    Returns:
        Single-row DataFrame with columns t_S{thresh} for each threshold
    """
    row = {f"t_S{thr}": first_time_below(s, t, thr) for thr in thresholds}
    return pd.DataFrame([row])


def ffill_data(save_path: str, df: pd.DataFrame | None = None, 
               path: str | None = None) -> None:
    """
    Function to postprocess the Kaplan-Meier output from model III

    Args:
        save_path: where to save the resulting csv
        df: dataframe instance (alternative to path)
        path: path to .csv file with KM estimate

    Returns:
        None, writes new file to save_path
    """
    # Baseline hazard
    l = 0.0016133681587490935

    if path:
        df = pd.read_csv(path)

    event_times = df["time"].values

    # Filling regular grid
    t_max = event_times.max()
    regular_grid = np.arange(0, np.floor(t_max) + 1, 1)

    full_time = np.union1d(regular_grid, event_times)
    full_time.sort()

    dense = pd.DataFrame({"time": full_time})

    dense = dense.merge(df, on="time", how="left")

    # Saving data with combined
    cols = ["S_organ(t)", "lower_CI", "upper_CI"]
    dense[cols] = dense[cols].ffill()

    dense["S_combined(t)"] = dense["S_organ(t)"] * np.exp(-l * dense["time"])
    dense["S_baseline(t)"] = np.exp(-l * dense["time"])

    dense.to_csv(save_path, index=False)

def frechet_hoeffding(data_brain, 
                      data_heart, 
                      data_lungs,
                      save_path: str | None = None) -> pd.DataFrame:
    """
    Function for Fréchet-Hoeffding bounds estimation.

    Args:
        data_brain: survival DataFrame for brain
        data_heart: survival DataFrame for heart
        data_lungs: survival DataFrame for lungs
        save_path: where to save the resulting csv (liver+LPC assumed S=1 everywhere)

    Returns:
        DataFrame with lower, upper bounds and independence case
    """
    
    # S(t) extraction
    st_organ_brain = data_brain["S_organ(t)"]
    st_organ_heart = data_heart["S_organ(t)"]
    st_baseline = data_heart["S_baseline(t)"]
    st_lungs = data_lungs["S_organ(t)"]

    # Aligning time, brain (or heart) as the baseline time
    t_common = data_brain["time"]
    t_baseline_time = data_heart["time"]
    t_lungs_time = data_lungs["time"]

    st_baseline_interp   = np.interp(t_common, t_baseline_time, st_baseline)
    st_heart_interp      = np.interp(t_common, t_baseline_time, st_organ_heart)
    st_lungs_filtered    = np.interp(t_common, t_lungs_time, st_lungs)

    S = np.vstack([st_organ_brain, st_heart_interp, st_lungs_filtered, np.ones(st_lungs_filtered.shape)])

    # Independence = prod(S_i), liver is 1.0 for all time points
    S_indep = np.prod([st_baseline_interp, st_organ_brain, st_heart_interp, st_lungs_filtered], axis=0)

    # Lower bound = max(0, sum_{i=1}^{n}(S_i) - (n - 1))
    L = st_baseline_interp * np.maximum(0, np.sum(S, axis=0) - (S.shape[0]-1))

    # Upper bound = min(S_i)
    U = st_baseline_interp * np.min(S, axis=0)

    threshold = 1/8e9
    t = data_brain["time"].values

    # Threshold calculation for the human population
    t_max   = first_time_below(U,       t, threshold)
    t_min   = first_time_below(L,       t, threshold)
    t_indep = first_time_below(S_indep, t, threshold)

    # Threshold = 0.01 and 0.5 for all cases
    t_001_indep = first_time_below(S_indep, t, 0.01)
    t_05_indep  = first_time_below(S_indep, t, 0.5)

    t_001_U = first_time_below(U, t, 0.01)
    t_05_U  = first_time_below(U, t, 0.5)

    t_001_L = first_time_below(L, t, 0.01)
    t_05_L  = first_time_below(L, t, 0.5)

    print(f"Lower bound: {np.round(t_min)}")
    print(f"Upper bound: {np.round(t_max)}")
    print(f"Perfect independence: {np.round(t_indep)}")

    data_dict = {"50%":{
            "counterdependence":t_05_L, "independence":t_05_indep, "codependence":t_05_U
            },
            "1%":{
            "counterdependence":t_001_L, "independence":t_001_indep, "codependence":t_001_U
            },
            "1/N":{
            "counterdependence":t_min, "independence":t_indep, "codependence":t_max
            }}
    
    df_times = pd.DataFrame(data_dict).T

    df_times.reset_index(inplace=True)

    df_times = df_times.rename(columns={"index":"percentile"})

    if save_path:
        df_times.to_csv(save_path)

    return df_times

def ffill_data_na(path: str | None = None, 
                  df: pd.Series | pd.DataFrame = None) -> pd.DataFrame | pd.Series:
    """
    Function to postprocess the Nelson-Aalen (NA) cumulative hazard estimate from model III

    Args:
        path: path to .csv file with NA estimate
        df: dataframe instance (alternative to path)

    Returns:
        DataFrame with forward-filled data on a regular time grid
    """
    if path:
        df = pd.read_csv(path)

    # Checking for empty df (if no events)
    if len(df) == 0:
        t_max = 100_000
    elif len(df) == 1 and df["time"].iloc[0] == 0:
        t_max = 100_000
    else:
        t_max = df["time"].max()
    
    # Constructing regular time grid
    regular_grid = np.arange(0, np.floor(t_max) + 1, 1)
    result = pd.DataFrame({"time": regular_grid})

    # Forward filling if not empty df, zeros otherwise
    if len(df) > 0:
        df_indexed = df.set_index("time")
        na_values = df_indexed["Nelson-Aalen"].reindex(regular_grid, method="ffill").fillna(0).values
    else:
        na_values = np.zeros_like(regular_grid)

    result["Nelson-Aalen"] = na_values
    return result

def smooth_lambdas(path: str | None = None, 
                   df: pd.Series | pd.DataFrame = None, 
                   eps=1e-4, k=5) -> pd.DataFrame | pd.Series:
    """
    Function for smoothing hazard rates for model III

    Args:
        path: path to .csv file with NA estimate
        df: dataframe instance (alternative to path)
        eps: lower bound to filter the h(t) values
        k: order of spline

    Returns:
        DataFrame with smooth h(t) values
    """

    if path:
        regular = ffill_data_na(path)
    else:
        regular = ffill_data_na(df=df)

    # Computing hazard rate
    regular["lambda"] = np.gradient(regular["Nelson-Aalen"], regular["time"])
    
    # Filtering out low values
    regular_filtered = regular[regular["lambda"] >= eps].copy()
    
    # Setting zero if there are no events
    if regular_filtered.empty:
        regular_full = regular.copy()
        regular_full["smooth_lambda"] = 0.0
        return regular_full[["time", "smooth_lambda"]]

    # Setting regular time grid
    time_fill = np.arange(0, regular_filtered["time"].iloc[0], 1)
    regular_add = pd.DataFrame({
        "time": time_fill,
        "Nelson-Aalen": np.zeros_like(time_fill),
        "lambda": np.zeros_like(time_fill),
        "smooth_lambda": np.zeros_like(time_fill)
    })

    # Setting spline smoothing
    try:
        k_use = min(k, len(regular_filtered) - 1)
        k_use = max(1, k_use)
        spline1 = UnivariateSpline(regular_filtered["time"], regular_filtered["lambda"], k=k_use, s=0)
        regular_filtered["smooth_lambda"] = spline1(regular_filtered["time"])
    except Exception:
        regular_filtered["smooth_lambda"] = regular_filtered["lambda"]

    regular_full = pd.concat([regular_add, regular_filtered], axis=0)
    return regular_full[["time", "smooth_lambda"]]

def save_na_estimates(regular, path: str):
    regular = regular[regular["smooth_lambda"]>=0]

    regular.to_csv(path)
