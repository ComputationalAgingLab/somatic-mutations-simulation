import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline

import os

def ffill_data(save_path: str, df: pd.DataFrame | None = None, path: str | None = None) -> None:
    """
    Function to postprocess the Kaplan-Meier output from model IIIA and IIIC

    Args: 
    * path to .csv file with KM estimate, 
    * save_path - where to save the resulting csv

    Output: 
    * None, writes new file to save_path
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
                      save_path: str | None = None) -> None:
    """
    Function for Frech\'et-Hoeffding bounds estimation.

    Args:
    * Three dataframes with survival probability (The fourth one - Liver+LPC is 1.0 everywhere)
    * save_path - where to save the resulting csv

    Output:
    * None, saves to .csv a dataframe with lower, upper bounds and independence case
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
    st_lungs_filtered  = np.interp(t_common, t_lungs_time, st_lungs)

    S = np.vstack([st_organ_brain, st_organ_heart, st_lungs_filtered, np.ones(st_lungs_filtered.shape)])

    # Independence = prod(S_i), liver is 1.0 for all time points
    S_indep = np.prod([st_baseline_interp, st_organ_brain, st_organ_heart, st_lungs_filtered], axis=0)

    # Lower bound = max(0, sum_{i=1}^{n}(S_i) - (n - 1))
    L = st_baseline_interp * np.maximum(0, np.sum(S, axis=0) - (S.shape[0]-1))

    # Upper bound = min(S_i)
    U = st_baseline_interp * np.min(S, axis=0)

    threshold = 1/8e9
    t = data_brain["time"].values

    def first_time_at_or_below(series, thresh):
        mask = series <= thresh
        if np.any(mask):
            return t[np.where(mask)[0][0]]
        else:
            return None

    # Threshold calculation for the human population
    t_max = first_time_at_or_below(U, threshold)    
    t_min  = first_time_at_or_below(L, threshold)     
    t_indep = first_time_at_or_below(S_indep, threshold)

    # Threshold = 0.01 and 0.5 for all cases
    t_001_indep = first_time_at_or_below(S_indep, 0.01)   
    t_05_indep = first_time_at_or_below(S_indep, 0.5)   

    t_001_U = first_time_at_or_below(U, 0.01)   
    t_05_U = first_time_at_or_below(U, 0.5)  

    t_001_L = first_time_at_or_below(L, 0.01)   
    t_05_L = first_time_at_or_below(L, 0.5)  

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

def ffill_data_na(path: str | None = None, df: pd.Series | pd.DataFrame = None):
    """
    Utility function for Nelson-Aalen estimate from Model III smoothing
    """
    if path:
        df = pd.read_csv(path)
    
    t_max = df["time"].max()
    regular_grid = np.arange(0, np.floor(t_max) + 1, 1)
    
    result = pd.DataFrame({"time": regular_grid})

    df_indexed = df.set_index("time")
    result["Nelson-Aalen"] = df_indexed["Nelson-Aalen"].reindex(regular_grid, method="ffill").fillna(0).values
    
    return result

def smooth_lambdas(path: str | None = None, df: pd.Series | pd.DataFrame = None, eps=1e-4, k=5):

    if path:
        regular = ffill_data_na(path)
    else:
        regular = ffill_data_na(df=df)

    regular["lambda"] = np.gradient(regular["Nelson-Aalen"], regular["time"])
    regular = regular[(regular["lambda"]>=eps)]

    time_fill = np.arange(0, regular["time"].iloc[0],1)
    regular_add = pd.DataFrame({"time": time_fill, 
                                 "Nelson-Aalen": np.zeros_like(time_fill),
                                 "lambda": np.zeros_like(time_fill),
                                 "smooth_lambda": np.zeros_like(time_fill)})
    
    spline1 = UnivariateSpline(regular["time"], regular["lambda"], k=k)

    regular["smooth_lambda"] = spline1(regular["time"])

    regular_full = pd.concat([regular_add, regular], axis=0)

    return regular_full[["time", "smooth_lambda"]]

def save_na_estimates(regular, path: str):
    regular = regular[regular["smooth_lambda"]>=0]

    regular.to_csv(path)
