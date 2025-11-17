import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor, as_completed

### Params conversion ###

def mean_se_to_lognorm_params(mean, se):
    if mean <= 0:
        raise ValueError("Mean must be > 0 for lognormal parameterization")
    if se <= 0:
        return np.log(mean), 1e-12
    cv2 = (se / mean) ** 2
    sigma_ln = np.sqrt(np.log(1.0 + cv2))
    mu_ln = np.log(mean) - 0.5 * sigma_ln ** 2
    return mu_ln, sigma_ln

def sample_lognormal_from_mean_se(mean, se, rng):
    mu_ln, sigma_ln = mean_se_to_lognorm_params(mean, se)
    return rng.lognormal(mean=mu_ln, sigma=sigma_ln)

### Events for the simulation ###

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

### Monte-Carlo single run ###

def single_run_worker(i, seed, initial_cond, param_fixed, param_specs, t_max,
                      save_trace_flag, mu_distribution, x_crit,
                      rhs_factory, organ_s):
    rng = np.random.default_rng(seed)

    sampled = {}
    for name, (mean, se) in param_specs.items():
        if mu_distribution == "lognormal":
            sampled[name] = sample_lognormal_from_mean_se(mean, se, rng)
        else:
            raise ValueError("Only lognormal is implemented")

    params = dict(param_fixed)
    params.update(sampled)

    if not organ_s:
        y0 = [
            sampled["K"],
            initial_cond["P0"]
        ]
    else:
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

### Monte-Carlo wrapper ###

def monte_carlo_parallel(n_runs, initial_cond, param_fixed, param_specs, crit,
                         t_max, n_workers, save_traces, mu_distribution, seed):
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
