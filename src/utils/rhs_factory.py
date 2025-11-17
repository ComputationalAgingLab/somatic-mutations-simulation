import numpy as np

### Model IIIA ODE ###

def rhs_three_a(params):
    mu = params.get("mu", 0.0)
    r = params.get("r", 0.0)
    H = params.get("H", 1.0)
    K = params.get("K", 1.0)

    def rhs(t, y):
        X, P = y
        if X < 1e-10: # Checking for near-zero
            return [0.0, 0.0]
        rho = r * (P / H) * (1 - X / K)
        dXdt = rho * X - mu * X
        dPdt = -2 * rho
        return [dXdt, dPdt]

    return rhs

### Model IIIB ODE ###

def rhs_three_b(params):
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

### Model IIIC ODE ###

def rhs_three_c(params):
    mu_b = params.get("mu_b", 0.0)
    r_b = params.get("r_b", 0.0)
    H = params.get("H", 1.0)
    K = params.get("K", 1.0)
    fbbfxx = params.get("fbbfxx", 1.0)

    def rhs(t, y):
        B, P = y

        if B < 1e-10:
            return [0.0, 0.0]

        q_bb = -3.4446824936018943

        q_xx = q_bb + np.log(fbbfxx)
        slope = 3.5619901404912166

        u_bb = slope * (1 - B/K) - q_bb
        u_xx = slope * (1 - B/K) - q_xx

        denom = 2 * np.exp(u_bb) + 1.0
        f_bb = np.exp(u_bb) / denom
        f_xx = np.exp(u_xx) / denom
        f_xb = 1.0 / denom

        total_div_fac = r_b * (P / H) * (1.0 - B / K)

        dBdt = - mu_b * B + total_div_fac * (f_bb - f_xx) * B
        dPdt = - (r_b * (1.0 / H) * (1.0 - B / K)) * (2.0 * f_bb + f_xb) * P

        return [dBdt, dPdt]

    return rhs

def rhs_base(organ: str, organ_s: str = None) -> callable:
    """
    Function to return ODE type for the simulation

    ** Args **
    - organ: base organ for the simulation
    - organ_s: if LPC is included for the liver
    """
    if (organ not in ["liver", "lungs"]):
        raise ValueError("Model three simulation only for liver and lungs.")
    
    if organ_s:
        if organ != "liver":
            return ValueError("LPC is modelled only when organ == liver.")
        return rhs_three_b
    else:
        if organ == "liver":
            return rhs_three_a
        else:
            return rhs_three_c
