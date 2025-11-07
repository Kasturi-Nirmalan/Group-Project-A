import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats


def euler_orbit_error_fraction(a=1.0, G=4*math.pi**2, dt=1e-4, fraction=0.1):

    # Circular orbit initial conditions
    x, y = a, 0.0
    vx, vy = 0.0, math.sqrt(G / a)
    
    T = 1.0  
    total_time = fraction * T
    nsteps = int(np.ceil(total_time / dt))
    
    for _ in range(nsteps):
        r = math.hypot(x, y)
        ax = -G * x / (r**3)
        ay = -G * y / (r**3)
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
    
    # Exact solution after the same fraction of orbit
    theta = 2 * math.pi * fraction
    x_true = a * math.cos(theta)
    error = abs(x - x_true)  # global error in x
    return error


def fit_error_vs_dt_fraction(dts=None, fraction=0.1):
    if dts is None:
        dts = np.array([1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
    dts = np.array(dts, dtype=float)

    errors = np.array([euler_orbit_error_fraction(dt=dt, fraction=fraction) for dt in dts])

    # Print raw errors
    print("dt\t\t error")
    for dt, err in zip(dts, errors):
        print(f"{dt:.1e}\t {err:.6e}")

    # Log-log linear regression
    logdt = np.log10(dts)
    logerr = np.log10(errors)
    slope, intercept, r_value, p_value, std_err = stats.linregress(logdt, logerr)
    n_lin = slope
    C_lin = 10**intercept
    print("\nLinear regression on log-log:")
    print(f"n (slope) = {n_lin:.6f} ± {std_err:.6f}  (R² = {r_value**2:.4f})")
    print(f"C = {C_lin:.6e}")

    def power_law(dt, C, n):
        return C * dt**n

    try:
        p0 = [errors[0] / dts[0], 1.0]
        popt, pcov = curve_fit(power_law, dts, errors, p0=p0, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        print("\ncurve_fit on raw data:")
        print(f"C_fit = {popt[0]:.6e} ± {perr[0]:.6e}")
        print(f"n_fit = {popt[1]:.6f} ± {perr[1]:.6f}")
    except Exception as ex:
        popt = None
        pcov = None
        print("\ncurve_fit failed:", ex)

    # Plot convergence
    plt.figure(figsize=(6,5))
    plt.loglog(dts, errors, 'o', label='data')
    plt.loglog(dts, C_lin * dts**n_lin, '--', label=f'logfit: n={n_lin:.3f}')
    if popt is not None:
        plt.loglog(dts, popt[0] * dts**popt[1], ':', label=f'curve_fit: n={popt[1]:.3f}')
    plt.xlabel('Δt')
    plt.ylabel(f'Global error in x (fraction={fraction})')
    plt.title("Euler Method Convergence (Circular Orbit, short fraction)")
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.show()

    # Residuals in log space
    fit_pred_log = intercept + slope * logdt
    residuals = logerr - fit_pred_log
    print("\nResiduals (log space):")
    for dt, res in zip(dts, residuals):
        print(f"dt={dt:.1e} residual={res:.3f} dex")

    return {
        'dts': dts,
        'errors': errors,
        'n_logfit': n_lin,
        'n_logfit_std_err': std_err,
        'C_logfit': C_lin,
        'curve_fit_popt': popt,
        'curve_fit_pcov': pcov,
        'residuals_log': residuals
    }

res = fit_error_vs_dt_fraction(fraction=0.1)
