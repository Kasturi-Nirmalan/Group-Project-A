import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

def initial_conditions(a=0.387, e=0.2056, G=4*math.pi**2):
    r_peri = a * (1 - e)
    v_peri = math.sqrt(G * (1 + e) / (a * (1 - e)))
    return r_peri, 0.0, 0.0, v_peri

def euler_orbit_energy(a, e, vx0, vy0, G=4*math.pi**2, dt=1e-4, num_orbits=1):
    r_peri = a * (1 - e)
    x, y = r_peri, 0.0
    vx, vy = vx0, vy0
    T = a**1.5
    total_time = num_orbits * T
    nsteps = int(np.ceil(total_time / dt))

    # initial energy
    r = math.hypot(x, y)
    E0 = 0.5 * (vx**2 + vy**2) - G / r

    for _ in range(nsteps):
        r = math.hypot(x, y)
        ax = -G * x / (r**3)
        ay = -G * y / (r**3)
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

    r = math.hypot(x, y)
    E = 0.5 * (vx**2 + vy**2) - G / r
    return abs(E - E0), T

def euler_accuracy_energy(a=0.387, e=0.2056, dts=None):
    if dts is None:
        dts = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    errors = []

    r_peri, y0, vx0, vy0 = initial_conditions(a, e)
    for dt in dts:
        err, T = euler_orbit_energy(a, e, vx0, vy0, dt=dt)
        errors.append(err)
        print(f"dt={dt:.1e}, energy error={err:.3e}")

    # Plot
    plt.figure(figsize=(6,5))
    plt.loglog(dts, errors, "o-", lw=2)
    plt.xlabel("Time step Δt (yr)")
    plt.ylabel("|ΔEnergy| (AU²/yr²)")
    plt.title("C3: Euler Energy Error vs Δt")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.show()

    pd.DataFrame({"dt": dts, "energy_error": errors}).to_csv("Y:/euler_energy_error_C3.csv", index=False)

euler_accuracy_energy()


