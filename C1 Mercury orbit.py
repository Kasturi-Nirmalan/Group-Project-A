import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# C2 initial conditions
def initial_conditions(a=0.387, e=0.2056, G=4*math.pi**2):
    r_peri = a * (1 - e)
    v_peri = math.sqrt(G * (1 + e) / (a * (1 - e)))
    x0, y0 = r_peri, 0.0
    vx0, vy0 = 0.0, v_peri
    return x0, y0, vx0, vy0, r_peri, v_peri

# C1 Euler
def euler_orbit(a, e, vx0, vy0, G=4*math.pi**2, dt=1e-4, num_orbits=1):
    r_peri = a * (1 - e)
    x, y = r_peri, 0.0
    vx, vy = vx0, vy0

    # Orbital period (Kepler's third law)
    T = a**1.5
    total_time = num_orbits * T
    nsteps = int(np.ceil(total_time / dt))

    times = np.zeros(nsteps+1)
    xs = np.zeros(nsteps+1)
    ys = np.zeros(nsteps+1)
    vxs = np.zeros(nsteps+1)
    vys = np.zeros(nsteps+1)

    times[0], xs[0], ys[0], vxs[0], vys[0] = 0, x, y, vx, vy

    for i in range(nsteps):
        r = math.hypot(x, y)
        ax = -G * x / (r**3)
        ay = -G * y / (r**3)

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        times[i+1] = times[i] + dt
        xs[i+1] = x
        ys[i+1] = y
        vxs[i+1] = vx
        vys[i+1] = vy

    return times, xs, ys, vxs, vys, T


def run_mercury_orbit():
    a = 0.387      # AU
    e = 0.2056     # dimensionless
    dt = 1e-4      # years per step
    num_orbits = 1 # simulate one full revolution

    x0, y0, vx0, vy0, r_peri, v_peri = initial_conditions(a, e)

    print("=== MERCURY ORBIT INITIAL CONDITIONS (C2) ===")
    print(f"Semi-major axis a = {a} AU")
    print(f"Eccentricity e = {e}")
    print(f"Perihelion distance r_p = {r_peri:.6f} AU")
    print(f"Perihelion speed v_p = {v_peri:.6f} AU/yr\n")

    times, xs, ys, vxs, vys, T = euler_orbit(a, e, vx0, vy0, dt=dt, num_orbits=num_orbits)

    df = pd.DataFrame({
        "time_yr": times,
        "x_AU": xs,
        "y_AU": ys,
        "vx_AU_per_yr": vxs,
        "vy_AU_per_yr": vys
    })
    csv_path = "Y:/mercury_euler_C1C2.csv"
    df.to_csv(csv_path, index=False)

    print("=== SIMULATION RESULTS (C1) ===")
    print(f"Kepler period (T) = {T:.6f} years")
    print(f"Total steps = {len(times)}")
    print(f"Data saved to: {csv_path}\n")

# Plot the orbit
    plt.figure(figsize=(6,6))
    plt.plot(xs, ys, label="Euler orbit")
    plt.scatter([0], [0], color="orange", s=60, label="Sun")
    plt.xlabel("x (AU)")
    plt.ylabel("y (AU)")
    plt.title(f"Mercury Orbit (Euler + Correct ICs)\nΔt={dt}, T={T:.4f} yr")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="upper right")
    plt.show()

run_mercury_orbit()
