import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy.optimize as sp

def time_estimation(dt, num_orbits):
    return(1.4828e-7 * (1/dt) * num_orbits)

def initial_conditions(a=0.387, e=0.2056, G=4*math.pi**2):
    
    r_peri = a * (1 - e)
    v_peri = math.sqrt(G * (1 + e) / (a * (1 - e)))
    x0, y0 = r_peri, 0.0
    vx0, vy0 = 0.0, v_peri
    return x0, y0, vx0, vy0, r_peri, v_peri


def euler_orbit(a, e, vx0, vy0, G=4*math.pi**2, dt=1e-4, num_orbits=1):
    
    r_peri = a * (1 - e)
    x, y = r_peri, 0.0
    vx, vy = vx0, vy0
    T = a**1.5
    total_time = num_orbits * T
    nsteps = int(np.ceil(total_time / dt))

    xs, ys = np.zeros(nsteps+1), np.zeros(nsteps+1)
    vxs, vys = np.zeros(nsteps+1), np.zeros(nsteps+1)
    xs[0], ys[0], vxs[0], vys[0] = x, y, vx, vy

    for i in range(nsteps):
        r = math.hypot(x, y)
        ax = -G * x / (r**3)
        ay = -G * y / (r**3)
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        xs[i+1], ys[i+1], vxs[i+1], vys[i+1] = x, y, vx, vy

    return xs, ys, vxs, vys, T

def analytic_ellipse(a, e, npts=400):

    theta = np.linspace(0, 2*np.pi, npts)
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def run_orbit(a=0.387, e=0.2056, dt=1e-4, num_orbits=1, G=4*math.pi**2):

    startTime = time.time()

    x0, y0, vx0, vy0, r_peri, v_peri = initial_conditions(a, e, G)
    xs, ys, vxs, vys, T = euler_orbit(a, e, vx0, vy0, G, dt, num_orbits)

    # Save trajectory
    df = pd.DataFrame({"x_AU": xs, "y_AU": ys})
    path = "/home/Isabella/Desktop/work/year 3/labs/week 6/code/"
    #df.to_csv(path, index=False)

    # Analytic ellipse for comparison
    xe, ye = analytic_ellipse(a, e)

    print("for dt=" + str(dt) + " and for N=" + str(num_orbits))
    print(time_estimation(dt, num_orbits))
    print("--- %s seconds true ---" % (time.time() - startTime) )

    return(time.time() - startTime)
    """
    # Plot
    plt.figure(figsize=(7,7))
    plt.plot(xs, ys, label="Euler orbit", lw=1.8)
    plt.plot(xe, ye, "--", color="gray", label="Kepler ellipse")
    plt.scatter(0, 0, color="orange", s=100, marker="*", label="☉ Sun")  # Sun symbol
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x (AU)")
    plt.ylabel("y (AU)")
    plt.title(f"Orbit Simulation (a={a} AU, e={e})\nΔt={dt}, T={T:.4f} yr")
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="black")
    plt.grid(True, alpha=0.3)
    plt.show()

    #print(f"Data saved to {path}")
    #print(f"Perihelion distance = {r_peri:.5f} AU")
    #print(f"Perihelion speed    = {v_peri:.5f} AU/yr")
    #print(f"Orbital period      = {T:.5f} yr")
    """

DT = []
DN = []


for Dt in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    DT.append(run_orbit(a=0.387, e=0.2056, dt=Dt, num_orbits=1))

for N in [1, 10, 100, 1000, 10000]:
    DN.append(run_orbit(a=0.387, e=0.2056, dt=1e-4, num_orbits=N))

print(DT)
print(DN)