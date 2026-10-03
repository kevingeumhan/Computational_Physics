import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipkinc
from scipy.integrate import trapezoid, simpson

from project1_ode import (m, g, L, omega_0, theta_0, E0, T_0, P0,
                 compute_period, simulate, first_crossing_time)

# ==========================================
# Integral: Parameters
# ==========================================
theta_1 = 1.0                          # Upper angle (rad), must satisfy theta1 < theta0
k2 = E0 / (2 * m * g * L)             # k^2 from the same initial energy as the ODE


def phi_limit(theta_up, k2):
    return np.arcsin(np.sin(theta_up / 2) / np.sqrt(k2))


phi_1 = phi_limit(theta_1, k2)


# ==========================================
# Integral: Integrand and Exact Solution
# ==========================================
def integrand(phi, k2):
    return 1.0 / (omega_0 * np.sqrt(1.0 - k2 * np.sin(phi)**2))


def compute_swing_time(phi_up, k2):
    return ellipkinc(phi_up, k2) / omega_0


t1_exact = compute_swing_time(phi_1, k2)


def f(phi):
    return integrand(phi, k2)


# ==========================================
# Integral: Algorithms
# ==========================================
def integrate(method, f, a, b, N):

    x = np.linspace(a, b, N + 1)
    y = f(x)
    h = (b - a) / N

    if method == 'riemann':
        # Left Riemann sum
        return h * np.sum(y[:-1])

    elif method == 'trapezoid':
        return h * (np.sum(y) - 0.5 * (y[0] + y[-1]))

    elif method == 'simpson':
        # Weights 1, 4, 2, 4, ..., 2, 4, 1
        return (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))

    elif method == 'scipy_trapezoid':
        return trapezoid(y, x)

    elif method == 'scipy_simpson':
        return simpson(y, x=x)


# ==========================================
# Integral: Comparison with SciPy
# ==========================================
def compare_with_scipy(N_check=64):
    print(f"\nHand vs. SciPy (N = {N_check}):")
    print(f"  |trapezoid - scipy_trapezoid| = "
          f"{abs(integrate('trapezoid', f, 0, phi_1, N_check) - integrate('scipy_trapezoid', f, 0, phi_1, N_check)):.3e}")
    print(f"  |simpson   - scipy_simpson|   = "
          f"{abs(integrate('simpson', f, 0, phi_1, N_check) - integrate('scipy_simpson', f, 0, phi_1, N_check)):.3e}")


# ==========================================
# Integral: Error vs. Number of Points
# ==========================================
def plot_error_vs_N(N_list=2 ** np.arange(2, 13)):     # Even numbers of intervals, 4 ... 4096
    int_err_rie = []
    int_err_trap = []
    int_err_simp = []
    int_err_strap = []
    int_err_ssimp = []

    for N in N_list:
        int_err_rie.append(abs(integrate('riemann', f, 0, phi_1, N) - t1_exact) / t1_exact)
        int_err_trap.append(abs(integrate('trapezoid', f, 0, phi_1, N) - t1_exact) / t1_exact)
        int_err_simp.append(abs(integrate('simpson', f, 0, phi_1, N) - t1_exact) / t1_exact)
        int_err_strap.append(abs(integrate('scipy_trapezoid', f, 0, phi_1, N) - t1_exact) / t1_exact)
        int_err_ssimp.append(abs(integrate('scipy_simpson', f, 0, phi_1, N) - t1_exact) / t1_exact)

    fig = plt.figure()
    plt.loglog(N_list, int_err_rie, 'o-', label='Riemann (left)', color='red')
    plt.loglog(N_list, int_err_trap, 'o-', label='Trapezoid', color='green')
    plt.loglog(N_list, int_err_simp, 'o-', label='Simpson', color='blue')
    plt.loglog(N_list, int_err_strap, 'x', label='SciPy trapezoid', color='black')
    plt.loglog(N_list, int_err_ssimp, '+', label='SciPy Simpson', color='black', markersize=10)
    plt.xlabel('Number of intervals N')
    plt.ylabel(r'$|t - t_{exact}| / t_{exact}$')
    plt.title(rf'Integral Error vs. N ($\theta_1$ = {theta_1} rad)')
    plt.legend()
    plt.savefig("error_vs_N.png")

    return fig


# ##########################################
# VERIFICATION 1: SMALL-ANGLE LIMIT
# ##########################################
# As theta0 -> 0, k -> 0 and the integrand -> 1/omega_0, so the quarter-period
# integral (phi1 = pi/2) gives T = 4 * (pi/2) / omega_0 = 2 pi / omega_0 = T_0.
# First correction: T / T_0 ~ 1 + theta0^2 / 16.

def plot_small_angle_limit(N_fixed=16, amplitudes=np.linspace(0.05, 3.0, 30)):
    Tint_rie = []
    Tint_trap = []
    Tint_simp = []

    for a in amplitudes:
        k2_a = np.sin(a / 2)**2
        f_a = lambda phi: integrand(phi, k2_a)
        Tint_rie.append(4 * integrate('riemann', f_a, 0, np.pi / 2, N_fixed))
        Tint_trap.append(4 * integrate('trapezoid', f_a, 0, np.pi / 2, N_fixed))
        Tint_simp.append(4 * integrate('simpson', f_a, 0, np.pi / 2, N_fixed))

    amp_fine = np.linspace(0.01, 3.1, 300)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(amp_fine, compute_period(amp_fine, 0.0) / T_0, label='Exact', color='black')
    plt.axhline(1.0, label='Small-angle limit', color='gray', linestyle='--')
    plt.plot(amp_fine, 1 + amp_fine**2 / 16, label=r'$1 + \theta_0^2/16$', color='gray', linestyle=':')
    plt.plot(amplitudes, np.array(Tint_rie) / T_0, 'x', label='Riemann (left)', color='red')
    plt.plot(amplitudes, np.array(Tint_trap) / T_0, 's', label='Trapezoid', color='green', mfc='none')
    plt.plot(amplitudes, np.array(Tint_simp) / T_0, 'o', label='Simpson', color='blue', mfc='none')
    plt.ylim(0.9, 3.0)
    plt.xlabel(r'Amplitude $\theta_0$ (rad)')
    plt.ylabel(r'$T / T_0$')
    plt.title(f'Period from the Integral vs. Amplitude (N = {N_fixed})')
    plt.legend()
    plt.savefig("small_angle_limit.png")

    return fig


# ##########################################
# VERIFICATION 2: CONSISTENCY WITH THE ODE
# ##########################################
# Launch the ODE from the bottom with the same energy E0 and record when RK4
# reaches each theta1. This must match the integral t(theta1). The quarter
# period from the integral must also match P0 from the ODE module.

def plot_consistency_with_ode(timestep=0.01, N=64, theta1_list=np.linspace(0.1, 0.95 * theta_0, 20)):
    w_bottom = np.sqrt(2 * E0 / (m * L**2))          # (1/2) m L^2 w^2 = E0 at theta = 0
    t_ode, th_ode, w_ode, _ = simulate('rk4', timestep, P0 / 4, th_init=0.0, w_init=w_bottom)

    t_int = []
    t_rk4_cross = []

    for th1 in theta1_list:
        t_int.append(integrate('simpson', f, 0, phi_limit(th1, k2), N))
        t_rk4_cross.append(first_crossing_time(t_ode, th_ode, w_ode, th1))

    t_int = np.array(t_int)
    t_rk4_cross = np.array(t_rk4_cross)
    print(f"\nMax |t_integral - t_RK4| over theta_1: {np.max(np.abs(t_int - t_rk4_cross)):.3e} s")

    T_quarter = integrate('simpson', f, 0, np.pi / 2, N)
    print(f"4 x quarter-period integral: {4 * T_quarter:.10f} s")
    print(f"P0 from the ODE module:      {P0:.10f} s")

    fig = plt.figure(figsize=(10, 5))
    plt.plot(theta1_list, t_int, 'o-', label='Integral (Simpson)', color='blue', mfc='none')
    plt.plot(theta1_list, t_rk4_cross, 'x', label='ODE (RK4) crossing time', color='red', markersize=9)
    plt.axhline(P0 / 4, label=r'$P_0/4$ (turning point)', color='gray', linestyle='--')
    plt.xlabel(r'$\theta_1$ (rad)')
    plt.ylabel(r'Time to swing from 0 to $\theta_1$ (s)')
    plt.title(rf'Integral vs. ODE ($\theta_0$ = {theta_0} rad)')
    plt.legend()
    plt.savefig("consistency_with_ode.png")

    return fig