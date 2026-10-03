import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk

# ==========================================
# ODE: Parameters
# ==========================================
# theta'' + (g/L) sin(theta) = 0
m = 1.0       # Bob mass (kg)
g = 1.0       # Gravitational acceleration (m/s^2)  -- g/L = 1 so omega0 matches the SHO script
L = 1.0       # Rod length (m)
omega_0 = np.sqrt(g / L)  # Small-angle angular frequency

theta_0 = 2.0  # Initial angle (rad) ~ 115 deg: strongly nonlinear regime
w_0 = 0.0      # Initial angular velocity (rad/s)


# ==========================================
# ODE: Energy
# ==========================================
def compute_energy(theta, w):
    """E = T + V = (1/2) m L^2 w^2 + m g L (1 - cos theta)"""
    return 0.5 * m * L**2 * w**2 + m * g * L * (1.0 - np.cos(theta))

E0 = compute_energy(theta_0, w_0)


# ==========================================
# ODE: Exact Period
# ==========================================
def compute_period(theta, w):
    E = compute_energy(theta, w)
    return 4.0 / omega_0 * ellipk(E / (2 * m * g * L))

T_0 = 2 * np.pi / omega_0          # Small-angle period
P0 = compute_period(theta_0, w_0)


# ==========================================
# ODE: Derivative
# ==========================================
def derivatives(theta, w):
    dthetadt = w
    dwdt = -(omega_0**2) * np.sin(theta)
    return dthetadt, dwdt


# ==========================================
# ODE: Algorithms
# ==========================================
def simulate(method, timestep, time_max, th_init=theta_0, w_init=w_0):
    N = int((time_max) / timestep)
    t = np.linspace(0, N * timestep, N + 1)

    theta = np.zeros(N + 1)
    w = np.zeros(N + 1)
    E = np.zeros(N + 1)

    theta[0], w[0] = th_init, w_init
    E[0] = compute_energy(theta[0], w[0])

    for i in range(N):
        if method == 'explicit_euler':
            dtheta, dw = derivatives(theta[i], w[i])
            theta[i+1] = theta[i] + timestep * dtheta
            w[i+1] = w[i] + timestep * dw

        elif method == 'symplectic_euler':
            _, dw = derivatives(theta[i], w[i])
            w[i+1] = w[i] + timestep * dw
            theta[i+1] = theta[i] + timestep * w[i+1]

        elif method == 'rk4':
            # Classical 4th-order Runge-Kutta
            dtheta1, dw1 = derivatives(theta[i], w[i])
            dtheta2, dw2 = derivatives(theta[i] + timestep * 0.5 * dtheta1, w[i] + timestep * 0.5 * dw1)
            dtheta3, dw3 = derivatives(theta[i] + timestep * 0.5 * dtheta2, w[i] + timestep * 0.5 * dw2)
            dtheta4, dw4 = derivatives(theta[i] + timestep * dtheta3, w[i] + timestep * dw3)

            theta[i+1] = theta[i] + timestep * (1/6) * (dtheta1 + 2*dtheta2 + 2*dtheta3 + dtheta4)
            w[i+1] = w[i] + timestep * (1/6) * (dw1 + 2*dw2 + 2*dw3 + dw4)

        E[i+1] = compute_energy(theta[i+1], w[i+1])

    return t, theta, w, E


# ==========================================
# ODE: Period Measurement
# ==========================================
def hermite_crossing(t, y, dydt, i):

    h = t[i+1] - t[i]
    y0, y1 = y[i], y[i+1]
    m0, m1 = dydt[i] * h, dydt[i+1] * h
    s = -y0 / (y1 - y0)            # Start from the linear guess
    for _ in range(10):            # Newton's method on the cubic
        p  = (2*s**3 - 3*s**2 + 1)*y0 + (s**3 - 2*s**2 + s)*m0 + (-2*s**3 + 3*s**2)*y1 + (s**3 - s**2)*m1
        dp = (6*s**2 - 6*s)*y0 + (3*s**2 - 4*s + 1)*m0 + (-6*s**2 + 6*s)*y1 + (3*s**2 - 2*s)*m1
        s -= p / dp
    return t[i] + s * h


def measure_period(t, theta, w):
    idx = np.where((theta[:-1] < 0) & (theta[1:] >= 0))[0]
    if len(idx) < 2:
        return np.nan      # Fewer than 2 crossings (e.g. pendulum went over the top)

    crossings = [hermite_crossing(t, theta, w, i) for i in idx]
    return (crossings[-1] - crossings[0]) / (len(crossings) - 1)


def first_crossing_time(t, theta, w, target):
    y = theta - target
    idx = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    if len(idx) == 0:
        return np.nan
    return hermite_crossing(t, y, w, idx[0])


# ##########################################
# VERIFICATION 1: ENERGY CONSERVATION
# ##########################################

# ==========================================
# ODE: Energy vs. Time
# ==========================================

timestep_fixed = 0.05
time_max = 50

def plot_energy_vs_time(timestep_fixed=0.05, time_max=50.0):
    t_exp, _, _, E_exp = simulate('explicit_euler', timestep_fixed, time_max)
    t_sym, _, _, E_sym = simulate('symplectic_euler', timestep_fixed, time_max)
    t_rk4, _, _, E_rk4 = simulate('rk4', timestep_fixed, time_max)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(t_exp, E_exp, label='Explicit Euler', color='red')
    plt.plot(t_sym, E_sym, label='Symplectic Euler', color='green')
    plt.plot(t_rk4, E_rk4, label='RK4', color='blue')
    plt.axhline(E0, label=r'$E_0$', color='black', linestyle='--')
    plt.ylim(1.3, 3)
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title(f'Energy vs. Time (h = {timestep_fixed})')
    plt.legend()
    plt.savefig("energy_vs_time.png")

    return fig


# ==========================================
# ODE: Energy vs. Timestep
# ==========================================
def plot_energy_vs_timestep(timestep=np.logspace(-3, -0.5, 30), time=20):
    error_exp = []
    error_sym = []
    error_rk4 = []

    for i in timestep:
        _, _, _, E_e = simulate("explicit_euler", i, time)
        _, _, _, E_s = simulate("symplectic_euler", i, time)
        _, _, _, E_r = simulate("rk4", i, time)

        error_exp.append(np.max(np.abs(E_e - E0)) / E0)
        error_sym.append(np.max(np.abs(E_s - E0)) / E0)
        error_rk4.append(np.max(np.abs(E_r - E0)) / E0)

    fig = plt.figure()
    plt.loglog(timestep, error_exp, label='Explicit Euler', color='red')
    plt.loglog(timestep, error_sym, label='Symplectic Euler', color='green')
    plt.loglog(timestep, error_rk4, label='RK4', color='blue')
    plt.xlabel('Timestep h (s)')
    plt.ylabel(r'max $|E - E_0| / E_0$')
    plt.title('Energy Error vs. Timestep')
    plt.legend()
    plt.savefig("energy_vs_timestep.png")

    return fig


# ##########################################
# VERIFICATION 2: PERIOD
# ##########################################

# ==========================================
# ODE: Period vs. Amplitude
# ==========================================
def plot_period_vs_amplitude(timestep_fixed=0.01, amplitudes=np.linspace(0.1, 3.0, 30)):
    period_exp = []
    period_sym = []
    period_rk4 = []

    for a in amplitudes:
        time = 2 * compute_period(a, 0.0)     # 2 periods -> 2 upward zero crossings
        t_e, th_e, w_e, _ = simulate("explicit_euler", timestep_fixed, time, th_init=a)
        t_s, th_s, w_s, _ = simulate("symplectic_euler", timestep_fixed, time, th_init=a)
        t_r, th_r, w_r, _ = simulate("rk4", timestep_fixed, time, th_init=a)

        period_exp.append(measure_period(t_e, th_e, w_e))
        period_sym.append(measure_period(t_s, th_s, w_s))
        period_rk4.append(measure_period(t_r, th_r, w_r))

    amp_fine = np.linspace(0.01, 3.1, 300)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(amp_fine, compute_period(amp_fine, 0.0) / T_0, label='Exact', color='black')
    plt.axhline(1.0, label='Small-angle limit', color='gray', linestyle='--')
    plt.plot(amplitudes, np.array(period_exp) / T_0, 'x', label='Explicit Euler', color='red')
    plt.plot(amplitudes, np.array(period_sym) / T_0, 's', label='Symplectic Euler', color='green', mfc='none')
    plt.plot(amplitudes, np.array(period_rk4) / T_0, 'o', label='RK4', color='blue', mfc='none')
    plt.xlabel(r'Amplitude $\theta_0$ (rad)')
    plt.ylabel(r'$T / T_0$')
    plt.title(f'Period vs. Amplitude (h = {timestep_fixed})')
    plt.legend()
    plt.savefig("period_vs_amplitude.png")

    return fig


# ==========================================
# ODE: Period vs. Timestep
# ==========================================
def plot_period_vs_timestep(timestep=np.logspace(-3, -0.5, 30), n_periods=3):
    time = n_periods * P0

    error_exp = []
    error_sym = []
    error_rk4 = []

    for i in timestep:
        t_e, th_e, w_e, _ = simulate("explicit_euler", i, time)
        t_s, th_s, w_s, _ = simulate("symplectic_euler", i, time)
        t_r, th_r, w_r, _ = simulate("rk4", i, time)

        error_exp.append(abs(measure_period(t_e, th_e, w_e) - P0) / P0)
        error_sym.append(abs(measure_period(t_s, th_s, w_s) - P0) / P0)
        error_rk4.append(abs(measure_period(t_r, th_r, w_r) - P0) / P0)

    fig = plt.figure()
    plt.loglog(timestep, error_exp, label='Explicit Euler', color='red')
    plt.loglog(timestep, error_sym, label='Symplectic Euler', color='green')
    plt.loglog(timestep, error_rk4, label='RK4', color='blue')
    plt.xlabel('Timestep h (s)')
    plt.ylabel(r'$|T - P_0| / P_0$')
    plt.title(rf'Period Error vs. Timestep ($\theta_0$ = {theta_0} rad)')
    plt.legend()
    plt.savefig("period_vs_timestep.png")
    return fig