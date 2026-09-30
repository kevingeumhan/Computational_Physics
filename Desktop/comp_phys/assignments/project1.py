


import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ODE: Parameters
# ==========================================
# theta'' + (g/L) sin(theta) = 0
m = 1.0       # Bob mass (kg)
g = 1.0       # Gravitational acceleration (m/s^2)  -- g/L = 1 so omega0 matches the SHO script
L = 1.0       # Rod length (m)
omega0 = np.sqrt(g / L)  # Small-angle angular frequency

theta0 = 2.0  # Initial angle (rad) ~ 115 deg: strongly nonlinear regime
w0 = 0.0      # Initial angular velocity (rad/s)

def compute_energy(theta, w):
    """E = T + V = (1/2) m L^2 w^2 + m g L (1 - cos theta)"""
    return 0.5 * m * L**2 * w**2 + m * g * L * (1.0 - np.cos(theta))

E0 = compute_energy(theta0, w0)
print(f"Initial Energy (E_0): {E0:.6f} J")


# ==========================================
# ODE: Derivatives
# ==========================================
def derivatives(theta, w):
    """Returns dtheta/dt and dw/dt"""
    dthetadt = w
    dwdt = -(omega0**2) * np.sin(theta)
    return dthetadt, dwdt


# ==========================================
# ODE: Algorithms
# ==========================================
def simulate(method, timestep, T_max, th_init=theta0, w_init=w0):
    """
    Simulates the pendulum trajectory up to T_max using step size h.
    """
    N = int(T_max / timestep)
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
            # Kick (update w with old theta), then drift (update theta with NEW w)
            _, dw = derivatives(theta[i], w[i])
            w[i+1] = w[i] + timestep * dw
            theta[i+1] = theta[i] + timestep * w[i+1]

        elif method == 'rk4':
            # Midpoint RK2
            dtheta1, dw1 = derivatives(theta[i], w[i])
            dtheta2, dw2 = derivatives(theta[i] + timestep * 0.5 * dtheta1, w[i] + timestep * 0.5 * dw1)
            dtheta3, dw3 = derivatives(theta[i] + timestep * 0.5 * dtheta2, w[i] + timestep * 0.5 * dw2)
            dtheta4, dw4 = derivatives(theta[i] + timestep * dtheta3, w[i] + timestep * dw3)

            theta[i+1] = theta[i] + timestep * (1/6) * (dtheta1 + 2*dtheta2 + 2*dtheta3 + dtheta4)
            w[i+1] = w[i] + timestep * (1/6) * (dw1 + 2*dw2 + 2*dw3 + dw4)


        E[i+1] = compute_energy(theta[i+1], w[i+1])

    return t, theta, w, E


# ==========================================
# ODE: Energy vs. Time
# ==========================================
h_fixed = 0.05
T_max = 50.0

t_exp, _, _, E_exp = simulate('explicit_euler', h_fixed, T_max)
t_sym, _, _, E_sym = simulate('symplectic_euler', h_fixed, T_max)
t_rk4, _, _, E_rk4 = simulate('rk4', h_fixed, T_max)


plt.figure(figsize=(10, 5))
plt.plot(t_exp, E_exp, label='Explicit Euler', color='red')
plt.plot(t_sym, E_sym, label='Symplectic Euler', color='green')
plt.plot(t_rk4, E_rk4, label='RK2 (Midpoint)', color='blue')
plt.ylim(1.3,3)
plt.show()