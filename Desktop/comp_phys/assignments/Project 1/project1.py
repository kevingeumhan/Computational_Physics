import matplotlib.pyplot as plt

import project1_ode
import project1_integral

# ==========================================
# Summary of exact values
# ==========================================
print(f"Initial Energy (E_0):        {project1_ode.E0:.6f} J")
print(f"Small-angle period (T_0):    {project1_ode.T_0:.6f} s")
print(f"Exact period (P_0):          {project1_ode.P0:.6f} s")
print(f"Upper limit phi_1:           {project1_integral.phi_1:.6f} rad")
print(f"Exact swing time t(theta_1): {project1_integral.t1_exact:.10f} s")

# ==========================================
# Part (a): ODE
# ==========================================
# Verification 1: Energy conservation
project1_ode.plot_energy_vs_time()
project1_ode.plot_energy_vs_timestep()

# Verification 2: Period
project1_ode.plot_period_vs_amplitude()
project1_ode.plot_period_vs_timestep()




# ==========================================
# Part (b): Definite integral
# ==========================================
# Numerical methods vs. SciPy and convergence
project1_integral.compare_with_scipy()
project1_integral.plot_error_vs_N()


# Verification 1: Small-angle limit
project1_integral.plot_small_angle_limit()

# Verification 2: Consistency with the ODE
project1_integral.plot_consistency_with_ode()

plt.show()