import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
N12_0 = 300  # initial N12 abundance
N13_0 = 1    # initial N13 abundance
N12_now = 3000  # present-day N12 abundance
N13_now = 500   # present-day N13 abundance
t_now = 10  # time in arbitrary units for present-day abundances

# Define the system of differential equations
def carbon_isotopes(y, t, alpha, beta):
    N12, N13 = y
    dN12_dt = alpha - beta * N12
    dN13_dt = beta * N12
    return [dN12_dt, dN13_dt]

# Time array
t = np.linspace(1, t_now, 500)

# Solving for alpha and beta using present-day constraints
def equations(params):
    alpha, beta = params
    sol = odeint(carbon_isotopes, [N12_0, N13_0], t, args=(alpha, beta))
    N12_final, N13_final = sol[-1]
    return [N12_final - N12_now, N13_final - N13_now]

# Initial guess for alpha and beta
from scipy.optimize import fsolve
alpha_beta_guess = [100, 0.01]  # initial guess
alpha, beta = fsolve(equations, alpha_beta_guess)
print(f'alpha = {alpha}, beta = {beta}')

# Solving the differential equations with the found alpha and beta
sol = odeint(carbon_isotopes, [N12_0, N13_0], t, args=(alpha, beta))
N12, N13 = sol.T

# Plotting the results
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

log_t = np.log10(t)
ax[0].plot(log_t, N12, label=r'$^{12}$C', color='blue')
ax[0].plot(log_t, N13, label=r'$^{13}$C', color='orange')
ax[0].set(xlabel='log Time', ylabel='Abundance')
ax[0].legend(loc='upper right')

# Adding text to explain initial and final conditions
ax[0].text(0.03, 0.9, r'$t=0$: [C/H]=-0.5, $^{12}$C/$^{13}$C = 300, $^{12}$C = 300, $^{13}$C = 1',
           transform=ax[0].transAxes, fontsize=10, fontweight='bold')
ax[0].text(0.03, 0.8, r'$t=now$: [C/H]=0.5, $^{12}$C/$^{13}$C = 60, $^{12}$C = 3000, $^{13}$C = 500',
           transform=ax[0].transAxes, fontsize=10, fontweight='bold')

# Plot the ratio of 12C/13C over time
ax[1].plot(log_t, N12 / N13, label=r'$^{12}$C/$^{13}$C', color='green')
ax[1].set_xlabel('log Time')
ax[1].set_ylabel(r'$^{12}$C/$^{13}$C')

# Adding a caption explaining the isotope ratio formula
ratio_eq = (r'$^{12}\text{C}/^{13}\text{C} (t) = \frac{\frac{\alpha}{\beta} + \left( N_{12}(0) - \frac{\alpha}{\beta} \right) e^{-\beta t}}{N_{13}(0) + \alpha t + \left( N_{12}(0) - \frac{\alpha}{\beta} \right) \left( 1 - e^{-\beta t} \right)}$ ' +
             '\n' + r'$\alpha$ is the rate of injection of $^{12}\text{C}$, ' +
             '\n' + r'$\beta$ is the rate at which $^{12}\text{C}$ converts to $^{13}\text{C}$')
ax[1].text(0.25, 0.6, ratio_eq, transform=ax[1].transAxes, fontsize=14, fontweight='normal')
ax[1].axhline(60, color='brown', ls='--', label='Observed $^{12}$C/$^{13}$C', lw=3., alpha=0.4)
ax[1].text(0.38, 0.23, r'Observed $^{12}$C/$^{13}$C = 60 in metal rich M dwarfs (young)',
              transform=ax[1].transAxes, fontsize=10, fontweight='normal', color='brown')
           

plt.tight_layout()
fig_name = '/home/dario/phd/retrieval_base/iso_ratio_evolution.pdf'
fig.savefig(fig_name)
print(f'--> Saved {fig_name}')
plt.show()
