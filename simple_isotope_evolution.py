import numpy as np  
import matplotlib.pyplot as plt


def twelve_C(t, twelve_C_0=500, k_12=29700):
    return twelve_C_0 + k_12 * t

def thirteen_C(t, thirteen_C_0=1, k_13=500):
    return thirteen_C_0 + k_13 * t

# initial conditions:
twelve_C_0 = 1e3
thirteen_C_0 = 1
C_ratio_0 = twelve_C_0 / thirteen_C_0

# present-day abundances:
twelve_C_now = twelve_C_0 * 10 # from [M/H] = -0.5 to [M/H] = 0.5
C_ratio_now = 60.0
thirteen_C_now = twelve_C_now / C_ratio_now

# rate constants:
k_13 = thirteen_C_now - thirteen_C_0
k_12 = ((1+k_13) * C_ratio_now) - twelve_C_0



t = np.logspace(-10, 2, 1000)

fig, ax = plt.subplots(2,1, figsize=(5,6), sharex=True, tight_layout=True)
ax[0].plot(t, twelve_C(t, twelve_C_0, k_12), label='12C')
ax[0].plot(t, thirteen_C(t, thirteen_C_0, k_13), label='13C')
ax[0].set_yscale('log')
ax[0].set_ylabel(f'Number of atoms\n(normalized to 12C/13C = {C_ratio_0:.0f} at t=0)')

ratio = twelve_C(t, twelve_C_0, k_12) / thirteen_C(t, thirteen_C_0, k_13)
ax[1].plot(t, ratio, label='12C/13C', color='darkgreen')
ax[1].set_ylabel(r'$^{12}C/^{13}C$')

ax[1].set_xlabel('Time / Gyr')
ax[1].set_xscale('log')

# equation for isotope ratio
eq = r'$\frac{^{12}C}{^{13}C} = \frac{^{12}C(0) + k_{12}t}{^{13}C(0) + k_{13}t}$'
# add arrow showing value as t goes to infinity
eq += '\n' + r'$t \rightarrow \infty, \frac{^{12}C}{^{13}C} \rightarrow \frac{k_{12}}{k_{13}}$'
ax[1].text(0.5, 2e2, eq, fontsize=14)

t_min = 1e-4
ratio_max = twelve_C(t_min, twelve_C_0, k_12) / thirteen_C(t_min, thirteen_C_0, k_13)
ax[1].set_xlim(1e-4, t[-1])
ax[1].set_yscale('log')
ax[1].axhline(C_ratio_now, color='k', ls='--', lw=0.5)

yticks = [30, 60, 120, 240, 480, 960]
ax[1].set_yticks(yticks)
ax[1].set_yticklabels([f'{y:.0f}' for y in yticks])
ax[1].set_ylim(30, yticks[-1])

ax[0].legend()
# plt.show()
path = '/home/dario/phd/retrieval_base/'
fig_name = path + 'simple_isotope_evolution.pdf'
fig.savefig(fig_name)
print(f'--> Saved {fig_name}')