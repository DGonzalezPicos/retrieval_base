import numpy as np
import matplotlib.pyplot as plt

p = np.logspace(-5, 2, 50) # 1e-5 to 1e2 bar
log_K_1 = -6.0
log_K_2 = -3.0
log_K_P = -2.0
log_p = np.log10(p)

def VMR_two_point_profile(log_K_1, log_K_2, log_K_P, log_p):
    
    top = log_p <= log_K_P
    log_K_slope = (log_K_1 - log_K_2) / (log_K_P - log_p[0])
    log_K = log_K_1 * np.ones(len(log_p))
    log_K[top] = log_K_1 + log_K_slope * (log_p[top] - log_K_P)
    K = 10**log_K
    return K

K = VMR_two_point_profile(log_K_1, log_K_2, log_K_P, log_p)


log_Na_1 = -6.0
log_Na_2 = -8.0
log_Na_P = -1.0

Na = VMR_two_point_profile(log_Na_1, log_Na_2, log_Na_P, log_p)

fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.plot(K, p)
ax.plot(Na, p)
ax.set_xlabel('K [cm2/molecule]')
ax.set_ylabel('Pressure [bar]')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(p.max(), p.min())
# ax.set_ylim(1e-20, 1e-10)
plt.show()
