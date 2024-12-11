import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erfcinv, lambertw as W
# create a range of ln_B values
ln_B = np.linspace(1, 200, 100)

# calculate sigma for each value of ln_B
B = np.exp(ln_B)  # Safe if ln_B isn't too large
x = -1 / (B * np.exp(1))
W_val = W(x, k=-1).real
p = np.exp(W_val)
sigma = np.sqrt(2) * erfcinv(p)
print(f'sigma = {sigma}')
sigma_ref = [3, 5, 10, 20]
for s in sigma_ref:
    idx = np.argmin(np.abs(sigma - s))
    print(f'sigma = {s}: ln_B = {ln_B[idx]:.1f}')

fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.plot(ln_B, sigma)
ax.set_xlabel('ln B')
ax.set_ylabel('sigma')
ax.grid()
plt.show()