import numpy as np
import matplotlib.pyplot as plt
import pathlib

path = pathlib.Path('/home/dario/phd/retrieval_base/')
file = path / 'paper/data/abunda.bncmrkTHIN_300'
assert file.exists(), f'File {file} does not exist'
data = np.loadtxt(file, skiprows=0)

# columns: t [Gyr], [Fe/H], 6th and 7th columns are 12C/13C mass, 13th and 14th are O16/O18 by mass

Z = data[:, 1]
c12 = data[:, 5] * 12
c13 = data[:, 6] * 13

c12c13 = c12 / c13

o16 = data[:, 12] * 16
o18 = data[:, 13] * 18

mask = (Z > -1)
Z = Z[mask]
c12c13 = c12c13[mask]
o16o18 = o16 / o18
o16o18 = o16o18[mask]

fig, ax = plt.subplots(2,1, figsize=(8,6), sharex=True)
ax[0].plot(Z, c12c13, label='data')
ax[0].set_ylabel('12C/13C')

ax[1].plot(Z, o16o18, label='data')
ax[1].set_ylabel('16O/18O')

ax[0].set_yscale('log')
ax[1].set_yscale('log')

plt.show()
