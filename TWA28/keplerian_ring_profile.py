# import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import interp1d
import time
from PyAstronomy import pyasl

from retrieval_base.auxiliary_functions import instr_broadening, pickle_load, get_path, apply_keplerian_profile

# Physical constants and parameters for line profile
G = 6.67430e-8  # Gravitational constant in cgs units (cm^3 g^-1 s^-2)
# M_star = 0.1 * 1.989e33  # Stellar mass in grams (for a solar-mass star)
Mjup = 1.898e30  # Jupiter mass in grams
M_star_Mjup = 20
M_star = 20 * Mjup
i_deg = 0.  # Disk inclination angle in degrees (45 degrees as an example)
i = np.radians(i_deg)  # Convert inclination to radians

rjup = 7.1492e9 # Jupiter radius in cm

rmin = 1 * rjup  # Inner disk radius in cm
rmax = 10 * rjup  # Outer disk radius in cm

nr = 20
ntheta = 60
radii = np.linspace(rmin, rmax, nr)  # Radii in cm
radii_Rjup = radii / rjup



# Plotting
fig, (ax_lp, ax) = plt.subplots(1,2, figsize=(12,5))

# create broadening kernel for gaussian broadening at a resolving power of R = 3000
R = 3000
resolution_element = 2.998e5 / R
print(f' resolution element = {resolution_element:.2f} km/s')

path = get_path(return_pathlib=True)


template = path / 'data/slab_models/12CO/g395h/slab_600K_N5e+17.npy'
wave, flux = np.load(template)

wmin = 4.63
wmax = 4.68
mask = (wave > wmin) & (wave < wmax)
wave = wave[mask]
flux = flux[mask]

start = time.time()
flux_s = apply_keplerian_profile(wave, flux, radii_Rjup, M_star_Mjup, i_deg, ntheta)
print(f'Elapsed time (fast): {1000*(time.time() - start):.2f} ms')
print(f' q16, q50, q84 = {np.percentile(flux_s, [16, 50, 84])}')

# find broadening function for the line profile by comparing flux and flux_s
# broadening kernel
svd = pyasl.SVD()
# Use 51 bins for the width of the broadening function.
# Needs to be an odd number.
svd.decompose(flux, 51)
# Obtain the broadening function needed to
# recover "observed" spectrum. Note that the
# edges (51/2 bins) will actually be neglected.
sv = svd.getSingularValues()
# b = svd.getBroadeningFunction(flux_s, wlimit=sorted(sv)[-len(sv)//2])
b = svd.getBroadeningFunction(flux_s)
b /= np.sum(b)
# Get the model, which results from the broadening
# function and the template; obtain the indices
# where it applies, too.
m, mind = svd.getModel(b, modelIndices=True)

# get radial velocity from modelindices
binsize = wave[1] - wave[0]

rv = svd.getRVAxis(binsize, np.mean(wave))

# start = time.time()
# flux_s_fast = apply_keplerian_profile_fast(wave, flux, velocities, radii, G, M_star, i, ntheta)
# print(f'Elapsed time (fast): {time.time() - start:.2f} seconds')
# Right panel: Keplerian line profile
# ax_lp.plot(velocities / 1e5, lp, color='blue', lw=2)
ax_lp.set_xlabel('Velocity (km/s)')
ax_lp.set_ylabel('Intensity (arbitrary units)')
ax_lp.set_title(f"Keplerian Line Profile (inclination = {np.degrees(i):.1f} degrees)")

ax_lp.plot(rv, b, color='red', lw=2, ls='--')
ax_lp.axvline(-42.6, color='black', lw=1.0, ls='--')
# ax_lp.plot(mind, m, color='green', lw=2, ls='--')

ax.plot(wave, flux, color='black', lw=1.0, label='Original')
ax.plot(wave, flux_s, color='blue', lw=1.0, label='Keplerian')
ax.set_xlabel('Wavelength (microns)')

plt.tight_layout()
fig_name = path / f'keplerian_line_profile_rmin{rmin/rjup:.0f}_rmax{rmax/rjup:.0f}_incl{i_deg:.0f}.pdf'
plt.savefig(fig_name)
print(f'--> Saved {fig_name}')
plt.show()
