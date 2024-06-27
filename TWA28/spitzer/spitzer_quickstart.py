import numpy as np
import matplotlib.pyplot as plt
import pathlib
from scipy.signal import medfilt

path = pathlib.Path(__file__).parent.absolute()
# print(path)
file = path / '1102-3430.txt'
wave, flux, err, flag = np.loadtxt(file, skiprows=0).T
mask = flag > 0
print(f' Number of flagged points: {np.sum(mask)}')

# mask points beyond 33 micron
mask_wave = wave > 33
flux[mask_wave] = np.nan

# clip 3 sigma outliers
flux_medfilt = medfilt(flux, kernel_size=5)
mask_clip = np.abs(flux - flux_medfilt) > 3*err
flux[mask_clip] = np.nan

wave_cm = wave * 1e-4  # [microns] -> [cm]
# convert Jy to [erg cm^{-2} s^{-1} Hz^{-1}]
flux *= 1e-23
# convert [erg cm^{-2} s^{-1} Hz^{-1}] -> [erg cm^{-2} s^{-1} cm^{-1}]
flux *= 2.998e10 / wave_cm**2 # wave in cm
# Convert [erg cm^{-2} s^{-1} cm^{-1}] -> [erg cm^{-2} s^{-1} nm^{-1}]
flux *= 1e-7

err = err * (1e-23 * 1e-3) * (2.998e10 / wave_cm**2) * 1e-7

# flux[mask] = np.nan

fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.errorbar(wave, flux, yerr=err, fmt='o', color='black')
# medfilt
# ax.plot(wave, flux_medfilt, color='blue')

ax.errorbar(wave[mask_clip], flux[mask_clip], yerr=err[mask_clip], fmt='o', color='red')


xlim = (np.nanmin(wave), np.nanmax(wave))
ax.set_xlim(xlim)
ax.set_xlabel('Wavelength (microns)')
ax.set_ylabel('Flux (mJy)')
plt.show()
