import matplotlib.pyplot as plt
import numpy as np
import pathlib

from astropy.io import fits
from scipy.ndimage import gaussian_filter
path  = pathlib.Path(__file__).parent
file = path / 'IFU_PCE' / 'comm_PCE_F290LP_G395H_IFU.fits'

with fits.open(file) as hdu:
    print(hdu.info())
    data = hdu[1].data
    
wave, pce = data['Wavelength'], data['Obs. PCE']
wave *= 1e9
# apply gaussian smoothing to the PCE curve
pce_smooth = gaussian_filter(pce, sigma=10)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(wave, pce, label='Observed')
ax.plot(wave, pce_smooth, label='Smoothed')
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel('PCE [electrons/photon]')
ax.set_title('PCE Curve')
ax.legend()
plt.show()

