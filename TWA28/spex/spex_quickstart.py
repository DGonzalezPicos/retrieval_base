import numpy as np
import matplotlib.pyplot as plt
import pathlib
from astropy.io import fits
from scipy.signal import medfilt

path = pathlib.Path(__file__).parent.absolute()
# print(path)
file = path / 'U40002_1102_3430_katelyn.fits'
with fits.open(file) as hdul:
    print(hdul.info())
    wave, flux, err = hdul[0].data
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.plot(wave, flux, color='black')
ax.fill_between(wave, flux-err, flux+err, color='black', alpha=0.5)

plt.show()