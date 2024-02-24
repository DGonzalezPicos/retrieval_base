import numpy as np 
import matplotlib.pyplot as plt

import pathlib
import pickle
import json

from astropy.io import fits



def sigma_clip(array, sigma=3, max_iter=5, return_mask=False):
    """Sigma clip an array by setting to NaN values that are more than sigma"""
    clipped = array.copy()
    for i in range(max_iter):
        mean = np.nanmean(clipped)
        std = np.nanstd(clipped)
        mask = np.abs(clipped - mean) < sigma * std
        clipped[~mask] = np.nan
        
    if return_mask:
        return clipped, mask
    return clipped

def read_data(file, units='mJy'):
    
    with fits.open(file) as hdul:
        data = hdul[1].data
        wave, flux, err = data['WAVELENGTH'], data['FLUX'], data['ERR'] # units [um, Jy, Jy]
    
    if units == 'mJy':
        flux *= 1e3
        err *= 1e3
    return wave, flux, err


targets = ['TWA27A', 'TWA27B', 'TWA28']
colors = ['navy', 'brown', 'darkgreen']
grisms = ['f100lp', 'f170lp', 'f290lp']
lw = 0.9

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                          gridspec_kw={'height_ratios': [1,1], 'wspace': 0.05, 'hspace': 0.01,
                                       'left': 0.1, 'right': 0.95, 'top': 0.95, 'bottom': 0.1})

for i, t in enumerate(targets):
    path = pathlib.Path(f'{t}/jwst')
    files = list(path.glob('*.fits'))
    # print(files)
    axi = ax[1] if t == 'TWA27B' else ax[0] 

    for j, file in enumerate(files):
        print(f'File {j+1} of {len(files)}')
        wave, flux, err = read_data(file)
        _, mask = sigma_clip(err, sigma=3, max_iter=5, return_mask=True)
        flux[~mask] = np.nan
        
        # for axi in ax:
        label = t if j == 0 else None
        axi.plot(wave, flux, label=label, color=colors[i], lw=lw)
            # axi.legend()
            
for axi in ax:            
    axi.set(ylabel='Flux / mJy')
    axi.legend(fontsize=12, loc='upper right', frameon=False)
ax[-1].set(xlabel='Wavelength / um')
fig.savefig('jwst_TWA_27_28.pdf')
# plt.show()


