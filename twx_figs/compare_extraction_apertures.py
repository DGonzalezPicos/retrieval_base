import numpy as np
import matplotlib.pyplot as plt

import pathlib
from retrieval_base.auxiliary_functions import get_path

path = get_path(return_pathlib=True)
target = 'TWA27A'
# grating = 'g140h'
grating = 'g395h'
n_ap_list = [4,5,6]

fig, ax = plt.subplots(2,1,figsize=(14, 5), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

wave_list, flux_list, err_list = [], [], []
for i, n_ap in enumerate(n_ap_list):
    file = path / target / 'jwst' / f'{grating}_s3d_extraction_{n_ap}ap.npy'
    if not file.exists():
        print(f'File {file} does not exist')
        continue
    wave, flux, err = np.load(file)

    ax[0].plot(wave, flux, label=f'nap={n_ap}')
    ax[0].fill_between(wave, flux-err, flux+err, alpha=0.5)
    
    
    
    wave_list.append(wave)
    flux_list.append(flux)
    err_list.append(err)
    
    if len(wave_list) < (i+1):
        print(f'No data to plot')
        continue
    
    residuals = flux_list[-1] / flux_list[-(i+1)]
    ax[1].plot(wave_list[-1], residuals, label=f'nap={n_ap}', color=ax[0].lines[-1].get_color())
    
ax[0].legend()
ax[0].set_ylabel('Flux')

ax[1].legend()
ax[1].set_ylabel('Error')
ax[1].set_xlabel('Wavelength (nm)')
plt.show()









