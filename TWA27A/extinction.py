
import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
import time


import retrieval_base.auxiliary_functions as af

path = af.get_path()
config_file = 'config_jwst.txt'
target = 'TWA27A'
# run = None
run = 'lbl15_KM_5'
wave, flux = np.load(pathlib.Path(path) / target / 'retrieval_outputs' / run / 'test_data/bestfit_model.npy')
wave_flat = wave.flatten()
flux_flat = flux.flatten()

wave_um = wave_flat * 1e-3 # [nm] -> [um]
Av = 3.0

# plot extinction law over wavelength range
fig, ax = plt.subplots(2,1, figsize=(12,7), gridspec_kw={'height_ratios': [2, 1],
                                                        'hspace': 0.12})
ax[0].plot(wave_um, flux_flat, color='darkgreen', lw=0.7, label='Best-fit model')

Av = np.arange(0.5, 10, 2.)
colors_Av = plt.cm.cividis(np.linspace(0, 1, len(Av)))

for i, Av_i in enumerate(Av):
    # ext = af.ism_extinction(Av_i, rv_red=3.1, wave=wave_um) # wave must be in [um]
    start = time.time()
    flux_ext = af.apply_extinction(flux_flat, wave_um, Av_i, rv_red=3.1)
    print(f' Slow version: {time.time()-start:.2e} s')
    
    
    ax[0].plot(wave_um, flux_ext, color=colors_Av[i], lw=0.7, label=f'Av={Av_i}')

    ax[1].plot(wave_um, flux_flat/flux_ext, color=colors_Av[i], lw=2.0)
    
ax[1].axhline(1.0, color='darkgreen', lw=0.7, ls='--')
ax[1].set(xlabel='Wavelength / um', ylabel='F / F * 10**(-0.4 Av)')

ax[0].set(ylabel='Flux / erg/s/cm2/nm')

ax[0].legend()
# fig_name = f'{conf.prefix}plots/extinction_law.pdf'
plt.show()
