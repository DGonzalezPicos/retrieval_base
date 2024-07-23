import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

import config_jwst as conf
from retrieval_base.spectrum_jwst import SpectrumJWST
from retrieval_base.auxiliary_functions import get_path

path = get_path()
config_file = 'config_jwst.txt'
target = 'TWA27b'
run = None

cwd = os.getcwd()
if target not in cwd:
    os.chdir(cwd + f'/{target}/')

gratings_dict = {'g140h': 'g140h-f100lp',
                     'g235h': 'g235h-f170lp', 
                     'g395h': 'g395h-f290lp'}
    
gratings_list = list(set(conf.constant_params['gratings']))
gratings = [gratings_dict[g] for g in gratings_list]

conf_data = conf.config_data['NIRSpec']
for key in ['data', 'plots']:
    pathlib.Path(f'{conf.prefix}{key}/').mkdir(parents=True, exist_ok=True)
# each grating has two filters, make list [a,b] to [a,a,b,b]
# gratings_list = [g.split('-')[0] for g in gratings for _ in range(2)]
# print(f'--> Loading data for {gratings_list}')

files = [f'jwst/{target}_{g}.fits' for g in gratings]
Nedge = conf_data.get('Nedge', 40)
spec = SpectrumJWST(Nedge=Nedge).load_gratings(files)
spec.reshape(spec.n_orders, 1)
# spec.fix_wave_nans() # experimental...
spec.sigma_clip_reshaped(use_flux=False, 
                            # sigma=3, # KM bands
                            sigma=conf_data.get('sigma_clip', 3),
                            width=conf_data.get('sigma_clip_width', 30),
                            max_iter=5,
                            fun='median', 
                            debug=False)

# load data from Sid

wave, flux, err = ([] for _ in range(3))
for i in [1,2]:
    file = f'jwst_sid/nirspec_395_{i}.txt'
    data_i = np.loadtxt(file, skiprows=0).T
    wave.append(data_i[0])
    flux.append(data_i[2])
    err.append(data_i[3])
    
wave = np.concatenate(wave)
flux = np.concatenate(flux)
err = np.concatenate(err)
    
    
wave *= 1e3 # [microns] -> [nm]
flux /= np.nanmedian(flux)
flux *= np.nanmedian(spec.flux)

wave_mask = (4568, 4580)
mask = (wave > wave_mask[0]) & (wave < wave_mask[1])
flux[mask] = np.nan

fig, ax = plt.subplots(2,1, figsize=(14,6), tight_layout=True, gridspec_kw={'height_ratios': [3,1]}, sharex=True)

for i in range(2):
    w = spec.wave[i,0,:]
    f = spec.flux[i,0,:]
    e = spec.err[i,0,:]
    ax[0].plot(w, f, 'k-', label='Manjavacas+2024' if i==0 else None)
    ax[0].fill_between(w, f-e, f+e, color='k', alpha=0.2)
    
    
    # file = f'jwst_sid/nirspec_395_{i+1}.txt'
    # wave, _, flux, err = np.loadtxt(file, skiprows=0).T
    # wave *= 1e3 # [microns] -> [nm]
    # flux /= np.nanmedian(flux)
    # flux *= np.nanmedian(f)
    # ax[0].plot(wave, flux, 'r-', label='Sid' if i==0 else None)

    # res = np.interp(wave, w,f) - flux
    # res /= flux
    # res *= 100
    # ax[1].plot(wave, res, 'r-')
    
ax[1].axhline(0, color='k', ls='-', lw=0.5)

# symmetric axis residuals plot
ylim = np.max(np.abs(ax[1].get_ylim()))
ax[1].set_ylim(-ylim, ylim)
ax[1].set_ylabel('Relative res. (M-S)/S [%]')


ax[0].legend()

ax[-1].set_xlabel('Wavelength / nm')
ax[0].set_ylabel('Flux / erg cm$^{-2}$ s$^{-1}$ nm$^{-1}$')
fig_name = f'{conf.prefix}plots/compare_data_sid_test.pdf'
fig.savefig(fig_name)
print(f'Figure saved as {fig_name}')