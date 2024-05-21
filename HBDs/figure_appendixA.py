'''Telluric correction with Molecfit'''

import numpy as np
import matplotlib.pyplot as plt
import pathlib
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')

from retrieval_base.auxiliary_functions import pickle_load
# plt.style.use('HBDs/my_science.mplstyle')
# plt.rcParams['text.usetex'] = False
# must use environment: Python 3.11.4 ('pycrires': conda)

target = 'TWA28'
prefix = 'final_full'

# current working directory
path = pathlib.Path('/home/dario/phd/retrieval_base')
data_path = path / f'{target}' / f'retrieval_outputs/{prefix}/test_data/'
# out_path = path / 'HBDs'
out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')

d_spec = pickle_load(data_path / 'd_spec_K2166.pkl')

# wave = d_spec.wave
# flux = d_spec.flux
# err = d_spec.err

wave = np.load(data_path / 'd_spec_wave_K2166.npy')
flux = np.load(data_path / 'd_spec_flux_K2166.npy')
err  = np.load(data_path / 'd_spec_err_K2166.npy')
flux_uncorr = np.load(data_path / 'd_spec_flux_uncorr_K2166.npy')
transm = np.load(data_path / 'd_spec_transm_K2166.npy')

# normalize flux_uncorr in the same way as flux
deep_lines = transm < d_spec.tell_threshold
f = np.where(deep_lines, np.nan, flux_uncorr)
# norm = getattr(np, f'nan{fun}')(f, axis=-1)
norm = np.nanmedian(f, axis=-1)
 # median flux per order
flux_uncorr /= norm[...,None]

n_orders, n_dets, _ = flux.shape

fig, ax = plt.subplots(n_orders, 1, figsize=(14,10), sharey=True,
                       gridspec_kw={'hspace':0.30})
# create gridspec for the figure showing two subplots below one wide subplot
# import matplotlib.gridspec as gridspec
# fig = plt.figure(figsize=(14, 6))
# gs = gridspec.GridSpec(2, 2, width_ratios=[1,1])
# ax = plt.subplot(gs[0, :])
# ax2 = plt.subplot(gs[1, 0])
# ax3 = plt.subplot(gs[1, 1])

color_data = 'k'
color_transm = 'brown'

scale = 1
wave_min = np.min(wave, axis=(1,2))
wave_max = np.max(wave, axis=(1,2))
for order in range(n_orders):
    ax[order].set_xlim(wave_min[order], wave_max[order])
    for det in range(n_dets):
        # if np.all(np.isnan(flux[order][det])):
        #     continue
        ax[order].plot(wave[order][det], flux[order][det] * scale, color=color_data, alpha=0.6,
                       label='Telluric corrected' if det == 0 else None)
        ax[order].fill_between(wave[order][det], scale*(flux[order][det] - err[order][det]), 
                               scale*(flux[order][det] + err[order][det]),
                        color=color_data, alpha=0.2)
        
        
        ax[order].plot(wave[order][det], flux_uncorr[order,det] * scale, color='r', alpha=0.4,
                       label='Uncorrected' if det == 0 else None)
        
        # ax.plot(wave[order][det], transm[order][det] * np.nanmedian(flux[order,det]),
        #         color=color_transm, alpha=0.5)
ax[0].legend(ncol=2, loc=(0.25, 1.01))
ax[-1].set(xlabel='Wavelength / nm')
# ax[len(ax)//2].set(ylabel='Flux (10$^{-15}$ erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$)')
ax[len(ax)//2].set(ylabel='Normalized flux')
fig.savefig(out_path / f'{target}_telluric_correction.pdf', bbox_inches='tight')
plt.show()
