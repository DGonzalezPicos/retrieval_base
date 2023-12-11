'''Telluric correction with Molecfit'''

import numpy as np
import matplotlib.pyplot as plt
import pathlib

plt.style.use('HBDs/my_science.mplstyle')
plt.rcParams['text.usetex'] = False
# must use environment: Python 3.11.4 ('pycrires': conda)

target = 'J0856'

# current working directory
path = pathlib.Path().absolute() / target
print(path)

wave = np.load(path / 'retrieval_outputs/freechem_1/test_data/d_spec_wave_K2166.npy')
flux = np.load(path / 'retrieval_outputs/freechem_1/test_data/d_spec_flux_K2166.npy')
err  = np.load(path / 'retrieval_outputs/freechem_1/test_data/d_spec_err_K2166.npy')
flux_uncorr = np.load(path / 'retrieval_outputs/freechem_1/test_data/d_spec_flux_uncorr_K2166.npy')
transm = np.load(path / 'retrieval_outputs/freechem_1/test_data/d_spec_transm_K2166.npy')
n_orders, n_dets, _ = flux.shape

fig, ax = plt.subplots(figsize=(14,3))
# create gridspec for the figure showing two subplots below one wide subplot
# import matplotlib.gridspec as gridspec
# fig = plt.figure(figsize=(14, 6))
# gs = gridspec.GridSpec(2, 2, width_ratios=[1,1])
# ax = plt.subplot(gs[0, :])
# ax2 = plt.subplot(gs[1, 0])
# ax3 = plt.subplot(gs[1, 1])

color_data = 'k'
color_transm = 'brown'


for order in range(n_orders):
    for det in range(n_dets):
        ax.plot(wave[order][det], flux[order][det], color=color_data)
        ax.fill_between(wave[order][det], flux[order][det] - err[order][det], flux[order][det] + err[order][det],
                        color=color_data, alpha=0.2)
        # ax.plot(wave[order][det], flux_uncorr[order][det], color='gray', alpha=0.5)
        ax.plot(wave[order][det], transm[order][det] * np.nanmedian(flux[order,det]),
                color=color_transm, alpha=0.5)
# ax.legend()
plt.show()