import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
# pdf pages, use path effects for white edge
import matplotlib.patheffects as path_effects
from matplotlib.backends.backend_pdf import PdfPages
import copy

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
# import config_jwst as conf

path = pathlib.Path(af.get_path(return_pathlib=True))
config_file = 'config_jwst.txt'
target = 'TWA28'
w_set='NIRSpec'

cwd = os.getcwd()
if target not in cwd:
    nwd = os.path.join(cwd, target)
    print(f'Changing directory to {nwd}')
    os.chdir(nwd)
    
runs = ['lbl12_G2G3_6', 'lbl15_G2G3_7']
colors = ['darkorange', 'navy']
species = '12CO_disk'
rv_max = 2000.0
rv_step = 5.0
rv_noise = 400.0
    
fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(6, 6), tight_layout=True, sharex=True,
                           gridspec_kw={'height_ratios': [3, 1]})

pe = path_effects.withStroke(linewidth=2, foreground='w', alpha=0.9)
for i, run in enumerate(runs):
    rv, CCF_SNR, ACF_SNR = np.loadtxt(path / target / f'retrieval_outputs/{run}/test_plots/CCF/RV_CCF_ACF_{species}.txt', unpack=True)
    
    ax.plot(rv, CCF_SNR, color=colors[i], path_effects=[pe], alpha=0.8)
    ax.plot(rv, ACF_SNR, color=colors[i], ls='--')
    
    CCF_RES = CCF_SNR - ACF_SNR
    ax_res.plot(rv, CCF_RES, color=colors[i], path_effects=[pe], alpha=0.8)
    
    rv_peak = rv[np.argmax(CCF_SNR)]
    snr_peak = np.max(CCF_SNR)
    print(f' Peak SNR at {rv_peak} km/s: {snr_peak}')
    ax.axvline(rv_peak, lw=1.0, label=f'SNR = {snr_peak:.1f}', color=colors[i], alpha=0.9) # color='darkgold'
    
    
[axi.axvspan(-rv_noise, rv_noise, color='gray', alpha=0.1) for axi in [ax, ax_res]]
ax.legend(frameon=False, fontsize=14, loc='upper right', bbox_to_anchor=(1, 1))
ax.set_ylabel('SNR')
ax_res.set(xlabel=r'RV / km s$^{-1}$', ylabel='CCF - ACF', xlim=(rv.min(), rv.max()))
ax_res.axhline(0, color='k', lw=0.5)

ylim_res = ax_res.get_ylim()
# make symmetric ylims for ax_res
max_res = np.abs(ylim_res).max()
ax_res.set_ylim(-max_res, max_res)
   
    
runs_label = '_'.join(runs)
ccf_path = path / target / f'retrieval_outputs/{runs[-1]}/test_plots/CCF/'
fig_name = ccf_path / f'CCF_SNR_{species}_{runs_label}.pdf'
fig.savefig(fig_name)
print(f' Saved {fig_name}')
plt.close(fig)  