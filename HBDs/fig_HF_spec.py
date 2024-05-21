from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters
from retrieval_base.auxiliary_functions import pickle_load
from retrieval_base.config import Config

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
# set fontsize to 16
# plt.rcParams.update({'font.size': 16})
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')

import pathlib
import pickle
import corner
import pandas as pd
import json

# import config_freechem_15 as conf

import os
# change cwd to the directory of this script
path = pathlib.Path(__file__).parent.absolute()
# os.chdir(path)


path = pathlib.Path('/home/dario/phd/retrieval_base')
# out_path = path / 'HBDs'
out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')

targets = dict(J1200='final_full',
               TWA28='final_full',
               J0856='final_full',
               )

colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')


target = 'J0856'
config_file = 'config_freechem.txt'
conf = Config(path=path, target=target, run=targets[target])
conf(config_file)
retrieval_id = targets[target]


data_path = pathlib.Path('/home/dario/phd/retrieval_base') / f'{target}'
print(data_path)


# bestfit_params = 
retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'
assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'

conf.prefix = f'{target}/' + conf.prefix[2:]
ret = Retrieval(
                conf=conf, 
                evaluation=True,
                plot_ccf=False,
                )
wave = ret.d_spec['K2166'].wave
bestfit_params, _ = ret.PMN_analyze()

def get_m_spec(bestfit_params):
    
    # Evaluate the model with best-fitting parameters
    for i, key_i in enumerate(ret.Param.param_keys):
        # Update the Parameters instance
        ret.Param.params[key_i] = bestfit_params[i]
        if key_i.startswith('log_'):
            ret.Param.params = ret.Param.log_to_linear(ret.Param.params, key_i)

        
    # Update the parameters
    ret.Param.read_PT_params(cube=None)
    ret.Param.read_uncertainty_params()
    ret.Param.read_chemistry_params()
    # ret.Param.read_cloud_params()
    ret.PMN_lnL_func()
    # TODO: finish this...
    return ret.m_spec['K2166']

m_spec = get_m_spec(bestfit_params)
phi = ret.LogLike['K2166'].phi.copy()
s = ret.LogLike['K2166'].s.copy()
m_spec.flux *= phi
bestfit_params_copy = bestfit_params.copy()


Cov = ret.Cov['K2166'].copy()

HF_idx = list(ret.Param.param_keys).index('log_HF')
bestfit_params_copy[HF_idx] = -20.
m_spec_noHF = get_m_spec(bestfit_params_copy)
m_spec_noHF.flux *= phi

orders = np.arange(4,6)

# gridspec with 2 columns and 6 rows
fig = plt.figure(figsize=(14, 4))
gs = fig.add_gridspec(8, 6, hspace=1.6, wspace=0.1)
ax_0 = fig.add_subplot(gs[:3, :])
ax_1 = fig.add_subplot(gs[3:, :2])
ax_2 = fig.add_subplot(gs[3:, 2:4])
ax_3 = fig.add_subplot(gs[3:, 4:6])
# ax_4 = fig.add_subplot(gs[3:, 6:8])
ax_regions = [ax_1, ax_2, ax_3
            #   , ax_4,
              ]

# first row is the full spectrum
m_HF = (m_spec.flux - m_spec_noHF.flux)
m_HF[~ret.d_spec['K2166'].mask_isfinite] = np.nan
m_HF -= np.nanmedian(m_HF, axis=-1)[:,:,None]
ax_0.plot(wave.flatten(), m_HF.flatten(), label='Model', color='magenta')

cenwaves = [2336.59, 2349.57, 2363.75
            # , 2432.9,
            ]
regions = [(cenwave-1, cenwave+1) for cenwave in cenwaves]
labels = [r'$\lambda_'+str(i)+'$' for i in range(1, len(regions)+1)]
for i, region in enumerate(regions):
    
    mask_region = (wave > region[0]) & (wave < region[1])
    cenwave = np.mean(region)
    # xticks = np.round([region[0]+0.5, cenwave, region[1]-0.5],0)
    xticks = [cenwave]
    xticks_labels = [r'$\lambda_'+str(i+1)+'$ = ' + f'{cenwave:.1f}']
    # set xticks and labels
    ax_regions[i].set_xticks(xticks)
    ax_regions[i].set_xticklabels(xticks_labels)
    ax_0.text(cenwave, 0.001, labels[i], fontsize=12, ha='center', va='bottom')

    order_i, det_i, _ = np.unravel_index(np.argmin(np.abs(ret.d_spec['K2166'].wave - cenwave)), ret.d_spec['K2166'].wave.shape)
    print(order_i)
    
    ax_regions[i].plot(wave[mask_region], m_spec.flux[mask_region], label='Model', color='magenta', lw=1.4)
    ax_regions[i].plot(wave[mask_region], m_spec_noHF.flux[mask_region], label='Model (no HF)', color=colors[target], lw=1.4)
    
    d = ret.d_spec['K2166'].flux[mask_region]
    mask_i = (wave[order_i,det_i] > region[0]) & (wave[order_i,det_i] < region[1])
    err = np.nan * np.ones_like(wave[order_i,det_i])
    
    err[ret.d_spec['K2166'].mask_isfinite[order_i,det_i]] = np.sqrt(np.diag(Cov[order_i,det_i].get_dense_cov())) * s[order_i,det_i]
    ax_regions[i].plot(wave[mask_region], d, label='Data', color='black', alpha=0.8)
    ax_regions[i].fill_between(wave[mask_region], d-err[mask_i], d+err[mask_i], color='black', alpha=0.2)
    # remove yticks except for the first plot
    ax_regions[i].set(xlim=region)
    if i != 0:
        ax_regions[i].set_yticks([])

ax_0.set(ylim=(-0.19, 0.06), xlim=(wave.min(), wave.max()), ylabel='Residuals')
ax_1.set(ylabel='Normalized Flux')
# add common xlabel and ylabel
fig.text(0.5, -0.01, 'Wavelength / nm', ha='center')
# fig.text(0.001, 0.5, 'Normalized Flux', va='center', rotation='vertical')
fig.savefig(out_path / f'{target}_HF_spec.pdf', bbox_inches='tight')

print(out_path / f'{target}_HF_spec.pdf')
# plt.show()


# fig, ax = plt.subplots(len(orders)*3, 1, figsize=(10, len(orders)*3),
#                        gridspec_kw={'hspace': 0.1, 'height_ratios': [3, 1, 0.5]*len(orders)})

# chi2_r = 0
# chi2_r_noHF = 0
# for i, order in enumerate(orders):
    
#     ax_spec = ax[i*3]
#     ax_res = ax[i*3+1]
#     # remove empty space
#     ax[i*3+2].remove()
#     for j, det in enumerate(range(3)):
#         x = wave[order, det]
#         d = ret.d_spec['K2166'].flux[order, det]
#         mask = ret.d_spec['K2166'].mask_isfinite[order, det]

#         err = np.nan * np.ones_like(d)
#         err[mask] = np.sqrt(np.diag(Cov[order, det].get_dense_cov()))
#         m = m_spec.flux[order, det]*phi[order,det]
#         m[~mask] = np.nan
        
#         m_noHF = m_spec_noHF.flux[order, det]*phi[order,det]
#         m_noHF[~mask] = np.nan
        
        
#         ax_spec.plot(x, d, label='Data', color='black')
#         ax_spec.plot(x, m, label='Model', color='red', alpha=0.9)
#         ax_spec.plot(x, m_noHF, label='Model (no HF)', color='blue', alpha=0.9)
        
#         # residuals
#         r = d - m
#         chi2_r += np.nansum((r/err)**2)
#         r_noHF = d - m_noHF
#         chi2_r_noHF += np.nansum((r_noHF/err)**2)
        
#         ax_res.plot(x, r, color='red',alpha=0.9)
#         ax_res.plot(x, r_noHF, color='blue',alpha=0.9)
#         ax_res.axhline(0, color='black', linestyle='-')
        
# print(f' HF: {chi2_r:.2f}, no HF: {chi2_r_noHF:.2f}')
# plt.show()