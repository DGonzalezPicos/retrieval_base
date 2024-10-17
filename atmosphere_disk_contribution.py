""" 
Plot the contributions from the atmosphere and disk to the total flux of the system.

date: 2024-10-17
"""
import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
# pdf pages
from matplotlib.backends.backend_pdf import PdfPages
import copy

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
# import config_jwst as conf

path = af.get_path()
config_file = 'config_jwst.txt'
target = 'TWA28'
w_set='NIRSpec'
run = 'lbl15_G2G3'

cwd = os.getcwd()
if target not in cwd:
    nwd = os.path.join(cwd, target)
    print(f'Changing directory to {nwd}')
    os.chdir(nwd)

def mad(x):
    return np.nanmedian(np.abs(x - np.nanmedian(x)))

conf = Config(path=path, target=target, run=run)(config_file)
ret = Retrieval(
    conf=conf, 
    evaluation=False
    )

bestfit_params, posterior = ret.PMN_analyze()
bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))

# remove disk blackbody
# bestfit_params_dict['R_d'] = 0.0

print(f' --> Best-fit parameters: {bestfit_params_dict}')
bestfit_params = np.array(list(bestfit_params_dict.values()))

ret.evaluate_model(bestfit_params)
ret.PMN_lnL_func()

# total flux: 
m_full = np.squeeze(np.copy(ret.LogLike[w_set].m_flux))

# blackbody disk contribution (inner disk)
bb_disk = np.squeeze(ret.m_spec[w_set].blackbody_disk(T=ret.Param.params.get('T_d'),
                                 R=ret.Param.params.get('R_d'),
                                 d=ret.Param.params.get('d_pc'),
                                 wave_cm=ret.d_spec[w_set].wave*1e-7))

# generate model without disk species to isolate the slab contribution
ret.pRT_atm[w_set].disk_species = []
ret.evaluate_model(bestfit_params)
ret.PMN_lnL_func()
print(f' ret.pRT_atm[w_set].disk_species = {ret.pRT_atm[w_set].disk_species}')
m_noslab = np.squeeze(ret.LogLike[w_set].m_flux)
m_slab = m_full - m_noslab

# n_orders = m_full.shape[0]
n_orders = 1
wave = np.squeeze(ret.d_spec[w_set].wave)
flux = np.squeeze(ret.d_spec[w_set].flux)
fig, ax = plt.subplots(2, 1, figsize=(14, 5*n_orders), sharex=True, tight_layout=True, gridspec_kw={'height_ratios': [3, 2]})

colors = dict(bb='brown', full='limegreen', slab='b', atmosphere='magenta', atmosphere_bb='orange')
# for i in range(n_orders):
n_edge = 10
for i in [6]:
    # ax[i].plot(wave[i], m_full[i], label='Total flux')
    ax[0].plot(wave[i, n_edge:], flux[i, n_edge:], label='Data', color='k')
    ax[0].plot(wave[i, n_edge:], m_full[i, n_edge:], label='Atm. + BB + Slab', color=colors['full'])
    ax[0].plot(wave[i], bb_disk[i], label='BB', color=colors['bb'])
    ax[0].plot(wave[i,n_edge:], m_noslab[i,n_edge:] - bb_disk[i,n_edge:], label='Atm.', color=colors['atmosphere'])
    ax[0].plot(wave[i], m_slab[i], label='Slab', color=colors['slab'])
    ax[0].set_ylabel('Flux / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$')
    
    
    residuals = flux - m_full
    residuals_noslab = flux - m_noslab
    ax[1].plot(wave[i, n_edge:], residuals[i, n_edge:], label=f'Atm. + BB + Slab (MAD={mad(residuals[i, n_edge:]):.2e})', color=colors['full'])
    ax[1].plot(wave[i, n_edge:], residuals_noslab[i, n_edge:], label=f'Atm. + BB (MAD={mad(residuals_noslab[i, n_edge:]):.2e})', color=colors['atmosphere_bb'])
    ax[1].plot(wave[i,],  m_slab[i], label='Slab', color=colors['slab'], ls='--')
    ax[1].axhline(0, color='k', lw=0.5, ls='-')
    ax[0].legend()
    ax[1].legend(ncol=3)
    
    
        
ax[-1].set_xlabel('Wavelength / nm')
plt.show()

