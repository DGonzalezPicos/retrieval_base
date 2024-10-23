""" 
Generate a model for G235+G395 with the best-fit parameters from G235 alone 
Inspect the residuals, disk emission?

date: 2024-09-17
"""
import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
# pdf pages
from matplotlib.backends.backend_pdf import PdfPages
import copy
from scipy.ndimage import gaussian_filter1d

from retrieval_base.retrieval import Retrieval
from retrieval_base.spectrum import ModelSpectrum
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
# import config_jwst as conf

path = pathlib.Path(af.get_path())
config_file = 'config_jwst.txt'
target = 'TWA28'
w_set='NIRSpec'

cwd = os.getcwd()
if target not in cwd:
    nwd = os.path.join(cwd, target)
    print(f'Changing directory to {nwd}')
    os.chdir(nwd)


def get_bestfit_params(run):
    conf = Config(path=path, target=target, run=run)(config_file)        
        
    ret = Retrieval(
        conf=conf, 
        evaluation=False
        )

    bestfit_params, posterior = ret.PMN_analyze()
    bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
    return bestfit_params_dict
# run with both gratings
# run = 'lbl15_K2'
run = 'lbl15_G2G3_3'


conf = Config(path=path, target=target, run=run)(config_file)        
    
ret = Retrieval(
    conf=conf, 
    evaluation=False
    )

bestfit_params, posterior = ret.PMN_analyze()
bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
# bestfit_params_dict['log_SiO'] = -6.0

# # remove disk blackbody
# bestfit_params_dict['R_d'] = 0.0

print(f' --> Best-fit parameters: {bestfit_params_dict}')
bestfit_params = np.array(list(bestfit_params_dict.values()))

ret.evaluate_model(bestfit_params)
ret.PMN_lnL_func()

m_flux_full = np.squeeze(ret.LogLike[w_set].m_flux)
chi2_full = ret.LogLike[w_set].chi_squared_red
wave = np.squeeze(ret.d_spec[w_set].wave)

n_orders = m_flux_full.shape[0]
d_wave = np.squeeze(ret.d_spec[w_set].wave)
d_flux = np.squeeze(ret.d_spec[w_set].flux)

fig_name = path / f'{target}/{conf.prefix}plots/bestfit_G235H_continuum.png'
kwargs = {'lw': 2.0, 'color': 'limegreen'}

fig, ax = plt.subplots(1, 1, figsize=(12,6), tight_layout=True, facecolor='none')
lw = kwargs.get('lw', 2.0)

sigma = 20

m_bb = ModelSpectrum(wave=d_wave, flux=np.ones_like(d_wave))
bb = m_bb.blackbody_disk(T=ret.Param.params['T_d'], R=ret.Param.params['R_d'], d=ret.Param.params['d_pc'], parallax=ret.Param.params['parallax'])
for i in range(n_orders):
    # ax.plot(wave[i], d_flux[i], color='k', lw=lw)

    ax.plot(wave[i], gaussian_filter1d(d_flux[i], sigma), color='k', lw=lw)
    mc = gaussian_filter1d(m_flux_full[i], sigma)
    
    ax.plot(wave[i], mc, lw=lw, color='limegreen', alpha=0.8, label='Atm. + Disk (BB)' if i == 0 else None)
    ax.plot(wave[i], mc - bb[i], lw=lw, color='r', alpha=0.8, label='Atm.' if i == 0 else None)
    ax.plot(wave[i], bb[i], lw=lw, color='royalblue', alpha=0.8, 
            label='Disk (BB)'+f'\nT = {ret.Param.params["T_d"]:.0f} K'+f'\nR = {ret.Param.params["R_d"]:.2f} R$_J$' if i == 0 else None)
    
ax.legend(frameon=False, loc='center left')
    
ax.set_yscale('log')
# remove top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# remove top and right axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# remove xtikcs and yticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_xlabel('Wavelength / nm')

   
ax.set_ylabel('Flux / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$')
ax.set_ylim(1e-17,None)
ax.set_facecolor('none')
# ax[1].set(xlabel='Wavelength / nm', ylabel='Residuals', xlim=(wave[0][0], wave[-1][-1]))
plt.show()
fig.savefig(fig_name, dpi=300, bbox_inches='tight', facecolor='none')
[print(f' --> Saved {fig_name}')]
    
