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

from retrieval_base.retrieval import Retrieval
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
d_flux = np.squeeze(ret.d_spec[w_set].flux)

fig_name = path / f'{target}/{conf.prefix}plots/bestfit_all.png'
kwargs = {'lw': 0.7, 'color': 'limegreen'}

fig, ax = plt.subplots(2, 1, figsize=(12,6), tight_layout=True, facecolor='none', sharex=True, gridspec_kw={'height_ratios': [3, 1]})
lw = kwargs.get('lw', 0.7)
for i in range(n_orders):
    
    ax[0].plot(wave[i], d_flux[i], color='k', lw=lw)
    ax[0].plot(wave[i], m_flux_full[i], lw=lw, color='limegreen')
    res_i = m_flux_full[i] - d_flux[i]
    # relative residual amplitude
    rra = np.nanmean(np.abs(res_i) / d_flux[i])
    
    ax[1].plot(wave[i], res_i, lw=lw, color='limegreen')
    text = 'RRA = ' if i == 0 else ''
    text += f'{rra*100:.2f}%'
    ax[1].text(0.02+i*0.12, 0.8, text, transform=ax[1].transAxes, color='k')

    ax[0].set_facecolor('none')
    ax[1].set_facecolor('none')
    
    ax[1].axhline(0, color='k', ls='-', lw=0.5)
    
ax[0].set_ylabel('Flux / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$')
ax[1].set(xlabel='Wavelength / nm', ylabel='Residuals', xlim=(np.nanmin(wave), np.nanmax(wave)))
plt.show()
fig.savefig(fig_name, dpi=300, bbox_inches='tight', facecolor='none')
[print(f' --> Saved {fig_name}')]
    
