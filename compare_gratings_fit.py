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

path = af.get_path()
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

bestfit_params_dict_G235 = get_bestfit_params('lbl15_K2')
# run with both gratings
run = 'lbl15_G2G3'


conf = Config(path=path, target=target, run=run)(config_file)        
    
ret = Retrieval(
    conf=conf, 
    evaluation=False
    )

bestfit_params, posterior = ret.PMN_analyze()
bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
bestfit_params_dict.update(bestfit_params_dict_G235)
bestfit_params_dict['log_SiO'] = -6.0

# remove disk blackbody
bestfit_params_dict['R_d'] = 0.0

print(f' --> Best-fit parameters: {bestfit_params_dict}')
bestfit_params = np.array(list(bestfit_params_dict.values()))

ret.evaluate_model(bestfit_params)
ret.PMN_lnL_func()

m_flux_full = np.squeeze(ret.LogLike[w_set].m_flux)
chi2_full = ret.LogLike[w_set].chi_squared_red
wave = np.squeeze(ret.d_spec[w_set].wave)

n_orders = m_flux_full.shape[0]

fig_name = f'{target}/retrieval_outputs/{conf.prefix}plots/compare_gratings_fit.pdf'
kwargs = {'lw': 0.7, 'color': 'orange'}

with PdfPages(fig_name) as pdf:
        
    lw = kwargs.get('lw', 0.7)
    color = kwargs.get('color', 'red')

    for order in range(n_orders):
        fig, ax = plt.subplots(2,1, figsize=(12,5), gridspec_kw={'height_ratios': [2, 1],
                                                                    'hspace': 0.10,
                                                                    'left': 0.07,
                                                                    'right': 0.98,
                                                                    'top': 0.95,
                                                                    'bottom': 0.1},
                                sharex=True)
        

        d_flux = np.squeeze(ret.d_spec[w_set].flux)
        ax[0].plot(wave[order,], d_flux[order], color='black', lw=lw, label='Data')
        ax[0].plot(wave[order,], m_flux_full[order,], color='limegreen', lw=lw, label=f'Full model (chi2={chi2_full:.2f})')
    
        res_data = d_flux[order] - m_flux_full[order,]
        ax[-1].plot(wave[order,], res_data, color='black', lw=lw)
        
        # res = m_flux_full[order,] - m_flux[order,]
        # ax[-1].plot(wave[order,], res, color=color, lw=lw)
    

        # ax[-1].axhline(0, color='b', lw=0.5, ls='--')
        ax[0].set_ylabel('Flux / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$')
        ax[-1].set_ylabel('Residuals')
        
        if order==n_orders-1:
            ax[-1].set_xlabel('Wavelength / nm')
        
        if order==0:
            ax[0].legend()
        pdf.savefig(fig)
        plt.close(fig)
    plt.close(fig)  
print(f'--> Saved {fig_name}')