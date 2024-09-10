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
target = 'TWA27A'
# run = None
run = 'lbl10_KM_4'
w_set='NIRSpec'

cwd = os.getcwd()
if target not in cwd:
    nwd = os.path.join(cwd, target)
    print(f'Changing directory to {nwd}')
    os.chdir(nwd)


conf = Config(path=path, target=target, run=run)(config_file)        
    
ret = Retrieval(
    conf=conf, 
    evaluation=False
    )

bestfit_params, posterior = ret.PMN_analyze()
bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))

# manual fix to ignore disk here
bestfit_params_dict['log_N_mol_12CO'] = -10.0
bestfit_params = np.array(list(bestfit_params_dict.values()))

ret.evaluate_model(bestfit_params)
ret.PMN_lnL_func()

m_flux_full = np.squeeze(ret.LogLike[w_set].m_flux)
chi2_full = ret.LogLike[w_set].chi_squared_red
wave = np.squeeze(ret.d_spec[w_set].wave)

disk_param_keys = ['T_ex', 'N_mol', 'A_au']


def plot_species(ret,
                 wave, 
                 m1,
                 chi2_1,
                 m2,
                chi2_2,
                  **kwargs):

    
    fig_name = f'{conf.prefix}plots/testing_disk_emission.pdf'
    n_orders = len(wave)
    
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
            ax[0].plot(wave[order,], m1[order,], color='limegreen', lw=lw, label=f'm1 ({chi2_1:.2f})')
            ax[0].plot(wave[order,], m2[order,], color=color, lw=lw, label=f'm2 ({chi2_2:.2f})')

            # ax[0].plot(wave[order,], m_flux[order,], color=color, lw=lw, label=
            res_1 = d_flux[order] - m1[order,]
            ax[1].plot(wave[order,], res_1, color='black', lw=lw)
            
            res_2 = d_flux[order] - m2[order,]
            ax[1].plot(wave[order,], res_2, color=color, lw=lw)
            
            disk_residuals = m2[order,] - m1[order,]
            ax[1].plot(wave[order,], disk_residuals, color='deepskyblue', lw=lw)
        

            ax[1].axhline(0, color='b', lw=0.5, ls='--')
            ax[0].set_ylabel('Flux / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$')
            ax[1].set_ylabel('Residuals')
            if order==0:
                ax[0].legend()
            if order==n_orders-1:
                ax[1].set_xlabel('Wavelength / nm')
                # ax[1].legend()
            pdf.savefig(fig)
            plt.close(fig)  
    print(f'--> Saved {fig_name}')
    plt.close(fig)
    
update_params = {'log_A_au_12CO': -2.0, 
                 'log_N_mol_12CO': 15.0,
                 'log_T_ex_12CO':np.log10(1000.),
                #  'log_12CO':bestfit_params_dict['log_12CO'] + 0.2,
                #  'log_H2O':-3.25,
                 }

# print(f'\n 12CO/H2O = {10**update_params["log_12CO"]/10**bestfit_params_dict["log_H2O"]:.2f}\n')
# update_params = {'log_12CO': -2.5}
new_params = copy.deepcopy(bestfit_params_dict)
new_params.update(update_params)

ret.evaluate_model(np.array(list(new_params.values())))
ret.PMN_lnL_func()

m2 = np.squeeze(ret.LogLike[w_set].m_flux)

plot_species(ret, wave, m_flux_full, chi2_full, m2, ret.LogLike[w_set].chi_squared_red)
