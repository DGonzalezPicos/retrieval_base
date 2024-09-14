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
run = 'lbl15_KM_6'
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

ret.evaluate_model(bestfit_params)
ret.PMN_lnL_func()

m_flux_full = np.squeeze(ret.LogLike[w_set].m_flux)
chi2_full = ret.LogLike[w_set].chi_squared_red
wave = np.squeeze(ret.d_spec[w_set].wave)

# save best-fit model as [wave, flux] in .npy file
np.save(f'{conf.prefix}data/bestfit_model.npy', [wave, m_flux_full])
print(f'--> Saved {conf.prefix}data/bestfit_model.npy')

# generate model without disk
generate_no_disk = True
if generate_no_disk:
    bestfit_params_dict['R_d'] = 0.0
    ret.evaluate_model(np.array(list(bestfit_params_dict.values())))
    ret.PMN_lnL_func()
    m_flux = np.squeeze(ret.LogLike[w_set].m_flux)
    # save best-fit model as [wave, flux] in .npy file
    np.save(f'{conf.prefix}data/bestfit_model_no_disk.npy', [wave, m_flux])

disk_param_keys = ['T_ex', 'N_mol', 'A_au']


def plot_species(ret,
                 wave, 
                 m_flux_full, 
                 chi2_full,
                 disk_species, 
                 bestfit_params_dict, 
                 emission=True,
                 overplot_extinction=False,
                  **kwargs):

    label = 'disk_' + 'blackbody' if (emission==False) else f'emission{disk_species}'
    fig_name = f'{conf.prefix}plots/bestfit_spec_{label}.pdf'
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
            
            bestfit_params_dict_copy = copy.deepcopy(bestfit_params_dict)
            if emission:
                bestfit_params_dict_copy[f'log_A_au_{disk_species}'] = -10.0
            else:
                bestfit_params_dict_copy['R_d'] = 0.0 

            ret.evaluate_model(np.array(list(bestfit_params_dict_copy.values())))
            ret.PMN_lnL_func()
            chi2 = ret.LogLike[w_set].chi_squared_red

            m_flux = np.squeeze(ret.LogLike[w_set].m_flux)
            d_flux = np.squeeze(ret.d_spec[w_set].flux)
            ax[0].plot(wave[order,], d_flux[order], color='black', lw=lw, label='Data')
            ax[0].plot(wave[order,], m_flux_full[order,], color='limegreen', lw=lw, label=f'Full model (chi2={chi2_full:.2f})')
            label = disk_species if emission else 'blackbody'
            ax[0].plot(wave[order,], m_flux[order,], color=color, lw=lw, label=f'w/o disk {label} (chi2={chi2:.2f})')

            # ax[0].plot(wave[order,], m_flux[order,], color=color, lw=lw, label=
            res_data = d_flux[order] - m_flux[order,]
            ax[1].plot(wave[order,], res_data, color='black', lw=lw)
            
            res = m_flux_full[order,] - m_flux[order,]
            ax[1].plot(wave[order,], res, color=color, lw=lw)
        

            ax[1].axhline(0, color='b', lw=0.5, ls='--')
            ax[0].set_ylabel('Flux / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$')
            ax[1].set_ylabel('Residuals')
            
            if order==n_orders-1:
                ax[1].set_xlabel('Wavelength / nm')
                
            if overplot_extinction:
                Av = 10.
                ax[0].plot(wave[order,], af.apply_extinction(m_flux_full[order,], wave[order] * 1e-3, Av), # wave must be in [um]
                           color='blue', lw=lw,  ls='--', label=f'Av={Av}')
            
            if order==0:
                ax[0].legend()
            pdf.savefig(fig)
            plt.close(fig)
        plt.close(fig)  
    print(f'--> Saved {fig_name}')
    
# plot_species(ret, wave, m_flux_full, chi2_full, '12CO', bestfit_params_dict, emission=True, apply_extinction=True)
plot_species(ret, wave, m_flux_full, chi2_full, '12CO', bestfit_params_dict, emission=False, overplot_extinction=True)

