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
# run = None
# run = 'lbl15_G2_4'
run = 'lbl12_G1G2G3_fastchem_1'
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

species_dict = {k[4:]:v[-1] for k,v in ret.conf.opacity_params.items()}

def plot_species(ret,
                 wave, 
                 m_flux_full, 
                 chi2_full,
                 line_species, 
                 bestfit_params_dict, 
                 overplot_extinction=False,
                  **kwargs):

    fig_name = f'{conf.prefix}plots/bestfit_spec_{line_species}.pdf'
    n_orders = len(wave)
    
    # if overplot_extinction:

    
    with PdfPages(fig_name) as pdf:
        
        lw = kwargs.get('lw', 0.7)
        color = kwargs.get('color', 'red')
        
        bestfit_params_dict_copy = copy.deepcopy(bestfit_params_dict)
            
        if conf.chem_mode == 'fastchem':
            bestfit_params_dict_copy[f'alpha_{line_species}'] = -14.0
            if line_species in conf.isotopologues_dict.keys():
                print(f'Found isotopologue {line_species} with ratio {conf.isotopologues_dict[line_species][0]}')
                log_ratio = conf.isotopologues_dict[line_species][0]
                bestfit_params_dict_copy[log_ratio] = 4.0
        else:
            bestfit_params_dict_copy[f'log_{line_species}'] = -14.0

        ret.evaluate_model(np.array(list(bestfit_params_dict_copy.values())))
        ret.PMN_lnL_func()
        chi2 = ret.LogLike[w_set].chi_squared_red

        m_flux = np.squeeze(ret.LogLike[w_set].m_flux)
        d_flux = np.squeeze(ret.d_spec[w_set].flux)

        for order in range(n_orders):
            fig, ax = plt.subplots(2,1, figsize=(12,5), gridspec_kw={'height_ratios': [2, 1],
                                                                        'hspace': 0.10,
                                                                        'left': 0.07,
                                                                        'right': 0.98,
                                                                        'top': 0.95,
                                                                        'bottom': 0.1},
                                   sharex=True)
            
            
            ax[0].plot(wave[order,], d_flux[order], color='black', lw=lw, label='Data')
            ax[0].plot(wave[order,], m_flux_full[order,], color='limegreen', lw=lw, label=f'Full model (chi2={chi2_full:.2f})')
            ax[0].plot(wave[order,], m_flux[order,], color=color, lw=lw, label=f'w/o {line_species} (chi2={chi2:.2f})')

            # ax[0].plot(wave[order,], m_flux[order,], color=color, lw=lw, label=
            res_data = d_flux[order] - m_flux[order,]
            ax[1].plot(wave[order,], res_data, color='black', lw=lw)
            
            res = m_flux_full[order,] - m_flux[order,]
            ax[1].plot(wave[order,], res, color=color, lw=lw)
        

            ax[1].axhline(0, color='r', lw=0.5)
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


for k, v in species_dict.items():
    plot_species(ret, wave, m_flux_full, chi2_full, k, bestfit_params_dict,
                #  color=ret.Chem.read_species_info(species, 'color')
                color='darkorange')
    
# plot_species(ret, wave, m_flux_full, chi2_full, 'H2O', bestfit_params_dict, color='blue')

