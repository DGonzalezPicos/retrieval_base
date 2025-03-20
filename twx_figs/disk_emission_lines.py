import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
# pdf pages
from matplotlib.backends.backend_pdf import PdfPages
import copy
import argparse

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config

target = 'TWA27A'
run = 'no_psf_corr_lbl10_G2G3_newGP_2'

path = af.get_path()
config_file = 'config_jwst.txt'
w_set='NIRSpec'
run_bestfit = None

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

params_dict = {}
for k in ret.Param.param_keys:
    # print(f' Setting {k} to mean of prior')    params_dict[k] = np.mean(ret.Param.param_priors[k])
    params_dict[k] = ret.Param.param_priors[k][1]
        
        
if run_bestfit == None:

    try:
        bestfit_params, posterior = ret.PMN_analyze()
        bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
    except:
        print(f'Error: No bestfit found for {target} {run}')
        bestfit_params_dict = {}

    
    
else:
    conf_bestfit = Config(path=path, target=target, run=run_bestfit)(config_file)        
    
    ret_bestfit = Retrieval(
        conf=conf_bestfit, 
        evaluation=False
        )
    
    bestfit_params, posterior = ret_bestfit.PMN_analyze()
    bestfit_params_dict = dict(zip(ret_bestfit.Param.param_keys, bestfit_params))
    

            

for k, v in bestfit_params_dict.items():
    if k in params_dict:
        params_dict[k] = v

params = np.array(list(params_dict.values()))

assert len(params) == len(ret.Param.param_keys), f'Number of parameters does not match'

ret.evaluate_model(params)
ret.PMN_lnL_func()

f = ret.d_spec[w_set].flux_unit_factor

m_flux_full = np.squeeze(ret.LogLike[w_set].m_flux) / f
chi2_full = ret.LogLike[w_set].chi_squared_red
wave = np.squeeze(ret.d_spec[w_set].wave)

species_dict = {k[4:]:v[-1] for k,v in ret.conf.opacity_params.items()}

        

def plot_species(ret,
                 wave, 
                 m_flux_full, 
                params_dict,
                orders,
                **kwargs):


    n_orders = len(wave)
    

        
    params_dict_copy = copy.deepcopy(params_dict)
    T_ex = params_dict_copy['T_ex_12CO']
    N_mol = params_dict_copy['log_N_mol_12CO']
    log_R_out = params_dict_copy['log_R_out']
    log_R_cav = params_dict_copy['log_R_cav']
    fig_name = f'{conf.prefix}plots/bestfit_spec_Tex{T_ex:.0f}_logNmol{N_mol:.0f}_logRcav{log_R_cav:.1f}_logRout{log_R_out:.1f}.pdf'

    
    with PdfPages(fig_name) as pdf:
        
        lw = kwargs.get('lw', 0.9)
        color = kwargs.get('color', 'red')
        
       
        ret.evaluate_model(np.array(list(params_dict_copy.values())))
        ret.PMN_lnL_func()
        chi2 = ret.LogLike[w_set].chi_squared_red
        # chi2 = 1.0

        m_flux = np.squeeze(ret.LogLike[w_set].m_flux) / f
        d_flux = np.squeeze(ret.d_spec[w_set].flux) / f
        Cov = ret.Cov[w_set]
        
        if orders is None:
            orders = range(n_orders)

        for order in orders:
            fig, ax = plt.subplots(2,1, figsize=(12,5), gridspec_kw={'height_ratios': [2, 1],
                                                                        'hspace': 0.10,
                                                                        'left': 0.07,
                                                                        'right': 0.98,
                                                                        'top': 0.95,
                                                                        'bottom': 0.1},
                                   sharex=True)
            
            
            mask_i = ret.d_spec[w_set].mask_isfinite[order,0]
            err_ij = Cov[order,0].get_err(mask=mask_i) / f

            
            ax[0].plot(wave[order,], d_flux[order], color='black', lw=lw, label='Data')
            
            chi2_full_order = np.nansum((d_flux[order] - m_flux_full[order,])**2 / err_ij**2) / mask_i.sum()
            chi2_order = np.nansum((d_flux[order] - m_flux[order,])**2 / err_ij**2) / mask_i.sum()
            
            ax[0].plot(wave[order,], m_flux_full[order,], color='limegreen', lw=lw, label=f'Full model (chi2={chi2_full_order:.2f})')
            ax[0].plot(wave[order,], m_flux[order,], color=color, lw=lw, label=f'new (chi2={chi2_order:.2f})', alpha=0.8)

            # ax[0].plot(wave[order,], m_flux[order,], color=color, lw=lw, label=
            # res_data = d_flux[order] - m_flux[order,]
            # ax[1].plot(wave[order,], res_data, color='black', lw=lw)
            
            # res = m_flux_full[order,] - m_flux[order,]
            # ax[1].plot(wave[order,], res, color=color, lw=lw, alpha=0.9)
            
            # residuals with full model
            res_full = d_flux[order] - m_flux_full[order,]
            mad_full = np.nanmedian(np.abs(res_full))
            ax[1].plot(wave[order,], res_full, color='black', lw=lw, label=f'MAD={mad_full:.2e}')                
            
            res = d_flux[order] - m_flux[order,]
            mad = np.nanmedian(np.abs(res))
            ax[1].plot(wave[order,], res, color=color, lw=lw, alpha=0.9, label=f'MAD={mad:.2e}')
            for sign in [-1, 1]:
                ax[1].axhline(mad_full*sign, color='black', lw=0.5, ls='--',zorder=-10)
                ax[1].axhline(mad*sign, color=color, lw=0.5, ls='--',zorder=-10)

            ax[1].axhline(0, color='r', lw=0.5)
            ax[0].set_ylabel('Flux / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$')
            ax[1].set_ylabel('Residuals')
            if order==0:
                ax[0].set_title(f'Tex={T_ex:.2f} K, Nmol={N_mol:.2e}')
            
            ax[0].legend()
            ax[1].legend(loc='upper right', frameon=False)
            if order==n_orders-1:
                ax[1].set_xlabel('Wavelength / nm')
                # ax[1].legend()
                
            

            pdf.savefig(fig)
            plt.close(fig)
    plt.close()
    print(f'--> Saved {fig_name}')


params_dict_copy = copy.deepcopy(params_dict)
params_dict_copy['T_ex_12CO'] = 900.0
params_dict_copy['log_N_mol_12CO'] = 14.0
params_dict_copy['log_N_mol_13CO'] = params_dict_copy['log_N_mol_12CO'] - params_dict_copy['log_12CO/13CO']
params_dict_copy['log_N_mol_H2O'] = params_dict_copy['log_N_mol_12CO'] + 0.0
params_dict_copy['log_R_cav'] = 1.7
params_dict_copy['log_R_out'] = 2.3 # best fit 1.09

orders = [8,9,10,11]

plot_species(ret, wave, m_flux_full, params_dict=params_dict_copy,
             orders=orders,
                color='dodgerblue')

# AU to cm
au_cm = 1.496e13
# rjup to cm
rjup_cm = 7.1492e9

radius_au = 10.0 
radius_rjup = radius_au * (au_cm/rjup_cm)

print(f'Radius = {radius_au:.2f} AU = {radius_rjup:.2f} RJup')
# Dust Sublimation Radius (Thermal Evaporation)
# if dust dominates, R_in ~ 0.04 AU
