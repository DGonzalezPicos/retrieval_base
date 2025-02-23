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
# run = 'lbl12_G1G2G3_fastchem_1'
# run = 'lbl12_G1_fastchem_1'
# run = 'lbl11_G2G3_fastchem_GP_1'
run = 'test_g395h'
w_set='NIRSpec'

run_bestfit = 'lbl11_G2G3_fastchem_GP_1'

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
    print(f' Setting {k} to mean of prior')
    params_dict[k] = np.mean(ret.Param.param_priors[k])
    # params_dict[k] = ret.Param.param_priors[k][1]
        
        
if run_bestfit == None:

    try:
        bestfit_params, posterior = ret.PMN_analyze()
        bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
    except:
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
                 chi2_full,
                 line_species, 
                 params_dict, 
                #  new_value,
                 overplot_extinction=False,
                  **kwargs):


    n_orders = len(wave)
    
    
    n = 4
    alpha_range = np.linspace(-3.0, 2.0, n)
    log_ratio_range = np.linspace(1.0, 4.0, n)
    log_range = np.linspace(-12.0, -4.5, n)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    
    ranges = {'alpha': alpha_range,
             'log_iso_ratio': log_ratio_range,
             'log': log_range}

        
    m_flux_list = []
    for i in range(n):
        params_dict_copy = copy.deepcopy(params_dict)    
        # print(f'line_species: {line_species}')
        # print(params_dict_copy.keys())
        # if conf.chem_mode == 'fastchem':
        if f'alpha_{line_species}' in list(params_dict_copy.keys()):
            params_dict_copy[f'alpha_{line_species}'] = ranges['alpha'][i]
            param_kind = 'alpha'
            bestfit_value = params_dict[f'alpha_{line_species}']
        elif line_species in conf.isotopologues_dict.keys():
            print(f'Found isotopologue {line_species} with ratio {conf.isotopologues_dict[line_species][0]}')
            log_ratio = conf.isotopologues_dict[line_species][0]
            params_dict_copy[log_ratio] = ranges['log_iso_ratio'][i]
            param_kind = 'log_iso_ratio'
            bestfit_value = params_dict[log_ratio]
        
        else:
            params_dict_copy[f'log_{line_species}'] = ranges['log'][i]
            param_kind = 'log'
            bestfit_value = params_dict[f'log_{line_species}']
        ret.evaluate_model(np.array(list(params_dict_copy.values())))
        ret.PMN_lnL_func()
        m_flux_list.append(np.squeeze(ret.LogLike[w_set].m_flux) / f)

    # if overplot_extinction:
    title = f'{line_species} ({conf.line_species_dict[line_species]})'
    fig_name = f'{conf.prefix}plots/test_{line_species}.pdf'

    
    with PdfPages(fig_name) as pdf:
        
        lw = kwargs.get('lw', 0.7)
        # color = kwargs.get('color', 'red')
        
       
        ret.evaluate_model(np.array(list(params_dict_copy.values())))
        ret.PMN_lnL_func()
        chi2 = ret.LogLike[w_set].chi_squared_red
        # chi2 = 1.0

        m_flux = np.squeeze(ret.LogLike[w_set].m_flux) / f
        d_flux = np.squeeze(ret.d_spec[w_set].flux) / f
        Cov = ret.Cov[w_set]
        

        for order in range(n_orders):
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
            res_full = d_flux[order] - m_flux_full[order,]
            ax[1].plot(wave[order,], res_full, color='limegreen', lw=lw)
        
            chi2_full_order = np.nansum((d_flux[order] - m_flux_full[order,])**2 / err_ij**2) / mask_i.sum()
            # label = line_species
            label = f'{param_kind}='
            label += f'{bestfit_value:.1e}'
            label += f' (chi2={chi2_full_order:.1f})'
            ax[0].plot(wave[order,], m_flux_full[order,], color='limegreen', lw=lw, label=label)

            for i in range(n):
                m_flux = m_flux_list[i]
                chi2_order = np.nansum((d_flux[order] - m_flux[order,])**2 / err_ij**2) / mask_i.sum()
                # label = line_species
                label = f'{param_kind}='
                label += f'{ranges[param_kind][i]:.1e}'
                label += f' (chi2={chi2_order:.1f})'
                ax[0].plot(wave[order,], m_flux[order,], color=colors[i], lw=lw, label=label)            
                
                res_new = d_flux[order,] - m_flux[order,]
                ax[1].plot(wave[order,], res_new, color=colors[i], lw=lw)
        

            ax[1].axhline(0, color='r', lw=0.5)
            ax[0].set_ylabel('Flux / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$')
            ax[1].set_ylabel('Residuals')
            if order==0:
                ax[0].set_title(title)
            
            ax[0].legend(ncol=2)
            if order==n_orders-1:
                ax[1].set_xlabel('Wavelength / nm')
                # ax[1].legend()
                
            

            pdf.savefig(fig)
        plt.close(fig)
    print(f'--> Saved {fig_name}')


# new_alphas = [-2.0, -1.0, 0.0, 1.0, 2.0]
for k, v in species_dict.items():
    if k != 'C2H2':
        continue
    plot_species(ret, wave, m_flux_full, chi2_full, k, params_dict,
                # new_value=new_value,
                #  color=ret.Chem.read_species_info(species, 'color')
                color='darkorange')
    
# plot_species(ret, wave, m_flux_full, chi2_full, 'H2O', bestfit_params_dict, color='blue')

