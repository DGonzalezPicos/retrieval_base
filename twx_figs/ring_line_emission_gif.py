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

parser = argparse.ArgumentParser()
parser.add_argument('--target', '-t', type=str, default='TWA28')
parser.add_argument('--run', '-r', type=str, default='freeslab_lbl10_G1G2G3_0')
args = parser.parse_args()

path = af.get_path()
config_file = 'config_jwst.txt'
target = args.target
run = args.run
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

def update_params_dict(params_dict, line_species, high_low='low'):
    
    params_dict_copy = copy.deepcopy(params_dict)
    if line_species in conf.isotopologues_dict.keys() and conf.chem_mode == 'fastchem':
        print(f'Found isotopologue {line_species} with ratio {conf.isotopologues_dict[line_species][0]}')
        log_ratio = conf.isotopologues_dict[line_species][0]
        new_value = 1.0 if high_low == 'high' else 4.0
        old_value = params_dict[log_ratio]
        params_dict_copy[log_ratio] = new_value
        format = log_ratio
        
    else:
        new_value = 3.0 if high_low == 'high' else -4.0
        old_value = params_dict.get(f'alpha_{line_species}')
        format = 'alpha'
        
        if old_value is None:
            new_value = -3.0 if high_low == 'high' else -14.0
            old_value = params_dict[f'log_{line_species}']
            params_dict_copy[f'log_{line_species}'] = new_value
            format = 'log'
            
        print(f'Setting {format}_{line_species} to {new_value:.2f} ({old_value:.2f})')
        params_dict_copy[f'{format}_{line_species}'] = new_value

    return params_dict_copy, format, old_value, new_value
        

def plot_species(ret,
                 wave, 
                 m_flux_full, 
                 line_species=None, 
                 params_dict={}, 
                 log_R_disk=0.0,
                 order=8,
                 color='C1',
                 frame_number=None,
                 **kwargs
                 ):


    n_orders = len(wave)
    assert order < n_orders, f'Order {order} is greater than the number of orders {n_orders}'
    orders = [order] 
    
    params_dict_copy = copy.deepcopy(params_dict)
    
    params_dict_copy['log_R_jup'] = log_R_disk
        
    title = 'w/o disk'
    fig_path = pathlib.Path(f'{conf.path}/{conf.target}/{conf.prefix}plots/disk_gif_frames')
    fig_path.mkdir(exist_ok=True)
    fig_name = fig_path / f'frame_{frame_number:03d}.png'
    
        
    lw = kwargs.get('lw', 0.9)    
    
    ret.evaluate_model(np.array(list(params_dict_copy.values())))
    ret.PMN_lnL_func()
    chi2 = ret.LogLike[w_set].chi_squared_red
    # chi2 = 1.0

    m_flux = np.squeeze(ret.LogLike[w_set].m_flux) / f
    d_flux = np.squeeze(ret.d_spec[w_set].flux) / f
    Cov = ret.Cov[w_set]
    

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
        
        ax[0].plot(wave[order,], m_flux_full[order,], color='dodgerblue', lw=lw, label='Full model ('+r'$\chi^2_{r}='+f'{chi2_full_order:.2f})$')
        ax[0].plot(wave[order,], m_flux[order,], color=color, lw=lw, label=f'Ring radius (log) ={log_R_disk:.2f} ('+r'$\chi^2_{r}='+f'{chi2_order:.2f})$')

        # ax[0].plot(wave[order,], m_flux[order,], color=color, lw=lw, label=
        res_data = d_flux[order] - m_flux_full[order,]
        # ax[1].plot(wave[order,], res_data, color='black', lw=lw, ls='', marker='o', markersize=2)
        
        # res = m_flux_full[order,] - m_flux[order,]
        res = d_flux[order] - m_flux[order,]
        ax[1].plot(wave[order,], res, color=color, lw=lw, alpha=0.9, ls='-', marker='o', markersize=3)
        
        ax[0].set_ylim(0.80e-15, 1.30e-15)
        ax[1].set_ylim(-5e-17, 5e-17)
        ax[0].set_xlim(np.nanpercentile(wave[order,], 15), np.nanpercentile(wave[order,], 85))

        ax[1].axhline(0, color='dodgerblue', lw=0.5)
        ax[0].set_ylabel('Flux / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$')
        ax[1].set_ylabel('Residuals')
        if order==0:
            ax[0].set_title(title)
        
        ax[0].legend()
        if order==n_orders-1:
            ax[1].set_xlabel('Wavelength / nm')
            # ax[1].legend()
            
        

    fig.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'--> Saved {fig_name}')


log_R_disk_range = np.linspace(0.0, 1.6, 14)
# get colors from cmap
colors = plt.cm.viridis(np.linspace(0.32, 0.94, len(log_R_disk_range)))
for i, log_R_disk in enumerate(log_R_disk_range):
    plot_species(ret, wave, 
                 m_flux_full.copy(), 
                 params_dict=params_dict.copy(),
                 log_R_disk=log_R_disk,
                 order=16,
                 color=colors[i],
                 frame_number=i)
