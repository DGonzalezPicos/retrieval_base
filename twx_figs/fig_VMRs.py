""" 
Plot VMRs as a function of altitude

date: 2025-01-21
"""
import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import copy
import matplotlib.cm as cm

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import seaborn as sns
# import config_jwst as conf
import colorcet as cc

path = pathlib.Path(af.get_path())
# path_figures = pathlib.Path('/home/dario/phd/retrieval_base/twx_figs')
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')

config_file = 'config_jwst.txt'
# target = 'TWA28'
w_set='NIRSpec'

runs = dict(
    TWA27A=['freeslab_lbl10_G1G2G3_0'],
    TWA28=[
        'freeslab_lbl10_G1G2G3_0', 
           ],
            )


def check_dir(target):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')

def get_bestfit_params(target,run):
    
    check_dir(target)
    
    conf = Config(path=path, target=target, run=run)(config_file)        
        
    ret = Retrieval(
        conf=conf, 
        evaluation=False
        )

    bestfit_params, _ = ret.PMN_analyze()
    bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
    return bestfit_params_dict

def get_VMR(target, run, cache=True):
    
    envelopes_dir = path / target / f'retrieval_outputs/{run}/test_data'/ 'envelopes'
    envelopes_dir.mkdir(parents=True, exist_ok=True)

    PT_envelopes_file = envelopes_dir / 'PT_envelopes.npy'

    VMR_envelopes_file = envelopes_dir / 'VMR_envelopes.npy'
    VMR_labels_file = envelopes_dir / 'VMR_labels.npy'

    check_dir(target)
    conf = Config(path=path, target=target, run=run)(config_file)
    
    posterior_file = f'{conf.prefix}data/bestfit_posteriors.npy'
    
    exist = (VMR_envelopes_file.exists() and VMR_labels_file.exists() and PT_envelopes_file.exists() and os.path.exists(posterior_file))

    if not cache or not exist:
        print(f' --> Calculating VMRs for {target} {run}')
        check_dir(target)
        conf = Config(path=path, target=target, run=run)(config_file)        
            
        ret = Retrieval(
            conf=conf, 
            evaluation=False
            )
        
        bestfit_params, posterior = ret.PMN_analyze()
        np.save(posterior_file, posterior)
        print(f' --> Saved posterior to {posterior_file}')
        bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
        # bestfit_params_dict['log_SiO'] = -6.0

        print(f' --> Best-fit parameters: {bestfit_params_dict}')
        bestfit_params = np.array(list(bestfit_params_dict.values()))

        ret.evaluate_model(bestfit_params)
        ret.evaluation = True
        ret.PMN_lnL_func()
        ret.get_PT_mf_envelopes(posterior)
        ret.Chem.get_VMRs_posterior(save_to=envelopes_dir)
        ret.copy_integrated_contribution_emission()

        # save PT envelopes as npy file with pressure and temperature envelopes
        PT_envelopes = np.vstack([ret.PT.pressure, ret.PT.temperature_envelopes, ret.PT.int_contr_em['NIRSpec']])
        # return np.array(ret.Chem.VMRs_envelopes), PT_envelopes, conf
        
    print(f' --> Found {VMR_envelopes_file}')
    VMR_envelopes_data = np.load(VMR_envelopes_file)
    VMR_labels_data = np.load(VMR_labels_file)
    # create a dictionary with the VMRs
    VMR_envelopes = dict(zip(VMR_labels_data, VMR_envelopes_data))
    PT_envelopes = np.load(PT_envelopes_file)
    
    free_params_keys = conf.free_params.keys()
    # print(f' --> free_params_keys: {free_params_keys}')
    # print(f' --> VMR_envelopes.keys: {VMR_envelopes.keys()}')
    posterior = dict(zip(free_params_keys, np.load(posterior_file).T))
    
    alpha_params = {k:posterior[f'alpha_{k}'] for k in VMR_labels_data if f'alpha_{k}' in free_params_keys}
    return VMR_envelopes, PT_envelopes, conf, alpha_params
    
fig, ax = plt.subplots(1,2, figsize=(9,3), sharey=True, sharex=True,
                       gridspec_kw=dict(wspace=0.15))

icf_colors = dict(TWA28='orange',
                  TWA27A='#0a74da')

def plot_target(target, run, ax, ax_icf=None, plot_species='all', color_species={}, ls_dict={}):
    VMR_envelopes, PT_envelopes, conf, alpha_params = get_VMR(target, run)
    # print(f' --> alpha_params: {alpha_params}')
    tex_labels = {k[4:]:v[0][-1].replace("\\log\\ ", "") for k,v in conf.opacity_params.items()}
    pressure = PT_envelopes[0]
    temperature = PT_envelopes[1:-1]
    icf = PT_envelopes[-1]

    if ax_icf is None:
        ax_icf = ax.twiny()
        ax_icf.set_xlim(-14, 0.)

    draw_icf = False
    if draw_icf:
        ax_icf.plot(-icf, pressure, color='black', lw=1.5, alpha=0.65, ls='-')
        ax_icf.fill_betweenx(pressure, -icf, 0.0, 
                            lw=0.0,
                            color='k',
                            alpha=0.2, 
                            zorder=-2)
    
    fill_icf = True
    if fill_icf:
        # Add gradient color to the fill
        p_gradient = np.logspace(np.log10(pressure.min()), np.log10(pressure.max()), len(pressure) * 5)
        icf_gradient = np.interp(p_gradient, pressure, icf)

        # weights = icf_gradient**(2) / icf_gradient.max()
        weights = (icf_gradient**(1) / icf_gradient.max()) * 0.2
        for i in range(len(p_gradient) - 1):
            
            # ax_icf.fill_betweenx(p_gradient[i:i+2], -icf_gradient[i:i+2], 0.0, 
            #                     lw=0.0,
            #                     color=icf_colors[target], 
            #                     alpha=max(weights[i], 0.2), 
            #                     zorder=-1)
            ax.axhspan(p_gradient[i], p_gradient[i+1], color='k', alpha=max(weights[i], 0.0), zorder=-1)

    ax_icf.set(yscale='log', xticks=[], yticks=[], ylim=(pressure.max(), pressure.min()))

    
    # cmap = cc.cm.glasbey_bw_minc_20_maxl_70
    cmap = cc.cm.glasbey_dark
    
    if plot_species == 'all':
        plot_species = VMR_envelopes.keys()
        
    for i, key in enumerate(plot_species):
        
        x1 = VMR_envelopes[key][0,:]
        x2 = VMR_envelopes[key][-1,:]
        color = color_species.get(key, cmap(i))
        color_species[key] = color
        
        # ls = ls_dict.get(key, '-' if i % 2 == 0 else '--')
        ls = ls_dict.get(key, '-')
        ls_dict[key] = ls
        if np.mean(abs(np.log10(x1) - np.log10(x2))) > 1.0:
            print(f' Skipping {key} because the VMR range is too large...')
            continue
        ax.fill_betweenx(pressure,
                            x1,
                            x2,
                            lw=0,
                            alpha=0.2, color=color)
            
        ax.plot(VMR_envelopes[key][1,:], pressure, 
                color=color,
                lw=1.5, alpha=0.75,
                label=tex_labels[key],
                ls = ls)
        
        plot_alpha_zero = False
        if plot_alpha_zero:
            alpha_i = np.median(alpha_params[key])
            print(f' key: {key}, alpha_i: {alpha_i}')
            ax.plot(VMR_envelopes[key][1,:] / 10.0**alpha_i, pressure, 
                    color=color,
                    lw=1.5, alpha=0.75,
                    ls = '--')
        
    ax.set(ylabel='Pressure / bar', 
        xlabel='VMR', yscale='log', 
        ylim=(pressure.max(), pressure.min()),
        xscale='log')
    ax.set_title('TWA ' + target.replace('TWA', ''))
    
    return color_species, ls_dict


color_species, ls_dict = {}, {}
plot_species = ['12CO', 'H2O','SiO','OH','HF', 'FeH','TiO', 'NaH', 'CO2', 'VO', 'CrH']

for t, target in enumerate(runs.keys()):
    target_runs = list(np.atleast_1d(runs[target]))
    for r, run_name in enumerate(target_runs):

        color_species, ls_dict = plot_target(target, run_name, ax[t], None, plot_species=plot_species, color_species=color_species, ls_dict=ls_dict)
    
ax[1].legend(loc=(1.01, 0.0), fontsize=10,
        ncol=1,
        frameon=False,
        handlelength=1.5,
        handletextpad=1.0,
        columnspacing=0.7)

ax[0].set_xlim(1e-9, 1e-2)
ax[1].set_ylabel('')
# plt.show()
species_label = '_'.join(plot_species)
fig_name = path_figures / f'fig_VMRs_{species_label}.pdf'
fig.savefig(fig_name, bbox_inches='tight')
print(f' --> Saved {fig_name}')
plt.close()