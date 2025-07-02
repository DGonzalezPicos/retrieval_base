""" 
Plot alpha for each species

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
from scipy.stats import norm
from matplotlib import patheffects as path_effects

path = pathlib.Path(af.get_path())
# path_figures = pathlib.Path('/home/dario/phd/retrieval_base/twx_figs')
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')

config_file = 'config_jwst.txt'
# target = 'TWA28'
w_set='NIRSpec'

# set global font size
plt.rcParams.update({'font.size': 12})
pe_white = [path_effects.withStroke(linewidth=2.0, foreground='w')]

runs = dict(
    TWA27A='freeslab_lbl10_G1G2G3_1',
    TWA28='freeslab_lbl10_G1G2G3_1',
            )


def check_dir(target):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')
        

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
        # ret.Chem.get_VMRs_posterior(save_to=envelopes_dir)
        
        # np.save(file_posterior, np.array(list(ret.Chem.VMRs_posterior.values())))
        np.save(VMR_envelopes_file, np.array(list(ret.Chem.VMRs_envelopes.values())))
        np.save(VMR_labels_file, np.array(list(ret.Chem.VMRs_posterior.keys())))
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
    posterior = dict(zip(free_params_keys, np.load(posterior_file).T))
    
    alpha_params = {k:posterior[f'alpha_{k}'] for k in VMR_labels_data if f'alpha_{k}' in free_params_keys}
    return VMR_envelopes, PT_envelopes, conf, alpha_params


def sigma_to_quantiles(sigma_levels):
    """
    Convert sigma levels to quantiles.
    
    Parameters:
    sigma_levels (list): List of sigma levels (e.g., [1, 2, 3])
    
    Returns:
    list: Corresponding quantiles
    """
    return [norm.cdf(s) - norm.cdf(-s) for s in sigma_levels]


target = 'TWA28'
run = runs[target]
colors = dict(
            # TWA28='orange',
            TWA28='#D55E00',
            TWA27A='#009E73',
                )

fig, ax = plt.subplots(1,1, figsize=(6,3.5))
sigma = [1,2,3]

def plot_target(target, run, ax, x_offset=0, species_list=[]):
    VMR_envelopes, PT_envelopes, conf, alpha_params = get_VMR(target, run)
    # Use the sigma_to_quantiles function to get quantiles
    q_pos = sigma_to_quantiles(sigma)
    # add negative quantiles
    q_neg = [1-q_i for q_i in q_pos]
    q = np.concatenate([q_neg, [0.5], q_pos])
    # sort from smallest to largest
    q = np.sort(q)

    alpha_quantiles = {k:np.quantile(alpha_params[k], q) for k in alpha_params.keys()}
    if len(species_list) == 0:
        pressure = PT_envelopes[0]
        temperature = PT_envelopes[1:-1]
        icf = PT_envelopes[-1]
        icf_peak = np.max(icf)
        pressure_peak = pressure[np.argmax(icf)]
        print(f' --> ICF peak: {icf_peak} at {pressure_peak} bar')

        VMR_median = {k:VMR_envelopes[k][1] for k in VMR_envelopes.keys()}
        VMR_peak = {k:VMR_median[k][np.argmax(icf)] for k in VMR_envelopes.keys()}
        # sort species by VMR_peak
        sorted_species = sorted(VMR_peak.keys(), key=lambda x: VMR_peak[x], reverse=True)
        print(f' --> Sorted species: {sorted_species}')
        ignore_species = ['Mg']
        print(f' --> ignore_species: {ignore_species}')
        species_list = [species for species in sorted_species if species not in ignore_species]
    
    # sort dictionary alpha_quantiles by the order of sorted_species
    alpha_quantiles = {k:alpha_quantiles[k] for k in species_list if k in alpha_quantiles.keys()}
    # print(f' --> alpha_quantiles.keys(): {alpha_quantiles.keys()}')
    species_list = []
    for i, species in enumerate(alpha_quantiles.keys()):
        # print(f' --> species: {species}: {alpha_quantiles[species]}')
        ns = len(sigma)
        for s in range(ns):
        # plot vertical line at alpha_quantiles[species][1], extending from alpha_quantiles[species][0] to alpha_quantiles[species][2]
            x = i + x_offset
            ax.plot([x,x], [alpha_quantiles[species][s], alpha_quantiles[species][-(s+1)]],
                    lw=4, ls='-', alpha=0.4, color=colors[target])
            
        # add vertical gray stripe to visually divide the species
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.05, lw=0)
        
        species_list.append(species)
            
    return species_list
            
      
      
species_list = []
# Store median alpha values for each target
median_alpha_by_target = {}
for t, target in enumerate(runs.keys()):
    x_offset = -0.2 + t*0.40
    # Get VMRs and alpha_params for this target
    VMR_envelopes, PT_envelopes, conf, alpha_params = get_VMR(target, runs[target])
    # Use the sigma_to_quantiles function to get quantiles
    q_pos = sigma_to_quantiles(sigma)
    q_neg = [1-q_i for q_i in q_pos]
    q = np.concatenate([q_neg, [0.5], q_pos])
    q = np.sort(q)
    # Get the list of species to plot (sorted, ignoring Mg)
    if len(species_list) == 0:
        pressure = PT_envelopes[0]
        icf = PT_envelopes[-1]
        VMR_median = {k:VMR_envelopes[k][1] for k in VMR_envelopes.keys()}
        VMR_peak = {k:VMR_median[k][np.argmax(icf)] for k in VMR_envelopes.keys()}
        sorted_species = sorted(VMR_peak.keys(), key=lambda x: VMR_peak[x], reverse=True)
        ignore_species = ['Mg']
        species_list = [species for species in sorted_species if species not in ignore_species]
    # Get alpha medians for the species in species_list
    alpha_medians = [np.median(alpha_params[k]) for k in species_list if k in alpha_params]
    if alpha_medians:
        median_alpha = np.median(alpha_medians)
        median_alpha_by_target[target] = median_alpha
        # Plot horizontal dashed line at median alpha
        ax.axhline(median_alpha, color=colors[target], ls='--', lw=1.5, alpha=0.8, zorder=-2)
    # Now plot the vertical lines as before
    species_list = plot_target(target, runs[target], ax, x_offset=x_offset, species_list=species_list)
    print(f' len(species_list): {len(species_list)} for {target}')


# make xticks at the bottom of the plot show the species_list
ax.set_xticks(range(len(species_list)))

replace_species = {
    'H2O': r'H$_2{}^{16}$O',
    'CO2': r'CO$_2$',
    '12CO': r'${}^{12}$CO',
}
species_list = [replace_species[species] if species in replace_species else species for species in species_list]
ax.set_xticklabels(species_list, rotation=55)
ax.set(ylabel='Abundance offset\n' + r'$\alpha$', ylim=(-1.6, 1.6))

# create custom handles for legend witht the color of each target, set lw of handles to 2
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=colors[target], lw=2) for target in runs.keys()]
legend_labels = [f'TWA {target[3:]}' for target in runs.keys()]
# change extent of legend to make it tighter, increase edgewidth
# change length of legend handles
for handle in handles:
    handle.set_linewidth(2.5)
    handle.set_alpha(0.8)

ax.legend(handles,
          legend_labels, 
          loc='upper left', 
          frameon=True, 
          framealpha=0.2,
          ncol=2,
          edgecolor='black',
          handlelength=1.5,
          )
# change length of legend handles

# add text with string "s" at bottom right
s = r'$\alpha=0 \rightarrow$' + ' chemical equilibrium at solar composition'
ax.text(0.57, 0.03, s, ha='center', va='bottom', fontsize=11, color='black',
        transform=ax.transAxes,
        path_effects=pe_white)

    

ax.set(xlim=(-0.8, len(species_list)))
ax.axhline(0.0, color='black', lw=0.5, ls='-')
# plt.show()
# save figure as pdf
fig.savefig(path_figures / 'fig_alpha.pdf', bbox_inches='tight')
print(f' --> Saved figure to {path_figures / "fig_alpha.pdf"}')

