from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters

import numpy as np
import matplotlib.pyplot as plt
# set fontsize to 16
# plt.rcParams.update({'font.size': 16})
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')

import pathlib
import pickle
import corner
import pandas as pd
import json

def get_FeH(chem):
    
    ignore_keys = ['H','H2', 'He', 'H-', 'e-', 'MMW']
    H = chem.mass_fractions_posterior['H'].mean(axis=-1)
    Z = np.zeros_like(H)
    for key, val in chem.mass_fractions_posterior.items():
        if key in ignore_keys:
            continue
        # print(f'Adding {key} to Z')
        Z += val.mean(axis=-1)
    # for Z/H = 0.0181 # Asplund et al. (2009)
    # log_FeH_solar = np.log10(0.0181 * 1e12) # = 10.257678574869184
    
    # [Q/H] = log10(Q/H) - log10(Q/H)_solar # Young and Wheeler (2022)
    log_FeH_solar = np.log10(0.0181) # = (10.258 - 12.)
    FeH = np.log10(Z/H) - log_FeH_solar
    return FeH

def get_CH(chem):
    H = chem.mass_fractions_posterior['H'].mean(axis=-1)
    C = np.zeros_like(H)
    keys_C = ['CO_high', 'CO_36_high', 'CO_28', 'CO_27']
    for key, val in chem.mass_fractions_posterior.items():
        if key in keys_C:
            C = val.mean(axis=-1)
            break
    log_CH_solar = 8.43 - 12 # Asplund et al. (2021)
    CH = np.log10(C/H) - log_CH_solar
    return CH

path = pathlib.Path('/home/dario/phd/retrieval_base')
# out_path = path / 'HBDs'
out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')

targets = dict(J1200='freechem_12', 
               TWA28='freechem_9', 
               J0856='freechem_9'
               )
colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')
corr_dict = dict()
# fig, ax = plt.subplots(1,1, figsize=(8,6))
# create custom legend
for i, (target, retrieval_id) in enumerate(targets.items()):
    data_path = pathlib.Path('/home/dario/phd/retrieval_base') / f'{target}'
    print(data_path)
    
    # bestfit_params = 
    retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'
    assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'
    
    # load json file with bestfit parameters
    with open(retrieval_path / 'test_data/bestfit.json', 'r') as f:
        bestfit_params = json.load(f)
        
    equal_weighted_file = retrieval_path / 'test_post_equal_weights.dat'
    posterior = np.loadtxt(equal_weighted_file)
    posterior = posterior[:,:-1]
    
    params = bestfit_params['params']
    chem = pickle.load(open(retrieval_path / 'test_data/bestfit_Chem.pkl', 'rb'))
    # logg = params['log_g']
    logg = posterior[:,3]
    # FeH = chem.FeH_posterior
    # FeH = 
    # FeH = get_FeH(chem)
    # FeH = get_CH(chem)
    FeH = chem.FeH_posterior
    print(f'Posterior shape = {posterior.shape}')
    samples = np.array([logg, FeH]).T

    # samples_dict = dict(zip(params.keys(), posterior.T))
    # calculate correlation between logg and Fe/H
    from scipy.stats import pearsonr
    corr, p = pearsonr(logg, FeH)
    print(f'Correlation between logg and Fe/H = {corr:.2f}')
    corr_dict[target] = corr

    

    # Make cornerplot with logg and Fe/H
    labels = [r'$\log g$', r'$\mathrm{[C/H]}$']
    if i == 0:
        fig = None
    # limits = [(3.0, 5.0), (-1.0, 1.5)]  # replace with your actual limits
    limits = [(2.5, 5.0), (-0.25, 1.7)]  # replace with your actual limits

    fig = corner.corner(samples, labels=labels, quantiles=[0.5],
                        show_titles=False, 
                        title_kwargs={"fontsize": 12}, 
                        color=colors[target], 
                        plot_density=True,
                        plot_datapoints=False,
                        plot_contours=True,
                        smooth=2.0, 
                        bins=40, 
                        hist_kwargs={'density': True,
                                     'fill': True,
                                     'alpha': 0.7,
                                     'edgecolor': 'k',
                                    'linewidth': 1.0,
                                     },
                        range=limits,
                        fig=fig)
    
    
    # fig.suptitle(target, fontsize=16)
fig.subplots_adjust(top=0.95)
# create manual legend
handles = [plt.Line2D([0], [0], color=colors[target], ls='-', lw=3., alpha=0.8) for target in targets.keys()]
labels = [f'{target}\n$r$ = {corr_dict[target]:.2f}' for target in targets.keys()]
# make legend entries bold, increase spacing between entries
leg = fig.legend(handles=handles, labels=labels, loc=(0.57, 0.6),
                 frameon=False, prop={'weight':'bold', 'size': 16},
                 )
plt.show()
save= True
if save:
    fig.savefig(out_path / f'fig5_logg_FeH.pdf', bbox_inches='tight', dpi=300)
    plt.close()