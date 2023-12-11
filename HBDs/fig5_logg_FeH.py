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

path = pathlib.Path('/home/dario/phd/retrieval_base')
# out_path = path / 'HBDs'
out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')

targets = dict(J1200='freechem_4', TWA28='freechem_1', J0856='freechem_1')
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
    FeH = chem.FeH_posterior
    print(f'Posterior shape = {posterior.shape}')
    samples = np.array([logg, chem.FeH_posterior]).T

    # samples_dict = dict(zip(params.keys(), posterior.T))
    # calculate correlation between logg and Fe/H
    from scipy.stats import pearsonr
    corr, p = pearsonr(logg, FeH)
    print(f'Correlation between logg and Fe/H = {corr:.2f}')
    corr_dict[target] = corr

    

    # Make cornerplot with logg and Fe/H
    labels = [r'$\log g$', r'$\mathrm{[Fe/H]}$']
    if i == 0:
        fig = None
    fig = corner.corner(samples, labels=labels, quantiles=[0.5],
                        show_titles=False, 
                        title_kwargs={"fontsize": 12}, 
                        color=colors[target], 
                        plot_density=True,
                        plot_datapoints=False,
                        smooth=1.0, 
                        bins=20, 
                        hist_kwargs={'density': True,
                                     'fill': True,
                                     'alpha': 0.6,
                                     'edgecolor': 'k'
                                     },
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
save= False
if save:
    fig.savefig(out_path / f'fig5_logg_FeH_{target}.png', bbox_inches='tight', dpi=300)
    plt.close()