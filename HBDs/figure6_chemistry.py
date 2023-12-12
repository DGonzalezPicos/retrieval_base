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

targets = dict(J1200='freechem_7', TWA28='freechem_3', J0856='freechem_3')
colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')

fig, ax = plt.subplots(1,2, figsize=(8,4))
n_bins = 30
solar_system = {'C/O': 0.54, '12C/13C': 89}

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
    C_O = chem.CO_posterior
    chem.C12C13_posterior = np.median(chem.mass_fractions_posterior['CO_high'] / chem.mass_fractions_posterior['CO_36_high'],axis=-1)
    chem.C16OC18O_posterior = np.median(chem.mass_fractions_posterior['CO_high'] / chem.mass_fractions_posterior['CO_28'],axis=-1)
    chem.H216OH218O_posterior = np.median(chem.mass_fractions_posterior['H2O_pokazatel_main_iso'] / chem.mass_fractions_posterior['H2O_181'],axis=-1)
    
    hist_args = {"color": colors[target], "alpha": 0.6, "fill": True, "edgecolor": "k",
                         "linewidth": 2.0, "histtype": "stepfilled", "density": False,
                         'bins': n_bins}

    ax[0].hist(C_O,  range=(0.45, 0.72), **hist_args)
    ax[1].hist(chem.C12C13_posterior, range=(10., 220.), **hist_args)
    labels = ['Sun', 'ISM'] if i == 0 else [None, None]
    for j, (key, val) in enumerate(solar_system.items()):
        ax[j].axvline(val, color='magenta', ls='--', lw=2.5, alpha=0.8, label=labels[0])
    ax[1].axvline(68, color='deepskyblue', ls='--', lw=2.5, alpha=0.8, label=labels[1])
    
    if len(ax) > 2:
        ax[2].hist(chem.C16OC18O_posterior, range=(10., 1220.), **hist_args)
        ax[3].hist(chem.H216OH218O_posterior, range=(0., 2220.), **hist_args)
    
# remove y-axis and top x-axis
xlabels = [r'C/O', r'$^{12}$C/$^{13}$C', 
           r'C$^{16}$O/C$^{18}$O',
           r'H$_2^{16}$O/H$_2^{18}$O'
           ]
for axi in ax:
    axi.yaxis.set_visible(False)
    axi.xaxis.set_ticks_position('bottom')
    axi.spines['top'].set_visible(False)
    axi.spines['right'].set_visible(False)
    axi.set(xlabel=xlabels.pop(0))

    for spine in ['top', 'right', 'left']:
        axi.spines[spine].set_visible(False)


# create manual legend
handles = [plt.Line2D([0], [0], color=colors[target], ls='-', lw=3., alpha=0.8) for target in targets.keys()]
labels = [f'{target}' for target in targets.keys()]
# make legend entries bold, increase spacing between entries
# leg = fig.legend(handles=handles, labels=labels, loc=(0.57, 0.6),
#                  frameon=False, prop={'weight':'bold', 'size': 16},
#                  )
ax[1].legend(frameon=False, prop={'weight':'bold', 'size': 16}, loc='upper right')
fig.tight_layout()
plt.show()
save= True
if save:
    fig.savefig(out_path / f'fig6_chemistry.pdf', bbox_inches='tight', dpi=300)
    plt.close()
