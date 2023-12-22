from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters
from retrieval_base.chemistry import Chemistry

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

targets = dict(J1200='freechem_9', 
               TWA28='freechem_5', 
               J0856='freechem_8'
               )
colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')

out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')

# targets = dict(J1200='freechem_8')

atomic_mass = {k:v[2] for k,v in Chemistry.species_info.items()}

solar = dict(H=12.00, He=10.93, C=8.43, N=7.83, O=8.69, F=4.56,
             Na=6.24, Mg=7.60, K=5.04, Ca=6.34, Ti=4.95, Fe=7.50,
             Fe_H=10.25768,
             )
def scale_to_solar(X_H, species):
    
    assert species in list(solar.keys()), f'Species {species} not in solar abundances.'
    # X_H_solar = (X_H - (np.log10(solar[species]) - 12))
    X_H_solar = (X_H - solar[species]) + 12.
    # X_H_solar -= solar[species]
    return X_H_solar

fig, ax = plt.subplots(1,1, figsize=(8,6))

for i, (target, retrieval_id) in enumerate(targets.items()):
    data_path = pathlib.Path('/home/dario/phd/retrieval_base') / f'{target}'
    print(data_path)
    
    # bestfit_params = 
    retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'
    assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'
        
    chem = pickle.load(open(retrieval_path / 'test_data/bestfit_Chem.pkl', 'rb'))
    # Ca_H = scale_to_solar(chem.mass_fractions_posterior['Ca'].mean(axis=-1), 'Ca')
    # convert from mass fractions to number fractions (divide by atomic mass)
    # H = chem.mass_fractions_posterior['H'].mean(axis=-1)
    # get number densities
    C = chem.mass_fractions_posterior['CO_high'].mean(axis=-1) / atomic_mass['12CO']
    C += chem.mass_fractions_posterior['CO_36_high'].mean(axis=-1) / atomic_mass['13CO']
    
    O = chem.mass_fractions_posterior['CO_high'].mean(axis=-1) / atomic_mass['12CO']
    O += chem.mass_fractions_posterior['CO_36_high'].mean(axis=-1) / atomic_mass['13CO']
    O += chem.mass_fractions_posterior['H2O_pokazatel_main_iso'].mean(axis=-1) / atomic_mass['H2O']
    
    F = chem.mass_fractions_posterior['HF_main_iso'].mean(axis=-1) / atomic_mass['HF']
    
    Na = chem.mass_fractions_posterior['Na_allard'].mean(axis=-1) / atomic_mass['Na']
    Ca = chem.mass_fractions_posterior['Ca'].mean(axis=-1) / atomic_mass['Ca']
    Ti = chem.mass_fractions_posterior['Ti'].mean(axis=-1) / atomic_mass['Ti']
    
    H = 2.* chem.mass_fractions_posterior['H2O_pokazatel_main_iso'].mean(axis=-1) / atomic_mass['H2O']
    H += chem.mass_fractions_posterior['HF_main_iso'].mean(axis=-1) / atomic_mass['HF']
    H += 2. * chem.mass_fractions_posterior['H2'].mean(axis=-1) / atomic_mass['H2']
    
    species = [C, O, F, Na, Ca, Ti]
    
    # log_HF = np.log10(chem.mass_fractions_posterior['HF_main_iso'].mean(axis=-1))
    # # plot corner with log_g and log_HF
    # corner
    
    log_HF_quantiles = np.quantile(log_HF, [0.16, 0.5, 0.84])
    print(f'Target {target} --> log HF = {log_HF_quantiles[1]:.2f} +{log_HF_quantiles[2]-log_HF_quantiles[1]:.2f} -{log_HF_quantiles[1]-log_HF_quantiles[0]:.2f}')
    # Fe_H = scale_to_solar(np.log10(np.sum(species, axis=0) / H), 'Fe_H')
    # Fe_H = np.log10(np.sum(species, axis=0) / H) - np.log10(0.0181)
    # Fe_H_quantiles = np.quantile(Fe_H, [0.16, 0.5, 0.84])

    species_name = ['C', 'O', 'F', 'Na', 'Ca', 'Ti']
    for i, X in enumerate(species):
        X_H = scale_to_solar(np.log10(X/H), species_name[i])
        # get quantiles for error bars
        quantiles = np.quantile(X_H, [0.16, 0.5, 0.84])
        yerr = np.array([[quantiles[1] - quantiles[0]], [quantiles[2] - quantiles[1]]])
        # if any yerr is larger than 2, dont plot
        if any(yerr > 1.2):
            continue
        ax.errorbar(i, quantiles[1], yerr=yerr,
                    color=colors[target], marker='o', ms=10, 
                    capsize=5, capthick=2, elinewidth=2, alpha=0.9)
    
    # Fe_H = np.log10(np.sum(species, axis=0) / H)
    # Fe_H -= (solar['Fe_H'] - 12.)
    # TODO: Fix Fe_H
    # Fe_H = chem.FeH_posterior.mean(axis=-1)
    # Fe_H_quantiles = np.quantile(Fe_H, [0.16, 0.5, 0.84])
    # ax.axhspan(Fe_H_quantiles[0], Fe_H_quantiles[2], color=colors[target], alpha=0.2)

    
    # plot histogram
    hist_args = {"color": colors[target], "alpha": 0.6, "fill": True, "edgecolor": "k",
                         "linewidth": 2.0, "histtype": "stepfilled", "density": True,
                         'bins': 30}
    # ax.hist(Ca_H, **hist_args, label=target)
    # ax.hist(C_H, **hist_args, label=target)
    
# species = 'Ca'
# ax.axvline(solar['Ca'], color='magenta', ls='--', lw=2.5, alpha=0.8, label='Sun')
# ax.set_xlabel(f'[{species}/H]')
# ax.set(xlim=(4, 12))
# remove y-axis and ticks
# replace x-ticks with species names
ax.set_xticks(np.arange(len(species_name)))
ax.set_xticklabels(species_name)

ax.axhline(0.0, color='magenta', ls='-', lw=3.5, alpha=0.3)

ax.legend()
plt.show()
# fig.savefig(out_path / f'figure7_{species}.pdf', bbox_inches='tight')
    