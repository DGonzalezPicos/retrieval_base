from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters
from retrieval_base.chemistry import Chemistry
atomic_mass = {k:v[2] for k,v in Chemistry.species_info.items()}

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

targets = dict(J1200='freechem_9', 
               TWA28='freechem_5', 
               J0856='freechem_9'
               )
colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')
# targets = dict(J0856='freechem_3')

# fig, ax = plt.subplots(1,1, figsize=(8,6))
# create custom legend
max_samples = 6000
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
    a = posterior[:,0]
    l = posterior[:,1]
    Rp = posterior[:,2]
    logg = posterior[:,3]
    epsilon = posterior[:,4]
    vsini = posterior[:,5]
    rv = posterior[:,6]
    
    HF = chem.mass_fractions_posterior['HF_main_iso'].mean(axis=-1) / atomic_mass['HF']
    Na = chem.mass_fractions_posterior['Na_allard'].mean(axis=-1) / atomic_mass['Na']
    Ca = chem.mass_fractions_posterior['Ca'].mean(axis=-1) / atomic_mass['Ca']
    Ti = chem.mass_fractions_posterior['Ti'].mean(axis=-1) / atomic_mass['Ti']
    
    
    
    FeH = chem.FeH_posterior
    CO = chem.CO_posterior
    
    posterior_12CO = chem.mass_fractions_posterior['CO_high'].mean(axis=-1) / atomic_mass['12CO']
    posterior_13CO = chem.mass_fractions_posterior['CO_36_high'].mean(axis=-1) / atomic_mass['13CO']
     
    chem.C12C13_posterior = posterior_12CO / posterior_13CO
    
    # posterior_C18O = chem.mass_fractions_posterior['CO_28'].mean(axis=-1) / atomic_mass['C18O']
    # chem.C18OC16O_posterior = posterior_12CO / posterior_C18O
    
    # posterior_H2O_181 = chem.mass_fractions_posterior['H2O_181'].mean(axis=-1) / atomic_mass['H2O_181']
    # chem.H216OH218O_posterior = chem.mass_fractions_posterior['H2O_pokazatel_main_iso'].mean(axis=-1) / posterior_H2O_181
    
    # print(f'Posterior shape = {posterior.shape}')
    samples = np.array([CO, FeH, chem.C12C13_posterior, 
                        # np.log10(chem.C18OC16O_posterior), 
                        # np.log10(chem.H216OH218O_posterior),
                        np.log10(HF),
                        np.log10(Na),
                        np.log10(Ca),
                        np.log10(Ti),
                        logg,
                        vsini,
                        epsilon,
                        rv,
                        # Rp,
                        # a, l,
                        ]).T

    # if samples.shape[0] > max_samples:
    #     print(f'Number of samples = {samples.shape[0]}. Taking only {max_samples} samples.')

    #     samples = samples[:max_samples,:]
    print(f'Number of samples = {samples.shape[0]}')
    # Make cornerplot with logg and Fe/H
    labels = [r'C/O', r'[Fe/H]', r'$^{12}$C/$^{13}$C',
                # r'log C$^{16}$O/C$^{18}$O', r'log H$_2^{16}$O/H$_2^{18}$O', 
                r'log HF', r'log Na', r'log Ca', r'log Ti', 
                r'$\log g$', r'$v \sin i$', r'$\epsilon_{\rm limb}$', r'$v_{\rm rad}$',
                ]

    pad_factor = 0.1
    lims= [(np.min(samples[:,i]), np.max(samples[:,i])) for i in range(samples.shape[-1])]
    pad = [(lims[i][1] - lims[i][0]) * pad_factor for i in range(samples.shape[-1])]
    lims = [(lims[i][0] - pad[i], lims[i][1] + pad[i]) for i in range(samples.shape[-1])]
    lims[2] = (0, 200)
    
    fig = corner.corner(samples, labels=labels, color=colors[target],
                        range=lims,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 18},
                        fontsize=16,
                        plot_density=True,
                        plot_datapoints=False,
                        plot_contours=True,
                        smooth=1.5, 
                        bins=30, 
                        max_n_ticks=3,
                        hist_kwargs={'density': False,
                                     'fill': True,
                                     'alpha': 0.7,
                                     'edgecolor': 'k',
                                    'linewidth': 1.0,
                                     },
                        )
    
    
    # collect all titles
    titles = [axi.title.get_text() for axi in fig.axes]
    for i, title in enumerate(titles):
        if len(title) > 30:
            title_split = title.split('=')
            titles[i] = title_split[0] + ' =\n' + title_split[1]
    # remove the original titles
    for i, axi in enumerate(fig.axes):
        fig.axes[i].title.set_visible(False)
        # increase fontsize of x and y labels
        fig.axes[i].xaxis.label.set_fontsize(20)
        fig.axes[i].yaxis.label.set_fontsize(20)
        # increse fontsize of tick labels
        fig.axes[i].tick_params(axis='both', which='major', labelsize=20)
        fig.axes[i].tick_params(axis='both', which='minor', labelsize=20)
        
        
    # add new titles
    for i, title in enumerate(titles):
        fig.axes[i].text(0.5, 1.05, title, fontsize=22,
                        ha='center', va='bottom',
                        transform=fig.axes[i].transAxes)
        
        
        
    fig.subplots_adjust(top=0.95)
    save= True
    if save:
        fig.savefig(out_path / f'full_cornerplot_{target}.pdf', bbox_inches='tight', dpi=300)   
        plt.close() 
# plt.show()
