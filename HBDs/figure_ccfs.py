from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters
from retrieval_base.chemistry import Chemistry
import retrieval_base.figures as figs
import retrieval_base.auxiliary_functions as af
from retrieval_base.callback import CallBack

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
import copy

targets = dict(J1200='freechem_8', TWA28='freechem_4', J0856='freechem_3')
targets = dict(J0856='freechem_3')
colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')
w_set = 'K2166'
out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')
line_species_to_plot = ['H2O', 
                        '12CO', 
                        '13CO',
                        # 'HF',
                        ]

species_plot_info = {k:v[0] for k,v in Chemistry.species_plot_info.items()}

fig, ax = plt.subplots(len(line_species_to_plot), 1, figsize=(8,6), sharex=True)

# rv = np.arange(-900,900+1e-6,1.)
rv = np.arange(-200., 200., 4.)
for t, (target, retrieval_id) in enumerate(targets.items()):
    data_path = pathlib.Path('/home/dario/phd/retrieval_base') / f'{target}'
    print(data_path)
    
    # bestfit_params = 
    retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'
    assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'
    
    ccf_files = sorted(retrieval_path.glob(f'test_data/CCF*.npy'))
    assert len(ccf_files) > 0, f'No CCF files found in {retrieval_path}.'
            
    for i, s in enumerate(line_species_to_plot):
        file = retrieval_path / f'test_data/CCF_{s}.npy'
        assert file.exists(), f'CCF file {file} does not exist.'
        rv, ccf,acf = np.load(file).T
        ax[i].plot(rv, ccf, color=colors[target], alpha=0.8, lw=1.5)
        ax[i].plot(rv, acf, color=colors[target], alpha=0.8, lw=1.5, ls='--')

for i, axi in enumerate(ax):
    axi.set_xlim(rv.min(), rv.max())
    axi.text(s=Chemistry.species_plot_info[line_species_to_plot[i]][1], x=0.05, y=0.8,
             transform=axi.transAxes, fontsize=22, weight='bold')
plt.show()