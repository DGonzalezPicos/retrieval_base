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

targets = dict(J1200='freechem_8', TWA28='freechem_4', J0856='freechem_3')
colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')

out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')

for i, (target, retrieval_id) in enumerate(targets.items()):
    data_path = pathlib.Path('/home/dario/phd/retrieval_base') / f'{target}'
    print(data_path)
    
    # bestfit_params = 
    retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'
    assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'
        
    chem = pickle.load(open(retrieval_path / 'test_data/bestfit_Chem.pkl', 'rb'))
    equal_weighted_file = retrieval_path / 'test_post_equal_weights.dat'
    posterior = np.loadtxt(equal_weighted_file)
    posterior = posterior[:,:-1]

    # logg = params['log_g']
    # logg = posterior[:,3]
    C_H = chem.FeH_posterior
    if i == 0:
        fig = None
        
    log_HF = np.log10(chem.mass_fractions_posterior['HF_main_iso'].mean(axis=-1))
    # linear fit to x=log_HF and y=logg
    # y = m * x + b
    # m = slope
    # b = intercept
    # m, b = np.polyfit(log_HF, logg, 1)
    m, b = np.polyfit(log_HF, C_H, 1)
    print(f'{target}: m = {m}, b = {b}')
    
    # calculate y when x = -8
    y = m * -8 + b
    # print(f'{target}: log g = {y} at log HF = -8')
    print(f'{target}: [Fe/H] = {y} at log HF = -8')
    
    # samples = np.vstack((logg, log_HF)).T
    samples = np.vstack((C_H, log_HF)).T
    
    hist_kwargs = {"color": colors[target], "alpha": 0.6, "fill": True, "edgecolor": "k",
                            "linewidth": 2.0, "histtype": "stepfilled", "density": False,}
    fig = corner.corner(samples, fig=fig, 
                        labels=[r'$\log g$', r'$\log_{10}$(HF)'],
                        color=colors[target],
                        hist_kwargs=hist_kwargs,
                        label_kwargs={'fontsize': 16},
                        show_titles=False,
                        levels=[0.68, 0.95],
                        plot_datapoints=False,
                        smooth=0.5,
                        no_fill_contours=True,
                        )
    # fig.suptitle(f'{target}', fontsize=16)
    
plt.show()
fig.savefig(out_path / f'{target}_logg_HF.pdf', bbox_inches='tight')    