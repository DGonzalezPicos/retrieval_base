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
import seaborn as sns

targets = dict(
                J1200='final_full',
                TWA28='final_full',
                J0856='final_full',
                )
targets = dict(reversed(list(targets.items())))

colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')

normalize_by_solar = True

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
        fig,ax = plt.subplots(1, 1, figsize=(6,4))
        # solar F_H 
        A_F_sun = 4.40 # +- 0.25, Maiorca+2014, 
        A_F_sun_err = 0.25
        nsamples = int(1e5)
        if normalize_by_solar:
            sun_samples = np.random.normal(loc=0.0, scale=A_F_sun_err, size=nsamples)
        else:
            sun_samples = np.random.normal(loc=A_F_sun, scale=A_F_sun_err, size=nsamples)
        # create KDE of solar F_H with a gaussian kernel
        kde = sns.kdeplot(sun_samples, color='magenta', linewidth=2.0, linestyle='--', label='Solar F/H', ax=ax, zorder=10)
        
                
        # remove y-axis
        ax.set_yticks([])
        ax.set_ylabel('')
        # remove upper x-axis and xticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # use tick_params to remove upper xticks
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        
        xlabel = 'Fluorine abundance to hydrogen $A$(F)' if not normalize_by_solar else 'Fluorine abundance [F/H]'
        xlim = np.array(([3.3, 5.3]))
        if normalize_by_solar:
            # xlim -= A_F_sun
            xlim = (-1, 1)
        ax.set(xlim=xlim, xlabel=xlabel)
        
    # log_HF = np.log10(chem.mass_fractions_posterior['HF_main_iso'].mean(axis=-1))
    log_HF = np.log10(chem.VMRs_posterior['HF'])
    
    # solar scaled abundance
    H = chem.mass_fractions['H']
    F_H = np.log10(1e12 * chem.VMRs_posterior['HF'] / H.mean())
    if normalize_by_solar:
        F_H -= A_F_sun
        
    F_H_quantiles = np.quantile(F_H, [0.16, 0.5, 0.84])
    print(f'{target}: [F/H] = {F_H_quantiles[1]:.2f} +{F_H_quantiles[2]-F_H_quantiles[1]:.2f} -{F_H_quantiles[1]-F_H_quantiles[0]:.2f}')

    # A_F_sun_kde = gaussian_kde(A_F_sun)
    # evaluate the KDE at the F_H values
    # A_F_sun = A_F_sun_kde(F_H)
    
    
    # F_H = F_H - A_F_sun
    # print(f'{target}: [F/H] = {F_H.mean()} +- {F_H.std()}')
            
    hist_kwargs = {"color": colors[target], "alpha": 0.65, "fill": True, "edgecolor": "k",
                            "linewidth": 2.0, "histtype": "stepfilled", "density": True,}
    
    
    ax.hist(F_H, bins=30, **hist_kwargs)
    hist_kwargs_edge = hist_kwargs.copy()
    hist_kwargs_edge['fill'] = False
    ax.hist(F_H, bins=30, **hist_kwargs_edge)
    
    # linear fit to x=log_HF and y=logg
    # y = m * x + b
    # m = slope
    # b = intercept
    # m, b = np.polyfit(log_HF, logg, 1)
    
    plot_HF_CH = False
    
    if plot_HF_CH:
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
                            labels=[r'C/H', r'$\log_{10}$(HF)'],
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
    
# plt.show()
# fig.savefig(out_path / f'{target}_logg_HF.pdf', bbox_inches='tight')

fig_name =  out_path / f'F_H_posterior.pdf' if normalize_by_solar else out_path / f'F_H_posterior_absolute.pdf'
fig.savefig(fig_name)
print(f'Saved {fig_name}')
plt.close(fig)