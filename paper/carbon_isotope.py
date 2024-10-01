from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

base_path = '/home/dario/phd/retrieval_base/'

target = 'gl880'

def main(target, label='', ax=None, run=None):
    if target not in os.getcwd():
        os.chdir(base_path + target)

    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    # find dirs in outputs
    print(f' outputs = {outputs}')
    dirs = [d for d in outputs.iterdir() if d.is_dir() and 'sphinx' in d.name and '_' not in d.name]
    runs = [int(d.name.split('sphinx')[-1]) for d in dirs]
    if run is None:
        run = 'sphinx'+str(max(runs))
    else:
        run = 'sphinx'+str(run)
        assert run in [d.name for d in dirs], f'Run {run} not found in {dirs}'
    print('Run:', run)

    config_file = 'config_freechem.txt'
    conf = Config(path=base_path, target=target, run=run)(config_file)

    ret = Retrieval(
                conf=conf, 
                evaluation=False,
                )

    bestfit_params, posterior = ret.PMN_analyze()

    param_keys = list(ret.Param.param_keys)
    Teff_id = param_keys.index('Teff')
    print(f' T_eff = {bestfit_params[Teff_id]}')
    log_carbon_ratio_id = param_keys.index('log_12CO/13CO')
    print(f' log(12CO/13CO) = {bestfit_params[log_carbon_ratio_id]}')

    # make scatter plot with one point corresponding to Teff vs log(12CO/13CO)
    # take uncertainties from posterior quantiles 
    q=[0.16, 0.5, 0.84]
    Teff_posterior = posterior[:, Teff_id]
    Teff_quantiles = np.quantile(Teff_posterior, q)

    carbon_isotope_posterior = 10.0**posterior[:, log_carbon_ratio_id]
    carbon_isotope_quantiles = np.quantile(carbon_isotope_posterior, q)

    ax_new = ax is None
    ax = ax or plt.gca()
# fig, ax = plt.subplots(1,1, figsize=(6,6))
    ax.errorbar(Teff_quantiles[1], carbon_isotope_quantiles[1], 
                xerr=[[Teff_quantiles[1]-Teff_quantiles[0]], [Teff_quantiles[2]-Teff_quantiles[1]]],
                yerr=[[carbon_isotope_quantiles[1]-carbon_isotope_quantiles[0]], [carbon_isotope_quantiles[2]-carbon_isotope_quantiles[1]]],
                fmt='o', label=label)
    if ax_new:
        ax.set_xlabel(r'$T_{\rm eff}$ (K)')
        ax.set_ylabel(r'$^{12}$C/$^{13}$C')
        
    return (Teff_quantiles, carbon_isotope_quantiles)
        

spirou_sample = {'880': [(3720, 4.72, 0.21), '17'],
                 '15A': [(3603, 4.86, -0.30), None],
                # '411': (3563, 4.84, 0.12), # TODO: double check this target
                # '752A': [(3558, 4.76, 0.10),None],
                '725A': [(3441, 4.87, -0.23),None],
                '725B': [(3345, 4.96, -0.30),None],
                '15B': [(3218, 5.07, -0.30),None],
                '905': [(2930, 5.04, 0.23),None],
}


targets = ['gl'+t for t in spirou_sample.keys()]

fig, ax = plt.subplots(1,1, figsize=(5,4), tight_layout=True)

Teff_list, C_ratio_list = [], []
for target in targets:
    Teff_t, C_ratio_t = main(target, ax=ax, label=target, run=spirou_sample[target[2:]][1])
    Teff_list.append(Teff_t)
    C_ratio_list.append(C_ratio_t)
    
    # check for binary companion, if so, plot a line connecting the two points
    if target.endswith('B'):
        target_A = target[:-1]+'A'
        if target_A not in targets:
            continue
        idx_A = targets.index(target_A)
        idx_B = targets.index(target)
        
        ax.plot([Teff_list[idx_A][1], Teff_list[idx_B][1]],
                [C_ratio_list[idx_A][1], C_ratio_list[idx_B][1]], 'k--')
        
        
    

solar = [89.0, 3.0] # review this value...
ax.axhspan(solar[0]-solar[1], solar[0]+solar[1], color='orange', alpha=0.3, label='Solar',lw=0)

ism = [69.0, 15.0]
ax.axhspan(ism[0]-ism[1], ism[0]+ism[1], color='green', alpha=0.2, label='ISM',lw=0)

ylim_min = 30.0
ylim_max = 180.0

ax.set_ylim(ylim_min, ylim_max)

ax.legend()
ax.set_xlabel(r'$T_{\rm eff}$ (K)')
ax.set_ylabel(r'$^{12}$C/$^{13}$C')
ax.legend(fontsize=8)
# plt.show()
fig_name = base_path + 'paper/figures/carbon_isotope.pdf'
fig.savefig(fig_name)
print(f'Figure saved as {fig_name}')
plt.close(fig)

