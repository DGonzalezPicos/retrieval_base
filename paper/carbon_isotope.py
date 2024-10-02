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

def main(target, label='', ax=None, run=None, **kwargs):
    if target not in os.getcwd():
        os.chdir(base_path + target)

    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    # find dirs in outputs
    # print(f' outputs = {outputs}')
    dirs = [d for d in outputs.iterdir() if d.is_dir() and 'sphinx' in d.name and '_' not in d.name]
    runs = [int(d.name.split('sphinx')[-1]) for d in dirs]
    print(f' {target}: Found {len(runs)} runs: {runs}')
    assert len(runs) > 0, f'No runs found in {outputs}'
    if run is None:
        run = 'sphinx'+str(max(runs))
    else:
        run = 'sphinx'+str(run)
        assert run in [d.name for d in dirs], f'Run {run} not found in {dirs}'
    # print('Run:', run)

    config_file = 'config_freechem.txt'
    conf = Config(path=base_path, target=target, run=run)(config_file)

    ret = Retrieval(
                conf=conf, 
                evaluation=False,
                )

    bestfit_params, posterior = ret.PMN_analyze()

    param_keys = list(ret.Param.param_keys)
    Teff_id = param_keys.index('Teff')
    # print(f' T_eff = {bestfit_params[Teff_id]}')
    log_carbon_ratio_id = param_keys.index('log_12CO/13CO')
    # print(f' log(12CO/13CO) = {bestfit_params[log_carbon_ratio_id]}')

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
    print(f' {target}: Teff = {Teff_quantiles[1]:.0f} +{Teff_quantiles[2]-Teff_quantiles[1]:.0f} -{Teff_quantiles[1]-Teff_quantiles[0]:.0f} K')
    print(f' {target}: log 12C/13C = {carbon_isotope_quantiles[1]:.2f} +{carbon_isotope_quantiles[2]-carbon_isotope_quantiles[1]:.2f} -{carbon_isotope_quantiles[1]-carbon_isotope_quantiles[0]:.2f}\n')
    # add black edge to points
    ax.errorbar(Teff_quantiles[1], carbon_isotope_quantiles[1], 
                xerr=[[Teff_quantiles[1]-Teff_quantiles[0]], [Teff_quantiles[2]-Teff_quantiles[1]]],
                yerr=[[carbon_isotope_quantiles[1]-carbon_isotope_quantiles[0]], [carbon_isotope_quantiles[2]-carbon_isotope_quantiles[1]]],
                fmt='o', label=label, alpha=0.9,
                    # markerfacecolor='none',  # Make the inside of the marker transparent (optional)
                 markeredgecolor='black', # Black edge color
                markeredgewidth=0.8,     # Thickness of the edge
                color=kwargs.get('color', 'k'),
    )
    

    if ax_new:
        ax.set_xlabel(r'$T_{\rm eff}$ (K)')
        ax.set_ylabel(r'$^{12}$C/$^{13}$C')
        
    return (Teff_quantiles, carbon_isotope_quantiles)
        

spirou_sample = {'880': [(3720, 4.72, 0.21, 6.868), '17'],
                 '15A': [(3603, 4.86, -0.30, 3.563), None],
                # '411': (3563, 4.84, 0.12), # TODO: double check this target
                '832': [(3590, 4.70, 0.06, 4.670),None],  # Tilipman+2021
                '752A': [(3558, 4.76, 0.10, 3.522),None], # Cristofari+2022
                '849':  [(3530, 4.78, 0.37, 8.803),None], # Cristofari+2022
                '725A': [(3441, 4.87, -0.23, 3.522),None],# Cristofari+2022
                '687': [(3413, 4.80, 0.10, 4.550),None], # Cristofari+2022
                '876' : [(3366, 4.80, 0.10, 4.672),None], # Moutou+2023, no measurement for logg, Z

                '725B': [(3345, 4.96, -0.30, 3.523),None],
                '699': [(3228.0, 5.09, -0.40, 1.827),None],
                '15B': [(3218, 5.07, -0.30, 3.561),None],
                '1151': [(3178, 4.71, -0.04, 8.043),None], # Lehmann+2024, I call it `gl` but it's `gj`
                '905': [(2930, 5.04, 0.23, 3.155),None],
}


targets = ['gl'+t for t in spirou_sample.keys()]
d_pc = {t: v[0][3] for t,v in spirou_sample.items()}

def distance_pc_to_parallax_mas(d_pc):
    return 1/d_pc * 1e3

parallax_mas = dict(zip(spirou_sample.keys(),
                        distance_pc_to_parallax_mas(np.array([v[0][3] for v in spirou_sample.values()]))))
print(parallax_mas)
# create colormap with distances in pc
norm = plt.Normalize(1.0, 9.0)
cmap = plt.cm.viridis


# replace gl1151 for gj1151
# targets = ['gj1151' if t == 'gl1151' else t for t in targets]

fig, ax = plt.subplots(1,1, figsize=(5,5), tight_layout=True)


Teff_dict = {}
C_ratio_dict = {}

for target in targets:
    
    color = cmap(norm(d_pc[target[2:]]))

    Teff_t, C_ratio_t = main(target, ax=ax, label=target, run=spirou_sample[target[2:]][1], color=color)
    if Teff_t is None:
        print(f'---> WARNING: Error with {target}, skipping...\n')
        continue
        
    # catch error and print it
    # except:
    #     # pr
    #     print(f'---> WARNING: Error with {target}, skipping...\n')
    #     continue
    Teff_dict[target] = Teff_t
    C_ratio_dict[target] = C_ratio_t
    
    # check for binary companion, if so, plot a line connecting the two points
    if target.endswith('B'):
        target_A = target[:-1]+'A'
        print(f'Looking for binary companion {target_A} from {target}...')
        if target_A not in targets:
            continue
        print(f'Found {target_A} from {target}...')
        # idx_A = targets.index(target_A)
        # idx_B = targets.index(target)
        Teff_A = Teff_dict[target_A][1]
        Teff_B = Teff_dict[target][1]
        
        C_ratio_A = C_ratio_dict[target_A][1]
        C_ratio_B = C_ratio_dict[target][1]
        
        ax.plot([Teff_A, Teff_B], [C_ratio_A, C_ratio_B], 'k--', lw=0.5)
        
        
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for color bar
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.03, aspect=20, location='right')
cbar.set_label('Distance (pc)')

solar = [89.0, 3.0] # review this value...
ax.axhspan(solar[0]-solar[1], solar[0]+solar[1], color='orange', alpha=0.3, label='Solar',lw=0)

ism = [69.0, 15.0]
ax.axhspan(ism[0]-ism[1], ism[0]+ism[1], color='brown', alpha=0.2, label='ISM',lw=0)

ylim_min = 30.0
ylim_max = 200.0

ax.set_ylim(ylim_min, ylim_max)

ax.legend(ncol=4, frameon=False, fontsize=8, loc=(0.0, 1.01))
ax.set_xlabel(r'$T_{\rm eff}$ (K)')
ax.set_ylabel(r'$^{12}$C/$^{13}$C')
# plt.show()
fig_name = base_path + 'paper/figures/carbon_isotope.pdf'
fig.savefig(fig_name)
print(f'Figure saved as {fig_name}')
plt.close(fig)

