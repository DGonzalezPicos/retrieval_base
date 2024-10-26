from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters
from retrieval_base.chemistry import Chemistry
atomic_mass = {k:v[2] for k,v in Chemistry.species_info.items()}


import numpy as np
import matplotlib.pyplot as plt
# set fontsize to 16
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')
plt.rcParams.update({'font.size': 16})

import pathlib
import pickle
import corner
import pandas as pd
import json

path = pathlib.Path('/home/dario/phd/retrieval_base')
save_transparent_to = pathlib.Path('/home/dario/phd/presentations/october24/')

# out_path = path / 'HBDs'
out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')
# targets = dict(J1200='freechem_15', 
#             #    TWA28='freechem_12', 
#             TWA28='rev_2',
#                J0856='freechem_13'
#                ) 

targets = dict(J1200='final_full',
                TWA28='final_full',
                J0856='final_full',
                )
# invert order of targets
targets = dict(reversed(list(targets.items())))

colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')

fig, ax = plt.subplots(2,1, figsize=(6,6))
n_bins = 30
# solar_system = {'C/O': 0.54, '12C/13C': 89}
# add solar values with uncertainties
solar_system = {
        'C/O': (0.59, 0.08), # updated value from Asplund et al. (2021)
        # 'C/O': (0.55, 0.02), # uncertainty propagated from Asplund et al. (2009)
         '12C/13C': (89.3, 0.2), # reference terrestrial value from Meija et al. (2016)
        # '12C/13C': (93.48, 0.68), # Lyons+2018
         } # Asplund et al. (2009) from Scott et al. (2006) using CO lines
# propagate uncertainty for C/O where log(C/H) = 8.43 +- 0.05 and log(O/H) = 8.69 +- 0.05



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
    # chem.get_mass_fractions()
    # logg = params['log_g']
    C_O = chem.CO_posterior
    print(f'C/O = {C_O.mean():.2f} +- {C_O.std():.2f}')
    if hasattr(chem, 'mass_fractions_posterior'):
        posterior_12CO = chem.mass_fractions_posterior['CO_high'].mean(axis=-1) / atomic_mass['12CO']
        posterior_13CO = chem.mass_fractions_posterior['CO_36_high'].mean(axis=-1) / atomic_mass['13CO']
        chem.C12C13_posterior = posterior_12CO / posterior_13CO

    if hasattr(chem, 'VMRs_posterior'):
        chem.C12C13_posterior = chem.VMRs_posterior['12_13CO']
       
     
    # chem.C12C13_posterior = np.median(chem.mass_fractions_posterior['CO_high'] / chem.mass_fractions_posterior['CO_36_high'],axis=-1)
    
    # chem.C16OC18O_posterior = np.median(chem.mass_fractions_posterior['CO_high'] / chem.mass_fractions_posterior['CO_28'],axis=-1)
    # chem.H216OH218O_posterior = np.median(chem.mass_fractions_posterior['H2O_pokazatel_main_iso'] / chem.mass_fractions_posterior['H2O_181'],axis=-1)
    
    hist_args = {"color": colors[target], "alpha": 0.5, "fill": True, "edgecolor": "k",
                         "linewidth": 2.0, "histtype": "stepfilled", "density": True,
                         'bins': n_bins}

    C_O_range = (0.50, 0.70)
    C_ratio_range = (0., 250.)
    ax[0].hist(C_O,  range=C_O_range, **hist_args)
    
    no_fill_args = hist_args.copy()
    no_fill_args['fill'] = False
    no_fill_args['alpha'] = 0.7
    
    ax[0].hist(C_O,  range=C_O_range, **no_fill_args)

    ax[1].hist(chem.C12C13_posterior, range=C_ratio_range, **hist_args)
    ax[1].hist(chem.C12C13_posterior, range=C_ratio_range, **no_fill_args)
    # labels = ['Sun', 'ISM'] if i == 0 else [None, None]

    if len(ax) > 2:
        ax[2].hist(chem.C16OC18O_posterior, range=(10., 1220.), **hist_args)
        ax[3].hist(chem.H216OH218O_posterior, range=(0., 2220.), **hist_args)
    
    
labels = ['Solar', 'ISM']

for j, (key, val) in enumerate(solar_system.items()):
    # ax[j].axvline(val, color='magenta', ls='--', lw=2.5, alpha=0.8, label=labels[0])
    # ax[j].axvspan(val[0]-val[1], val[0]+val[1], color='magenta', alpha=0.1, label=labels[0])
    # scatter point with errorbars
    
    if j == 0:
        ax[j].errorbar(val[0], 14, xerr=val[1], color='magenta', 
                        ls='none', 
                        ms=5, marker='s',
                        lw=2.5, alpha=0.8, label=labels[0])

    else:
        ax[j].axvline(val[0], color='magenta', ls='-', lw=3.5, alpha=0.5, label=labels[0])
        

# ISM_C_ratio = (69, 6) # Wilson (1999)
ISM_C_ratio = (68, 15) # Milam et al. (2005)


kde = False

if kde:
    # create a gaussian kde to plot the measurent of the ISM C12/C13 ratio, use sns.kdeplot
    g_samples = np.random.normal(ISM_C_ratio[0], ISM_C_ratio[1], 5000)
    # use sns.kdeplot to plot the gaussian kde, fill the area under the curve
    import seaborn as sns
    # set the alpha of the fill to 0.6, make it smooth
    kdeplot_args = {'color': 'deepskyblue', 'lw': 2.5, 'ax': ax[1], 'ls': '--',
                    'bw_adjust': 3.0}
    sns.kdeplot(g_samples, label=labels[1],alpha=0.8, **kdeplot_args)
    sns.kdeplot(g_samples, fill=True, alpha=0.2,  **kdeplot_args)

# plot a data point with errorbars on the x-axis
ax[1].errorbar(ISM_C_ratio[0], 0.014, xerr=ISM_C_ratio[1], color='deepskyblue', 
                ls='none', 
                ms=7, marker='s',
                lw=3.0, alpha=0.9, label=labels[1])
# ax[1].axvline(68, color='deepskyblue', ls='--', lw=3.5, alpha=0.6, label=labels[1])
# ax[1].axvspan(ISM_C_ratio[0]-ISM_C_ratio[1], ISM_C_ratio[0]+ISM_C_ratio[1], color='deepskyblue', 
#               alpha=0.3, label=labels[1])
    # sns.kdeplot(g_samples, color='deepskyblue', lw=2.5, alpha=0.18, ax=ax[1], ls='--',
#             fill=True, , bw_adjust=3.0
# ax[1].hist(g_samples, range=C_ratio_range, bins=n_bins, color='deepskyblue', alpha=0.3, density=True, histtype='stepfilled')


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
ax[0].set(xlim=(0.50, 0.70))
ax[1].set(xlim=(0, 200))
ax[1].legend(frameon=False, prop={'weight':'bold', 'size': 16}, loc='upper right')
fig.tight_layout()
plt.show()
save= True
if save:
    fig.savefig(out_path / f'fig6_chemistry.pdf', bbox_inches='tight', dpi=300)
    if save_transparent_to is not None:
        fig.savefig(save_transparent_to / f'fig6_chemistry.png', dpi=300, transparent=True)
    print(f'Saved figure in {out_path / f"fig6_chemistry.pdf"}')
    plt.close()
