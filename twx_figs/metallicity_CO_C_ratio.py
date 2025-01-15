"""PLot histograms for the metallicity and C/O and isotope ratios
"""
import numpy as np
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt
import pathlib
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config

path = af.get_path(return_pathlib=True)
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
config_file = 'config_jwst.txt'
target = 'TWA28'
w_set='NIRSpec'

runs = dict(
    TWA27A='lbl15_G1G2G3_fastchem_0',
    TWA28='lbl12_G1G2G3_freechem_1',
            )
colors = dict(TWA28={'data':'k', 'model':'orange'},
              TWA27A={'data':'#733b27', 'model':'#0a74da'})


def load_data(target, run, cache=True):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')
        
    conf = Config(path=path, target=target, run=run)(config_file) 
    CO_file = f'{conf.prefix}data/CO_posterior.npy'
    CH_file = f'{conf.prefix}data/CH_posterior.npy'
    # mass_fractions_posterior_file = f'{conf.prefix}data/mass_fractions_posterior.npy'
    VMRs_posterior_file = f'{conf.prefix}data/VMRs_posterior.npy'
    files = [CO_file, CH_file, VMRs_posterior_file]
    
    if cache and all(os.path.exists(file) for file in files):
        CO_posterior = np.load(CO_file)
        CH_posterior = np.load(CH_file)
        VMRs_posterior = np.load(VMRs_posterior_file, allow_pickle=True).item()
    else:
        
        ret = Retrieval(
                conf=conf, 
                evaluation=False
                )
        
        _, posterior = ret.PMN_analyze()
        ret.get_PT_mf_envelopes(posterior=posterior)
        CO_posterior = ret.Chem.CO_posterior
        CH_posterior = ret.Chem.FeH_posterior
        
        ret.Chem.get_VMRs_posterior()
        VMRs_posterior = ret.Chem.VMRs_posterior
        # mass_fractions_posterior_raw = ret.Chem.mass_fractions_posterior # keys are line species e.g. 'CO_high
        # mass_fractions_posterior = {k: mass_fractions_posterior_raw[v] for k, v in conf.line_species_dict.items()}  # rename keys to species names e.g. '12CO'
        
        
        np.save(CO_file, CO_posterior)
        np.save(CH_file, CH_posterior)
        np.save(VMRs_posterior_file, VMRs_posterior)
        for file in files:
            print(f'Saved {file}')
        
    return CO_posterior, CH_posterior, VMRs_posterior

# CO, CH are the MEAN values of the atmosphere

# three columns, one row, histograms with only the bottom axis
fig, ax = plt.subplots(1, 3, figsize=(10, 3), sharex='col')
axes = ax.flatten()
axes_dict = {'C/O': ax[0], '[C/H]': ax[1], '12C/13C': ax[2]}
# plot the histograms
bins = 20
alpha = 0.65

def plot_hist(ax, CO_posterior, CH_posterior, isotope_ratios, color, edge=True, density=True, label=None):
    
    htypes = ['stepfilled', 'step']
    for ht in htypes:
        # ec = 'k' if ht=='step' else None
        ec = 'k'
        label = label if ht == 'stepfilled' else None
        ax[0].hist(CO_posterior, bins=bins, alpha=alpha, color=color, density=density, histtype=ht, edgecolor=ec)
        ax[1].hist(CH_posterior, bins=bins, alpha=alpha, color=color, density=density, histtype=ht, edgecolor=ec)
        ax[2].hist(isotope_ratios['12C/13C'], bins=bins, alpha=alpha, color=color, label=label, density=density, histtype=ht, edgecolor=ec)

for t, target in enumerate(runs.keys()):
    CO_posterior, CH_posterior, VMRs_posterior = load_data(target, runs[target], cache=True)

    isotope_ratios = {'12C/13C': VMRs_posterior['12CO'] / VMRs_posterior['13CO'],
    }

    plot_hist(ax, CO_posterior, CH_posterior, isotope_ratios, colors[target]['model'], edge=True, density=True,
              label='TWA '+target.replace('TWA', ''))
    
# load CRIRES posteriors
file_crires = path / target / f'retrieval_outputs/final_full/test_data/bestfit_Chem.pkl'
chem_crires = af.pickle_load(file_crires)
crires = {
    'C/O' : chem_crires.VMRs_posterior['C/O'],
    '[C/H]' : chem_crires.VMRs_posterior['Fe/H'],
    '12C/13C' : chem_crires.VMRs_posterior['12_13CO']
    }

for key, axi in axes_dict.items():
    # plot hist
    axi.hist(crires[key], 
             bins=bins, 
             alpha=0.5, 
            #  label='CRIRES',
             density=True, 
             color=colors['TWA28']['model'],
             histtype='stepfilled', 
             edgecolor='k',
             ls='--',
             label='TWA 28 (CRIRES' + r'$\mathrm{^{+}}$)',
             )
    
    axi.hist(crires[key], 
             bins=bins, 
             alpha=0.7, 
            #  label='TWA 28 (CRIRES+)',
             density=True, 
             color=colors['TWA28']['model'],
             histtype='step', 
             edgecolor='k',
             ls='--',
             )



# solar value
solar = {'C/O':(0.59, 0.08),
        #  'CH':(0.0,
         '12C/13C':(93.5, 3.1),
         'color': 'magenta',
         'label': 'Solar'}
ism = {'12C/13C':(68, 14),
       'color': 'mediumseagreen',
       'label': 'ISM'}
# ax[0].errorbar(solar['C/O'][0], 0, xerr=solar['C/O'][1], color='k', linestyle='--', label='Solar')
# ax[1].axvline(solar['CH'][0], color='k', linestyle='--', label='Solar')

eb_args = dict(fmt='o', markersize=6, markeredgecolor='k', markeredgewidth=1.2)
for value in [solar, ism]:
    eb_args['color'] = value['color']
    eb_args['label'] = value['label']

    if value.get('C/O') is not None:
        ax[0].errorbar(value['C/O'][0], 55, xerr=value['C/O'][1], **eb_args)
    
    if value.get('12C/13C') is not None:
        ax[2].errorbar(value['12C/13C'][0], 0.05, xerr=value['12C/13C'][1], **eb_args)

axes[0].set_xlabel('C/O')
axes[1].set_xlabel('[C/H]')
axes[2].set_xlabel(r'$\mathrm{^{12}C}/\mathrm{^{13}C}$')
# add separation between columns of legend
leg = axes[2].legend(loc='upper right', ncol=2, frameon=False,
                     columnspacing=24)

# remove the top, right and left spines

def remove_spines(ax):
    spines = ['top', 'right', 'left']
    for spine in spines:
        ax.spines[spine].set_visible(False)
    # remove yticks
    ax.set_yticks([])

[remove_spines(axi) for axi in axes]

xlims = [(0.44, 0.69), (0.0, 1.0), (30, 150)]
for axi, xlim in zip(axes, xlims):
    axi.set_xlim(xlim)
# TODO: plot each target on a separate row, compare freechem and fastchem??
# plt.show()

# fig_name = path / 'twx_figs' / 'metallicity_CO_C_ratio.pdf'
fig_name = path_figures / 'metallicity_CO_C_ratio.pdf'
fig.savefig(fig_name, bbox_inches='tight')
print(f'Saved {fig_name}')