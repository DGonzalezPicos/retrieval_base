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
from tabulate import tabulate

path = af.get_path(return_pathlib=True)
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
config_file = 'config_jwst.txt'
w_set='NIRSpec'

runs = dict(
    # TWA27A=['lbl11_G1G2G3_fastchem_0'],
    TWA28=[
        # 'lbl11_G1G2G3_fastchem_0', 
           ('lbl11_G2G3_fastchem_GP_0', 'G2+G3 (GP)'), 
           ('lbl11_G2G3_fastchem_0', 'G2+G3'),
           ],
            )
colors = dict(TWA28={'data':'k', 
                     'model':['brown', 'darkgreen', 'darkblue'], 
                    #  'model_labels':['G1+G2+G3', 'G2+G3', 'G2'],
                     'crires': 'orange'},
              TWA27A={'data':'#733b27',
                      'model':['#0a74da'],
                      'model_labels':['G1+G2+G3']
                      })

def load_data(target, run, cache=True):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')
        
    conf = Config(path=path, target=target, run=run)(config_file) 
    
    PT_VMRs_COH_file = f'{path}/{target}/retrieval_outputs/{run}/test_data/temperature_VMRs_COH.npy'

    log_g_posterior_file = f'{conf.prefix}data/log_g_posterior.npy'
    files = [PT_VMRs_COH_file, log_g_posterior_file]
    
    posterior = None
    ret = Retrieval(
                conf=conf, 
                evaluation=False
                )
    if not cache or not all(os.path.exists(file) for file in files):
        
        
        _, posterior = ret.PMN_analyze()
        log_g_index = list(ret.Param.param_keys).index('log_g')
        log_g_posterior = posterior[:,log_g_index]
        np.save(log_g_posterior_file, log_g_posterior)
        print(f'Saved {log_g_posterior_file}')
        
    
    log_g_posterior = np.load(log_g_posterior_file)
        
    _ = ret.get_PT_mf_envelopes(posterior=posterior, n_samples=None, cache=cache)
        
    return ret.Chem, log_g_posterior

# three columns, one row, histograms with only the bottom axis
fig, ax = plt.subplots(1, 4, figsize=(10, 3), sharex='col')
axes = ax.flatten()
axes_dict = {'C/O': ax[0], '[C/H]': ax[1], '12C/13C': ax[2], 'log_g': ax[3]}
# plot the histograms 
bins = 20
alpha = 0.65

def plot_hist(ax, CO_posterior, CH_posterior, isotope_ratios, color, log_g_posterior=None, edge=True, density=True, label=None,
              fill=True):
    
    htypes = ['step']
    if fill:
        htypes.append('stepfilled')
        htypes = htypes[::-1]
        
    for ht in htypes:
        # ec = 'k' if ht=='step' else None
        ec = 'k'
        label = label if ht == 'stepfilled' else None
        ax[0].hist(CO_posterior, bins=bins, alpha=alpha, color=color, density=density, histtype=ht, edgecolor=ec)
        ax[1].hist(CH_posterior, bins=bins, alpha=alpha, color=color, density=density, histtype=ht, edgecolor=ec)
        ax[2].hist(isotope_ratios['12C/13C'], bins=bins, alpha=alpha, color=color, label=label, density=density, histtype=ht, edgecolor=ec)
        
        if len(ax) > 3:
            assert log_g_posterior is not None, 'log_g_posterior is required'
            ax[3].hist(log_g_posterior, bins=bins, alpha=alpha, color=color, label=label, density=density, histtype=ht, edgecolor=ec)

def print_quantiles(target, run, log_g_posterior, CO_posterior, CH_posterior, isotope_ratios, q):
    headers = ["Parameter", "3σ Lower", "1σ Lower", "Median", "1σ Upper", "3σ Upper"]
    
    # Format quantiles to two decimal places
    def format_quantiles(quantiles):
        return [f"{q:.2f}" for q in quantiles]
    
    data = [
        ["log_g", *format_quantiles(af.quantiles(log_g_posterior, q=q))],
        ["C/O", *format_quantiles(af.quantiles(CO_posterior, q=q))],
        ["[C/H]", *format_quantiles(af.quantiles(CH_posterior, q=q))],
        ["12C/13C", *format_quantiles(af.quantiles(isotope_ratios["12C/13C"], q=q))]
    ]
    print(f' ** {target} {run} **')
    print(tabulate(data, headers=headers, tablefmt="pretty"))

for t, target in enumerate(runs.keys()):
    target_runs = list(np.atleast_1d(runs[target]))
    for r, run_name in enumerate(target_runs):
        run = run_name[0]
        label = run_name[1]
        
        chem, log_g_posterior = load_data(target, run, cache=True)
        
        CO_posterior = np.mean(chem.COH_posterior['C'] / chem.COH_posterior['O'], axis=-1)
        CH_posterior = af.solar_metallicity(np.mean(chem.COH_posterior['C'], axis=-1), 
                                            np.mean(chem.COH_posterior['H'], axis=-1))
        isotope_ratios = {'12C/13C': np.mean(chem.VMRs_posterior['12CO'] / chem.VMRs_posterior['13CO'], axis=-1),
        }
        
        # Define quantiles
        q = [0.5-0.997/2, 0.5-0.68/2, 0.5, 
             0.5+0.68/2, 0.5+0.997/2
             ]
        
        # Print quantiles in a formatted table
        print_quantiles(target, run, log_g_posterior, CO_posterior, CH_posterior, isotope_ratios, q)
        
        plot_hist(ax, CO_posterior, CH_posterior, isotope_ratios, colors[target]['model'][r], log_g_posterior=log_g_posterior, edge=True, density=True,
                # label=colors[target]['model_labels'][r], fill=(r==0))
                label='TWA ' + target.replace('TWA', '') + f"\n({label})",
                fill=True)
    
# load CRIRES posteriors
if target == 'TWA28':
    file_crires = path / target / f'retrieval_outputs/final_full/test_data/bestfit_Chem.pkl'
    chem_crires = af.pickle_load(file_crires)

    log_g_crires_file = path / target / f'retrieval_outputs/final_full/test_data/log_g_posterior.npy'
    if os.path.exists(log_g_crires_file):
        log_g_crires = np.load(log_g_crires_file)
    else:
        import pymultinest

        conf = Config(path=path, target=target, run='final_full')('config_freechem.txt')
        analyzer = pymultinest.Analyzer(
                    n_params=len(conf.free_params), 
                    outputfiles_basename=conf.prefix
                    )
        posterior = analyzer.get_equal_weighted_posterior()
        posterior = posterior[:,:-1]
        log_g_index = list(conf.free_params).index('log_g')
        log_g_crires = posterior[:,log_g_index]
        print(f'log_g_crires shape: {log_g_crires.shape}')
        np.save(log_g_crires_file, log_g_crires)
        print(f'Saved {log_g_crires_file}')
    crires = {
        'C/O' : chem_crires.VMRs_posterior['C/O'],
        '[C/H]' : chem_crires.VMRs_posterior['Fe/H'],
        '12C/13C' : chem_crires.VMRs_posterior['12_13CO'],
        'log_g' : log_g_crires
        }

    for key, axi in axes_dict.items():
        # if key == 'log_g':
        #     continue
        # plot hist
        axi.hist(crires[key], 
                bins=bins, 
                alpha=0.5, 
                #  label='CRIRES',
                density=True, 
                color=colors['TWA28']['crires'],
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
                color=colors['TWA28']['crires'],
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
axes[3].set_xlabel('log g')
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

xlims = [(0.40, 0.69), (-1, 1.0), (30, 150), (2.8, 4.5)]
for axi, xlim in zip(axes, xlims):
    axi.set_xlim(xlim)
# TODO: plot each target on a separate row, compare freechem and fastchem??
# plt.show()

# fig_name = path / 'twx_figs' / 'metallicity_CO_C_ratio.pdf'
fig_name = path_figures / 'metallicity_CO_C_ratio.pdf'
fig.savefig(fig_name, bbox_inches='tight')
print(f'Saved {fig_name}')
plt.close('all')