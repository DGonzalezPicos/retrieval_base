""" 
Generate a model for G235+G395 with the best-fit parameters from G235 alone 
Inspect the residuals, disk emission?

date: 2024-09-17
"""
import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
# increase font size
# plt.style.use('/home/dario/phd/retsupjup/GQLupB/paper/gqlupb.mplstyle')
# pdf pages
from matplotlib.backends.backend_pdf import PdfPages
import copy

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import seaborn as sns
# import config_jwst as conf

path = pathlib.Path(af.get_path())
# path_figures = pathlib.Path('/home/dario/phd/retrieval_base/twx_figs')
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')

config_file = 'config_jwst.txt'
# target = 'TWA28'
w_set='NIRSpec'


def check_dir(target):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')

def get_bestfit_params(target,run):
    
    check_dir(target)
    
    conf = Config(path=path, target=target, run=run)(config_file)        
        
    ret = Retrieval(
        conf=conf, 
        evaluation=False
        )

    bestfit_params, _ = ret.PMN_analyze()
    bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
    return bestfit_params_dict
# run with both gratings
# run = 'lbl15_K2'
# run = 'lbl15_G2G3_3'
# run = 'lbl12_G1G2G3_fastchem_1'

targets = dict(TWA28={'run': 'lbl12_G1G2G3_fastchem_1', 'teff': (2382, 42)},
                TWA27A={'run': 'lbl15_G1G2G3_fastchem_0', 'teff': (2430, 20)})


def get_PT(path, target, run):
    
    envelopes_dir = path / target / f'retrieval_outputs/{run}/test_data'/ 'envelopes'
    envelopes_dir.mkdir(parents=True, exist_ok=True)

    PT_envelopes_file = envelopes_dir / 'PT_envelopes.npy'
    
    if PT_envelopes_file.exists():
        print(f' --> Found {PT_envelopes_file}')
        PT_envelopes_data = np.load(PT_envelopes_file)
        pressure = PT_envelopes_data[0]
        temperature = PT_envelopes_data[1:-1]
        icf = PT_envelopes_data[-1]
        print(f' --> Loaded PT_envelopes.npy with shape {temperature.shape}')
    else:
        print(f' Calculating PT envelopes for {run}')
        check_dir(target)
        conf = Config(path=path, target=target, run=run)(config_file)        
            
        ret = Retrieval(
            conf=conf, 
            evaluation=False
            )

        bestfit_params, posterior = ret.PMN_analyze()
        bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
        # bestfit_params_dict['log_SiO'] = -6.0

        print(f' --> Best-fit parameters: {bestfit_params_dict}')
        bestfit_params = np.array(list(bestfit_params_dict.values()))

        ret.evaluate_model(bestfit_params)
        ret.evaluation = True
        ret.PMN_lnL_func()
        ret.get_PT_mf_envelopes(posterior)
        # save PT envelopes as npy file with pressure and temperature envelopes
        ret.copy_integrated_contribution_emission()
        np.save(PT_envelopes_file, np.vstack([ret.PT.pressure, ret.PT.temperature_envelopes, ret.PT.int_contr_em['NIRSpec']]))
        print(f' --> Saved {PT_envelopes_file}')
        return ret.PT.pressure, ret.PT.temperature_envelopes, ret.PT.int_contr_em['NIRSpec']
        
    return pressure, temperature, icf


def plot_envelopes(p, t_env, ax=None, cf=None, **kwargs):
    
    ax = ax or plt.gca()
    assert len(t_env.shape) > 1, f'Expected 2D array, got {t_env.shape}'
    assert t_env.shape[0] == 7, f'Expected 7 envelopes, got {t_env.shape[0]}'
    
    color = kwargs.pop('color', 'brown')
    alpha = kwargs.pop('alpha', 0.2)
    label = kwargs.pop('label', '')
    
    for i in range(3):
        ax.fill_betweenx(p, 
                            t_env[i,:],
                            t_env[-(i+1),:], color=color, alpha=alpha, lw=0, 
                            label=label if i ==0 else '',
                            # ls='--',
                            )
    ax.plot(t_env[3,:], p, color=color, lw=0.5, ls=kwargs.pop('ls', '-'), alpha=0.75)

    if cf is not None:
        fill_cf = kwargs.pop('fill_cf', False)
        ax_cf = ax.twiny()
        ls = kwargs.pop('ls_cf', ':')
        lw = kwargs.pop('lw_cf', 2.5)
        ax_cf.plot(cf, p, color=color, lw=lw, ls=ls, alpha=0.75)
        ax_cf.set_xticks([])
        ax_cf.set_yticks([])
        # ax_cf.set_yticks([], minor=True)
        ax_cf.set_xlim(0, np.max(cf)*4.5)
        if fill_cf:
            ax_cf.fill_between(cf, p, color=color, alpha=0.05)
        
    return ax

# colors = {'CRIRES': 'green', 'G235': 'navy', 'G235+G395': 'brown'}
colors = dict(TWA28={'data':'k', 'model':'darkorange', 'crires': 'orange'},
              TWA27A={'data':'#733b27', 'model':'#0a74da'})

fig, ax = plt.subplots(1,1,figsize=(4,4), tight_layout=True)

def plot_crires(ax):
    run_full = 'final_full'
    p, t, cf = get_PT(path, 'TWA28', run=run_full)
    ax = plot_envelopes(p, t, ax=ax, cf=cf, color=colors['TWA28']['crires'] , alpha=0.2, label='TWA 28\n' + r'(CRIRES$^{+}$)', fill_cf=True,
                        ls='--', ls_cf='--', lw_cf=1.0)


for target in ['TWA28', 'TWA27A']:
    Teff = targets[target]['teff']
    # label_teff = r'T$_{\rm eff}$' + f' = {Teff[0]} K'
    label_teff = f'{Teff[0]:.0f} K'
    # ax.axvspan(Teff[0]-Teff[1], Teff[0]+Teff[1], color=colors[target]['model'], alpha=0.3, label=label_teff, lw=0, zorder=-1)
    ax.axvline(Teff[0], color=colors[target]['model'], ls=':', lw=2, zorder=-10, alpha=0.6, label=label_teff)
    
    if target == 'TWA28':
        plot_crires(ax)
    
    p, t, cf = get_PT(path, target, targets[target]['run'])
    ax = plot_envelopes(p, t, ax=ax, cf=cf, color=colors[target]['model'], alpha=0.3, fill_cf=True,
                        label='TWA ' + target.replace('TWA', ''),
                        ls_cf='-', lw_cf=1.0)
    
    



ax.set(yscale='log', ylim=(np.max(p), np.min(p)), ylabel='Pressure / bar', xlabel='Temperature / K')
ax.set_xlim(None, 5000)
# make legend labels bold
# ax.legend(fontsize=8, 
ax.legend(prop={'size': 14, 'weight': 'bold'}, loc='upper right')
# make the order of legend reverse
handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(handles[::-1], labels[::-1], prop={'size': 10, 'weight': 'bold'}, loc='upper right',
          frameon=False, ncol=1)

# for lh in leg.get_lines():
#     lh.set_alpha(1.0)
    
for p, patch in enumerate(leg.get_patches()):
    # print(patch)
    patch.set_alpha(0.55)
    # add edge to patch
    if p == 2:
        patch.set_edgecolor('orange')
        patch.set_linewidth(0.95)
        patch.set_linestyle('dashed')

# plt.show()
fig_name = path_figures / f'fig_PTs.pdf'
fig.savefig(fig_name, bbox_inches='tight')
# also save as transparent png
# fig.savefig(fig_name.with_suffix('.png'), dpi=300, transparent=True)
print(f' --> Saved {fig_name}')
plt.close(fig)