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
plt.style.use('/home/dario/phd/retsupjup/GQLupB/paper/gqlupb.mplstyle')
# pdf pages
from matplotlib.backends.backend_pdf import PdfPages
import copy

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import seaborn as sns
# import config_jwst as conf

path = pathlib.Path(af.get_path())
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')
config_file = 'config_jwst.txt'
target = 'TWA28'
w_set='NIRSpec'

cwd = os.getcwd()
if target not in cwd:
    nwd = os.path.join(cwd, target)
    print(f'Changing directory to {nwd}')
    os.chdir(nwd)


def get_bestfit_params(run):
    conf = Config(path=path, target=target, run=run)(config_file)        
        
    ret = Retrieval(
        conf=conf, 
        evaluation=False
        )

    bestfit_params, posterior = ret.PMN_analyze()
    bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
    return bestfit_params_dict
# run with both gratings
# run = 'lbl15_K2'
run = 'lbl15_G2G3_3'


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
                            t_env[-i,:], color=color, alpha=alpha, lw=0, 
                            label=label if i ==0 else '',
        )
    ax.plot(t_env[3,:], p, color=color, lw=0.5)
    
    if cf is not None:
        fill_cf = kwargs.pop('fill_cf', False)
        ax_cf = ax.twiny()
        ax_cf.plot(cf, p, color=color, lw=2.5, ls=':')
        ax_cf.set_xticks([])
        ax_cf.set_yticks([])
        # ax_cf.set_yticks([], minor=True)
        ax_cf.set_xlim(0, np.max(cf)*4.0)
        if fill_cf:
            ax_cf.fill_between(cf, p, color=color, alpha=0.05)
        
    return ax

colors = {'CRIRES': 'green', 'G235': 'navy', 'G235+G395': 'brown'}

fig, ax = plt.subplots(1,1,figsize=(5,5), tight_layout=True)
run_full = 'final_full'
p, t, cf = get_PT(path, target, run=run_full)
ax = plot_envelopes(p, t, ax=ax, cf=cf, color=colors['CRIRES'] , alpha=0.3, label=r'CRIRES$^{+}$', fill_cf=True)


p, t, cf = get_PT(path, target, run='lbl10_G2_3')
ax = plot_envelopes(p, t, ax=ax, cf=cf, color=colors['G235'], alpha=0.3, label='G235', fill_cf=True)

p, t, cf = get_PT(path, target, run)
ax = plot_envelopes(p, t, ax=ax, cf=cf, color=colors['G235+G395'], alpha=0.3, label='G235+G395', fill_cf=True)

Teff = (2382, 42) # Cooper+2024
label_teff = r'T$_{\rm eff}$' + f' = {Teff[0]} K'
ax.axvspan(Teff[0]-Teff[1], Teff[0]+Teff[1], color='k', alpha=0.1, label=label_teff, lw=0, zorder=-1)
# path_full = path / target / f'retrieval_outputs/{run_full}/test_data'
# PT = af.pickle_load(path_full / 'bestfit_PT.pkl')
# PT_file = path_full / 'envelopes/PT_envelopes.npy'
# np.save(PT_file, np.vstack([PT.pressure, PT.temperature_envelopes, PT.int_contr_em]))
# print(f' --> Saved {PT_file}')

ax.set(yscale='log', ylim=(np.max(p), np.min(p)), ylabel='Pressure / bar', xlabel='Temperature / K')
ax.set_xlim(None, 5000)
# make legend labels bold
# ax.legend(fontsize=8, 
ax.legend(prop={'size': 14, 'weight': 'bold'}, loc='upper right')
# plt.show()
fig_name = path_figures / f'{target}_PT_envelopes.pdf'
fig.savefig(fig_name, dpi=300)
# also save as transparent png
fig.savefig(fig_name.with_suffix('.png'), dpi=300, transparent=True)
print(f' --> Saved {fig_name}')
plt.close(fig)