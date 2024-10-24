""" 
Compare C/O posteriors of different runs, also CRIRES+

date: 2024-10-24
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

envelopes_dir = path / target / f'retrieval_outputs/{run}/test_data'/ 'envelopes'
envelopes_dir.mkdir(parents=True, exist_ok=True)

# VMR_envelopes_file = envelopes_dir / 'VMR_envelopes.npy'
# VMR_envelopes_species_file = envelopes_dir / 'VMR_envelopes_species.npy'

def get_CO(path, target, run, CO_key='C/O'):
    
    data_dir = path / target / f'retrieval_outputs/{run}/test_data'

    VMR_posteriors_file = data_dir / 'VMR_posteriors.npy'
    VMR_labels_file = data_dir / 'VMR_labels.npy'


    if all([f.exists() for f in [VMR_labels_file, VMR_posteriors_file]]):
        print(f' --> Found {VMR_posteriors_file}')
        VMR_labels = np.load(VMR_labels_file)
        VMR_posteriors_values = np.load(VMR_posteriors_file)
        VMRs_posteriors = dict(zip(VMR_labels, VMR_posteriors_values))
        
        
    else:
        conf = Config(path=path, target=target, run=run)(config_file)        
            
        ret = Retrieval(
            conf=conf, 
            evaluation=False
            )

        bestfit_params, posterior = ret.PMN_analyze()
        print(f' posterior.shape = {posterior.shape}')
        bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))

        print(f' --> Best-fit parameters: {bestfit_params_dict}')
        bestfit_params = np.array(list(bestfit_params_dict.values()))

        ret.evaluate_model(bestfit_params)
        ret.evaluation = True
        # ret.PMN_lnL_func()
        ret.get_PT_mf_envelopes(posterior)
        

        ret.Chem.get_VMRs_posterior(save_to=data_dir)
        
        VMRs_posteriors = ret.Chem.VMRs_posterior
        
    assert 'C/O' in VMRs_posteriors.keys(), f'C/O not in VMRs_posteriors.keys(), {VMRs_posteriors.keys()}'
    return VMRs_posteriors['C/O']

def hist(CO, ax, label=None, **kwargs):
    fill = kwargs.pop('fill', False)
    ax.hist(CO, bins=20, histtype='step', lw=2, label=label if (not fill) else '',
            density=True, **kwargs)
    if fill:
        ax.hist(CO, bins=20, histtype='stepfilled', lw=2, alpha=0.5, density=True,label=label,**kwargs)
    return ax


fig, ax = plt.subplots(1,1,figsize=(4,4), tight_layout=True)
colors = {'CRIRES': 'green', 'G235': 'navy', 'G235+G395': 'brown'}
run = {'CRIRES': 'final_full', 'G235': 'lbl10_G2_3', 'G235+G395': 'lbl15_G2G3_3'}
# plot 1D histograms

for dataset in list(colors.keys()):
    if dataset == 'CRIRES':
        print(f' --> Skipping {dataset}') # CRIRES not implemented yet...
        continue
        
    hist(get_CO(path, target, run[dataset]), ax, label=dataset, color=colors[dataset], fill=True)
    

# crires file
crires_file = path / target / f'retrieval_outputs/final_full/test_data/bestfit_Chem.pkl'
crires = af.pickle_load(crires_file)
hist(crires.CO_posterior, ax, label='CRIRES', color=colors['CRIRES'], fill=True)

ax.set(xlim=(0.55, 0.68), xlabel='C/O')
# remove y-axis and yticks
ax.set_yticks([])
# remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# remove xticks from top axis
ax.tick_params(axis='x', top=False)
# remove x-axis
ax.legend(frameon=False)
fig_name = path / target / f'retrieval_outputs/{run["G235+G395"]}/test_data/envelopes/CO_posteriors.png'
fig.savefig(fig_name, dpi=300, transparent=True)
plt.show()