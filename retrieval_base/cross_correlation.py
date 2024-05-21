import numpy as np
import matplotlib.pyplot as plt
import pathlib

import os
cwd = os.getcwd()
if 'dgonzalezpi' in cwd:
    path = '/home/dgonzalezpi/retrieval_base/'
if 'dario' in cwd:
    path = '/home/dario/phd/retrieval_base/'
    
from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.config import Config
import retrieval_base.figures as figs
# import config_freechem as conf


out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')

config_file = 'config_freechem.txt'
target = 'J0856'
if target not in path:
    os.chdir(path+target)

# run = 'final_no13CO' # important to set this to the correct run 
run = 'final_full' # important to set this to the correct run
    
conf = Config(path=path, target=target, run=run)
conf(config_file)

ret = Retrieval(
    conf=conf, 
    evaluation=True,
    plot_ccf=True,
    )
bestfit_params, _ = ret.PMN_analyze()
bestfit_params_dict = dict(zip(ret.Param.param_keys, bestfit_params))
ret.Param.params.update(bestfit_params_dict)

for i, key_i in enumerate(ret.Param.param_keys):
    # Update the Parameters instance
    ret.Param.params[key_i] = bestfit_params[i]

    if key_i.startswith('log_'):
        ret.Param.params = ret.Param.log_to_linear(ret.Param.params, key_i)

# Update the parameters
ret.Param.read_PT_params(cube=None)
ret.Param.read_uncertainty_params()
ret.Param.read_chemistry_params()
# ret.Param.read_cloud_params()

species = {
            '13CO': 'CO_36_high',
           'H2O_181': 'H2O_181_HotWat78',
           }
ret.get_species_contribution(species=list(species.values()))
ret.PMN_lnL_func()

w_set = 'K2166'
fig, ax = plt.subplots(2,1, figsize=(6,6), 
                       sharex=True, 
                       tight_layout=True)

for h, species_h in enumerate(species.keys()):
    figs.plot_ax_CCF(
        ax=ax[h],
        d_spec=ret.d_spec[w_set],
        m_spec=ret.m_spec[w_set],
        pRT_atm=ret.pRT_atm_broad[w_set],
        m_spec_wo_species=ret.m_spec_species[w_set][species_h],
        pRT_atm_wo_species=ret.pRT_atm_species[w_set][species_h],
        LogLike=ret.LogLike[w_set],
        Cov=ret.Cov[w_set],
        rv=np.arange(-1000, 1000+1, 1),
        color=ret.Chem.species_plot_info[species_h][0],
        species_h=species_h,
        prefix=conf.prefix,
        )
    ax[h].set_title(species_h)
    
outfig = conf.prefix+ f'plots/CCF.pdf'
fig.savefig(outfig)
plt.show()