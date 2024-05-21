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

path = pathlib.Path('/home/dario/phd/retrieval_base')
targets = dict(
                J1200='final_full',
                TWA28='final_full',
                J0856='final_full',
                )
targets = dict(reversed(list(targets.items())))

colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')

out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')

fig, ax = plt.subplots(1,1, figsize=(6,4))
n_bins = 30
o_ratio_range = (1.5, 3.5)


def linear_to_log(value, error):
    return np.log10(value), error / value / np.log(10)
solar = 525 # Lyon+2018
solar_err = 21

ism = 527 # wilson+1999
ism_err = 30

# propagate uncertainty in the log space
# log_solar = np.log10(solar)
# log_solar_err = solar_err / solar / np.log(10)
log_solar, log_solar_err = linear_to_log(solar, solar_err)
log_ism, log_ism_err = linear_to_log(ism, ism_err)

for i, (target, retrieval_id) in enumerate(targets.items()):
    data_path = path / f'{target}'
    print(data_path)
    
    # bestfit_params = 
    retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'
    assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'
    
    # load json file with bestfit parameters
    # with open(retrieval_path / 'test_data/bestfit.json', 'r') as f:
    #     bestfit_params = json.load(f)
        
    # equal_weighted_file = retrieval_path / 'test_post_equal_weights.dat'
    # posterior = np.loadtxt(equal_weighted_file)
    # posterior = posterior[:,:-1]
    
    # params = bestfit_params['params']
    chem = pickle.load(open(retrieval_path / 'test_data/bestfit_Chem.pkl', 'rb'))
    
    o_ratio = np.log10(chem.VMRs_posterior['H2_16_18O'])
    hist_args = {"color": colors[target], "alpha": 0.6, "fill": True, "edgecolor": "k",
                         "linewidth": 2.0, "histtype": "stepfilled", "density": True,
                         'bins': n_bins}
    
    ax.hist(o_ratio, range=o_ratio_range, label=target, **hist_args)
    
    hist_args_edge = hist_args.copy()
    hist_args_edge['fill'] = False
    ax.hist(o_ratio, range=o_ratio_range, **hist_args_edge)
    
    
ax.yaxis.set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.set(xlabel=xlabels.pop(0))

# ax.axvspan(log_solar-log_solar_err, log_solar+log_solar_err, color='magenta', alpha=0.5, label='Solar')
ax.axvline(log_solar, color='magenta', linestyle='--', label='Solar', lw=2.)
ax.axvspan(log_ism-log_ism_err, log_ism+log_ism_err, color='cyan', alpha=0.5, label='ISM')
# leg = ax.legend()
# flip order of first three labels
leg = ax.legend()
handles = leg.legendHandles
labels = [label.get_text() for label in leg.get_texts()]
sort = [2,1,0,3,4]

ax.legend(np.array(handles)[sort], np.array(labels)[sort], loc='upper right', frameon=False)


ax.set(xlabel=r'$\log {^{16}\rm O}/{^{18} \rm O}$')
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
          
fig.tight_layout()
fig.savefig(out_path / 'oxygen_isotope_ratio.pdf')