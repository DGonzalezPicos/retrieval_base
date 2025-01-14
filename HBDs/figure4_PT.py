from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters

import numpy as np
import matplotlib.pyplot as plt
# set fontsize to 16
# plt.rcParams.update({'font.size': 16})
plt.style.use('/home/dario/phd/retrieval_base/HBDs/my_science.mplstyle')

import pathlib
import pickle
import corner
import pandas as pd
import json
save_transparent_to = pathlib.Path('/home/dario/phd/presentations/october24/')

path = pathlib.Path('/home/dario/phd/retrieval_base')
# out_path = path / 'HBDs'
out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')

# targets = dict(J1200='freechem_16', 
#                TWA28='freechem_13', 
#                J0856='freechem_14'
#                )
# targets = dict(J1200='freechem_15', 
#                TWA28='freechem_12', 
#                J0856='freechem_13'
#                )
targets = dict(J1200='final_full',
                TWA28='final_full',
                J0856='final_full',
                )
colors = dict(J1200='royalblue', TWA28='seagreen', J0856='indianred')

fig, ax = plt.subplots(1,1, figsize=(6,6), constrained_layout=True)

for i, (target, retrieval_id) in enumerate(targets.items()):
    data_path = pathlib.Path('/home/dario/phd/retrieval_base') / f'{target}'
    print(data_path)
    
    # bestfit_params = 
    retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'
    assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'
    
    PT = pickle.load(open(retrieval_path / 'test_data/bestfit_PT.pkl', 'rb'))
    ax.fill_betweenx(PT.pressure, PT.temperature_envelopes[0], PT.temperature_envelopes[-1], color=colors[target], alpha=0.2)
    ax.fill_betweenx(PT.pressure, PT.temperature_envelopes[1], PT.temperature_envelopes[-2], color=colors[target], alpha=0.4)
    # ax.plot(PT.temperature, PT.pressure, color=colors[target], lw=2.5, label=target)
    # ax.plot(PT.temperature_posterior[-1,:], PT.pressure, color=colors[target], lw=2.5, label=target)
    ax.plot(PT.temperature_envelopes[3], PT.pressure, color=colors[target], lw=2.0, label=target)

    # plot integrated contribution function
    icf = np.load(retrieval_path / 'test_data/bestfit_int_contr_em_K2166.npy')
    print(f'shape of icf = {icf.shape}')
    ax_icf = ax.twiny()
    # make the zero on the right side of the axis
    ax_icf.plot(icf, PT.pressure, color=colors[target], lw=2.5, label='ICF', ls='-', alpha=0.7)
    ax_icf.fill_betweenx(PT.pressure, icf, 0., color=colors[target], alpha=0.2)
    # ax_icf.invert_xaxis()
    ax_icf.set(xlim=(0, 4.5*icf.max()))
    # remove xticks from ax_icf
    ax_icf.set_xticks([])
        
ax.set(ylim=(PT.pressure.max(), PT.pressure.min()), ylabel='Pressure [bar]', xlabel='Temperature [K]',
        yscale='log')
ax.set_xlim(1000., 5000.)
# increase padding of x-axis labels
ax.tick_params(axis='x', pad=10)

# adjust linewidth of handle
ax.legend(frameon=False, prop={'weight':'bold', 'size': 20}, handlelength=2)
# increase linewidth of handles in legend and alpha value
for legobj in ax.legend_.legendHandles:
    legobj.set_linewidth(3.0)
    legobj.set_alpha(0.9)
# remove minor yticks
ax.minorticks_off()
plt.show()

save = True
if save:
    fig.savefig(out_path / f'fig4_bestfit_PT.pdf', bbox_inches='tight', dpi=300)
    print(f'Saved figure in {out_path / f"fig4_bestfit_PT.pdf"}')
    if save_transparent_to is not None:
        fig.savefig(save_transparent_to / f'fig4_bestfit_PT.png', dpi=300, transparent=True)
        print(f'Saved transparent figure in {save_transparent_to / f"fig4_bestfit_PT.png"}')