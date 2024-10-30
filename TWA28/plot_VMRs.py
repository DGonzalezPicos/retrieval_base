""" Plot VMR envelopes """
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import seaborn as sns

from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config
import config_jwst as conf

path = pathlib.Path('/home/dario/phd/retrieval_base')
config_file = 'config_jwst.txt'
target = 'TWA28'
cwd = os.getcwd()
if target not in cwd:
    print(f'Changing directory to {target}')
    os.chdir(target)

run = 'lbl15_G2_4' # contains G2+G3 (bad naming)
conf = Config(path=path, target=target, run=run)(config_file)        
            
chem = af.pickle_load(conf.prefix + 'data/bestfit_Chem.pkl')
p = chem.pressure

# pick colorpalette with more than 20 colors
colors = sns.color_palette('tab20', n_colors=len(chem.VMRs_envelopes))
fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
for k, v in chem.VMRs_envelopes.items():
    
    # check that is is constrained
    if abs(np.log10(v[0].mean()) - np.log10(v[-1].mean())) > 2:
        print(f' Skipping {k} because it is not constrained')
        continue
    color = colors.pop(0)
    # add white edge on the fill_betweenx
    ax.fill_betweenx(y=p,
                     x1=v[0], x2=v[-1],
                        label=k,
                        color=color,
                        alpha=0.5,
                        lw=0.5,
                        edgecolor='white',
                        # zorder=1,
                        )
    ax.plot(v[1], p, color=color, lw=1)
    # random offset for text label
    offset = np.random.uniform(0.35, 1.7)
    v_max = np.max(v)
    p_max = p[np.argmax(v[1])]
    ax.text(s=k, x=v_max, y=min(p_max*offset, 20 * offset), fontsize=8, color='black', transform=ax.transData)
    # add text label next to line
    # ax.text(v[0][len(v[0])//2], p[0], k, fontsize=8, color='black', transform=ax.transData)
    
ax.legend(ncol=4, loc=(0.0, 1.01))
ax.set(ylabel='Pressure [bar]', xlabel='VMR', xscale='log', yscale='log', ylim=(np.max(p),np.min(p)),
       xlim=(1e-9, 1e-2))
# plt.show()
fig_name = f'{conf.prefix}plots/VMR_envelopes.pdf'
fig.savefig(fig_name)
print(f' --> Saved {fig_name}')