from retrieval_base.retrieval import Retrieval
import retrieval_base.figures as figs
from retrieval_base.config import Config
from retrieval_base.auxiliary_functions import pickle_load, pickle_save, read_spirou_sample_csv
# import config_freechem as conf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import corner

import scienceplots
# reset to default
plt.style.use('default')
# plt.style.use(['latex-sans'])
plt.style.use(['sans'])
# enable latex
# plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "font.size": 11,
})

base_path = '/home/dario/phd/retrieval_base/'
nat_path = '/home/dario/phd/nat/figures/'
run ='fc5'
fs =14
# target = 'gl205'

def plot_target(target, ax, color, label=None, species=['OH','H2O'], cache=True, ax_ice=None):

    if target not in os.getcwd():
        os.chdir(base_path + target)
        
    outputs = pathlib.Path(base_path) / target / 'retrieval_outputs'
    config_file = 'config_freechem.txt'

    posterior_samples_file = outputs / run / outputs / f'posterior_samples.npy'
    posterior_labels_file = outputs / run / outputs / f'posterior_labels.npy'
    Chem_file = outputs / run / 'test_data/bestfit_Chem.pkl'
    PT_file = outputs / run / 'test_data/bestfit_PT.pkl'

    conf = Config(path=base_path, target=target, run=run)(config_file)


    if not Chem_file.exists() or not cache:
        ret = Retrieval(
                    conf=conf, 
                    evaluation=False,
                    )

        _, samples = ret.PMN_analyze()
        # samples = samples.T
        ret.get_PT_mf_envelopes(samples)
        ret.Chem.get_VMRs_posterior()
        pickle_save(Chem_file, ret.Chem)
        
    # oh = 
    Chem = pickle_load(Chem_file)
    PT = pickle_load(PT_file)
    p = Chem.pressure
    
    
        
    for i, s in enumerate(species):
        env = Chem.VMRs_envelopes[s]
        log_env = np.log10(env[:,len(env)//2])
        if log_env[-1] - log_env[0] > 3:
            print(f' Unconstrained {target}: {log_env[-1] - log_env[0]} for {species[i]}')
            ax[i].plot(env[1], p, color=color, lw=1, ls='--', label=label, zorder=-1) # 1sigma upper limits
        else:   
            ax[i].fill_betweenx(p, env[0], env[2], color=color, alpha=0.3, lw=0)
            ax[i].plot(env[1], p, color=color, lw=1)
        
    
    ax[0].set(yscale='log', xscale='log', ylim=(np.max(p), np.min(p)))
    ax[1].set(xscale='log', ylim=(np.max(p), np.min(p)))
    
    ax[-1].fill_betweenx(p, PT.temperature_envelopes[1], PT.temperature_envelopes[-2], color=color, alpha=0.5, lw=0)
    ax[-1].plot(PT.temperature_envelopes[3], p, color=color, lw=1)
    
    if ax_ice is not None:
        ice = PT.int_contr_em['spirou']
        # ice /= np.median(ice)
        ax_ice.fill_between(ice, p, color=color, alpha=0.01, lw=0)
        ax_ice.plot(ice, p, color=color, lw=1.0, alpha=0.8, zorder=color[0], ls='dotted')
    

    
species = ['OH','H2O']
n = len(species) + 1

fig, ax = plt.subplots(1,n, figsize=(7+n,3), sharey=True, gridspec_kw={'wspace': 0.04})
ax_ice = ax[-1].twiny()

df = read_spirou_sample_csv()
names = df['Star'].to_list()
ignore_targets = ['gl3622']

teff =  dict(zip(names, [float(t.split('+-')[0]) for t in df['Teff (K)'].to_list()]))
norm = plt.Normalize(3000.0, 3900.0)
cmap = plt.cm.coolwarm_r

for i, name in enumerate(names):
    print(f'---> {name}')

    target = name.replace('Gl ', 'gl')
    if target in ignore_targets:
        print(f'---> Skipping {target}...')
        continue
    color = cmap(norm(teff[name]))
    # print(f' Target = {target}, color = {color}')
    plot_target(target, ax, color, label=name, species=species, cache=True, ax_ice=ax_ice)
    
    # if i > 1:
    #     break # testing
        
ax[0].set(ylabel='Pressure / bar')

xlims = [ax[0].get_xlim(), ax[1].get_xlim()]
# xmin = min([x[0] for x in xlims])
xmin = 1e-8
xmax = max([x[1] for x in xlims])
ax[0].set_xlim(xmin, xmax)
ax[0].set_xlabel(r'$X$'+'(OH)')
ax[1].set_xlim(xmin, xmax)
ax[1].set_xlabel(r'$X$'+'(H'+r'$_2$'+r'$^{16}$'+'O)')

ax[-1].set(xlabel='Temperature / K', xlim=(1200.0, 6200.0))
ax_ice_max = ax_ice.get_xlim()[1]
ax_ice.set_xlim(0, ax_ice_max*4.0)
# remove xticks from ax_ice
ax_ice.set_xticks([])

# add plot labels: a,b,c
for i, axi in enumerate(ax):
    axi.text(0.90, 0.90, f'{chr(97+i)}', transform=axi.transAxes, fontsize=12, fontweight='bold')


# plt.show()
fig_name = nat_path + 'oh_h2o_pt.pdf'
fig.savefig(fig_name, bbox_inches='tight')
print(f' Saved {fig_name}')
plt.close(fig)