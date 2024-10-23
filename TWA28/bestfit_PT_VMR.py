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

PT_envelopes_file = envelopes_dir / 'PT_envelopes.npy'
VMR_envelopes_file = envelopes_dir / 'VMR_envelopes.npy'
VMR_envelopes_species_file = envelopes_dir / 'VMR_envelopes_species.npy'

if all([f.exists() for f in [PT_envelopes_file, VMR_envelopes_file, VMR_envelopes_species_file]]):
    print(f' --> Found all files in {envelopes_dir}')
    PT_envelopes_data = np.load(PT_envelopes_file)
    pressure = PT_envelopes_data[0]
    temperature_envelopes = PT_envelopes_data[1:-1]
    int_contr_em = PT_envelopes_data[-1]
    
    VMR_envelopes_data = np.load(VMR_envelopes_file)
    VMR_envelopes_species = np.load(VMR_envelopes_species_file)
    sort = np.argsort([v[3] for v in VMR_envelopes_data])[::-1]
    
    VMR_envelopes_species = VMR_envelopes_species[sort]
    VMR_envelopes_data = VMR_envelopes_data[sort]
    
    VMR_envelopes ={k:v for k, v in zip(VMR_envelopes_species, VMR_envelopes_data)}
    
else:
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
    

    ret.Chem.get_VMRs_posterior()
    ret.Chem.get_VMRs_envelopes()
    # save VMR envelopes as npy file with pressure and VMR envelopes
    np.save(VMR_envelopes_file, np.array(list(ret.Chem.VMRs_envelopes.values())))
    np.save(VMR_envelopes_species_file, list(ret.Chem.VMRs_envelopes.keys()))
    print(f' --> Saved {VMR_envelopes_file}')
    print(f' --> Saved {VMR_envelopes_species_file}')
    
    # save PT envelopes as npy file with pressure and temperature envelopes
    ret.copy_integrated_contribution_emission()
    np.save(PT_envelopes_file, np.vstack([ret.PT.pressure, ret.PT.temperature_envelopes, ret.PT.int_contr_em['NIRSpec']]))
    print(f' --> Saved {PT_envelopes_file}')
    
    pressure = ret.PT.pressure
    temperature_envelopes = ret.PT.temperature_envelopes
    int_contr_em = ret.PT.int_contr_em['NIRSpec']
    
    VMR_envelopes_species = list(ret.Chem.VMRs_envelopes.keys())
    VMR_envelopes = ret.Chem.VMRs_envelopes
    
fig, (ax_pt, ax_vmr) = plt.subplots(1, 2, figsize=(12, 6), sharey=True,
                                    gridspec_kw={'wspace': 0.1})
ax_pt.set_facecolor('none')
ax_vmr.set_facecolor('none')

pt_color = 'royalblue'
vmr_colors = sns.color_palette('tab20', n_colors=len(VMR_envelopes_species))
# temperature envelopes
for i in range(3):
    ax_pt.fill_betweenx(pressure, 
                        temperature_envelopes[i,:],
                        temperature_envelopes[-1,:], color=pt_color, alpha=0.2, lw=0)
ax_pt.plot(temperature_envelopes[3,:], pressure, color=pt_color, lw=2)
ax_contr = ax_pt.twiny()
ax_contr.plot(int_contr_em, pressure, color='navy', lw=2, zorder=-1, alpha=0.6, ls=':')
ax_contr.fill_between(int_contr_em, pressure, color='navy', alpha=0.05)
# remove xticks
ax_contr.set_xticks([])
# ax_contr

ax_contr.set_xlim(0, 2*np.max(int_contr_em))
ax_pt.set(xlabel='Temperature / K', ylabel='Pressure / bar',
        yscale='log', ylim=(pressure.max(), pressure.min()))

color_i = 0

replace = {'H2O_181': 'H$_2$$^{18}$O',
           'H2O': 'H$_2$O',
           '13CO': '$^{13}$CO',
           'C18O': 'C$^{18}$O',
              '12CO': '$^{12}$CO',
              'CO2': 'CO$_2$',
              'H2S': 'H$_2$S',
              'C2H2': 'C$_2$H$_2$',
}
for s, species in enumerate(VMR_envelopes_species):
    if '/' in species:
        continue
    
    e = VMR_envelopes[species]
    loge = np.log10(e)
    
    if abs(loge[0] - loge[-1]) > 3:
        continue

    for i in range(3):
        ax_vmr.fill_betweenx(pressure,
                            e[i] * np.ones_like(pressure),
                            e[-i] * np.ones_like(pressure), alpha=0.2, color=vmr_colors[color_i], lw=0)
    label = replace.get(species, species)
    ax_vmr.plot(e[3]*np.ones_like(pressure), pressure, lw=2, label=label, color=vmr_colors[color_i])
    color_i += 1
print(f' detected {color_i} species')
ax_vmr.set(xlabel='VMR', yscale='log', xscale='log', xlim=(1e-11, 1e-2))
ax_vmr.legend(frameon=False, fontsize=12)
# plt.show()
fig.savefig(envelopes_dir / 'PT_VMR_envelopes.png', dpi=300, bbox_inches='tight', facecolor='none')
print(f' --> Saved {envelopes_dir / "PT_VMR_envelopes.png"}')
plt.close(fig)



    