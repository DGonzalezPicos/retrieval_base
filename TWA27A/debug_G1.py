import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import json

from retrieval_base.retrieval import Retrieval
from retrieval_base.config import Config
import retrieval_base.auxiliary_functions as af

path = pathlib.Path(af.get_path(return_pathlib=True))
config_file = 'config_jwst.txt'

target = 'TWA27A'
w_set='NIRSpec'
run = 'lbl20_G1_2_freechem'

conf = Config(path=path, target=target, run=run)(config_file)

cwd = os.getcwd()
if target not in cwd:
    nwd = os.path.join(cwd, target)
    print(f'Changing directory to {nwd}')
    os.chdir(nwd)

d_spec  = af.pickle_load(conf.prefix+f'data/d_spec_{w_set}.pkl')
d_spec.fix_wave_nans()
pRT_atm = af.pickle_load(conf.prefix+f'data/pRT_atm_{w_set}.pkl')
pRT_atm.d_wave = d_spec.wave

run_bf = 'lbl15_G2G3_8'

# PT = af.pickle_load(path / target / 'retrieval_outputs' / run_bf / 'test_data' / 'bestfit_PT.pkl')
bestfit = json.load(open(path / target / f'retrieval_outputs/{run_bf}/test_data/bestfit.json'))
bestfit['params'].update({'gratings' : ['g140h'] *4})
# bestfit['params'].update({'log_g' : 6.0})
temperature = np.array(bestfit['temperature'])
pressure = np.array(bestfit['pressure'])
chem = af.pickle_load(path / target / 'retrieval_outputs' / run_bf / 'test_data' / 'bestfit_Chem.pkl')


# generate new spectrum with pRT_atm
m_spec = pRT_atm(mass_fractions=chem.mass_fractions,
                 temperature=temperature,
                 params=bestfit['params'],
                 get_contr=False,
                 get_full_spectrum=False,
                 calc_flux_fast=True,
                 )

old_hminus = chem.mass_fractions['H-'].copy()
Hminus = 6e-9
# chem.mass_fractions['H-'] = Hminus * np.ones_like(temperature)
chem.mass_fractions['H-'] = Hminus * np.ones_like(temperature)
m_spec_new = pRT_atm(mass_fractions=chem.mass_fractions,
                     temperature=temperature,
                     params=bestfit['params'],
                     get_contr=False,
                     get_full_spectrum=False,
                     calc_flux_fast=True,
                     )

n_orders = len(d_spec.wave)
fig, (ax, axh) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [3, 1]})

labels = ['fastchem', 'free']
colors = ['darkorange', 'darkgreen']
axh.plot(old_hminus, pressure, label=labels[0], color=colors[0])
axh.plot(chem.mass_fractions['H-'], pressure, label=labels[1], color=colors[1])
axh.set_ylabel('Pressure (bar)')
axh.set_xlabel('H-')
axh.set(ylim=(np.max(pressure), np.min(pressure)), yscale='log', xscale='log',
        xlim=(1e-12, 1e-5))
# axh.legend()

for i in range(n_orders):
    w = d_spec.wave[i,0]
    f = d_spec.flux[i,0]
    m = m_spec.flux[i,0]
    m_new = m_spec_new.flux[i,0]
    
    ax.plot(w, f, label='data' if i==0 else None,
            color='k')
    ax.plot(w, m, label=labels[0] if i==0 else None,
            alpha=0.7, color=colors[0])
    ax.plot(w, m_new, label=labels[1] if i==0 else None,
            alpha=0.7, color=colors[1])
    
    # ax.set_ylim(0, 1.1)
    # ax.set_xlim(5000, 6000)
    
ax.legend()
    
plt.show()