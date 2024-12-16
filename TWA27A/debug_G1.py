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
run = 'lbl20_G1_3'

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

chem = af.pickle_load(path / target / 'retrieval_outputs' / run_bf / 'test_data' / 'bestfit_Chem.pkl')
# generate new spectrum with pRT_atm
m_spec = pRT_atm(mass_fractions=chem.mass_fractions,
                 temperature=np.array(bestfit['temperature']) + 600.0,
                 params=bestfit['params'],
                 get_contr=False,
                 get_full_spectrum=False,
                 calc_flux_fast=True,
                 )


n_orders = len(d_spec.wave)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

for i in range(n_orders):
    w = d_spec.wave[i,0]
    f = d_spec.flux[i,0]
    m = m_spec.flux[i,0]
    
    ax.plot(w, f, label='data', color='k')
    ax.plot(w, m, label='model', alpha=0.7)
    # ax.set_ylim(0, 1.1)
    # ax.set_xlim(5000, 6000)
    
plt.show()