from retrieval_base.retrieval import pre_processing, Retrieval
from retrieval_base.parameters import Parameters
from retrieval_base.auxiliary_functions import pickle_load

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

import config_freechem_15 as conf

import os
# change cwd to the directory of this script
path = pathlib.Path(__file__).parent.absolute()
os.chdir(path)



path = pathlib.Path('/home/dario/phd/retrieval_base')
# out_path = path / 'HBDs'
out_path = pathlib.Path('/home/dario/phd/Hot_Brown_Dwarfs_Retrievals/figures/')

targets = dict(J1200='freechem_15'
               )

target = 'J1200'
retrieval_id = targets[target]


data_path = pathlib.Path('/home/dario/phd/retrieval_base') / f'{target}'
print(data_path)


# bestfit_params = 
retrieval_path = data_path / f'retrieval_outputs/{retrieval_id}'
assert retrieval_path.exists(), f'Retrieval path {retrieval_path} does not exist.'
wave = np.load(f'{conf.prefix}data/d_spec_wave_K2166.npy')


generate_m_spec_species = False
if generate_m_spec_species:
    ret = Retrieval(
                conf=conf, 
                evaluation=True,
                plot_ccf=True,
                )
                
    ret.PMN_callback_func(
        n_samples=None, 
        n_live=None, 
        n_params=None, 
        live_points=None, 
        posterior=None, 
        stats=None,
        max_ln_L=None, 
        ln_Z=None, 
        ln_Z_err=None, 
        nullcontext=None
        )
    
m_spec = pickle_load(f'{conf.prefix}data/bestfit_m_spec_K2166.pkl')
m_spec_species = pickle_load(f'{conf.prefix}data/m_spec_species.pkl')['K2166']
species = list(m_spec_species.keys())
my_species = {'H2O': '#2563f4', # blue
              '12CO': '#0BDA51', #'limegreen'
              '13CO': '#dc2c40', # red
              }

order = 1
fig, ax = plt.subplots(len(my_species), 1, figsize=(16, 6),
                       sharex=True,
                       gridspec_kw=dict(hspace=0.06)
)

flux_factor = 1e15
for i, species_i in enumerate(my_species.keys()):
    print(f' Plotting {species_i}')
    xlim = [wave[order,:].min()-0.2, wave[order,:].max()+2.5]
    ax[i].set_xlim(xlim)
    
    
    for det in range(3):
        label = f'{species_i}' if det == 0 else None
        ax_list = [ax[i-1], ax[i]] if species_i == '13CO' else [ax[i]]
        for axi in ax_list:
            res = flux_factor*(m_spec.flux[order,det] - m_spec_species[species_i].flux[order, det])
            res[np.abs(res)<1e-6] = 0.0
            axi.plot(wave[order,det], res,
                    alpha=0.7, label=label, color=my_species[species_i])
        
        
        ax[i].legend(frameon=True, loc='lower right', prop={'weight':'bold', 'size': 20})
ax[-1].set_xlabel('Wavelength [nm]')

# one y-label for all subplots
fig.text(0.055, 0.5, 'Flux [10$^{-15}$ erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$]', va='center', rotation='vertical')

outfig = f'{conf.prefix}plots/m_spec_species_order{order}.pdf'
fig.savefig(outfig)
fig.savefig(outfig.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f'Figure saved as {outfig}')
plt.close(fig)
# plt.show()
    