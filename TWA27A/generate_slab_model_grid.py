""" Generate individual slab models for different species"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib

from retrieval_base.auxiliary_functions import get_path
from retrieval_base.slab_model import Disk
from retrieval_base.spectrum import ModelSpectrum

path = pathlib.Path(get_path())
disk_species = [
            'H2O',
            '12CO',
            '13CO',
            ]
colors = ['navy', 'darkorange', 'darkgreen']
 # disk_species = disk_species
wave_step = 1e-5

gratings = {
            'g235h': [(1.58, 2.4), (2.4, 3.24)],
            # 'g395h': (2.86, 5.32),
            'g395h': [(2.80, 4.1), (4.1, 5.40)],
            # 'g': [(1.62, 2.38), (2.38, 3.22), (3.22, 4.1), (4.1, 5.32)],
            }


T_ex = 500.0 # K
N_mol = 1e17 # cm^-2
# A_au = 1e-3 # au^2
A_au = 1.0 # au^2
# d_pc = 59.17 # pc (16.46 mas = 59.17 pc)
d_pc = 1.0 # pc
title = "+".join(disk_species) + ' emission at' + f' T_ex={T_ex:.0f} K' + f' (N_mol={N_mol:.0e}' + r' cm$^{-2}$' + f', A={A_au:.0e}' + r' au$^2$)' + f', d={d_pc:.2f} pc'

disk_params = { "T_ex": np.array([np.array([T_ex])]),
                "N_mol": np.array([np.array([N_mol]),]),
                "A_au": np.array([np.array([A_au])]),
                "dV": np.array([np.array([1.0])]),
                # "distance": d_pc, # pc
                "d_pc": d_pc, # pc
    }

# grating = 'g395h'
# assert grating in gratings.keys(), f'grating {grating} not in {gratings.keys()}'
# gratings_list = ['g235h', 'g395h']
gratings_list = list(gratings.keys())
fig, ax = plt.subplots(len(disk_species),1, figsize=(12,len(disk_species)*3), tight_layout=True)
ax[0].set_title(title)
path_slab = path / 'data/slab_models'
path_slab.mkdir(parents=True, exist_ok=True)

# ylim = (-1e-16, 5.5e-15)
ylim = (None, None)

for i, disk_species_i in enumerate(disk_species):
    
    ax[i].set_ylabel('Flux / erg s$^{-1}$ cm$^{-2}$ $\mu$nm$^{-1}$')
    # ax[i].set_ylim(ylim)
    ax[i].text(0.05, 0.8, disk_species_i, transform=ax[i].transAxes, fontsize=18, fontweight='bold')
    
    for grating in gratings_list:
        # file_name = f'{path}/TWA27A/slab_model_{disk_species_i}_{grating}.npy'
        file_name = path_slab / f'slab_model_{disk_species_i}_{grating}.npy'
        wave_slab = []
        flux_slab = []
        
        for j, wave_range_j in enumerate(gratings[grating]):
            wmin, wmax = wave_range_j
            disk = Disk(molecules=[disk_species_i],
                        wave_range=(wmin, wmax),
                        wave_step=None,
                        grating=None,
                        path_to_moldata=str(path / 'data/hitran'),
                        )
            
            
    
            wave_i = np.arange(wmin, wmax, wave_step)
            wave_obs_i = np.arange(wave_i.min()+0.01, wave_i.max()-0.01, wave_step * 10)

            
            disk.set_fine_wgrid(wave_i)
            flux_disk = disk(disk_params, wave=wave_obs_i)
            
            m_spec = ModelSpectrum(wave_obs_i, flux_disk)
            m_spec.flux = m_spec.instr_broadening_nirspec(m_spec.wave, m_spec.flux, grating=grating)
            print(f' Mean flux (@ {grating}): {m_spec.flux.mean()}')
            wave_slab += m_spec.wave.tolist()
            flux_slab += m_spec.flux.tolist()
            

            label = f'T_ex={T_ex:.0f} K' if i==0 else None
            # ax[i].plot(disk.slab.fine_wgrid, flux_disk, color=colors[j], alpha=0.8, label=label)

            print(f' m_spec.wave.shape = {m_spec.wave.shape}, m_spec.flux.shape = {m_spec.flux.shape}')
            
            ax[i].plot(m_spec.wave, m_spec.flux, color=colors[i], alpha=0.8, label=label)
            

        wave_slab = np.array(wave_slab)
        flux_slab = np.array(flux_slab)
        np.save(file_name, np.array([wave_slab, flux_slab]))
        print(f'--> Saved {file_name}')
        
    if i == len(disk_species) - 1:
        ax[i].set_xlabel('Wavelength / um')

# fig_name = f'{path}/TWA27A/slab_model_individual_{"_".join(disk_species)}_{grating}.png'
fig_name = path_slab / f'slab_model_individual_{"_".join(disk_species)}_{"_".join(gratings_list)}.pdf'
fig.savefig(fig_name); print(f'--> Saved {fig_name}'); plt.close(fig)