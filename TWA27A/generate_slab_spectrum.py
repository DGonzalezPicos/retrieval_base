import numpy as np
import matplotlib.pyplot as plt
import pathlib

from retrieval_base.auxiliary_functions import get_path
from retrieval_base.slab_model import Disk
from retrieval_base.spectrum import ModelSpectrum

path = get_path()
disk_species = [
            # 'H2O',
            '12CO',
            ]
 # disk_species = disk_species
wave_step = 1e-5
wave_range = [
                # [2.5, 3.2],
            #   [3.2, 4.2],
              [4.1, 5.3]]

wmin = np.min([w[0] for w in wave_range])
wmax = np.max([w[1] for w in wave_range])

T_d = 510.0 # K
R_d = 14.0  # Rjup
add_blackbody_disk = (T_d > 0.0)

disk = Disk(molecules=disk_species,
    # wave_range=(wmin, wmax),
    wave_range=(wmin, wmax),
    wave_step=None,
    grating=None,
    path_to_moldata=path+'data/hitran',
    )

T_ex_range = np.arange(400., 710., 100.)[::-1]
colors = plt.cm.inferno_r(np.linspace(0, 1, 1+len(T_ex_range)))

# gratings = ['g235h','g395h','g395h']
gratings = ['g395h']
fig, ax = plt.subplots(len(wave_range),1, figsize=(12,len(wave_range)*3), tight_layout=True)
ax = np.atleast_1d(ax)

N_mol = 1e17 # cm^-2
A_au = 1e-3 # au^2
d_pc = 59.0 # pc

title = "+".join(disk_species) + ' emission at different T_ex'
title += f' (N_mol={N_mol:.0e}' + r' cm$^{-2}$' + f', A={A_au:.0e}' + r' au$^2$)'
if add_blackbody_disk:
    title += f'\nBlackbody disk: T_d={T_d:.0f} K, R_d={R_d:.0f} Rjup'

for j, T_ex in enumerate(T_ex_range):
    disk_params = {"T_ex": np.array([np.array([T_ex]), np.array([T_ex])]),
                "N_mol": np.array([np.array([N_mol]), np.array([N_mol])]),
                    "A_au": np.array([np.array([A_au]), np.array([A_au])]),
                    "dV": np.array([np.array([1.0]), np.array([1.0])]),
                    "distance": d_pc, # pc
    }
    for i, wave_range_i in enumerate(wave_range):
        wave_i = np.arange(wave_range_i[0], wave_range_i[1], wave_step)
        wave_obs_i = np.arange(wave_i.min()+0.1, wave_i.max()-0.1, wave_step * 10)
        
        disk.set_fine_wgrid(wave_i)
        flux_disk = disk(disk_params, wave=wave_obs_i)
        
        m_spec = ModelSpectrum(wave_obs_i, flux_disk)
        m_spec.flux = m_spec.instr_broadening_nirspec(m_spec.wave, m_spec.flux, grating=gratings[i])

        label = f'T_ex={T_ex:.0f} K' if i==0 else None
        # ax[i].plot(disk.slab.fine_wgrid, flux_disk, color=colors[j], alpha=0.8, label=label)
        if add_blackbody_disk:
            m_spec.add_blackbody_disk(T_d, R_d, d=d_pc, wave_cm=m_spec.wave * 1e-4)
            
        print(f' m_spec.wave.shape = {m_spec.wave.shape}, m_spec.flux.shape = {m_spec.flux.shape}')
        
        ax[i].plot(m_spec.wave, m_spec.flux, color=colors[j], alpha=0.8, label=label)

ax[-1].set_xlabel('Wavelength / um')
fig.text(0.005, 0.5, 'Flux / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$', va='center', rotation='vertical')
ax[0].legend()
ax[0].set_title(title)

fig_name = f'{path}/TWA27A/slab_model_{"_".join(disk_species)}.pdf'
if add_blackbody_disk:
    fig_name = fig_name.replace('.pdf', f'_blackbody_T{int(T_d)}_R{int(R_d)}.pdf')

fig.savefig(fig_name); print(f'--> Saved {fig_name}'); plt.show()
