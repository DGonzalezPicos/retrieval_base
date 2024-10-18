""" Generate individual slab models for different species"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib

from retrieval_base.auxiliary_functions import get_path
from retrieval_base.slab_model import Disk
from retrieval_base.spectrum import ModelSpectrum

path = pathlib.Path(get_path())
testing = True

disk_species = [
            'H2O',
            '12CO',
            '13CO',
            ]

colors = ['navy', 'darkorange', 'darkgreen']
 # disk_species = disk_species
# wave_step = 1e-4 if testing else 1e-5
wave_step = 1e-5

gratings = {
            'g235h': [(1.58, 2.4), (2.4, 3.24)],
            # 'g395h': (2.86, 5.32),
            'g395h': [(2.76, 4.10), (4.10, 5.34)],
            # 'g': [(1.62, 2.38), (2.38, 3.22), (3.22, 4.1), (4.1, 5.32)],
            }
if testing:
    disk_species = ['12CO']
    # gratings = {'g395h':[(4.3, 4.8)]}
    gratings = {'g395h': [(4.10, 5.34)],}

    

T_ex = 500.0 # K
N_mol = 1e17 # cm^-2
# A_au = 1e-3 # au^2
A_au = 1.0 # au^2
# d_pc = 59.17 # pc (16.46 mas = 59.17 pc)
d_pc = 1.0 # pc

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

path_slab = path / 'data/slab_models'
path_slab.mkdir(parents=True, exist_ok=True)

# ylim = (-1e-16, 5.5e-15)
ylim = (None, None)

def slab_model(species: str = '12CO', 
               grating: str = 'g395h',
               disk_params: dict = disk_params,
                path: pathlib.Path = path_slab, # path to HITRAN data
                wave_step: float = 1e-5,
                ):
               
    folder = path / 'data/slab_models' / species / grating
    folder.mkdir(parents=True, exist_ok=True)
    
    T_ex = disk_params['T_ex'][0][0]
    N_mol = disk_params['N_mol'][0][0]
    print(f'--> Generating slab model for {species} at T_ex={T_ex:.0f} K, N_mol={N_mol:.0e} cm^-2')
    file_name = folder / f'slab_{T_ex:.0f}K_N{N_mol:.0e}.npy'
    # print(f'--> Saving to {file_name}')
    
    
    wave_slab, flux_slab = [], []
    
    # here we use a **global** variable `gratings` define at the beginning of the script
    for j, wave_range_j in enumerate(gratings[grating]):
        assert len(wave_range_j) == 2, f'len(wavelength range) {len(wave_range_j)} != 2'
        wmin, wmax = wave_range_j
        disk = Disk(molecules=[species],
                    wave_range=(wmin, wmax),
                    wave_step=None,
                    grating=None,
                    path_to_moldata=str(path / 'data/hitran'),
                    )
    

        wave_i = np.arange(wmin, wmax, wave_step)
        wave_obs_i = np.arange(wave_i.min()+0.01, wave_i.max()-0.01, wave_step * 10)

        
        disk.set_fine_wgrid(wave_i)
        flux_disk = disk(disk_params, wave=wave_obs_i)
        
        
        # for plotting purposes....
        m_spec = ModelSpectrum(wave_obs_i, flux_disk)
        m_spec.flux = m_spec.instr_broadening_nirspec(m_spec.wave, m_spec.flux, grating=grating)
        print(f' Mean flux (@ {grating}): {m_spec.flux.mean()}')
        wave_slab += m_spec.wave.tolist()
        flux_slab += m_spec.flux.tolist()
        
    wave_slab = np.array(wave_slab)
    flux_slab = np.array(flux_slab)
    np.save(file_name, np.array([wave_slab, flux_slab]))
    print(f'--> Saved {file_name}')
    
    
T_ex_range = np.arange(300.0, 600.0+100.0, 100.0)
N_mol_range = np.logspace(15, 20, 6)

def grid(T_ex_range, N_mol_range, 
         species='12CO', 
         grating='g395h', 
         disk_params=disk_params, 
         path=path, wave_step=wave_step):
    for T_ex in T_ex_range:
        for N_mol in N_mol_range:
            disk_params['T_ex'] = np.array([np.array([T_ex])])
            disk_params['N_mol'] = np.array([np.array([N_mol])])
            slab_model(species=species, grating=grating, disk_params=disk_params, path=path, wave_step=wave_step)
            
    return None

# slab_model(species='12CO', grating='g395h', disk_params=disk_params, path=path, wave_step=wave_step)
grid(T_ex_range,
     N_mol_range, 
     species='13CO', 
     grating='g395h',
     disk_params=disk_params, 
     path=path,
     wave_step=wave_step)