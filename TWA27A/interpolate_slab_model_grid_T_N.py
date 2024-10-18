""" Interpolate spectral models on a grid of temperature and column density
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
from scipy.interpolate import griddata, RegularGridInterpolator
# pdf pages
from matplotlib.backends.backend_pdf import PdfPages

from retrieval_base.auxiliary_functions import get_path
from retrieval_base.slab_model import Disk
from retrieval_base.spectrum import ModelSpectrum

path = pathlib.Path(get_path())
disk_species = '13CO'

def load_slab(species: str = '12CO', grating: str = 'g395h', disk_params: dict = None, path: pathlib.Path = path):
    """ Load slab model for a given species and grating"""
    
    folder = path / 'data/slab_models' / species / grating
    assert folder.exists(), f'folder {folder} does not exist'
    
    T_ex = disk_params['T_ex'][0][0]
    N_mol = disk_params['N_mol'][0][0]
    file_name = folder / f'slab_{T_ex:.0f}K_N{N_mol:.0e}.npy'
    assert file_name.exists(), f'file {file_name} does not exist'

    wave, flux = np.load(file_name)
    return wave, flux

def load_grid(species: str = '12CO', grating: str = 'g395h', disk_params: dict = None, path: pathlib.Path = path,
              T_ex_range: np.ndarray = np.arange(100.0, 600.0, 200.0), N_mol_range: np.ndarray = np.logspace(12, 18, 1)):
    """ Load slab model grid for a given species and grating"""
    
    wave, flux = [], []
    for T_ex in T_ex_range:
        wave_j, flux_j = [], []
        for N_mol in N_mol_range:
            disk_params['T_ex'][0][0] = T_ex
            disk_params['N_mol'][0][0] = N_mol
            wave_ij, flux_ij = load_slab(species=species, grating=grating, disk_params=disk_params, path=path)
            wave_j.append(wave_ij)
            flux_j.append(flux_ij)
        wave.append(wave_j)
        flux.append(flux_j)
        
    return np.array(wave), np.array(flux)

T_ex = 500.0 # K
N_mol = 1e18 # cm^-2
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


T_ex_range = np.arange(300.0, 600.0+100.0, 100.0)
N_mol_range = np.logspace(15, 20, 6)

wave, flux = load_grid(species=disk_species, grating='g395h', disk_params=disk_params, path=path,
                T_ex_range=T_ex_range, N_mol_range=N_mol_range)

# shape wave: (4,6,12200) = (T_ex, N_mol, wave)
# use scipy grid interpolation to interpolate to a new pair or (T_ex, N_mol)

# Interpolation setup
T_ex_new = 611.0
N_mol_new = 1.5e18

# check that the new values are within the grid, else clip to min/max
T_ex_new = np.clip(T_ex_new, T_ex_range.min(), T_ex_range.max())
N_mol_new = np.clip(N_mol_new, N_mol_range.min(), N_mol_range.max())
print(f' Interpolating to T_ex={T_ex_new:.0f} K, N_mol={N_mol_new:.0e} cm^-2')


flux_interpolator = RegularGridInterpolator((T_ex_range, N_mol_range), flux, method='linear', bounds_error=False, fill_value=None)
flux_new = flux_interpolator((T_ex_new, N_mol_new))

fig, ax = plt.subplots(1,1, figsize=(12,6), tight_layout=True)

# plot closest point of the grid
idx_T_ex = np.argmin(np.abs(T_ex_range - T_ex_new))
idx_N_mol = np.argmin(np.abs(N_mol_range - N_mol_new))
ax.plot(wave[idx_T_ex, idx_N_mol], flux[idx_T_ex, idx_N_mol], label=f'grid point T_ex={T_ex_range[idx_T_ex]:.0f} K, N_mol={N_mol_range[idx_N_mol]:.0e} cm^-2', lw=0.7)

# plot interpolated point
ax.plot(wave[idx_T_ex, idx_N_mol], flux_new, label=f'interpolated point T_ex={T_ex_new:.0f} K, N_mol={N_mol_new:.0e} cm^-2', lw=0.7)
ax.legend()
plt.show()