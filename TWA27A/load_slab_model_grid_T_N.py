""" Generate individual slab models for different species"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
# pdf pages
from matplotlib.backends.backend_pdf import PdfPages

from retrieval_base.auxiliary_functions import get_path
from retrieval_base.slab_model import Disk
from retrieval_base.spectrum import ModelSpectrum

path = pathlib.Path(get_path())
disk_species = 'H2O'
    
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

def plot(wave, flux, label='', **kwargs):
    ax.plot(wave, flux, label=label, **kwargs)
    return ax

def plot_grid(wave, flux, ax,**kwargs):
    
    ax = np.atleast_1d(ax)
    k = -1
    lw = kwargs.get('lw', 0.7)
    labels = kwargs.get('labels', None)
    for i, wave_i in enumerate(wave):
        for j, wave_ij in enumerate(wave_i):
            k += 1
            flux_ij = flux[i][j]
            ax_k = ax[k] if len(ax) > 1 else ax
            
            label = labels[k] if labels is not None else ''
            ax_k.plot(wave_ij,flux_ij, label=label)
            ax_k.legend()
            
    return None
    
    
    
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
# get sns color palette
import seaborn as sns
colors = sns.color_palette('husl', len(T_ex_range) * len(N_mol_range))

wave, flux = load_grid(species=disk_species, grating='g395h', disk_params=disk_params, path=path, T_ex_range=T_ex_range, N_mol_range=N_mol_range)

labels = [f'T_ex={T_ex:.0f} K, N_mol={N_mol:.0e} cm^-2' for T_ex in T_ex_range for N_mol in N_mol_range]

pdf_name = path / f'data/slab_models/slab_{disk_species}_model_grid.pdf'

with PdfPages(pdf_name) as pdf:
    for i, T_ex in enumerate(T_ex_range):
        for j, N_mol in enumerate(N_mol_range):
            

            fig, ax = plt.subplots(1,1, figsize=(12,4), tight_layout=True)

            ax.plot(wave[i,j], flux[i,j], label=f'T_ex={T_ex:.0f} K, N_mol={N_mol:.0e} cm^-2',
                    color=colors[i*len(N_mol_range) + j])
            ax.legend()
            if i == 0 and j == 0:
                ax.set_title(f'{disk_species} slab model grid (at 1pc, 1 AU$^2$)')
            if (i == len(T_ex_range) - 1) and (j == len(N_mol_range) - 1):
                ax.set_xlabel('Wavelength / $\mu$m')
                ax.set_ylabel('Flux / erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$')
            pdf.savefig(fig)
            plt.close(fig)

    print(f'--> Saved {pdf_name}')