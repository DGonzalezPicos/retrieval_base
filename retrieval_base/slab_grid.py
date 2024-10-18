""" Class to create, load and interpolate a grid of slab models
generated with IRIS for a given species, grating and for a given range of T_ex and N_mol

The scaling of size (A_au) is performed at a later stage in the retrieval

We only consider the wavelength range of (4.1, 5.34) um for the g395h grating where the 12CO, 13CO lines are located
H2O emission is lower than the CO lines and is not considered here, might only affect the >4.8 um range at less than 1% flux level
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
from scipy.interpolate import griddata, RegularGridInterpolator


from retrieval_base.auxiliary_functions import get_path
from retrieval_base.slab_model import Disk
from retrieval_base.spectrum import ModelSpectrum

class SlabGrid:
    
    wave_range = (4.20, 5.30) # fixed to the range of the g395h grating (reddest filter)
    wave_step = 1e-5 # um, fixed and tested visually (must be small)
    
    def __init__(self, species: str = '12CO', grating: str = 'g395h', path: pathlib.Path = None,
                 T_ex_range: np.ndarray = np.arange(100.0, 600.0, 200.0), N_mol_range: np.ndarray = np.logspace(12, 18, 1)):
        """ Initialize SlabGrid object
        
        Parameters
        ----------
        species: str
            Molecule species
        grating: str
            Grating used to generate the slab models
        path
            Path to the slab models
        T_ex_range: np.ndarray
            Range of excitation temperatures
        N_mol_range: np.ndarray
            Range of column densities
        """
        
        self.species = species
        self.grating = grating
        assert self.grating == 'g395h', f'Only grating g395h is implemented, {self.grating} not implemented'
        self.path = path
        self.path_to_moldata = str(path / 'data/hitran')
        self.T_ex_range = T_ex_range
        self.N_mol_range = N_mol_range
        
        
        self.folder = self.path / 'data/slab_models' / self.species / self.grating
        self.folder.mkdir(parents=True, exist_ok=True)
        
        # default disk params
        self.disk_params = { "T_ex": np.array([np.array([500.0])]),
                            "N_mol": np.array([np.array([1e18]),]),
                            "A_au": np.array([np.array([1.0])]),
                            "dV": np.array([np.array([1.0])]),
                            "d_pc": 1.0, # pc
                            }
        
    def update_disk_params(self, attr: str = 'T_ex', value: float=500.0):
        """ Update disk parameters"""
        
        self.disk_params[attr] = np.array([np.array([value])])
        return self
    
    @property 
    def file_name(self):
        """ Get file name"""
        
        T_ex = self.disk_params['T_ex'][0][0]
        N_mol = self.disk_params['N_mol'][0][0]
        return self.folder / f'slab_{T_ex:.0f}K_N{N_mol:.0e}.npy'
        
    def get_flux(self, T_ex: float, N_mol: float):
        """ Get flux for a given T_ex and N_mol"""
        
        self.update_disk_params('T_ex', T_ex)
        self.update_disk_params('N_mol', N_mol)
        
        disk = Disk(molecules=[self.species],
                    wave_range=self.wave_range,
                    wave_step=None,
                    grating=None,
                    path_to_moldata=self.path_to_moldata,
                    )
        
        wave = np.arange(self.wave_range[0], self.wave_range[1], self.wave_step)
        wave_obs = np.arange(wave.min()+0.01, wave.max()-0.01, self.wave_step * 10)
        
        disk.set_fine_wgrid(wave)
        flux_disk = disk(self.disk_params, wave=wave_obs)
        
        m_spec = ModelSpectrum(wave_obs, flux_disk)
        m_spec.flux = m_spec.instr_broadening_nirspec(m_spec.wave, m_spec.flux, grating=grating)
        print(f' Mean flux (@ {grating}): {m_spec.flux.mean()}')
        
        wave_slab = np.array(m_spec.wave)
        flux_slab = np.array(m_spec.flux)
        np.save(self.file_name, np.array([wave_slab, flux_slab]))
        print(f'--> Saved {self.file_name}')
        return m_spec
    
    def get_grid(self, 
                 T_ex_range: np.ndarray=np.arange(300.0, 600.0+1.0, 100.0),
                 N_mol_range: np.ndarray=np.logspace(15, 20, 6),
                 cache: bool = True):
        
        flux_ij = []
    
        for T_ex in T_ex_range:
            flux_j = []
            for N_mol in N_mol_range:
                self.update_disk_params('T_ex', T_ex)
                self.update_disk_params('N_mol', N_mol)
                if cache and self.file_name.exists():
                    print(f'[get_grid] Loading {self.file_name}...')
                    wave, flux = np.load(self.file_name)
                else:
                    m = self.get_flux(T_ex, N_mol)
                    wave, flux = m.wave, m.flux
                flux_j.append(flux)
                
            flux_ij.append(flux_j)
            
        self.wave_grid = np.array(wave)
        self.flux_grid = np.array(flux_ij)
        
        self.T_ex_range = T_ex_range
        self.N_mol_range = N_mol_range
        
        return self

    
    def load_interpolator(self):
        
        assert hasattr(self, 'wave_grid'), f'wave_grid not found, run get_grid() first' 
        assert hasattr(self, 'flux_grid'), f'flux_grid not found, run get_grid() first'
        
        self.interpolator = RegularGridInterpolator((self.T_ex_range, self.N_mol_range), self.flux_grid, method='linear',
                                                    bounds_error=False, fill_value=None)
        del self.flux_grid # save memory
        return self
    
    def interpolate(self, T_ex_new: float, N_mol_new: float, A_au: float = 1.0, d_pc: float = 1.0):
        """ Interpolate to new values of T_ex and N_mol"""
        
        assert hasattr(self, 'interpolator'), f'interpolator not found, run load_interpolator() first'
        
        T_ex_new = np.clip(T_ex_new, self.T_ex_range.min(), self.T_ex_range.max())
        N_mol_new = np.clip(N_mol_new, self.N_mol_range.min(), self.N_mol_range.max())
        print(f' Interpolating to T_ex={T_ex_new:.0f} K, N_mol={N_mol_new:.0e} cm^-2')
        
        flux_new = self.scale_flux(self.interpolator((T_ex_new, N_mol_new)), A_au, d_pc)
        return flux_new
    
    @classmethod
    def scale_flux(self, flux: np.ndarray, A_au: float = 1.0, d_pc: float = 1.0):
        """ Scale the flux to the given A_au and d_pc"""
        # convert au to cm**2
        # unit conversion already multiplied by 1 au**2 and divided by (1pc)**2
                
        factor = A_au / d_pc**2
        
        return flux * factor
    
    def combine_interpolators(self, other):
        """ Combine two interpolators"""
        
        assert hasattr(other, 'interpolator'), f'interpolator not found in other, run load_interpolator() first'
        # check they have the same T_ex and N_mol ranges
        assert np.allclose(self.T_ex_range, other.T_ex_range), f'T_ex_range not the same'
        assert np.allclose(self.N_mol_range, other.N_mol_range), f'N_mol_range not the same'
        
        
        interpolator_dict = {self.species: self.interpolator, other.species: other.interpolator}
        # return interpolator_dict, 
        
        
    
    
if __name__ =='__main__':
    
    path = pathlib.Path(get_path())
    species = '13CO'
    grating = 'g395h'
    
    slab = SlabGrid(species=species, grating=grating, path=path)
    # m = slab.get_flux(T_ex, N_mol)
    T_ex_range = np.arange(300.0, 600.0+100.0, 100.0)
    N_mol_range = np.logspace(15, 20, 6)
    slab.get_grid(T_ex_range, N_mol_range, cache=True)
    
    slab.load_interpolator()
    flux_new = slab.interpolate(611.0, 1.5e18)
    print(f'--> Done')
        
        
        
        
        
        
    