import numpy as np
import iris as iris
from iris import setup
from iris import spectrum as sp
import pathlib
import time

from jax import jit, vmap
from jax.scipy.signal import fftconvolve
import jax.numpy as jnp

from spectres import spectres

# from broadpy import InstrumentalBroadening
# from broadpy.utils import load_nirspec_resolution_profile

# constants
c_kms = 2.99792458e5 # km/s
c_cms = 2.99792458e10 # cm/s

class Disk:
    
    def __init__(self, 
                 molecules=['12CO'], # 12CO 
                 wave_range=(1.90, 5.2), # um
                 wave_step=1e-5, # um, this seems enough for NIRSpec, check for other setups
                 grating=None,
                 path_to_moldata='data/hitran',):
        self.molecules = molecules
       
        
        self.path_to_moldata = path_to_moldata
        
        if not pathlib.Path(self.path_to_moldata).exists():
            print(f'Creating directory {self.path_to_moldata}')
            pathlib.Path(self.path_to_moldata).mkdir(parents=True, exist_ok=True)
            
        # only CO implemented for now...
        CO_iso = {'12CO': 1, '13CO': 2, 'C18O': 3}
        for mol in self.molecules:
            assert mol in CO_iso.keys(), f'{mol} not in {CO_iso.keys()}'
            setup.setup_linelists(mol, 'CO', CO_iso[mol], path_to_moldata=path_to_moldata)
        
        if wave_step is not None:
            self.fine_wgrid = jnp.arange(wave_range[0], wave_range[1], wave_step)
        self.slab = iris.slab(molecules=self.molecules,
                     wlow=wave_range[0],
                     whigh=wave_range[1],
                     path_to_moldata=self.path_to_moldata)

        
        # if grating in ['g235h', 'g395h', 'g140h']:
        #     self.grating = grating
        #     self.load_fwhm_nirspec(grating=self.grating)
        
    def set_properties(self, distance=100.,
                          T_ex=np.array([600.]),
                          N_mol=np.array([1e17]),
                          A_au=np.array([1.0]),
                          dV=np.array([1.0]),
                          ):
        ''' Set the properties of the disk 
        
        Parameters
        ----------
        distance : float
            Distance to the source in pc
        T_ex : array
            Excitation temperatures for each molecule in K
        N_mol : array
            Column densities in cm^-2
        A_au : array
            Emitting areas in au^2
        dV : array
            Turbulent line widths in km/s (line FWHM)
        '''
        
        assert len(T_ex) == len(self.molecules), 'T_ex must have the same length as molecules'
        assert len(N_mol) == len(self.molecules), 'N_mol must have the same length as molecules'
        assert len(A_au) == len(self.molecules), 'A_au must have the same length as molecules'
        assert len(dV) == len(self.molecules), 'dV must have the same length as molecules'
        
        self.slab.setup_disk(distance, T_ex, N_mol, A_au, dV)
        # return self
        
    def set_obs_wgrid(self, obs_wgrid):
        ''' Set the grid for the disk
        
        Parameters
        ----------
        obs_wgrid : array
            Wavelength grid of the observation
        R : float
            Instrument resolving power
        '''
        # check obs_grid is contained in fine_grid
        # assert np.all(obs_wgrid >= self.fine_wgrid[0]), 'obs_wgrid must be contained in fine_wgrid'
        # assert np.all(obs_wgrid <= self.fine_wgrid[-1]), 'obs_wgrid must be contained in fine_wgrid'
        
        # self.slab.setup_grid(self.fine_wgrid, obs_wgrid, R)
        self.slab.obs_wgrid = jnp.array(obs_wgrid)
        return self
    
    def set_fine_wgrid(self, fine_wgrid):
        self.slab.fine_wgrid = jnp.array(fine_wgrid)
        return self
    
    def calc_flux(self):
        ''' Calculate the flux density of the disk '''
        self.flux = sp.compute_total_fdens(self.slab.catalog, 
                                           self.slab.distance, 
                                           self.slab.A_au, 
                                           self.slab.T_ex, 
                                           self.slab.N_mol, 
                                           self.slab.dV, 
                                           self.slab.fine_wgrid,
                                           )
        return self
    
    # def load_fwhm_nirspec(self, grating='g235h'):
    #     ''' Load the FWHM of the NIRSpec resolution profile '''
    #     _, resolution = load_nirspec_resolution_profile(grating=grating, wave=self.fine_wgrid*1e3)
    #     self.fwhm = 2.99792458e5 / (resolution)
    #     return self
    
    # def broaden(self, fwhm=None):
    #     ''' Broaden the flux density of the disk '''
    #     assert (fwhm is not None) or hasattr(self, 'fwhm'), 'fwhm must be provided or load_fwhm_nirspec must be called'
    #     fwhm = fwhm if fwhm is not None else self.fwhm
            
    #     self.flux = InstrumentalBroadening(self.slab.fine_wgrid, self.flux)(fwhm=fwhm, kernel='gaussian_variable')
    #     return self
    
    def resample(self):
        ''' Resample the flux density of the disk '''
        self.flux = spectres(self.slab.obs_wgrid, self.slab.fine_wgrid, self.flux)
        return self
    
    def __call__(self, params, 
                 wave=None, # this is the fine_wgrid in um
                 ):
        
        # self.set_properties(distance=params.get('distance', 100.0),
        #                     T_ex=params.get('T_ex', np.array(np.array([600.]))),
        #                     N_mol=params.get('N_mol', np.array(np.array([1e17]))),
        #                     A_au=params.get('A_au', np.array(np.array([1.0]))),
        #                     dV=params.get('dV', np.array(np.array([1.0]))),
        #                     )
        
        self.slab.setup_disk(distance=params.get('d_pc', 100.0),
                             T_ex=params.get('T_ex', np.array([600.])),
                             N_mol=params.get('N_mol', np.array([1e17])),
                             A_au=params.get('A_au', np.array([1.0])),
                             dV=params.get('dV', np.array([1.0])),
                             )
        assert hasattr(self.slab, 'distance'), f'Distance not set'
        
        if wave is not None:
            self.fine_wgrid = jnp.array(wave)
        assert hasattr(self.slab, 'fine_wgrid'), 'fine_wgrid not set'
        
        # start = time.time()
        # if not hasattr(self.slab, 'obs_wgrid'):
        #     self.set_obs_wgrid(wave)
            # self.set_grid(obs_wgrid=wave,
                        #   R=params.get('R', 3200),
                        #   )
        # end = time.time()
        # print(f'Setup time: {end-start:.2e} s')
        # self.calc_flux()
        # start = time.time()
        self.flux = calc_flux_jit(self.slab.catalog, self.slab.distance, self.slab.A_au, self.slab.T_ex, self.slab.N_mol, self.slab.dV, self.slab.fine_wgrid)
        # end = time.time()
        # print(f'calc_flux_jit time: {end-start:.2f} s')
        
        # convert Jy to [erg cm^{-2} s^{-1} nm^{-1}]
        wave_cm = self.slab.fine_wgrid * 1e-4 # [um] -> [cm]
        self.flux = self.flux * 1e-23 * (c_cms/wave_cm**2)
        # [erg cm^{-2} s^{-1} cm^{-1}] -> [erg cm^{-2} s^{-1} nm^{-1}]
        self.flux *= 1e-7
        
        # start = time.time()
        # self.broaden(fwhm=self.fwhm)
        # print(f'Instrumental broadening time: {time.time()-start:.2f} s')
        # # self.flux = instrumental_broadening_jit(self.slab.fine_wgrid, self.flux, self.fwhm)
        
        # start = time.time()
        # self.resample()
        # print(f'Resampling time: {time.time()-start:.2f} s')
        return self.flux
    
@jit
def calc_flux_jit(catalog, distance, A_au, T_ex, N_mol, dV, fine_wgrid):
    return sp.compute_total_fdens(catalog, distance, A_au, T_ex, N_mol, dV, fine_wgrid)

if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    import time
    
    # get directory of script
    dir_path = pathlib.Path(__file__).parent.absolute()
    
    # wmin, wmax = 4.1, 5.3
    wmin, wmax = 1.9, 2.6
    disk = Disk(molecules=['12CO',],
                wave_range=(wmin, wmax),
                wave_step=1e-5,
                grating='g395h',
                path_to_moldata=str(dir_path/'../data/hitran'),
                )
    
    params = {'distance': 59.0,
              'T_ex': np.array([np.array([800.])]),
                'N_mol': np.array([np.array([1e17])]),
                'A_au': np.array([np.array([2.0])]),
                'dV': np.array([np.array([2.0])]),
              'R': 3200,
              }
    
    wave = np.load('../TWA28/wave_NIRSPec.npy')[-1] * 1e-3
    wave = wave[~np.isnan(wave)]
    mask = (wave >= wmin) & (wave <= wmax)
    wave = wave[mask]
    
    wave_step_obs = np.median(np.diff(wave))
    print(f' Wave step of observation: {wave_step_obs:.2e}')
    disk.set_obs_wgrid(obs_wgrid=wave)
    # disk.set_fine_wgrid()
    
    
    time_list = []
    n = 2
    for i in range(n):
        start = time.time()
        flux = disk(params, wave)
        time_list.append(time.time()-start)
        print(f' {i+1}/{n} Time taken: {time_list[-1]:.2f} s\n')
        
    
    print(f'Time (ignoring first): {np.mean(time_list[1:]):.2f} +- {np.std(time_list[1:]):.2f} s')
    
    plot = True
    
    if plot:
        fig, ax = plt.subplots(1,1, figsize=(10,6))
        # ax.plot(wave, disk.flux, label='CO')
        ax.plot(disk.slab.fine_wgrid, flux, label='12CO')
        ax.legend()
        ax.set(xlabel='Wavelength / um', ylabel='Flux / erg cm$^{-2}$ s$^{-1}$ nm$^{-1}$',
            #    xlim=(4.71, 4.92),
               )
        
        fig_name = f'CO_emission_wave_{wmin:.2f}-{wmax:.2f}.pdf'
        fig.savefig('../'+fig_name, bbox_inches='tight')
        print(f'Saved figure: {fig_name}')
        plt.show()        