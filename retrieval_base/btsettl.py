""" Load BT-Settl models from xarray files """
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import xarray as xr
import copy
from scipy.interpolate import RegularGridInterpolator
from spectres import spectres
from petitRADTRANS import nat_cst as nc

from retrieval_base.spectrum_jwst import SpectrumJWST
import retrieval_base.auxiliary_functions as af
from retrieval_base.resample import Resample

class BTSettl:
    
    path_suffix = 'dario/phd' if 'dario' in os.environ['HOME'] else 'dgonzalezpi'
    base_path = pathlib.Path(f'/home/{path_suffix}')
    path = base_path / 'BT-Settl'
    file = path / 'BTSETTL.nc'
    
    def __init__(self):
        
        pass
    
    def load_full_dataset(self):
        self.model = xr.open_dataset(self.file)
        self.wave = self.model.grid.wavelength.data * 1e3 # [um] -> [nm]
        self.teff_grid = self.model.par1.data
        self.logg_grid = self.model.par2.data
        return self
    
    def load_dataset(self, file):
        print(f' Loading {file}')
        dataset = xr.open_dataset(file)
        # self.dataset = dataset
        self.flux_grid = dataset.flux.data # shape (n_wave, n_teff, n_logg)
        print(f' BTSETTL grid shape: {self.flux_grid.shape}')
        self.flux_units = dataset.flux.attrs['units']
        self.wave = dataset.wave.data # units should be in [nm]
        self.wave_unit = 'nm'
        self.teff_grid = dataset.teff.data
        self.logg_grid = dataset.logg.data
        return self
    
    def set_interpolator(self):
        
        assert hasattr(self, 'wave'), 'Load grid first'
        assert hasattr(self, 'flux_grid'), 'Load grid first'
        assert hasattr(self, 'teff_grid'), 'Load grid first'
        assert hasattr(self, 'logg_grid'), 'Load grid first'
        
        self.interpolator = RegularGridInterpolator((self.wave, self.teff_grid, self.logg_grid), self.flux_grid, method='linear')
        # remove attributes to save memory
        del self.flux_grid
        return self

    
        
    def prepare_grid(self, teff_range=[], logg_range=[], wave=None, out_res=3000,
                     file_out=None):
        
        # select a range of teff and logg
        if len(teff_range) == 0:
            teff_range = [1200, 3000]
        if len(logg_range) == 0:
            logg_range = [2.5, 5.5]
            
        # select a range of teff and logg
        flux_grid = np.copy(self.model.grid.data)
        mask_teff = np.logical_and(self.teff_grid >= teff_range[0], self.teff_grid <= teff_range[1])
        mask_logg = np.logical_and(self.logg_grid >= logg_range[0], self.logg_grid <= logg_range[1])
        # mask_wave = 
        
        if wave is None:
            # file_NIRSpec = self.path / 'wave_NIRSPec.npy'
            # wave = np.load(file_NIRSpec)
            # print(f' cenwave NIRSpec: {np.nanmedian(wave)}')
            wave = np.arange(900, 35000, 0.05) # [nm], NIRSPec pixel spacing ~ 0.5 nm, so 10x oversampling
            
        pad = 0.05
        mask_wave = np.logical_and(self.wave >= np.nanmin(wave)*(1-pad), self.wave <= np.nanmax(wave)*(1+pad))
        
        # print shapes of grid and masks
        print(f'flux_grid shape: {flux_grid.shape}')
        print(f'mask_wave shape: {mask_wave.shape}')
        print(f'mask_teff shape: {mask_teff.shape}')
        print(f'mask_logg shape: {mask_logg.shape}')
        
            
        flux_grid = flux_grid[mask_wave,][:, mask_teff, :][:, :, mask_logg]
        # print(f'flux_grid shape: {flux_grid.shape}')
        npix, nteff, nlogg = flux_grid.shape
        # interpolate onto the NIRSpec wavelength grid
        flux_grid_int = np.zeros((len(wave), nteff, nlogg))
        for i in range(nteff):
            for j in range(nlogg):
                f_ij = af.instr_broadening(wave, flux_grid[:,i,j], out_res=out_res, in_res=1e6)
                flux_grid_int[:, i, j] = np.interp(wave, self.wave[mask_wave], f_ij)
                
        print(f'flux_grid_int shape: {flux_grid_int.shape}')                
        
        # enable overwrite
        dataset = xr.DataArray(flux_grid_int, coords=[wave, self.teff_grid[mask_teff], self.logg_grid[mask_logg]],
                                    dims=['wave', 'teff', 'logg'], name='flux', attrs={'units': 'erg s^-1 cm^-2 nm^-1'},
                                    )
        # save as .nc file
        file_out = self.path / 'BTSETTL_NIRSPec.nc' if file_out is None else file_out
        dataset.to_netcdf(file_out, mode='w')
        print(f' Saved {file_out}')
        # close file
        dataset.close()
        return self
    
    
        
    
    def get_spec(self, teff, logg, wave=None):
        
        assert hasattr(self, 'interpolator'), f'Interpolator not set. Call set_interpolator() first'
        
        self.teff = teff
        assert (teff >= self.teff_grid.min()) and (teff <= self.teff_grid.max()), f'Invalid teff {teff}. Must be in range {self.teff_grid.min()} - {self.teff_grid.max()}'
        
        # teff_range = np.arange(1200, 3000+1, 100)
        # assert self.teff in teff_range, f' Invalid teff {teff}. Must be in {teff_range}'
        self.logg = logg
        assert (logg >= self.logg_grid.min()) and (logg <= self.logg_grid.max()), f'Invalid logg {logg}. Must be in range {self.logg_grid.min()} - {self.logg_grid.max()}'
        # logg_range = np.arange(2.5, 5.5+0.5, 0.5)
        
        # self.flux = self.model.grid.data[self.mask, teff_index, logg_index]
        self.flux = self.interpolator((self.wave, self.teff, self.logg))
        if wave is not None:
            self.flux = np.interp(wave, self.wave, self.flux)
            # del self.wave
        return self
    
    def rebin_to_NIRSpec(self, wave=None, out_res=3000., flatten=True):
        
        assert hasattr(self, 'wave'), 'Load grid first'
        
        if wave is None:
            file_NIRSpec = self.path / 'wave_NIRSPec.npy'
            wave_NIRSpec = np.load(file_NIRSpec) # shape (2*n_grisms, n_wave)
        else:
            assert len(wave.shape) == 2, f'Invalid wave shape {wave.shape}, must be (2*n_grisms, n_wave)'
            wave_NIRSpec = wave
            
        if out_res is not None:
            self.flux = af.instr_broadening(self.wave, self.flux, out_res=out_res, in_res=1e6)
            
        if flatten:
            wave_NIRSpec = wave_NIRSpec.flatten()
            # nans = np.isnan(wave_NIRSpec)
            # nans = np.zeros_like(wave_NIRSpec, dtype=bool)
            wave_NIRSpec = wave_NIRSpec
            
            self.flux = np.interp(wave_NIRSpec, self.wave, self.flux)
            self.wave = wave_NIRSpec
        else:
            self.flux = np.array([np.interp(wave_NIRSpec[i], self.wave, self.flux) for i in range(wave_NIRSpec.shape[0])])
            self.wave = wave_NIRSpec
            
        return self
    
    def scale_flux(self, R_jup=1.0, d_pc=10.0):
            
        assert hasattr(self, 'wave'), 'Load grid first'
        # convert to cm
        R_cm = R_jup * 7.1492e9
        d_cm = d_pc * 3.0857e18
        # scale flux
        self.flux *= (R_cm / d_cm)**2
        return self
    
    def apply_flux_factor(self, factor):
        
        self.flux_factor = factor
        if isinstance(self.flux, list):
            self.flux = [f * factor for f in self.flux]
        else:
            self.flux *= factor
        return self
    
    def rebin(self, Nbin=None, Nbin_spitzer=None):
        self.Nbin = Nbin
        self.Nbin_spitzer = Nbin_spitzer
        
        if self.Nbin_spitzer is not None:
            spec_jwst = SpectrumJWST(wave=self.wave[:-1], flux=self.flux[:-1]).rebin(self.Nbin)
            spec_spitzer = SpectrumJWST(wave=self.wave[-1], flux=self.flux[-1]).rebin(self.Nbin_spitzer)
            
            # self.wave = np.vstack((spec_jwst.wave, spec_spitzer.wave))
            # self.flux = np.vstack((spec_jwst.flux, spec_spitzer.flux))
            # print(f' spec_jwst.wave.shape: {spec_jwst.wave.shape}')
            # print(f' spec_spitzer.wave.shape: {spec_spitzer.wave.shape}')
            self.wave = af.make_array([*spec_jwst.wave, spec_spitzer.wave])
            self.flux = af.make_array([*spec_jwst.flux, spec_spitzer.flux])
        else:
            spec = SpectrumJWST(wave=self.wave, flux=self.flux).rebin(self.Nbin)
            self.wave = spec.wave
            self.flux = spec.flux
            
        return self
    
    # def resample(self, wave_step=None, new_wave=None):
    #     if new_wave is None:
    #         assert wave_step is not None, 'Either wave_step or new_wave must be provided'
    #         new_wave = np.arange(np.nanmin(self.wave), np.nanmax(self.wave), wave_step)
        
    #     if isinstance(new_wave, list):
    #         new_flux = []
    #         for i in range(len(new_wave)):
    #             # print( f' Resampling {i+1}/{len(new_wave)}')
    #             # print(self.wave)
    #             # print(self.flux)
    #             # print(new_wave[i])
    #             new_flux.append(spectres(new_wave[i], self.wave, self.flux))
    #         self.wave = new_wave
    #         self.flux = new_flux
    #         return self
    #     else:
    #         self.flux = spectres(new_wave, self.wave, self.flux)
    #         assert np.isnan(self.flux).sum() < np.size(self.flux), 'ALL NaNs in resampled flux'
    #         self.wave = new_wave
    #         return self
    def resample(self, wave_step=None, new_wave=None):
        if new_wave is None:
            assert wave_step is not None, 'Either wave_step or new_wave must be provided'
            new_wave = np.arange(np.nanmin(self.wave), np.nanmax(self.wave), wave_step)
        
        if isinstance(new_wave, list):
            new_flux = []
            for i in range(len(new_wave)):
                re = Resample(wave=self.wave, flux=self.flux)(new_wave[i])
                new_flux.append(re[0])
                
            self.wave = new_wave
            self.flux = new_flux
            return self
        else:
            self.flux = Resample(wave=self.wave, flux=self.flux)(new_wave)[0]
            assert np.isnan(self.flux).sum() < np.size(self.flux), 'ALL NaNs in resampled flux'
            self.wave = new_wave
            return self
    
    
    def blackbody_disk(self, T=None, R=None, d=None, parallax=None, wave_cm=None, add_flux=True,
                       flux_factor=1.0):
        ''' Calculate the emission of a disk from a single blackbody
        
        Parameters:
        T : float
            Temperature of the disk in [K]
        R : float
            Radius of the disk in [R_jup], equivalent to: R^2 = R_out^2 - R_in^2
        d : float, optional
            Distance to the system in [pc]
        parallax : float, optional
            Parallax of the system in [mas]
        wave_cm : ndarray, optional
            Wavelength array in [cm]
        
        Returns:
        F_disk : ndarray
            Flux of the disk in [erg s^-1 cm^-2 nm^-1]
        
        '''
        assert (d is not None) or (parallax is not None), 'Either distance [pc] or parallax [mas] must be provided'
        assert add_flux, 'add_flux must be True'
        # store attributes in dictionary disk_blackbody
        self.blackbody_disk_args = {'T': T, 'R': R, 'd': d, 'parallax': parallax}
        if parallax is not None:
            d = 1e3/parallax # distance in [pc]
        if wave_cm is None:
            assert self.wave_unit == 'nm', 'Invalid wave unit, must be [nm]'
            if isinstance(self.wave, list):
                wave_cm = [wave * 1e-7 for wave in self.wave]
            else:
                wave_cm = self.wave * 1e-7 # [nm] -> [cm]
                
        if isinstance(wave_cm, list):
            flux_disk, new_flux = [], []
            for i in range(len(wave_cm)):
                bb = af.blackbody(wave_cm=wave_cm[i], T=T)
                bb *= (R*nc.r_jup_mean / (d * nc.pc))**2
                bb *= flux_factor
                flux_disk.append(bb)
                new_flux.append(self.flux[i] + bb)
            self.flux_disk = flux_disk
            self.flux = new_flux
            self.wave = [wave_cm[i] * 1e7 for i in range(len(wave_cm))]
            return self
        else:
            # flux of a blackbody disk in units of [erg s^-1 cm^-2 nm^-1]
            bb = af.blackbody(wave_cm=wave_cm, T=T)
            # the factor of R^2 is to scale the flux to the disk size
            bb *= (R*nc.r_jup_mean / (d * nc.pc))**2
            
            self.flux_disk = bb 
        # if add_flux:
            self.flux += self.flux_disk
            return self
        # return bb
    
    def plot(self, ax=None, **kwargs):
        
        ax = ax or plt.gca()
        
        if isinstance(self.wave, list):
            for i in range(len(self.wave)):
                ax.plot(self.wave[i], self.flux[i], **kwargs)
            return ax
        
        if getattr(self, 'Nbin', None) is not None:
            ax.scatter(self.wave, self.flux, **kwargs)
        else:
            ax.plot(self.wave, self.flux)
        return ax
    
    def copy(self):
        return copy.deepcopy(self)
        
        
        
if __name__ == '__main__':
    
    bt = BTSettl()
    
    create_grid = False
    if create_grid:
        bt.load_full_dataset()
        fg = bt.prepare_grid(teff_range=[2000,2900], logg_range=[2.5, 5.0], out_res=3000)
    else:
        bt.load_dataset(bt.path / 'BTSETTL.nc')
        bt.set_interpolator()
        # bt.get_spec(teff=2320, logg=4.0)
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        
        # for teff in range(2300, 2500, 25):
        for logg in np.arange(3.6, 3.7, 0.2):
            bt.get_spec(teff=2400, logg=logg)
            ax.plot(bt.wave, bt.flux, label=f'logg={logg}', alpha=0.3)
            bt.resample(wave_step=20.)
            ax.scatter(bt.wave, bt.flux, label=f'logg={logg}', 
                       alpha=0.9, 
                       s=5.,
                       color=ax.lines[-1].get_color())

            
        ax.legend()
        plt.show()
        
    
    
    