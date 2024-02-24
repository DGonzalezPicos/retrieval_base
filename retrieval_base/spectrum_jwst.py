import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle
import os

from PyAstronomy import pyasl
import petitRADTRANS.nat_cst as nc
# from petitRADTRANS.retrieval import rebin_give_width as rgw

import retrieval_base.auxiliary_functions as af
import retrieval_base.figures as figs


class SpectrumJWST:
    
    # spectral resolution = 2700 for grisms g140h, g235h, g395h
    settings = dict(
        g140h_f100lp = (0.97, 1.82),
        g235h_f170lp = (1.66, 3.05),
        g395h_f290lp = (2.87, 5.28),
    )
    
    resolution = 2700.
    w_set = 'G395H_F290LP'
    # expected units to work with
    wave_unit = 'nm'
    flux_unit = 'Jy'
    
    def __init__(self, wave=None, flux=None, err=None, target=None, grism=None, file=None):
        self.wave = wave
        self.flux = flux
        self.err = err
        self.target = target
        self.grism = grism
        self.file = file
        if self.file is not None:
            print(f'Reading {self.file}')
            self.read_data(units='erg/s/cm2/nm')
            
            
    def read_data(self, grism=None, units='mJy'):
        with fits.open(self.file) as hdul:
            data = hdul[1].data
            self.wave, self.flux, self.err = data['WAVELENGTH'], data['FLUX'], data['ERR'] # units [um, Jy, Jy]
        
        self.wave *= 1e3 # um to nm
        
        self.wave_unit = 'nm'
        self.flux_unit = 'Jy'
        
        if grism is not None:
            # file name must be in the format 'TWA28_g235h-f170lp.fits'
            self.grism = self.file.split('_')[1].split('-')[0]
        
        if units == 'mJy':
            self.flux *= 1e3
            self.err *= 1e3
            self.flux_unit = 'mJy'
        if units == 'erg/s/cm2/nm':
            # [Jy] --> [erg/s/cm2/Hz]
            self.flux *= 1e-23
            self.err  *= 1e-23
            
            # [erg/s/cm2/Hz] --> [erg/s/cm2/cm]
            wave_cm = self.wave * 1e-7
            self.flux *= nc.c / (wave_cm**2)
            self.err  *= nc.c / (wave_cm**2)
            
            # [erg/s/cm2/cm] --> [erg/s/cm2/nm]
            self.flux *= 1e-7
            self.err  *= 1e-7
            self.flux_unit = 'erg/s/cm2/nm'
        return self
    
    @property
    def mask_isfinite(self):
        return np.isfinite(self.flux)
    
    def split_grism(self, break_wave, keep=1, grism=None):
        '''Split data of one grisms into chunks'''
        mask = self.wave < break_wave
        
        wave0, flux0, err0 = self.wave[mask], self.flux[mask], self.err[mask]
        wave1, flux1, err1 = self.wave[~mask], self.flux[~mask], self.err[~mask]
        print(f'Shape of first chunk: {wave0.shape}')
        print(f'Shape of second chunk: {wave1.shape}')
        
        if keep == 0:
            self.wave, self.flux, self.err = wave0, flux0, err0
            print(f' Keeping first chunk with cenwave {np.median(self.wave):.2f} nm')

        if keep == 1:
            self.wave, self.flux, self.err = wave1, flux1, err1
            print(f' Keeping second chunk with cenwave {np.median(self.wave):.2f} nm')
        
        return self
    
    def sigma_clip(self, array=None, sigma=3, width=5, max_iter=5, fun='median'):
        '''Sigma clip the spectrum'''
        array = self.flux if array is None else array
        clip = af.sigma_clip(array, sigma=sigma, width=width, 
                              max_iter=max_iter, fun=fun, replace=False)
        print(f'Clipped {np.sum(clip)} points')
        self.flux[clip] = np.nan
        return self
    
    def reshape(self, n_orders=1, n_dets=1):
        self.n_orders = n_orders
        self.n_dets = n_dets
        
        attrs = ['wave', 'flux', 'err']
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).reshape(n_orders, n_dets, -1))
        return self
        
    def prepare_for_covariance(self, prepare_err_eff=True):
        
        assert hasattr(self, 'wave') and hasattr(self, 'flux'), 'No data to prepare'
        assert hasattr(self, 'n_orders') and hasattr(self, 'n_dets'), 'No reshaped data'
        
        # Make a nested array of ndarray objects with different shapes
        self.separation = np.empty((self.n_orders, self.n_dets), dtype=object)

        if prepare_err_eff:
            self.err_eff = np.empty((self.n_orders, self.n_dets), dtype=object)
            # self.flux_eff = np.empty((self.n_orders, self.n_dets), dtype=object)
        
        # Loop over the orders and detectors
        for i in range(self.n_orders):
            for j in range(self.n_dets):
                
                # Mask the arrays, on-the-spot is slower
                mask_ij = self.mask_isfinite[i,j]
                wave_ij = self.wave[i,j,mask_ij]

                # Wavelength separation between all pixels within order/detector
                separation_ij = np.abs(wave_ij[None,:] - wave_ij[:,None])
                
                self.separation[i,j] = separation_ij

                if prepare_err_eff:
                    # Arithmetic mean of the squared flux-errors
                    self.err_eff[i,j] = np.nanmean(self.err[i,j,mask_ij])
        
        return self
    
    
    def find_gaps(self, size=10, debug=True):
        # find gaps in the spectrum 
        # identify clusters of NaNs in flux
        # clusters with size > size are considered gaps
        nans = np.isnan(self.flux)
        gaps = np.zeros_like(self.flux)
        gap = 0
        for i in range(len(nans)):
            if nans[i]:
                gap += 1
            else:
                if gap > size:
                    gaps[i-gap:i] = 1
                gap = 0
                
        # find indices of gaps
        gap_indices = np.where(gaps)[0]
        gap_indices = np.split(gap_indices, np.where(np.diff(gap_indices) != 1)[0]+1)
        gap_indices = [np.array(gap) for gap in gap_indices]
        
        if debug:
            fig, ax = plt.subplots()
            ax.plot(self.wave, self.flux)
            ax.fill_between(self.wave, 0, gaps, alpha=0.3)
            ax.set_xlabel(f'Wavelength [{self.wave_unit}]')
            ax.set_ylabel(f'Flux [{self.flux_unit}]')
            plt.show()
        
        return gaps, gap_indices
    
    def plot(self, ax=None, **kwargs):
        
        assert hasattr(self, 'wave') and hasattr(self, 'flux'), 'No data to plot'
        if ax is None:
            fig, ax = plt.subplots()
        color = kwargs.pop('color', 'k')
        lw = kwargs.pop('lw', 1)
        ax.plot(self.wave, self.flux, color=color, lw=lw, **kwargs)
        if hasattr(self, 'err'):
            ax.fill_between(self.wave, self.flux-self.err, self.flux+self.err, alpha=0.3, color=color)
        ax.set_xlabel(f'Wavelength [{self.wave_unit}]')
        ax.set_ylabel(f'Flux [{self.flux_unit}]')
        return ax
        
    
    
    
if __name__ == '__main__':
    import pathlib
    path = pathlib.Path('TWA28/jwst/')
    # spec = SpectrumJWST(file=path/'TWA28_g235h-f170lp.fits')
    spec = SpectrumJWST(file=path/'TWA28_g395h-f290lp.fits')
    spec.split_grism(4155., keep=1)
    # spec.sigma_clip(sigma=3, width=5, max_iter=5, fun='median')
    spec.sigma_clip(spec.err, sigma=3, width=50, max_iter=5, fun='median')
    spec.plot()
    plt.show()
    
    