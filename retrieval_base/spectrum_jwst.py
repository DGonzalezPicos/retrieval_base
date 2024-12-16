import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
import pickle
import os
import copy
from spectres import spectres 

from PyAstronomy import pyasl
import petitRADTRANS.nat_cst as nc
# from petitRADTRANS.retrieval import rebin_give_width as rgw

import retrieval_base.auxiliary_functions as af
import retrieval_base.figures as figs
from retrieval_base.resample import Resample

from broadpy import InstrumentalBroadening
from broadpy.utils import load_nirspec_resolution_profile

class SpectrumJWST:
    
    # spectral resolution = 2700 for gratings g140h, g235h, g395h
    settings = dict(
        g140h_f100lp = (0.97, 1.45, 1.82),
        g235h_f170lp = (1.66, 2.45, 3.18),
        g395h_f290lp = (2.87, 4.15, 5.28),
    )
    
    resolution = 2700.
    w_set = 'NIRSpec' # JWST/NIRSpec
    # expected units to work with
    wave_unit = 'nm'
    flux_unit = 'Jy'
    n_dets = 1 # default
    
    def __init__(self, wave=None, flux=None, err=None, target=None, grating=None, file=None,
                 Nedge=10):
        self.wave = wave
        self.flux = flux
        self.err = err
        self.target = target
        self.grating = grating
        self.file = file
        self.Nedge = Nedge
        if self.file is not None:
            print(f'Reading {self.file}')
            self.read_data(units='erg/s/cm2/nm')
            
        # if self.grating is not None: # deprecated
            # self.split_grating(keep='both')
            
            
    def read_data(self, grating=None, units='mJy'):
        with fits.open(self.file) as hdul:
            data = hdul[1].data
            self.wave, self.flux, self.err = data['WAVELENGTH'], data['FLUX'], data['ERR'] # units [um, Jy, Jy]
            print(f' [SpectrumJWST.read_data] self.wave.shape = {self.wave.shape}')
            print(f' [SpectrumJWST.read_data] Wave range: {np.nanmin(self.wave):.2f} - {np.nanmax(self.wave):.2f} um')
        self.wave *= 1e3 # um to nm
        
        self.wave_unit = 'nm'
        self.flux_unit = 'Jy'
        
        # if grating is not None:
            # file name must be in the format 'TWA28_g235h-f170lp.fits'
        self.grating = str(self.file).split('_')[1].split('.fits')[0].replace('-', '_')
        print(f'grating: {self.grating}')
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
            
        # drop pixels with nan values in wavelength solution
        nans_in = np.isnan(self.wave)
        self.wave = self.wave[~nans_in]
        self.flux = self.flux[~nans_in]
        self.err = self.err[~nans_in]
        
            
        # split into two filters
        split_id = self.flux.shape[0] // 2
        print(f' Splitting data into two filters at {split_id}')
        # self.wave = np.array([self.wave[:split_id], self.wave[split_id:]])
        # self.flux = np.array([self.flux[:split_id], self.flux[split_id:]])
        # self.err = np.array([self.err[:split_id], self.err[split_id:]])
        attrs = ['wave', 'flux', 'err']
        for attr in attrs:
            setattr(self, attr, af.make_array([getattr(self, attr)[:split_id], getattr(self, attr)[split_id:]]))
        print(f' [SpectrumJWST.read_data] self.wave.shape = {self.wave.shape}')
        # change shape into (1, n_pixels)
        # self.wave = self.wave[None, :]
        # self.flux = self.flux[None, :]
        # self.err = self.err[None, :]
        
        if self.Nedge > 0:
            self.clip_det_edges(n=self.Nedge)
            
        
        # fix nans in self.wave
        # self.fix_wave_nans()
        # self.n_orders = 1
        return self
    
    def load_gratings(self, files):
        # self.gratings = [f.split('_')[1].split('-')[0] for f in files]
        
        if len(files) == 1:
            self.__init__(file=files[0], Nedge=self.Nedge)
        else:
            spec_list = []
            for f in files:
                spec_list.append(SpectrumJWST(file=f, Nedge=self.Nedge))
                
            self += spec_list
            print(f'Loaded {len(spec_list)} gratings')
            
        print(f' shape of wave: {self.wave.shape}')
        print(f' shape of flux: {self.flux.shape}')
        return self
    
    def select_filter(self, filter_id=0):
        
        assert len(self.flux.shape) == 3, f'Flux array must be 3D, not {self.flux.shape}'
        assert self.flux.shape[0] > 1, f'Flux array must have more than one filter, not {self.flux.shape[0]}'
        
        attrs = ['wave', 'flux', 'err']
        for attr in attrs:
            setattr(self, attr, getattr(self, attr)[filter_id][None,...])
            
        print(f' Selected filter {filter_id} with cenwave {np.nanmedian(self.wave):.2f} nm')
        print(f' shape of wave: {self.wave.shape}')
        return self
    
    
    def __add__(self, spec_list):
        """ Add a list of SpectrumJWST objects with proper padding"""
        attrs = ['wave', 'flux', 'err']
        # Determine the maximum length of the wave arrays in the input chunks
        n = max([len(getattr(spec, 'wave')[0]) for spec in spec_list])
        # print(f'Padding arrays to length {n}')
        
        for i, spec in enumerate(spec_list):
            for attr in attrs:
                if hasattr(spec, attr):
                    n_orders = len(getattr(spec, attr))
                    print(f'[__add__] n_orders = {n_orders}')
                    # attr_pad = np.nan * np.ones((2, n))
                    attr_pad = np.nan * np.ones((n_orders, n))
                    for order in range(n_orders):
                        n_order = len(getattr(spec, attr)[order])
                        if n_order < n:
                            print(f' Padding order {order} from {n_order} to {n}...')
                            attr_pad[order] = np.pad(getattr(spec, attr)[order],
                                                        (0, n-len(getattr(spec, attr)[order])),
                                                        mode='constant', constant_values=np.nan)
                        else:
                            attr_pad[order] = getattr(spec, attr)[order]
                    setattr(spec_list[i], attr, attr_pad)
                    # print(f' shape of {attr}: {getattr(spec_list[i], attr).shape}')
        new_wave = np.vstack([spec.wave for spec in spec_list])
        # sort first axis
        idx = np.argsort(np.nanmedian(new_wave, axis=1))
        
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, np.vstack([getattr(spec, attr) for spec in spec_list])[idx])
        # self.set_n_orders()
        assert self.n_orders > 1, 'No data loaded'
        self.grating = np.array([spec.grating for spec in spec_list])[idx]
        # Stack the arrays                    
        return self
    
    @property
    def gratings(self):
        return self.grating # alias
    
    @property
    def nans(self):
        return (np.isnan(self.wave) | np.isnan(self.flux))
    
    
    def fix_wave_nans(self):
        '''Replace NaNs in the wavelength array with linear interpolation'''
        for i in range(self.n_orders):
            for j in range(self.n_dets):
                wave_ij = self.wave[i, j]  # Extract the 1D array
                mask = np.isnan(wave_ij)
                if np.any(mask):  # Check if there are any NaNs to interpolate
                    wave_ij[mask] = np.interp(
                        np.flatnonzero(mask), 
                        np.flatnonzero(~mask), 
                        wave_ij[~mask]
                    )
                    # No need to reassign back to self.wave[i, j] since wave_ij is a reference

        # Assert to ensure all NaNs are replaced
        assert np.sum(np.isnan(self.wave)) == 0, f'Wave array contains {np.sum(np.isnan(self.wave))} NaNs'
        return self
    
    def flux_scaling(self, factor):
        
        self.flux_factor = factor
        if isinstance(self.flux, list):
            self.flux = [f_i * factor for f_i in self.flux]
            if hasattr(self, 'err'):
                self.err = [e_i * factor for e_i in self.err]
                
        else:
            self.flux *= factor
            if hasattr(self, 'err'):
                self.err *= factor
        return self

    
    @property
    def mask_isfinite(self):
        return np.isfinite(self.flux)
    @property
    def n_orders(self):
        if isinstance(self.wave, (list, np.ndarray)):
            return len(self.wave)
        return 1
            
    
    def split_grating(self, break_wave=None, keep=1, grating=None, fig_name=None):
        '''Split data of one gratings into chunks'''
        assert hasattr(self, 'grating'), f' No grating attribute found in the SpectrumJWST object'
        # mask = self.wave < break_wave
        mask = self.wave < self.settings[self.grating][1]*1e3
        
        wave0, flux0, err0 = self.wave[mask], self.flux[mask], self.err[mask]
        wave1, flux1, err1 = self.wave[~mask], self.flux[~mask], self.err[~mask]
        print(f'Shape of first chunk: {wave0.shape}')
        print(f'Shape of second chunk: {wave1.shape}')
        
        if fig_name:
            self._plot_chunks(wave0, flux0, err0, wave1, flux1, err1, fig_name)

        if keep == 0:
            self._update_data(wave0, flux0, err0)
            print(f' Keeping first chunk with cenwave {np.nanmedian(self.wave):.2f} nm')
        elif keep == 1:
            self._update_data(wave1, flux1, err1)
            print(f' Keeping second chunk with cenwave {np.nanmedian(self.wave):.2f} nm')
        elif keep == 'both':
            self._update_data_both(wave0, flux0, err0, wave1, flux1, err1)

        return self

    def _update_data(self, wave, flux, err):
        self.wave, self.flux, self.err = wave, flux, err

    def _update_data_both(self, wave0, flux0, err0, wave1, flux1, err1):
        n = max(len(wave0), len(wave1))
        self.wave, self.flux, self.err = (np.nan * np.ones((2, n)) for _ in range(3))

        self.wave[0, :len(wave0)] = wave0
        self.flux[0, :len(wave0)] = flux0
        self.err[0, :len(wave0)] = err0

        self.wave[1, :len(wave1)] = wave1
        self.flux[1, :len(wave1)] = flux1
        self.err[1, :len(wave1)] = err1
        if self.Nedge > 0:
            self.clip_det_edges(n=self.Nedge)

        print(f' Keeping both chunks with cenwaves {np.nanmedian(self.wave[0]):.2f} nm and {np.nanmedian(self.wave[1]):.2f} nm')

    def _plot_chunks(self, wave0, flux0, err0, wave1, flux1, err1, fig_name):
        fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
        ax[0].plot(wave0, flux0, label='First chunk', color='k')
        ax[1].plot(wave1, flux1, label='Second chunk', color='k')
        ax[0].errorbar(wave0.min(), np.nanmedian(flux0), yerr=np.nanmean(err0), fmt='o', color='r', markersize=4)
        ax[1].errorbar(wave1.min(), np.nanmedian(flux1), yerr=np.nanmean(err1), fmt='o', color='r', markersize=4)
        ax[0].set(ylabel=f'Flux / {self.flux_unit}')
        ax[1].set(xlabel=f'Wavelength / {self.wave_unit}', ylabel=f'Flux / {self.flux_unit}')
        fig.savefig(fig_name)
        print(f'--> Saved {fig_name}')
        
    def plot_orders(self, fig_name=None, grid=False, **kwargs): # DEPRECATED
        # use PDF pages to save a page for every order
        
        color = kwargs.pop('color', 'k')
        kwargs['lw'] = kwargs.pop('lw', 0.8)
        with PdfPages(fig_name) as pdf:
            for i in range(self.n_orders):
                for j in range(self.n_dets):
                    fig, ax = plt.subplots(1,1, figsize=(12, 4), gridspec_kw={'hspace': 0.1,
                                                                              'left': 0.05, 
                                                                              'right': 0.98,
                                                                              'top': 0.95,
                                                                                'bottom': 0.12})
                    ax.plot(self.wave[i,j], self.flux[i,j], label=f'Order {i}', color=color, alpha=0.9, **kwargs)
                    if hasattr(self, 'err'):
                        ax.fill_between(self.wave[i,j], self.flux[i,j]-self.err[i,j], self.flux[i,j]+self.err[i,j], alpha=0.3, color=color)
                        
                    nans_ij = np.isnan(self.flux[i,j])
                    print(f' Order {i}, detector {j} has {np.sum(nans_ij)}/{len(nans_ij)} NaNs')
                    # if hasattr(self, 'flux_uncorr'):
                        # ax.plot(self.wave[i,j], self.flux_uncorr[i,j], label='Uncorrected', color='r', alpha=0.5)
                        # ax.fill_between(self.wave[i,j], self.flux_uncorr[i,j]-self.err_uncorr[i,j], self.flux_uncorr[i,j]+self.err_uncorr[i,j], alpha=0.3, color='r')
                    ax.set(xlabel=f'Wavelength / {self.wave_unit}', ylabel=f'Flux / {self.flux_unit}',
                           xlim=(np.nanmin(self.wave[i,j]), np.nanmax(self.wave[i,j])))
                    if grid:
                        ax.grid()
                    pdf.savefig(fig)
                    plt.close(fig)
        print(f'--> Saved {fig_name}')
        return self
        
    def sigma_clip_reshaped(self,
                            use_flux=True,
                            sigma=3, 
                            width=5, 
                            max_iter=5, 
                            fun='median',
                            fig_name=False,
                            ):
        
        array = self.flux if use_flux else self.err
        assert use_flux == False, f'Only sigma-clip on errors is implemented'
        debug = (fig_name != None)
        if debug:
            # keep a copy of the original data
            self.flux_uncorr = self.flux.copy()
            self.err_uncorr = self.err.copy()
            fig, ax = plt.subplots(self.n_orders, 1, figsize=(10, 8), tight_layout=True)
            
        for order in range(self.n_orders):
            for det in range(self.n_dets):
                nans_in = np.isnan(self.err[order,det])
                # print(f' Average SNR (BEFORE) = {np.nanmean(self.flux[order,det]/self.err[order,det]):.1f}')

                clip  = af.sigma_clip(y=np.copy(self.err[order,det]), sigma=sigma, width=width, 
                                max_iter=max_iter, fun=fun, replace=False,
                                replace_w_fun=True)
                
                # self.flux[order,det,clip] = np.nan
                # self.flux[order,det,:] = clip # this is the flux with bad values replaced by the function values
                if debug:
                    # print(f' Clipped {np.sum(np.isnan(clip)) - np.sum(nans_in)} points in order {order}, detector {det}')
                    ax[order].plot(self.wave[order,det], self.err[order,det],
                                   label=f'std={np.nanstd(self.err[order,det]):.2e}', color='k')
                    
                    ax[order].plot(self.wave[order,det], clip, label=f'std={np.nanstd(clip):.2e}', color='r')
                    ax[order].legend(frameon=False)
                self.err[order,det,] = clip
                # print(f' Mean error {order},{det} = {np.nanmean(self.err[order,det]):.2e}')
                print(f' Mean SNR ~ {np.nanmean(self.flux[order,det]/self.err[order,det]):.1f}')
        if debug:
            fig.savefig(fig_name)
            print(f'--> Saved {fig_name}')
            plt.close(fig)
        return self
    
    def reshape(self, n_orders=1, n_dets=1):
        # self.n_orders = n_orders
        self.n_dets = n_dets
        shape_in = self.flux.shape
        attrs = ['wave', 'flux', 'err']
        
        rs = lambda x: x.reshape(n_orders, n_dets, -1) if n_dets > 0 else x.reshape(n_orders, -1)
        for attr in attrs:
            if hasattr(self, attr):
                # setattr(self, attr, getattr(self, attr).reshape(n_orders, n_dets, -1))
                setattr(self, attr, rs(getattr(self, attr)))
        shape_out = self.flux.shape
        print(f' Reshaped data from {shape_in} to {shape_out}')
        return self
    
    def squeeze(self):
        attrs = ['wave', 'flux', 'err']
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).squeeze())
        # self.n_orders = self.flux.shape[0]
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
    
    def clip_det_edges(self, n=20):
        assert len(self.flux.shape) == 2, f'Data must be reshaped to (orders, pixels) instead of {self.flux.shape}'
        print(f' Clipping {n} pixels from the edges of each order')
        self.flux[..., :n] = np.nan
        self.flux[...,-n:] = np.nan
        return self
        
    def plot(self, ax=None, style='plot', **kwargs):
        
        assert hasattr(self, 'wave') and hasattr(self, 'flux'), 'No data to plot'
        if ax is None:
            fig, ax = plt.subplots()
        color = kwargs.pop('color', 'k')
        lw = kwargs.pop('lw', 1)
        
        for i in range(self.n_orders):
            
            if style in ['plot', 'scatter']:
                getattr(ax, style)(self.wave[i], self.flux[i], color=color, lw=lw, **kwargs)
                if hasattr(self, 'err'):
                    ax.fill_between(self.wave[i], self.flux[i]-self.err[i], self.flux[i]+self.err[i], alpha=0.3, color=color)
                    
            if style=='errorbar':
                ax.errorbar(self.wave[i], self.flux[i], yerr=self.err[i], color=color, lw=lw, fmt='o', **kwargs)
        ax.set_xlabel(f'Wavelength / {self.wave_unit}')
        ax.set_ylabel(f'Flux / {self.flux_unit}')
        return ax
        
    
    def flatten(self):
        attrs = ['wave', 'flux', 'err']
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).flatten())
                
        self.n_orders = 1
        return self
    
    def rebin(self, nbin=100):
        """ Bin spectrum every `nbin` data points by taking the mean of each bin. 
        Propagate the errors by taking the square root of the sum of the squared errors in each bin.
        """
        
        if len(self.flux.shape) > 1:
            # print(f' Flattening from {self.flux.shape} to 1D arrays {np.prod(self.flux.shape)}')
            assert len(self.flux.shape) == 2, f'Flux array must be 2D, not {self.flux.shape}'
            n_orders = self.flux.shape[0]
            wave_rb, flux_rb, err_rb = ([] for _ in range(3))
            for i in range(n_orders):
                out = af.rebin(self.wave[i], self.flux[i], nbin, 
                               err=self.err[i] if getattr(self, 'err', None) is not None else None)
                wave_rb.append(out[0])
                flux_rb.append(out[1])
                
                if len(out) > 2:
                    err_rb.append(out[2])
                    
            # self.wave = np.array(wave_rb)
            self.wave = af.make_array(wave_rb)
            self.flux = af.make_array(flux_rb)
            if len(err_rb) > 0:
                self.err =  af.make_array(err_rb)
            
        else:
            
            out = af.rebin(self.wave, self.flux, nbin, err=getattr(self, 'err', None))
            self.wave, self.flux = out[:2]
            if len(out) > 2:
                self.err = out[2]
        return self
    
    
    def copy(self):
        """ Return a copy of the Spectrum instance. """
        return copy.deepcopy(self)

    
    def resample_old(self, wave_step, replace_wave_flux=True, use_mean_err=True):
            
        # Resample the model spectrum onto a new wavelength grid using spectres
        self.wave_step = wave_step
        
        if len(self.flux.shape) > 1:
            new_wave, new_flux, new_err = ([] for _ in range(3))
            for i in range(self.n_orders):
                assert wave_step < (np.nanmax(self.wave[i]) - np.nanmin(self.wave[i])), 'Wave step too large'

                new_wave.append(np.arange(np.nanmin(self.wave[i]), np.nanmax(self.wave[i]), wave_step))
                nans_i = np.isnan(self.flux[i]) | np.isnan(self.wave[i])
                # assert np.sum(nans_i) == 0, 'NaNs in flux'
                if all(nans_i):
                    print(f'All NaNs in order {i}')
                    continue
                # print(new_wave[-1])
                # print(self.wave[i][~nans_i])
                # print(self.flux[i][~nans_i])
                # print(self.err[i][~nans_i])
                re = spectres(new_wave[-1], self.wave[i][~nans_i], self.flux[i][~nans_i], 
                              spec_errs=self.err[i][~nans_i], 
                              fill=np.nan, 
                              verbose=False)
                assert np.sum(np.isnan(re[0])) < np.size(re[0]), 'All NaNs in rebinned flux'
                new_flux.append(re[0])
                if use_mean_err:
                    new_err.append(np.mean(self.err[i][~nans_i]) * np.ones_like(re[1]))
                else:
                    new_err.append(re[1])
            
            if replace_wave_flux:
                self.flux = new_flux
                self.err = new_err
                self.wave = new_wave
                
                return self
        
            return new_wave, new_flux, new_err
        
        else:
            assert wave_step < np.nanmax(self.wave) - np.nanmin(self.wave), 'Wave step too large'
            new_wave = np.arange(np.nanmin(self.wave), np.nanmax(self.wave), wave_step)
            re = spectres(new_wave, self.wave, self.flux, spec_errs=self.err, fill=np.nan)
            self.wave, self.flux, self.err = new_wave, re[0], re[1]
            return self
        
    def resample(self, wave_step):
        
        
        assert self.n_orders > 1, 'Works with multiple orders only, at least [1, n_pixels]'
        
        new_wave, new_flux, new_err = ([] for _ in range(3))
        for i in range(self.n_orders):
            
            # print(new_wave)
            nans_i = np.isnan(self.flux[i]) | np.isnan(self.wave[i])
            # ignore nans
            if all(nans_i):
                print(f'All NaNs in order {i}')
                continue
            if len(self.wave[i][~nans_i]) < 2:
                print(f'Not enough data points in order {i}')
                continue
            new_wave_i = np.arange(np.nanmin(self.wave[i][~nans_i]), np.nanmax(self.wave[i][~nans_i]), wave_step)
            if len(new_wave_i) < 4:
                print(f'Not enough data points in order {i} after resampling')
                continue
            new_flux_i, new_cov_i = Resample(wave=self.wave[i][~nans_i], flux=self.flux[i][~nans_i], 
                                         flux_err=self.err[i][~nans_i])(new_wave_i)
            new_wave.append(new_wave_i)
            new_flux.append(new_flux_i)
            new_err.append(np.sqrt(np.diag(new_cov_i)))
        
        self.wave = new_wave
        self.flux = new_flux
        self.err = new_err
        return self
    
    
    def scatter_overlapping_points(self, plot=False):
        
        
        self.err_s = np.ones(self.n_orders)
        for i in range(1,self.n_orders):
            wave_0 = np.squeeze(self.wave[i-1])
            wave_1 = np.squeeze(self.wave[i])
            
            mask_0 = (wave_0 > np.nanmin(wave_1)) & (wave_0 < np.nanmax(wave_1))
            if np.sum(mask_0) > 0:
                print(f'Order {i-1} has {np.sum(mask_0)} overlapping points with order {i}')
                flux_0 = np.squeeze(self.flux[i-1])
                flux_1 = np.squeeze(self.flux[i])
                
                mask_1 = (wave_1 > np.nanmin(wave_0)) & (wave_1 < np.nanmax(wave_0))

                # interpolate 0 on 1 and calculate the difference
                flux_0_interp = np.interp(wave_1[mask_1], wave_0, flux_0)
                diff = flux_0_interp - flux_1[mask_1]
                # use scatter as new uncertainty for the overlapping points
                scatter = np.nanstd(diff)
                print(f' Scatter in overlapping points: {scatter:.2e}')
                snr_overlap = np.nanmean(flux_0_interp) / scatter
                print(f' SNR in overlapping points: {snr_overlap:.2f}')
    
                
                # input SNR
                snr = np.nanmean(flux_0_interp) / np.nanmean(self.err[i-1])
                print(f' SNR in order {i-1}: {snr:.2f}')
                
                # scaling factor
                self.err_s[i] = snr / snr_overlap if self.err_s[i]==1.0 else np.mean([self.err_s[i], snr / snr_overlap])
                print(f' Scaling uncertainty by {self.err_s[i]:.2f}')
    
                if plot:
                    fig, ax = plt.subplots(2,1, figsize=(10,5), gridspec_kw={'height_ratios': [2,1]},sharex=True)
                    ax[0].plot(wave_0, flux_0, label='Order 0', color='k', alpha=0.4)
                    ax[0].plot(wave_1, flux_1, label='Order 1', color='r', alpha=0.4)
                    ax[0].scatter(wave_1[mask_1], flux_0_interp, color='r', s=3)
                    ax[0].scatter(wave_1[mask_1], flux_1[mask_1], color='k', s=3)
                    ax[0].legend()
                    ax[1].scatter(wave_1[mask_1], diff, color='k', s=3)
                    plt.show()
                    
        return self
            
    def apply_error_scaling(self, mode='same', default=100.):
        
        assert hasattr(self, 'err_s'), 'No error scaling factors found'
        assert mode == 'same', 'Only same scaling is implemented' # TODO: enable different scalings for the orders
        # find the err_s that is greater than 1
        shape_in = self.err.shape
        if np.any([s > 1 for s in self.err_s]):
            print(f'Applying error scaling factor (max): {np.max(self.err_s)}')
        
            self.err = [self.err[i] * np.max(self.err_s) for i in range(self.n_orders)]
        else:
            self.err = [self.err[i] * default for i in range(self.n_orders)]
            
        self.err = np.array(self.err).reshape(shape_in)
        return self              
           
    def mask_lines(self, lines={'weird_line':(2050, 2080)}):
        
        for name, wrange in lines.items():
            mask = (self.wave > wrange[0]) & (self.wave < wrange[1])
            if np.sum(mask) > 0:
                print(f' Masking {name} ({wrange}) with {np.sum(mask)} points.')
                self.flux[mask] = np.nan
        return self        
        
            
            
        
        
        
        

    
if __name__ == '__main__':
    import pathlib
    path = pathlib.Path('TWA28/jwst/')
    gratings = [
            'g140h-f100lp', 
              'g235h-f170lp', 
              'g395h-f290lp']
    gratings_prefix = ['g140h', 'g235h', 'g395h']
    files = [path/f'TWA28_{g}.fits' for g in gratings]

    spec = SpectrumJWST(Nedge=0).load_gratings(files)
    
    wave_g = [spec.wave[i, np.isfinite(spec.wave[i])] for i in range(spec.n_orders)]
    fig, ax = plt.subplots(1,1, figsize=(10,5), tight_layout=True)
    wave_add_list = []
    grating_list = []
    for i in range(spec.n_orders):
        
        # n_add = 4000 - len(wave_g[i])
        n_add = 60
        pixels = np.arange(n_add, len(wave_g[i]) + n_add)
        pixels_add = np.arange(0, len(wave_g[i]) + n_add*2)
        ax.plot(pixels, wave_g[i], label=gratings_prefix[i])
        # fit 2nd order polynomial to the data
        p = np.polyfit(pixels, wave_g[i], 2)
        wave_add = np.polyval(p, pixels_add)
        print(f' Fit for {gratings_prefix[i]}: {p}')
        print(f' New shape of wave: {wave_add.shape}')
        print(f' New wave range for {gratings_prefix[i]}: {np.nanmin(wave_add):.2f} - {np.nanmax(wave_add):.2f} nm')
        ax.plot(pixels_add, wave_add, color='r', lw=0.5
                , label=f'{gratings_prefix[i]} fit')
        
        wave_add_list.append(wave_add * 1e-3) # save in micron
        grating_list.append([gratings_prefix[i]]*len(wave_add))
        
    ax.set(ylabel='Pixel', xlabel='Wavelength / nm')
    ax.legend()
    # plt.show()
    fig_name = path/'nirspec_wavegrid_TWA28.png'
    fig.savefig(fig_name)
    plt.close(fig)
    
    # save .txt with two columns: wavelength, grating
    
    save = True
    
    if save:
        wave = np.concatenate(wave_add_list)
        grating = np.concatenate(grating_list)
        file_out = path/'nirspec_wavegrid_TWA28.txt' # with extended range of 60 pixels (each side)
        with open(file_out, 'w') as f:
            for w, g in zip(wave, grating):
                f.write(f'{w:12.8f} {g}\n')
        print(f'--> Saved {file_out}')
    # spec.reshape(spec.n_orders, 1)
    # spec.sigma_clip_reshaped(use_flux=False, 
    #                                 sigma=3, 
    #                                 width=31, 
    #                                 max_iter=5,
    #                                 fun='median', 
    #                                 debug=False)
    # spec.squeeze()
    
    # spec.scatter_overlapping_points(plot=True)
    # spec.apply_error_scaling()
    # plot = False
    # if plot:
    #     fig, ax = plt.subplots(1,1, figsize=(10,5))
        
        
    #     spec.plot(ax, style='plot', color='k', alpha=0.8)
    #     spec.resample(wave_step=50.)
    #     spec.plot(ax, style='errorbar', color='r', alpha=0.8, ms=4.)
    #     spec.plot(ax, style='plot', color='r', alpha=0.8)
    #     plt.show()
    
    
    
    
    