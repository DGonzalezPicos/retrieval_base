import numpy as np
import matplotlib.pyplot as plt

import pathlib
import xarray as xr

import os
cwd = os.getcwd()

    


class SonoraElfOwl:
    
    # base = pathlib.Path('/home/dario/phd/SonoraElfOwl')
    if 'dgonzalezpi' in cwd:
        base = pathlib.Path('/home/dgonzalezpi/retrieval_base/sonora_elfowl')
    if 'dario' in cwd:
        base = pathlib.Path('/home/dario/phd/SonoraElfOwl')
        
    available_T = [[1000,1200], [1600, 1800], [2200, 2400]]
    
    flux_unit_labels = {
        'erg/s/cm^2/cm': r'erg s$^{-1}$ cm$^{-2}$ cm$^{-1}$',
        'erg/s/cm^2/nm': r'erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$',
        'Jy' : 'Jy'
    }
    def __init__(self, teff, log_g=3.5, co=0.5, mh=0.0, logkzz=2.0, wave_range=None):
    
        # check teff falls within the available range
        if not any([teff >= T[0] and teff <= T[1] for T in self.available_T]):
            raise ValueError(f'teff must be within the range {self.available_T}')
        # check what range teff falls into
        for T_range in self.available_T:
            if teff >= T_range[0] and teff <= T_range[1]:
                self.T_range = T_range
                break
        print(f'Using Sonora opacities for teff = {teff}')
        print(f'{teff} falls into the range {self.T_range}')
        
        
        # check if the file exists
        self.path = self.base / f'output_{self.T_range[0]:.1f}_{self.T_range[1]:04d}'
        print(f'Checking if {self.path} exists')
        
        self.teff = teff
        # self.g = np.round(10.0**logg, 0)
        self.log_g = log_g
        self.g = 10.0**log_g
        available_g = [17.0, 31.0, 56.0, 100.0, 178.0, 316.0, 562.0, 1000.0, 1780.0, 3160.0]
        if self.g not in available_g:
        # find closest value
            self.g = available_g[np.argmin(np.abs(np.log10(self.g) - np.log10(available_g)))]
            print(f' Using g={self.g:.2f} as closest value to g={10**self.log_g:.2f}')
            self.log_g = np.log10(self.g)
        self.logkzz = logkzz
        available = [2.0, 4.0, 7.0, 8.0, 9.0]   
        assert self.logkzz in available, f'logkzz must be one of {available}'
        self.co = co # carbon to oxygen ratio with respect to solar
        self.mh = mh # metallicity with respect to solar
        self.wave_range = wave_range # [um]

        self.file = self.path / f'spectra_logzz_{self.logkzz:.1f}_teff_{self.teff:.1f}_grav_{self.g:.1f}_mh_{self.mh:.1f}_co_{self.co:.1f}.nc'
        assert self.file.exists(), f'File {self.file} does not exist'


    def load_data(self):
        ds = xr.open_dataset(self.file)
        
        self.wave = ds['wavelength'].values
        
        sort = np.argsort(self.wave)
        self.flux = ds['flux'].values[sort]
        self.wave = self.wave[sort]

        self.wave_unit = 'um'
        self.flux_unit = 'erg/s/cm^2/cm'
        
        if self.wave_range is not None:
            print(f' Applying wave range {self.wave_range}')
            wave_min, wave_max = self.wave_range
            mask = (self.wave >= wave_min) & (self.wave <= wave_max)
            self.wave, self.flux = self.wave[mask], self.flux[mask]
            # make sure it is sorted
            
        # transform flux units to erg/s/cm^2/nm
        self.flux *= 1e-7 # convert to nm
        self.flux_unit = 'erg/s/cm^2/nm'
            
        return self
    
    def get_VMRs(self):
        # get the volume mixing ratios
                   
        # check if species in ds
        ds = xr.open_dataset(self.file)
        self.VMRs = {}
        for attr in ds:
            if attr in ['temperature', 'flux']:
                continue
            self.VMRs[attr] = ds[attr].values

        
        return self
    
    def to_nm(self):
        if self.wave_unit == 'um':
            self.wave *= 1e3
            self.wave_unit = 'nm'
            if self.wave_range is not None:
                self.wave_range = [w*1e3 for w in self.wave_range]
        return self
    
    def to_Jy(self):
        assert self.flux_unit == 'erg/s/cm^2/nm', f'flux_unit must be erg/s/cm^2/nm, not {self.flux_unit}'
        # [erg/s/cm^2/nm] = [erg/s/cm^2/Hz]
        wave_cm = self.wave * 1e-7 # [nm] -> [cm]
        
        
        c = 2.998e10 # [cm/s]
        self.flux /= c / wave_cm**2 # [erg/s/cm^2/Hz]
        self.flux *= 1e7 # [erg/s/cm^2/nm] -> [erg/s/cm^2/cm]

        self.flux *= 1e23 # [Jy]
        self.flux_unit = 'Jy'
        return self
    
    def __str__(self):
        out = f'** Sonora(teff={self.teff}, g={self.g}, logkzz={self.logkzz}) **\n'
        if hasattr(self, 'wave'):
            out += f'- Loaded data from {self.file}\n'
            out += f'- Wavelength range {self.wave.min():.1f} - {self.wave.max():.1f} nm\n'
            out += f'- Number of points {len(self.wave)}\n'
        if hasattr(self, 'VMRs'):
            out += f'- Volume mixing ratios available for: {list(self.VMRs.keys())}\n'
        return out
    
    def __repr__(self):
        return self.__str__()
    
    def scale_flux(self, distance_pc=313.0, radius_rsun=0.08):
        '''Convert the flux to the observed flux at a given distance and radius'''
        self.distance_pc = distance_pc
        self.radius_rsun = radius_rsun
        # convert to cm
        self.distance_cm = self.distance_pc * 3.086e18 # 1 pc = 3.086e18 cm
        self.radius_cm = self.radius_rsun * 6.96e10 # 1 Rsun = 6.96e10 cm
        # convert to erg/s/cm^2/nm
        self.flux *= (self.radius_cm / self.distance_cm)**2
        return self
    
    def plot_spectrum(self, ax=None, **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(self.wave, self.flux, **kwargs)
        ax.set_xlabel(f'Wavelength [{self.wave_unit}]')
        ax.set_ylabel(f'Flux [{self.flux_unit_labels[self.flux_unit]}]')
        return ax
    
    def load_PT(self):
        '''Load the temperature and pressure profile'''
        ds = xr.open_dataset(self.file)
        self.temperature = ds['temperature'].values
        self.pressure = ds['pressure'].values
        return self
    
    def get_dlnT_dlnP(self):
        '''Compute the temperature gradient'''
        assert hasattr(self, 'temperature'), 'Temperature profile not loaded'
        self.dlnT_dlnP = np.gradient(np.log(self.temperature), np.log(self.pressure))
        return self
        
    def plot_PT(self, ax=None, fig_name=None, **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        
        ax.plot(self.temperature, self.pressure, **kwargs)
        ax.set_xlabel('Temperature [K]')
        ax.set(ylabel='Pressure [bar]', yscale='log', ylim=(np.max(self.pressure), np.min(self.pressure)))
        if fig_name is not None:
            plt.savefig(fig_name)
            print(f'Figure saved as {fig_name}')
        return ax
    
        
        
        
        
        
if __name__ == '__main__':
    
    T_range = np.arange(2200, 2400+1, 100)
    # T_range = np.arange(1000, 1200+1, 100)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True,
                           gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    
    for i, T in enumerate(T_range):
        seo = SonoraElfOwl(T)
        seo.load_PT().get_dlnT_dlnP()
        ax[0] = seo.plot_PT(ax=ax[0], label=f'{T} K', color=f'C{i}')
        ax[1].plot(seo.dlnT_dlnP, seo.pressure, label=f'{T} K', color=f'C{i}')
        
    ax[0].legend()
    ax[1].set(xlabel=r'$\nabla T$')
    fig.savefig(f'PT_profiles{T_range[0]:.0f}_{T_range[-1]:.0f}.pdf')
    print(f' Figure saved as PT_profiles{T_range[0]:.0f}_{T_range[-1]:.0f}.pdf')
    plt.show()