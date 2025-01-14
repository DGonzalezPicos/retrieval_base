import numpy as np
import matplotlib.pyplot as plt
import pathlib
from scipy.interpolate import RegularGridInterpolator, Akima1DInterpolator, NearestNDInterpolator, LinearNDInterpolator
import pickle
## species
# ['H2H2', 'H2He', 'HMFF', 'HMBF', 'H2O', 'CO', 'CO2', 'CH4', 'NH3', 
# 'H2S', 'PH3', 'HCN', 'C2H2', 'TiO', 'VO', 'SiO',
# 'FeH', 'CaH', 'MgH', 'CrH', 'AlH', 'TiH', 'Na', 'K', 'Fe', 'Mg', 'Ca', 'C', 'Si', 'Ti', 'O',
# 'FeII', 'MgII', 'TiII', 'CaII', 'CII', 'N2', 'AlO', 'SH', 'OH', 'NO', 'SO2']
class SPHINX:

    path = pathlib.Path(__file__).parent.absolute()
    
    # grid range
    Teff_bounds = (2000.0, 4000.0, 100.0)
    logg_bounds = (4.0, 5.5, 0.25)
    Z_bounds = (-1.0, 1.0, 0.25)
    C_O_bounds = (0.3, 0.9, 0.2)
    
    grid_attrs = ['Teff', 'logg', 'Z', 'C_O']
    n_species = 0 # default
    
    pressure_full = np.logspace(-5., 2., 40) # WARNING: manually fixed...
    
    
    def __init__(self, Teff=2300, logg=4.0, Z=0.0, C_O=0.5, path=None):
        
        self.__in_grid(Teff, logg, Z, C_O)
        self.set_attrs(Teff, logg, Z, C_O)
        
        self.path = self.path if path is None else pathlib.Path(path) # path should point to dir with (ATMS, ABUNDANCES)
        
        # self.file = self.path / 'ATMS' / f'Teff_{self.Teff:.1f}_logg_{self.logg}_logZ_{self.sign}{abs(self.Z)}_CtoO_{self.C_O}_atms.txt'
        self.file = self.get_file(kind='atms')
        assert self.file.exists(), f'File {self.file} does not exist'
        
    def set_attrs(self, Teff, logg, Z, C_O):
        self.Teff = Teff
        self.logg = logg
        idx = 1 if (self.logg % 1) == 0 else 2
        self.logg = np.round(self.logg, idx)
        
        self.Z = abs(Z)
        idx = 1 if (self.Z % 1) == 0 else 2
        self.Z = np.round(self.Z, idx)
        self.sign = '+' if Z >= 0 else '-'
        self.C_O = C_O
        return self
        
    def __in_grid(self, Teff, logg, Z, C_O):
        attrs = ['Teff', 'logg', 'Z', 'C_O']
        for attr, val in zip(attrs, [Teff, logg, Z, C_O]):
            bounds = getattr(self, f'{attr}_bounds')[:2]
            assert val >= bounds[0] and val <= bounds[1], f'{attr} must be in range {bounds}'
        return True
        
        
    def get_file(self, kind='atms', update=False):
        
        folder = {'atms': 'ATMS', 'mixing_ratios': 'ABUNDANCES'}
        assert kind in folder.keys(), f'kind must be in {folder.keys()}'
        
        file_stem = f'Teff_{self.Teff:.1f}_logg_{self.logg}_logZ_{self.sign}{self.Z}_CtoO_{self.C_O:.1f}_{kind}'
        file = self.path / folder[kind] / f'{file_stem}.txt'
        assert file.exists(), f'File {file} does not exist'
        if update:
            self.file = file
            return self
        return file
        
    def load_PT(self):

        self.temperature, self.pressure = np.loadtxt(self.file, unpack=True)
        
        self.gradient = np.gradient(np.log10(self.temperature), np.log10(self.pressure))
        return self
    
    def load_abundances(self, species={}, ignore_ions=True):
        
        file = self.get_file(kind='mixing_ratios')
        # print(f' Loading abundances from {file}')
        with open(file) as f:
            header = f.readline()   
        header = header.split()[1:]
        # remove commas from keys
        header = [s.replace(',', '') for s in header]
        # remove units from pressure (index 1)
        header = header[:1] + header[2:]
        # remove irrelevant species
        irrelevant = []
        if ignore_ions:
            irrelevant += [s for s in header if 'II' in s] # ignore ionized species
        irrelevant += ['N2', 'NO']
        irrelevant += ['H2H2', 'H2He', 'HMFF', 'HMBF']
        
        if len(species) > 0:
            irrelevant = [s for s in header if s not in species and s != 'Pressure']
        

        ignore_idx = [header.index(s) for s in irrelevant]

        # header is first line
        data = np.loadtxt(file, unpack=True)
        self.species = [s for s in header if s not in irrelevant]
        data = data[[header.index(s) for s in self.species], :]
        
        self.pressure_ab = data[0, :]
        self.abundances = data[1:, :]
        self.species = self.species[1:] # ignore "Pressure", first element
        self.all_species = header[1:]
        return self
    
    def plot_abundances(self, ax=None, min_abundance=1e-6, show_temperature=True,
                        save=False, **kwargs):
        
        if not hasattr(self, 'abundances'):
            self.load_abundances()
        if show_temperature and not hasattr(self, 'temperature'):
            self.load_PT()
            
        count_upper = lambda x: sum(1 for c in x if c.isupper())
        atomic_mask = np.array([count_upper(s) == 1 for s in self.species])
        fig, ax = plt.subplots(1, 2, figsize=(14, 7), 
                               sharex=True, 
                               sharey=True, 
                               tight_layout=True)
        
        if kwargs.get('grid', False):
            # set low alpha for grid
            ax[0].grid(alpha=0.5)
            ax[1].grid(alpha=0.5)
        
        if show_temperature:
            ax_temp = ax[0].twiny()
            ax_temp.plot(self.temperature, self.pressure, color='brown', lw=5., alpha=0.4, zorder=-1)
            ax_temp.set(xlabel='Temperature / K')
            
            if kwargs.get('temperature_range', None) is not None:
                ax_temp.set_xlim(kwargs['temperature_range'])
        for i, ab in enumerate(self.abundances):
            if np.max(ab) < min_abundance:
                continue
            axi = ax[1] if atomic_mask[i+1] else ax[0]
            # get last linestyle
            
            ls_last = axi.lines[-1].get_linestyle() if len(axi.lines) > 0 else '-'
            ls = '--' if ls_last == '-' else '-'
            axi.plot(ab, self.pressure_ab, label=f'{self.species[i+1]}', lw=3., alpha=0.9,
                     ls=ls)
            
        ax[0].legend(ncol=5, fontsize=12, loc=(0.0, 1.08), frameon=False)
        ax[1].legend(ncol=5, fontsize=12, loc=(0.0, 1.01), frameon=False)
        ax[0].set(xlabel='Abundance', ylabel='Pressure (bar)',
            yscale='log', xscale='log',
            ylim=(np.max(self.pressure_ab), np.min(self.pressure_ab)),
            xlim=(1e-10, 1e-2))

        fig.suptitle(f'Teff={self.Teff:.0f}, logg={self.logg:.1f}, logZ={self.Z:.1f}, C/O={self.C_O:.1f}', fontsize=20)
        if save:
            fig_name = self.path / f'abundances_Teff_{self.Teff:.0f}_logg_{self.logg}_Z_{self.sign}{abs(self.Z)}_CtoO_{self.C_O}.pdf'
            fig.savefig(fig_name)
            print(f'Saved {fig_name}')
            # plt.show()
            plt.close(fig)
        return ax
            
    
    def plot_PT(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.temperature, self.pressure, **kwargs)
        ax.set(xlabel='Temperature / K', ylabel='Pressure / bar', ylim=(self.pressure.max(), self.pressure.min()), yscale='log')
        return ax
    
    def plot_gradient(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.gradient, self.pressure, **kwargs)
        ax.set(xlabel='dlnT/dlnP', ylabel='Pressure / bar', ylim=(self.pressure.max(), self.pressure.min()), yscale='log')
        return ax
    
    def __init_grid(self, pressure=None):
        
        pressure = self.pressure_full if pressure is None else pressure
        attrs = ['Teff', 'logg', 'Z', 'C_O']
        for attr in attrs:
            bounds = getattr(self, f'{attr}_bounds')[:2]
            step = getattr(self, f'{attr}_bounds')[-1]
            setattr(self, attr+'_grid', np.arange(bounds[0], bounds[1]+step, step))
        
        self.temperature_grid = np.zeros((
            len(self.Teff_grid), len(self.logg_grid), len(self.Z_grid), len(self.C_O_grid), len(pressure)))
        if self.n_species > 0:
            self.vmr_grid = np.zeros((*self.temperature_grid.shape, self.n_species))
            
        return self
    
    def check_grid(self):
        self.__init_grid()
        all_nans = []
        for i, Teff in enumerate(self.Teff_grid):
            for j, logg in enumerate(self.logg_grid):
                for k, Z in enumerate(self.Z_grid):
                    for l, C_O in enumerate(self.C_O_grid):
                        self.set_attrs(Teff, logg, Z, C_O)
                        self.file = self.get_file(kind='atms')
                        assert self.file.exists(), f'File {self.file} does not exist'
                        self.load_PT()
                        nans = np.isnan(self.temperature)
                        if np.all(nans):
                            all_nans.append((Teff, logg, Z, C_O))
                        # self.file = self.get_file(kind='mixing_ratios')
                        # assert self.file.exists(), f'File {self.file} does not exist'
        
        if len(all_nans) > 0:
            print(f'All Nans in {len(all_nans)} files')
            print(all_nans)
        return all_nans
    
    def fix_grid(self, all_nans=[]):
        '''given list of all_nans file, create grid ignoring these and interpolate over them'''
        
        Teff_grid_nonans, logg_grid_nonans, Z_grid_nonans, C_O_grid_nonans = [], [], [], []
        temperature_grid_nonans = []
        for i, Teff in enumerate(self.Teff_grid):
            for j, logg in enumerate(self.logg_grid):
                for k, Z in enumerate(self.Z_grid):
                    for l, C_O in enumerate(self.C_O_grid):
                        if (Teff, logg, Z, C_O) in all_nans:
                            continue
                        Teff_grid_nonans.append(Teff)
                        logg_grid_nonans.append(logg)
                        Z_grid_nonans.append(Z)
                        C_O_grid_nonans.append(C_O)
                        self.set_attrs(Teff, logg, Z, C_O)
                        self.file = self.get_file(kind='atms')
                        self.load_PT()
                        temperature_grid_nonans.append(self.interp_makima(self.pressure_full, self.pressure, self.temperature, smooth_nans=True))
                        
        # temperature_grid_nonans 
        # create interpolator for non-rectilinear grid
        interpolator = NearestNDInterpolator((Teff_grid_nonans, logg_grid_nonans, Z_grid_nonans, C_O_grid_nonans), temperature_grid_nonans)
        return self
    
    @classmethod
    def get_attrs(cls, f):
        Teff = float(f.name.split('_')[1])
        logg = float(f.name.split('_')[3])
        Z_sign = f.name.split('_')[5][0]
        Z_abs = float(f.name.split('_')[5][1:])
        Z = Z_abs * (-1)**(Z_sign == '-')
        
        C_O = float(f.name.split('_')[7])
        return Teff, logg, Z, C_O
        
    def load_interpolator(self, species=[], cache=True):
        
        interp_pickle = self.path / 'sphinx_interpolator.pkl'
        if interp_pickle.exists() and cache:
            print(f' Loading interpolator from {interp_pickle}')
            with open(interp_pickle, 'rb') as f:
                interp = pickle.load(f)
            self.temp_interpolator = interp['temp_interpolator']
            self.vmr_interpolator = interp['vmr_interpolator']
            return self
        
        # load all files in grid
        files = sorted((self.path / 'ATMS').glob('*atms.txt'))
        assert len(list(files)) > 0, 'No files found'
        self.n_species = len(species)  
        
        # create interpolator for 4D grid 
        points = []
        temps = []
        # create empty dict for species with empty lists
        vmrs = {k:v for k, v in zip(species, [[] for _ in range(len(species))])}
        for f in files:
            Teff, logg, Z, C_O = self.get_attrs(f)
            # TODO:
            # check_bounds = self.check_bounds({'Teff': (value, min, max),
            #                                   'logg': (value, min, max),
            # ...})
            # if not check_bounds:
            #     continue
            print(f'Checking {f.name:90}', end='\r')
            self.set_attrs(Teff=Teff, logg=logg, Z=Z, C_O=C_O)
            self.get_file(kind='atms', update=True)
            self.load_PT()
            nans = np.isnan(self.temperature)
            if np.all(nans):
                print(f'[SPHINX.load_interpolator] All nans for {f.name}')
                continue
            
            if self.n_species > 0:
                self.load_abundances(species=species)
                nans_ab = np.array([np.isnan(ab_i).all() for ab_i in self.abundances])
                
                if np.any(nans_ab):
                    print(f'[SPHINX.load_interpolator] ANY nans for {f.name} abundances')
                    continue
                # vmr = np.array([self.interp_makima(self.pressure_full, self.pressure_ab, ab_i) for ab_i in self.abundances])
                for i, s in enumerate(species):
                    vmrs[s].append(self.interp_makima(self.pressure_full, self.pressure_ab, self.abundances[i]))
                
            
            points.append((Teff, logg, Z, C_O))
            temps.append(self.interp_makima(self.pressure_full, self.pressure, self.temperature, smooth_nans=True))
            
        self.pressure = self.pressure_full
        print(f'[SPHINX.load_interpolator] Loaded {len(points)} files')
        print(f'[SPHINX.load_interpolator] Preparing temperature interpolator')
        points = np.array(points)
        temps = np.array(temps)
        self.temp_interpolator = LinearNDInterpolator(points, temps, fill_value=np.nan)
        
        if self.n_species > 0:
            print(f'[SPHINX.load_interpolator] Preparing VMRs interpolator')
            # vmrs = np.array(vmrs)
            self.vmr_interpolator = {}
            for i, s in enumerate(species):
                print(f'[SPHINX.load_interpolator] Preparing VMRs interpolator for {s}')
                
                # print(f' --> shape of vmrs: {np.array(vmrs[s]).shape}')
                self.vmr_interpolator[s] = LinearNDInterpolator(points, np.array(vmrs[s]), fill_value=np.nan)
                
        del self.abundances
        del self.temperature
        
        # save pickle with interpolators
        interp = {'temp_interpolator': self.temp_interpolator, 'vmr_interpolator': self.vmr_interpolator}
        with open(interp_pickle, 'wb') as f:
            pickle.dump(interp, f)
        print(f'[SPHINX.load_interpolator] Saved interpolators to {interp_pickle}')
        
        return self
    
    def load_PT_grid_old(self, species={}):
        
        # load all files in grid
        files = sorted((self.path / 'ATMS').glob('*atms.txt'))
        assert len(list(files)) > 0, 'No files found'
        # self.file = files[0]
        # self.load_PT()
        self.n_species = len(species)
        self.__init_grid() # init vmr grid if len(species) > 0
        for i, Teff in enumerate(self.Teff_grid):
            for j, logg in enumerate(self.logg_grid):
                for k, Z in enumerate(self.Z_grid):
                    for l, C_O in enumerate(self.C_O_grid):
                        self.set_attrs(Teff, logg, Z, C_O)
                        self.file = self.get_file(kind='atms')
                        self.load_PT()
                        if self.n_species > 0:
                            self.load_abundances(species=species)
                            self.vmr_grid[i, j, k, l,...] = np.array([np.interp(self.pressure_full, self.pressure_ab, ab_i) for ab_i in self.abundances]).T
                        
                        if len(self.pressure) != len(self.pressure_full):
                            # self.temperature = np.interp(self.pressure_full, self.pressure, self.temperature)
                            nans = np.isnan(self.temperature)
                            # assert not np.all(nans), f'ALL Nans in temperature: {np.sum(nans)} for Teff: {Teff}, logg: {logg}, Z: {Z}, C_O: {C_O}'
                            if np.any(nans):
                                print(f' --> Before interpolation: Nans in temperature: {np.sum(nans)}')
                            if np.all(nans):
                                print(f' WARNING: ALL Nans in temperature: {np.sum(nans)} for Teff: {Teff}, logg: {logg}, Z: {Z}, C_O: {C_O}')
                            # print shapes
                            # print(f'Pressure: {self.pressure.shape}, Pressure_full: {self.pressure_full.shape}')
                            # print(f'Temperature: {self.temperature.shape}')
                            self.temperature = self.interp_makima(self.pressure_full, self.pressure, self.temperature)
                            nans = np.isnan(self.temperature)
                            if np.any(nans):
                                print(f'After interpolation: Nans in temperature: {np.sum(nans)}')
                                print(f'Teff: {Teff}, logg: {logg}, Z: {Z}, C_O: {C_O}')
                                # print(f'Pressure: {self.pressure}')
                                # print(f'Temperature: {self.temperature}')
                                # break
                        self.temperature_grid[i, j, k, l, :] = self.temperature
                            
        # interpolator
        self.temp_interpolator = RegularGridInterpolator((self.Teff_grid, self.logg_grid, self.Z_grid, self.C_O_grid),
                                                    self.temperature_grid, method='linear', bounds_error=True, fill_value=None) 
        if self.n_species > 0:
            self.vmr_interpolator = RegularGridInterpolator((self.Teff_grid, self.logg_grid, self.Z_grid, self.C_O_grid),
                                                    self.vmr_grid, method='linear', bounds_error=True, fill_value=None)
        self.pressure = self.pressure_full
        delattrs = self.grid_attrs + ['temperature']
        if self.n_species > 0:
            delattrs += ['vmr']
            delattr(self, 'pressure_ab')
            delattr(self, 'abundances')

        [delattr(self, attr+'_grid') for attr in delattrs]
            
        return self
    
    @classmethod
    def interp_makima(self, new_x, x, y, smooth_nans=True):
        nans = np.isnan(y)
        new_y = Akima1DInterpolator(x[~nans], y[~nans], method='makima')(new_x)
        if smooth_nans:
            new_y = self.smooth_nans(new_y)
        return new_y
        
    @classmethod
    def smooth_nans(cls, x):
        nans = np.isnan(x)
        x[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), x[~nans])
        return x
        
        
    
    def interpolate_PT(self, Teff, logg, Z, C_O):
        
        self.__in_grid(Teff, logg, Z, C_O)
        self.set_attrs(Teff, logg, Z, C_O)
        assert hasattr(self, 'temp_interpolator'), 'Load PT grid first'
        
        return self.temp_interpolator([Teff, logg, Z, C_O])[0]
        # set up regular grid interpolator
        
    def interpolate_vmr(self, Teff, logg, Z, C_O):
        
        self.__in_grid(Teff, logg, Z, C_O)
        self.set_attrs(Teff, logg, Z, C_O)
        assert hasattr(self, 'vmr_interpolator'), 'Load PT grid first'
        
        return self.vmr_interpolator([Teff, logg, Z, C_O])[0]
    
    def interpolate(self, Teff, logg, Z, C_O):
        self.__in_grid(Teff, logg, Z, C_O)
        self.set_attrs(Teff, logg, Z, C_O)
        assert hasattr(self, 'temp_interpolator'), 'Load PT grid first'
        assert hasattr(self, 'vmr_interpolator'), 'Load PT grid first'
        return self.temp_interpolator([Teff, logg, Z, C_O])[0], self.vmr_interpolator([Teff, logg, Z, C_O])[0]
        

    
if __name__ == '__main__':
    
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 7), gridspec_kw={'width_ratios': [3, 2], 'wspace': 0.01},
                            sharey=True)
    lw = 3.
    Teff_range = np.arange(2900, 4000, 100)
    logg = 4.5
    colors = plt.cm.inferno(np.linspace(0, 1, len(Teff_range)))
    for i, Teff in enumerate(Teff_range):
        sphinx = SPHINX(Teff=Teff, logg=logg, Z=0.0, C_O=0.5, path='SPHINX')
        sphinx.load_PT()
        sphinx.load_abundances()
        sphinx.plot_PT(ax=ax[0], color=colors[i], lw=lw)
        sphinx.plot_gradient(ax=ax[1], color=colors[i], lw=lw)
        
    
    
    # move y-ticks and labels to the right
    ax[1].axvline(0, color='k', lw=2, ls='--')
    ax[0].set_ylim(1e2, 1e-5)
    ax[0].set_xlim(1000, 10500)
    ax[1].yaxis.tick_right()
    ax[1].set_ylabel('')
    
    # create colorbar
    im = ax[1].scatter([], [], c=[], cmap=plt.cm.inferno, vmin=Teff_range.min(), vmax=Teff_range.max())
    fig.colorbar(im, ax=ax[1], label='Teff', orientation='vertical', location='right',
                    fraction=0.20, pad=0.00, ticks=Teff_range)
    
    teff_label = '_'.join([f'{Teff_range.min()}', f'{Teff_range.max()}'])
    fig_name = f'SPHINX_{teff_label}_logg_{logg}.pdf'
    fig.savefig(fig_name)
    # plt.show()
    plt.close(fig)
    print(f'Saved {fig_name}')
    
    
    