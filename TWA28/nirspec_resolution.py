import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pathlib

class NIRSpecResolution:
    
    default_path = pathlib.Path('TWA28/jwst/')
    
    def __init__(self, gratings, path=None):
        self.gratings = gratings
        self.path = pathlib.Path(path) if path is not None else self.default_path
        self.files = [self.path/f'jwst_nirspec_{g}_disp.fits' for g in gratings]
        assert all([f.exists() for f in self.files]), 'Files not found'
        
        self._load_files()
        
    def _load_files(self):
        assert len(self.files) > 0, 'No files to load'
        
        self.wave, self.disp, self.resolution = [], [], []
        for f in self.files:
            with fits.open(f) as hdul:
                # print(hdul.info())
                data = hdul[1].data
                self.wave.append(data['WAVELENGTH'])
                self.disp.append(data['DLDS'])
                self.resolution.append(data['R'])
                
        
        return self
    
    def __str__(self):
        out = f'NIRSpec resolution data loaded from {self.path}\n'
        for i, g in enumerate(self.gratings):
            out += f' grating {g} (R ~ ({np.min(self.resolution[i]):.0f}, {np.max(self.resolution[i]):.0f}))\n'
        return out
    def __repr__(self):
        return self.__str__()
    
    def plot_resolution(self):
    
        assert hasattr(self, 'wave'), 'No data loaded'
        assert hasattr(self, 'resolution'), 'No data loaded'
        
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        for i, (w, r) in enumerate(zip(self.wave, self.resolution)):
            ax.plot(w, r, label=f'{self.gratings[i]} (R ~ {np.mean(r):.0f})')

        ax.set_xlabel('Wavelength [nm]')    
        ax.set_ylabel('Resolution')
        ax.legend(frameon=False)
        # plt.show()
        return self
        
        
    def __call__(self, wave, flux, grating):
        """
        Broadens the spectrum by the instrumental resolution that scales linearly with wavelength.

        Parameters:
        wave : ndarray
            Wavelength array in [um]
        flux : ndarray
            Flux array.
        out_res : float
            Minimum output resolution at the start of the wavelength array.
        in_res : float, optional
            Input resolution (default is 1e6).

        Returns:
        flux_LSF : ndarray
            Flux array after applying instrumental broadening.
        """
        assert grating in self.gratings, 'grating not found'

        # Calculate the resolution array with the correct profile
        res = np.interp(wave, self.wave[self.gratings.index(grating)], self.resolution[self.gratings.index(grating)])

        # Calculate the sigma for the LSF at each wavelength point
        sigma_array = np.sqrt(1/res**2) / (2 * np.sqrt(2 * np.log(2)))

        # Calculate the average spacing between wavelength points
        spacing = np.nanmean(2 * np.diff(wave) / (wave[1:] + wave[:-1]))

        # Convert the sigma_LSF to pixel units
        sigma_gauss_filter = sigma_array / spacing

        # Create a Gaussian kernel for each wavelength point
        max_sigma = np.nanmax(sigma_gauss_filter)
        kernel_half_width = int(3 * max_sigma) # WARNING: Manually set to 99.9% of gaussian (3 sigma)
        x = np.arange(-kernel_half_width, kernel_half_width + 1)

        # Use broadcasting to create a 2D array of Gaussian kernels
        kernels = np.exp(-0.5 * (x[None, :] / sigma_gauss_filter[:, None]) ** 2)
        kernels /= kernels.sum(axis=1)[:, None]

        # Pad the flux array to handle edge effects
        flux_padded = np.pad(flux, (kernel_half_width, kernel_half_width), mode='reflect')

        # Create a matrix where each row is a shifted version of the flux array
        flux_matrix = np.array([flux_padded[i:i + len(flux)] for i in range(2 * kernel_half_width + 1)]).T

        # Perform the convolution using matrix multiplication
        return np.einsum('ij, ij->i', kernels, flux_matrix)
            
        
if __name__ == '__main__':
    path = pathlib.Path('TWA28/jwst/')
    gratings = ['g140h', 'g235h', 'g395h']
    files = [path/f'jwst_nirspec_{g}_disp.fits' for g in gratings]
    nr = NIRSpecResolution(gratings)

    # g = 'g235h'
    g = 'g395h'
    g_idx = nr.gratings.index(g)
    wave_g = nr.wave[g_idx]
    # create fake spectrum to test broadening function
    x = np.linspace(wave_g[0], wave_g[-1], 20 * len(wave_g))
    # add sharp lines as narrow gaussian peaks
    flux = np.ones_like(x)
    for i in np.linspace(0, len(x)-1, 20).astype(int):
        flux -= 0.5 * np.exp(-0.5 * (x - x[i])**2 / 1e-7)

    fig, ax = plt.subplots(2,1, figsize=(8,6), gridspec_kw={'height_ratios': [1, 5], 'hspace':0.05}, sharex=True)
    ax[0].plot(nr.wave[g_idx], nr.resolution[g_idx], label=f'{g} (R ~ {np.mean(nr.resolution[g_idx]):.0f})', color='k')
    overlap = np.interp(x, nr.wave[g_idx], nr.resolution[g_idx])
    print(f' R ~ ({np.min(overlap):.0f}, {np.max(overlap):.0f})')
    ax[0].plot(x, overlap, color='magenta', ls='--')


    ax[1].plot(x, flux, label='Original')
    ax[1].plot(x, nr(x, flux, g), label='Broadened')
    ax[1].legend(frameon=False)

    ax[0].set(ylabel='Resolution')
    ax[1].set(xlabel = 'Wavelength / um', ylabel='Flux')
    plt.show()

    