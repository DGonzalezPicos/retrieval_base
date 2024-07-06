import numpy as np


class Resample:
    
    
    def __init__(self, wave=None, flux=None, flux_err=None):
        
        self.wave = wave
        self.flux = flux
        self.flux_err = flux_err
        
        
        self.fluxcov = np.diag(self.flux_err**2) if self.flux_err is not None else None
        
        if self.wave is not None:
            self.bin_edges = self._parse_wavelength_information(self.wave, bin_edges=None)
        
    # @staticmethod    
    def _parse_wavelength_information(self, wave, bin_edges):
        """Parse wavelength information and return a set of bin edges.

        Either the central wavelength for each bin can be passed as ``wave``, or
        the bin edges can be passed directly as ``bin_edges``. This function will
        recover the bin edges from either input and verify that they are a valid
        monotonically-increasing list.
        """
        # Make sure that a valid combination of inputs was given.
        valid_count = 0
        if wave is not None:
            valid_count += 1
        if bin_edges is not None:
            valid_count += 1
        if valid_count != 1:
            raise ValueError('must specify exactly one of wave or bin_edges')

        # Extract the bin starts and ends.
        if wave is not None:
            # print(f'[resample.py] wave.shape {wave.shape}')
            bin_edges = self._recover_bin_edges(np.asarray(wave))

        # Make sure that the bin ends are larger than the bin starts.
        if np.any(bin_edges[1:] <= bin_edges[:-1]):
            raise ValueError('wavelength must be monotonically increasing')

        return bin_edges

    @staticmethod
    def _recover_bin_edges(wave):
        """Recover the edges of a set of wavelength bins given the bin centers.

        This function is designed to work for standard linear binning along with
        other more exotic forms of binning such as logarithmic bins. We do a second
        order correction to try to get the bin widths as accurately as possible.
        For linear binning there is only machine precision error with either a
        first or second order estimate.

        For higher order binnings (eg: log), the fractional error is of order (dA /
        A)**2 for linear estimate and (dA / A)**4 for the second order estimate
        that we do here.

        Parameters
        ----------
        wave : array-like
            Central wavelength values of each wavelength bin.

        Returns
        -------
        bin_edges : `~numpy.ndarray`
            The recovered edges of each wavelength bin.
        """
        wave = np.asarray(wave)
        assert len(wave) > 3, 'must have at least 3 points to recover bin edges'
        # First order estimate
        o1 = (wave[:-1] + wave[1:]) / 2.

        # Second order correction
        o2 = 1.5*o1[1:-1] - (o1[2:] + o1[:-2]) / 4.

        # Estimate front and back edges
        f2 = 2 * wave[1] - o2[0]
        f1 = 2 * wave[0] - f2
        b2 = 2 * wave[-2] - o2[-1]
        b1 = 2 * wave[-1] - b2

        # Stack everything together
        bin_edges = np.hstack([f1, f2, o2, b2, b1])

        return bin_edges


    def __call__(self, new_wave):
        # print(f' len(new_wave): {len(new_wave)}')
        new_bin_edges = self._parse_wavelength_information(new_wave, bin_edges=None)
        new_bin_starts = new_bin_edges[:-1]
        new_bin_ends = new_bin_edges[1:]

        old_bin_starts = self.bin_edges[:-1]
        old_bin_ends = self.bin_edges[1:]

        # Generate a weight matrix for the transformation.
        overlap_starts = np.max(np.meshgrid(old_bin_starts, new_bin_starts),
                                axis=0)
        overlap_ends = np.min(np.meshgrid(old_bin_ends, new_bin_ends), axis=0)
        overlaps = overlap_ends - overlap_starts
        overlaps[overlaps < 0] = 0

        # Normalize by the total overlap in each bin to keep everything in
        # units of f_lambda
        total_overlaps = np.sum(overlaps, axis=1)
        if np.any(total_overlaps == 0):
            raise ValueError("new binning not contained within original "
                             "spectrum")
        weight_matrix = overlaps / total_overlaps[:, None]

        new_flux = weight_matrix.dot(self.flux)
        if self.fluxcov is None:
            new_fluxcov = None
        else:
            new_fluxcov = weight_matrix.dot(self.fluxcov.dot(weight_matrix.T))

        return new_flux, new_fluxcov
    
if __name__ == '__main__':
    from spectres import spectres, spectres_numba # comparison with spectres
    import time
    # create a random spectrum
    wave = np.linspace(3000, 10000, 1000)
    flux = np.sin(wave / 5000 * np.pi) + np.cos(wave / 3000 * np.pi)
    flux += np.random.normal(0, 0.1, wave.size)
    # add some random nans
    # flux[np.random.choice(np.arange(wave.size), size=wave.size//10)] = np.nan
    
    flux_err = np.random.uniform(0.01, 0.1, wave.size)
    
    # create a resample object
    
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    # from cycler import cycler
    # import seaborn as sns

    # # Get the "deep" palette colors
    # deep_palette = sns.color_palette("colorblind", 12)
    
    # def rgb_to_hex(rgb):
    #     return '{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

    # deep_palette_hex = [rgb_to_hex(color) for color in deep_palette]
    # print(deep_palette_hex)
    
    
    # sns.set_palette(sns.color_palette('colorblind'))

    
    snr = np.abs(np.mean(flux) / np.mean(flux_err))
    ax.errorbar(wave, flux, yerr=flux_err, label=f'OG (SNR={snr:.2f})', fmt='o', color='k', alpha=0.6)
    # for i in range(7):
    for i in range(5,7):
        
        start = time.time()
        resample = Resample(wave=wave, flux=flux, flux_err=flux_err)
        new_wave = np.linspace(3500, 9000, wave.size//(i+2))
        new_flux, new_fluxcov = resample(new_wave)
        print(f' Resample: {(time.time() - start)*1e3:.2f} ms')
        
        
        snr = np.abs(np.mean(new_flux) / np.mean(np.sqrt(np.diag(new_fluxcov))))
        ax.errorbar(new_wave, new_flux, yerr=np.sqrt(np.diag(new_fluxcov)), label=f'New (SNR={snr:.2f})',
                    fmt='o', alpha=0.9)
        
        # spectres
        start = time.time()
        new_flux_s, new_err = spectres(new_wave, wave, flux, spec_errs=flux_err)
        print(f' Spectres: {(time.time() - start)*1e3:.2f} ms')
        snr_s = np.abs(np.mean(new_flux_s) / np.mean(new_err))
        ax.errorbar(new_wave, new_flux_s, yerr=new_err, label=f'Spectres (SNR={snr_s:.2f})',
                    fmt='x', alpha=0.9)
        
        # spectres numba
        start = time.time()
        new_flux_s, new_err = spectres_numba(new_wave, wave, flux, spec_errs=flux_err)
        print(f' Spectres numba: {(time.time() - start)*1e3:.2f} ms')
        snr_s = np.abs(np.mean(new_flux_s) / np.mean(new_err))
        ax.errorbar(new_wave, new_flux_s, yerr=new_err, label=f'Spectres numba (SNR={snr_s:.2f})',
                    fmt='x', alpha=0.9)
        
        
    ax.legend()
    plt.show()
    
    print(f' Better to use `spectres` then...')
    print(f' `spectres_numba` is the fastest')
    
    
    