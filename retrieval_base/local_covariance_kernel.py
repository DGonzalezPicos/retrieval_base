import numpy as np

class LocalCovarianceKernel:
    def __init__(self, wave, flux, err, lck_width=2):
        self.wave = wave
        self.separation = (wave[None,:] - wave[:,None])
        self.flux = flux
        self.err = err
        self.err[self.err == 0] = np.nan
        self.err2_inv = 1 / err**2
        self.lck_width = lck_width
        self.lck_width_wavelength = lck_width * np.nanmean(np.diff(self.wave))

    def gaussian_kernel(self, x, mu, sigma_wavelength=1.0):
        
        return np.exp(-(x - mu)**2 / (2 * sigma_wavelength**2))

    def merge_regions(self, regions, chi2_pp):
        
        regions = np.sort(regions, axis=0)
        merged = []
        chi2_region = []

        def add_region(merged, chi2_region, region):
            merged.append(region)
            mask_region = (self.wave >= region[0]) & (self.wave <= region[1])
            # chi2_region.append(np.nanmax(chi2_pp[mask_region]) / np.sum(mask_region))
            chi2_region.append(np.nanmax(chi2_pp[mask_region]))
            return merged, chi2_region

        for i, region in enumerate(regions):
            if i > 0:
                if region[0] <= current_region[1]:  # Check if regions overlap
                    current_region = np.array([current_region[0], max(current_region[1], region[1])])
                else:
                    current_region = region
                    merged, chi2_region = add_region(merged, chi2_region, current_region)
            else:
                current_region = region
                merged, chi2_region = add_region(merged, chi2_region, current_region)

        return np.array(merged), np.array(chi2_region)

    def select_max_regions(self, regions, chi2_regions, n_max_regions=5):
        chi2_sort = np.argsort(chi2_regions)[::-1]
        regions = regions[chi2_sort][:n_max_regions]
        chi2_regions = chi2_regions[chi2_sort][:n_max_regions]
        # print(regions)
        wave_sort = np.argsort(regions[:, 0])
        regions = regions[wave_sort]
        chi2_regions = chi2_regions[wave_sort]

        return regions, chi2_regions

    def scale_factors_regions(self, regions, chi2_regions):
        s_lck = np.ones(self.wave.shape)
        for region, chi2_region in zip(regions, chi2_regions):
            mask_region = (self.wave >= region[0]) & (self.wave <= region[1])
            s_lck[mask_region] = np.sqrt(chi2_region) * self.gaussian_kernel(self.wave[mask_region], np.median(self.wave[mask_region]), self.lck_width_wavelength / 2.355)
        return s_lck
    
    def correlated_kernel(self, trunc_dist=4.0):
        
        kernels = np.zeros((self.wave.shape[0], self.wave.shape[0]))
        for region, chi2_region in zip(self.regions, self.chi2_regions):
            
            kernel = np.zeros((self.wave.shape[0], self.wave.shape[0]))

            r_0 = np.mean(region)
            r_i = np.abs(self.wave[None,:] - r_0)
            r_j = np.abs(self.wave[:,None] - r_0)
            r2 = r_i**2 + r_j**2
            # print(f' r2.shape {r2.shape}')
            w_ij = (self.separation < trunc_dist * self.lck_width_wavelength / 2.355)
            # print(f' w_ij.shape {w_ij.shape}')
            
            # print(f' self.s.shape {self.s.shape}')
            kernel[w_ij] = chi2_region * np.exp(-0.5 * r2[w_ij] / (self.lck_width_wavelength / 2.355)**2)
            kernels += kernel
            
        
        return kernels
            

    def __call__(self, m_flux, sigma_threshold=5.0, n_max_regions=None):
        self.n_max_regions = n_max_regions
        res = self.flux - m_flux
        # nans = np.isnan(res)

        chi2_pp = res**2 * self.err2_inv
    
        mean_chi2_pp = np.nanmean(chi2_pp)
        std_chi2_pp = np.nanstd(chi2_pp)

        self.mask = chi2_pp > (sigma_threshold * std_chi2_pp) + mean_chi2_pp
        lck_center = self.wave[self.mask]
        self.regions = np.array([lck_center - self.lck_width_wavelength / 2, lck_center + self.lck_width_wavelength / 2]).T
        if len(self.regions) == 0:
            # print(' No regions found')
            self.s = np.ones(self.wave.shape)
            return self.s
        
        # print(f' Number of regions: {len(self.regions)}')
        self.regions, self.chi2_regions = self.merge_regions(self.regions, chi2_pp)

        if self.n_max_regions is not None:
            self.regions, self.chi2_regions = self.select_max_regions(self.regions, self.chi2_regions, self.n_max_regions)
        self.s_regions = self.scale_factors_regions(self.regions, self.chi2_regions)
        # print(f' Regions: {self.regions}')
        # print(f' Chi2 regions: {self.chi2_regions}')
        # print(f' Scaling factors: {self.s_regions[self.mask]}')
        mask_rest = ~self.mask
        chi2_rest = np.nansum(chi2_pp[mask_rest]) / np.sum(mask_rest)
        self.s_rest = np.sqrt(chi2_rest)

        self.s = np.where(self.mask, self.s_regions, self.s_rest)
        
        # self.kernel = self.correlated_kernel()
        return self.s
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    
    np.random.seed(1234)
    wave = np.linspace(0, 2, 400)
    p = 0.2
    flux = 5.0 + np.sin(wave / p)
    snr = 5.0
    
    model = flux.copy()
    
    noise = np.random.normal(0, 1/snr, wave.shape)
    flux += noise
    err = np.sqrt(flux) / snr * np.ones(wave.shape)
    
    # add outliers
    def add_random_outlier(flux, amplitude=1.0, width=2, n_outliers=2):
        n_outliers = np.random.randint(1, n_outliers)
        for i in range(n_outliers):
            i = np.random.randint(0, len(flux))
            flux[i:i+width] += np.random.normal(amplitude, amplitude/2)
        return flux
    flux = add_random_outlier(flux, 1.0, 2, 5)
    # flux[1008:1016] += 1.0
    
    res = flux - model
    chi2_pp = res**2 * err**-2
    
    
    lck = LocalCovarianceKernel(wave, flux, err, lck_width=4)
    lck.s = lck(model, sigma_threshold=3.0, n_max_regions=5)
    print(f' Scaling factors:')
    print(f' Outliers: {lck.s_regions[lck.mask]}')
    print(f' Rest: {lck.s_rest:.2f}')
    # print(lck.s)
    
    fig, ax = plt.subplots(3,1, figsize=(14,5), sharex=True, gridspec_kw={'height_ratios':[3,2,2]})
    ax[0].plot(wave, flux, label='data', color='k')
    ax[0].fill_between(wave, flux - lck.s*err, flux + lck.s*err, alpha=0.2, color='k', lw=0)
    ax[0].plot(wave, model, label='model', color='darkorange')
    ax[0].legend()
    
    ax[1].plot(wave, res, '.', label='res', color='k')
    ax[1].fill_between(wave, res - lck.s*err, res + lck.s*err, alpha=0.2, color='k', lw=0)
    ax[1].axhline(0, color='darkorange', ls='-', lw=0.5)
    ax[1].legend()
    
    # ax[2].plot(wave, lck.s, label='lck.s', color='k')
    # ax[2].legend()
    ax[2].plot(wave, chi2_pp, label='chi2_pp', color='k')
    for region in lck.regions:
        ax[2].axvspan(region[0], region[1], alpha=0.2, color='r', lw=0)
    ax[2].legend()
    ax[0].set_ylabel('Flux')
    ax[1].set_ylabel('Residuals')
    ax[2].set_ylabel(r'$\chi^2_{pp}$')
    ax[2].set_xlabel('Wavelength')
    plt.show()