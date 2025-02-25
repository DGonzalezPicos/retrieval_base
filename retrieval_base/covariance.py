import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky_banded, cho_solve_banded

def get_Covariance_class(err, mode=None, **kwargs):

    assert mode is 'GP', 'Only GP mode is currently implemented'
    # Use a GaussianProcesses instance
    return GaussianProcesses(err, **kwargs)
    
class Covariance:
    
    c_km_s = 299792.458  # speed of light in km/s
    
    def __init__(self, x, err, max_length_scale=None, truncate=4.0, scale_amplitude=False):
        """
        Initialize the Covariance class.
        
        Parameters
        ----------
        x : array_like
            1D array of wavelengths.
        err : array_like
            1D array of measurement uncertainties (must be finite).
        """
        self.x = np.atleast_1d(np.array(x))
        self.err = np.atleast_1d(np.array(err))
        assert self.x.shape == self.err.shape, "x and err must have the same shape."
        assert np.all(np.isfinite(self.err)), "Error array contains non-finite values (NaNs or infs)."

        self.max_length_scale = max_length_scale
        self.truncate = truncate
        self.scale_amplitude = scale_amplitude
        self.max_value = None
        if self.max_length_scale is not None:
            assert self.max_length_scale > 0, "max_length_scale must be positive."
            assert self.truncate > 0, "truncate must be positive."
            self.max_value = self.max_length_scale * self.truncate

        if self.x is not None:
            self.x_ij = self.full_to_banded(self.separation_matrix(), 
                                            max_value=self.max_value)
            
            
    def separation_matrix(self, x=None, x2=None):
        """
        Compute the pairwise velocity separation matrix.
        
        Parameters
        ----------
        x : array_like, optional
            If provided, use this wavelength array; otherwise use self.x.
        
        Returns
        -------
        r : ndarray
            Matrix with elements 
            r_ij = 0.5 * c_km_s * |λ_i - λ_j| / (λ_i + λ_j)
        """
        if x is None:
            x = self.x
        else:
            x = np.atleast_1d(np.array(x))
        if x2 is None:
            x2 = self.x
        else:
            x2 = np.atleast_1d(np.array(x2))
        return 0.5 * self.c_km_s * np.abs(x[:, None] - x2[None, :]) / (x[:, None] + x2[None, :])

    @staticmethod
    def hann_window(r, r0):
        """
        Apply a Hann window taper to separations.
        
        Parameters
        ----------
        r : array_like
            Separation (scalar, vector or matrix).
        r0 : float
            Truncation distance.
        
        Returns
        -------
        w : ndarray
            Tapering weight: 0.5*(1+cos(pi*r/r0)) for r<=r0; 0 for r > r0.
        """
        return np.where(r <= r0, 0.5 * (1 + np.cos(np.pi * r / r0)), 0.0)

    def matern_kernel(self, r, a, l, nu=1.5, r0=None):
        """
        Construct a Matérn kernel (here implemented for ν = 3/2) with an optional Hann taper.
        
        Parameters
        ----------
        r : ndarray
            Velocity separation matrix (km/s).
        a : float
            Amplitude of the kernel.
        l : float
            Correlation length scale (km/s).
        nu : float, optional
            Smoothness parameter; currently only supports nu=1.5.
        r0 : float, optional
            Truncation distance for the Hann window. If None, no taper is applied.
        
        Returns
        -------
        K : ndarray
            Covariance matrix computed using the Matérn ν=3/2 kernel.
        """
        if np.abs(nu - 1.5) > 1e-6:
            raise NotImplementedError("Only nu=1.5 (Matérn 3/2) is currently implemented.")
        # Matérn 3/2 kernel:
        K = a**2 * (1.0 + np.sqrt(3) * r / l) * np.exp(-np.sqrt(3) * r / l)
        if r0 is not None:
            K *= self.hann_window(r, r0)
        return K

    def identify_outliers(self, resid, threshold=3.0):
        """
        Identify outliers in the residuals using a robust statistic.
        
        Parameters
        ----------
        resid : array_like
            Residuals for which outliers are to be identified.
        threshold : float, optional
            Threshold in units of robust sigma (default is 3.0).
        
        Returns
        -------
        outlier_indices : ndarray
            Indices of the identified outliers.
        amplitudes : ndarray
            Additional amplitudes (excess over threshold) for each outlier.
        sigma_robust : float
            Robust estimate of the standard deviation.
        """
        resid = np.atleast_1d(np.array(resid))
        median = np.median(resid)
        mad = np.median(np.abs(resid - median))
        sigma_robust = 1.4826 * mad  # conversion factor for Gaussian data
        
        outlier_indices = np.where(np.abs(resid - median) > threshold * sigma_robust)[0]
        # Extra amplitude above the threshold
        amplitudes = np.abs(resid[outlier_indices]) - threshold * sigma_robust
        amplitudes[amplitudes < 0] = 0.0
        return outlier_indices, amplitudes, sigma_robust

    def local_kernel(self, x, a_L, mu_local, sigma_local, truncate=4.0, k=None):
        """
        Build a local covariance kernel for a single feature.
        The kernel is given by a Gaussian function of the velocity separation from mu_local,
        tapered with a Hann window.
        
        Parameters
        ----------
        x : array_like
            Wavelengths (typically the ones in the region of interest).
        a_L : float
            Local kernel amplitude.
        mu_local : float
            Central wavelength of the local feature.
        sigma_local : float
            Width of the local feature (km/s).
        
        Returns
        -------
        K_L : ndarray
            Local covariance matrix.
        """
        # Compute separation between each pixel and the local feature centre.
        # x = np.atleast_1d(np.array(x))
        r_local = 0.5 * self.c_km_s * np.abs(x - mu_local) / (x + mu_local)
                
        # Set truncation for the Hann window (e.g., 4 * sigma_local)
        r0_local = truncate * sigma_local
        w_local = self.hann_window(r_local, r0_local)
        profile = np.exp(-0.5 * (r_local / sigma_local)**2)
        weighted_profile = w_local * profile
        return self.full_to_banded(a_L**2 * np.outer(weighted_profile, weighted_profile), k=k)
    
    def effective_variance(self, b=0.0):
        """
        Compute the effective variance array, which is the diagonal of the covariance matrix scaled by the factor b as in
        sigma2 = sigma2_0 * 10.0**b
        """
        return (self.err**2 * 10.0**b)
    
    def local_covariance(self, residuals, residuals_threshold=4.0, sigma_local=1.0, truncate=4.0, k=None):
        """
        Compute the local covariance matrix from the residuals.
        """
        K_local = np.zeros_like(self.C)
        # print(f'[Covariance.local_covariance]: K_local.shape = {K_local.shape}')
        
        outlier_indices, amplitudes, sigma_robust = self.identify_outliers(residuals, threshold=residuals_threshold)
        for a_local, idx in zip(amplitudes, outlier_indices):
            if a_local > 0:
                mu_local = self.x[idx]
                K_local += self.local_kernel(self.x, a_local, mu_local, sigma_local, truncate=truncate, k=k)
        return K_local
    
    def __call__(self, 
                 params={}, 
                 residuals=None,
                 jitter=0.0):
        """
        Compute the covariance matrix for a given set of parameters.
        """
        
        self.C = self.full_to_banded(np.diag(self.effective_variance(b=params.get('b', 0.0))),
                                     k=self.x_ij.shape[0])
        
        a_G = params.get('a_G', 0.0)
        if self.scale_amplitude:
            a_G *= np.median(self.C[0])**0.5
            
        self.C += self.matern_kernel(self.x_ij, 
                                    a_G, # default is no global covariance
                                    params.get('l_G', 1.0), 
                                    nu=1.5, 
                                    r0=self.max_value)
        
        if residuals is not None:
            # Identify outliers from the residuals:
            K_local = self.local_covariance(residuals, 
                                            residuals_threshold=params.get('local_threshold', 4.0),
                                            sigma_local=params.get('local_sigma', 1.0),
                                            truncate=self.truncate,
                                            k=self.x_ij.shape[0])
            if np.any(K_local > 0.0):
                self.C += K_local
            
        if jitter > 0.0:
            # typical value for jitter is 1e-6
            self.C += jitter * np.mean(self.C[0]) * np.eye(len(self.x))
        
        # Cholesky factorize
        self.L = self.cholesky_banded(self.C, get_logdet=True)
        return 
    
    @property
    def cov(self):
        'alias for the covariance matrix'
        if hasattr(self, 'L'):
            return self.L
        else:
            return self.C
    
    @classmethod
    def full_to_banded(cls, array, max_value=None, k=None):

        # Make banded covariance matrix
        banded_array = []
        k = len(array) if k is None else k
        for i in range(k):
            # Retrieve the i-th diagonal
            diag_i = np.diag(array, k=i)

            if (diag_i == 0).all() and (i != 0) and i >= k:
                # There are no more non-zero diagonals coming
                break
            
            if max_value is not None:
                if (diag_i > max_value).all():
                    break

            # Only store the non-zero diagonals
            # Pad the diagonals to the same sizes
            banded_array.append(
                np.concatenate((diag_i, np.zeros(i)))
                )
        
        # Convert to array for scipy
        banded_array = np.asarray(banded_array)

        return banded_array
    
    def banded_to_full(self, Ab):
        
        # Full covariance matrix
        A = np.zeros((Ab.shape[1], Ab.shape[1]))
        
        for i, diag_i in enumerate(Ab):

            if i != 0:
                diag_i = diag_i[:-i]

            # Fill upper diagonals
            A += np.diag(diag_i, k=i)
            if i != 0:
                # Fill lower diagonals
                A += np.diag(diag_i, k=-i)

        return A
    
    def cholesky_banded(self, Ab, get_logdet=False):
        """
        Compute the Cholesky factorization of a symmetric positive-definite 
        banded matrix A given in banded form Ab.

        Parameters
        ----------
        Ab : array_like
            Banded representation of A.

        Returns
        -------
        L : ndarray
            The Cholesky factor in banded form.
        """
        self.L = cholesky_banded(Ab, lower=True, overwrite_ab=False, check_finite=True)
        # Calculate and store logdet
        if get_logdet:
            self.logdet = self.logdet_from_cholesky(self.L, lower=True)
        return self.L

    def solve_banded_system(self, L, b, lower=True):
        """
        Solve a linear system A x = b using the banded Cholesky factorization.

        Parameters
        ----------
        L : array_like
            The banded Cholesky factor of A.
        b : array_like
            Right-hand side vector or matrix.
        lower : bool, optional
            Whether L is in lower banded form. Default is False.

        Returns
        -------
        x : ndarray
            The solution to A x = b.
        """
        return cho_solve_banded((L, lower), b)

    def logdet_from_cholesky(self, L, lower=True):
        """
        Calculate the log-determinant of A given its banded Cholesky factor.

        Parameters
        ----------
        L : array_like
            The banded Cholesky factor of A.
        lower : bool, optional
            Whether L is in lower banded form. Default is False.

        Returns
        -------
        logdet : float
            The log determinant of A.
        """
        if lower:
            diag = L[0, :]
        else:
            diag = L[-1, :]
        return 2.0 * np.sum(np.log(diag))

    def build_composite_covariance(self, resid, b, global_params, sigma_local, 
                                   truncate=4.0, 
                                   threshold=3.0, jitter=1e-6):
        """
        Construct the composite covariance matrix.
        
        The composite covariance is defined as:
            C = b * diag(err^2) + K_global + K_local,
        where K_global is the global (stationary) covariance kernel (here Matérn 3/2),
        and K_local is the sum of local kernels generated from identified outliers in resid.
        
        Parameters
        ----------
        resid : array_like
            The residuals (data - model).
        b : float
            A scaling factor for the diagonal noise (typically close to 1).
        global_params : dict
            Dictionary with keys 'a_G', 'l'. Optionally 'r0' (default r0 = 4 * l).
        sigma_local : float
            The width (in km/s) to be used for the local kernels.
        threshold : float, optional
            Outlier threshold in units of robust sigma (default 3.0).
        
        Returns
        -------
        C_total : ndarray
            The composite covariance matrix.
        components : dict
            A dictionary with individual components: 
            {'noise': M1, 'global': K_global, 'local': K_local}.
        """
        self.jitter = jitter
        # Noise covariance
        M1 = b * np.diag(self.effective_variance())
        
        # Global covariance kernel:
        a_G = global_params.get('a_G')
        l = global_params.get('l')
        r0_global = global_params.get('r0', 4 * l)
        r_full = self.separation_matrix()
        K_global = self.matern_kernel(r_full, a_G, l, nu=1.5, r0=r0_global)
        
        # Identify outliers from the residuals:
        K_local = self.local_covariance(resid, 
                                       residuals_threshold=threshold, 
                                       sigma_local=sigma_local, 
                                       truncate=truncate)
        
        C_total = M1 + K_global + K_local + self.jitter * np.eye(len(self.x))
        components = {'noise': M1, 'global': K_global, 'local': K_local}
        return C_total, components
    
    def draw_random_samples(self, C=None, n_draws=200):
        if C is None:
            C = self.C
        return np.random.multivariate_normal(np.zeros(C.shape[0]), C, size=n_draws)
    
    def envelope_contours(self, C=None, draws=None, n_draws=200):
        if C is None:
            C = self.C
        if draws is None:
            draws = self.draw_random_samples(C=C, n_draws=n_draws)
        lower1, upper1 = np.percentile(draws, [16, 84], axis=0)
        lower2, upper2 = np.percentile(draws, [2.5, 97.5], axis=0)
        lower3, upper3 = np.percentile(draws, [0.15, 99.85], axis=0)
        return (lower1, upper1), (lower2, upper2), (lower3, upper3)

    def plot_decomposition(self, resid, b, global_params, sigma_local, truncate=4.0, region=None, threshold=3.0, n_draws=200):
        """
        Plot the full spectrum residuals and illustrate the decomposition of the covariance matrix.
        
        The left panels show the covariance matrices (trivial noise, noise+global, noise+global+local)
        in the specified region; the right panels show the residuals with random draws from each covariance
        along with envelope contours.
        
        Parameters
        ----------
        resid : array_like
            Residuals (data - model) corresponding to self.x.
        b : float
            Noise scaling factor.
        global_params : dict
            Global kernel parameters ('a_G', 'l', optionally 'r0').
        sigma_local : float
            Width (km/s) for the local kernels.
        region : tuple or None, optional
            If a tuple (xmin, xmax) is provided, only that wavelength region is plotted.
            If None, the full spectrum is used.
        threshold : float, optional
            Threshold for outlier detection (default 3.0).
        truncate : float, optional
            Truncation distance for the local kernels (default 4.0).
        n_draws : int, optional
            Number of random draws for envelope estimation (default 200).
        """
        # Select the region of interest.
        if region is not None:
            xmin, xmax = region
            mask = (self.x >= xmin) & (self.x <= xmax)
        else:
            mask = np.ones_like(self.x, dtype=bool)
        
        x_region = self.x[mask]
        resid_region = np.atleast_1d(np.array(resid))[mask]
        n_region = len(x_region)
        r_region = self.separation_matrix(x=x_region)
        
        # Build decomposed covariance matrices for the region.
        M1 = b * np.diag(self.err[mask]**2)
        a_G = global_params.get('a_G')
        l = global_params.get('l')
        r0_global = global_params.get('r0', 4 * l)
        K_global_reg = self.matern_kernel(r_region, a_G, l, nu=1.5, r0=r0_global)
        # Identify outliers in the region:
        outlier_indices, amplitudes, _ = self.identify_outliers(resid_region, threshold=threshold)
        K_local_reg = np.zeros((n_region, n_region))
        for a_local, idx in zip(amplitudes, outlier_indices):
            if a_local > 0:
                mu_local = x_region[idx]
                K_local_reg += self.local_kernel(x_region, a_local, mu_local, sigma_local, truncate=truncate)
        
        M2 = M1 + K_global_reg
        M3 = M2 + K_local_reg

        # Generate random draws for each covariance matrix.
        draws_all = [self.draw_random_samples(x=x_region, C=M1, n_draws=n_draws),
                     self.draw_random_samples(x=x_region, C=M2, n_draws=n_draws),
                     self.draw_random_samples(x=x_region, C=M3, n_draws=n_draws)]
        # Compute the envelopes for each covariance matrix.
        env_M1 = self.envelope_contours(x=x_region, C=M1, draws=draws_all[0], n_draws=n_draws)
        env_M2 = self.envelope_contours(x=x_region, C=M2, draws=draws_all[1], n_draws=n_draws)
        env_M3 = self.envelope_contours(x=x_region, C=M3, draws=draws_all[2], n_draws=n_draws)
        
        # Set up the plot: 3 rows (for each decomposition) x 2 columns.
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12), sharey='col', sharex='col',
                                 gridspec_kw={'width_ratios': [1, 2]})
        cmap = plt.cm.Blues
        titles_left = ["Trivial Noise (Diagonal)",
                       "Noise + Global Covariance",
                       "Noise + Global + Local Covariance"]
        cov_matrices = [M1, M2, M3]
        envs = [env_M1, env_M2, env_M3]
        # draws_all = [draws_all[0], draws_all[1], draws_all[2]]
        extent = [x_region[0], x_region[-1], x_region[0], x_region[-1]]
        for i in range(3):
            # Left: Heat map of the covariance matrix.
            im = axes[i, 0].imshow(cov_matrices[i], origin='lower', aspect='auto', cmap=cmap, extent=extent)
            axes[i, 0].set_title(titles_left[i])
            axes[i, 0].set_ylabel("Wavelength (Å)")
            # axes[i, 0].set_xticks([])
            plt.colorbar(im, ax=axes[i, 0])
            
            # Right: Residuals and random draws.
            ax = axes[i, 1]
            ax.errorbar(x_region, resid_region, yerr=self.err[mask], fmt='ko', markersize=4, label="Residuals")
            idx_sample = np.random.choice(n_draws, size=min(10, n_draws), replace=False)
            for draw in draws_all[i][idx_sample]:
                ax.plot(x_region, draw, color='gray', alpha=0.5, lw=1)
            
            (l1, u1), (l2, u2), (l3, u3) = envs[i]
            ax.fill_between(x_region, l3, u3, color='orangered', alpha=0.1, label='3σ')
            ax.fill_between(x_region, l2, u2, color='orangered', alpha=0.2, label='2σ')
            ax.fill_between(x_region, l1, u1, color='orangered', alpha=0.3, label='1σ')
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylabel("Flux Residual")
            if i == 0:
                ax.legend(loc='upper right', ncol=4)
                ax.set_title("Random draws from covariance")
                
            if i == 2:
                ax.set_xlabel("Wavelength (Å)")
        
        # share y-axis limits between the second column
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data:
    num_points = 100
    wavelength = np.linspace(5040, 5060, num_points)  # in Å
    
    # Define a "true" spectral model: a continuum with a Gaussian absorption line.
    def true_model(w):
        A = 0.3       # absorption depth
        centre = 5050 # centre of the line (Å)
        sigma = 1.0   # width (Å)
        return 1.0 - A * np.exp(-0.5 * ((w - centre) / sigma)**2)
    
    model_flux = true_model(wavelength)
    err = 0.02 * np.ones(num_points)
    
    # Simulate white noise
    noise = np.random.randn(num_points) * err
    # Introduce outliers
    n_outliers = 3
    w_outliers = 2
    outlier_indices = np.random.choice(num_points, size=5, replace=False)
    # noise[outlier_indices] += np.random.choice([0.2, -0.2], size=5)
    for outlier_idx in outlier_indices:
        outlier_region = np.arange(outlier_idx-w_outliers, outlier_idx+w_outliers)
        sign = np.random.choice([1, -1])
        noise[outlier_region] += sign * np.random.choice([0.05, 0.2], size=len(outlier_region))
    
    # Synthetic observed flux and residuals (data - model)
    flux = model_flux + noise
    residuals = flux - model_flux
    region = (5030, 5050)  # illustrative wavelength region, plotting only part of the spectrum

    
    # Instantiate the Covariance class.
    cov_obj = Covariance(wavelength, err)
    
    truncate = 4.0
    length_scale = 10.0
    x_ij = cov_obj.separation_matrix()
    x_ij_b = cov_obj.full_to_banded(x_ij, max_value=length_scale * truncate)
    cov = cov_obj.matern_kernel(x_ij, a=10.0, l=length_scale, nu=1.5, r0=truncate * length_scale)
    cov_b = cov_obj.full_to_banded(cov, k=x_ij_b.shape[0])
    cov_full = cov_obj.banded_to_full(cov_b)
    
    print(f' cov.shape {cov.shape}')
    print(f' cov_full.shape {cov_full.shape}')
    print(f' cov_b.shape {cov_b.shape}')
    assert np.allclose(cov, cov_full), f'cov and cov_full are not close'
    
    L = cov_obj.cholesky_banded(cov_b)
    print(f' L.shape {L.shape}')
    print(f' logdet {cov_obj.logdet}')
    
    # test banded solve
    a = np.random.randn(cov_b.shape[1])
    solve = cov_obj.solve_banded_system(L, a)
    print(f' solve.shape {solve.shape}')
    
    plot_cov = False
    if plot_cov:
        # Plot data and residuals and fill region with error bars
        fig, ax = plt.subplots(2,1, figsize=(10,8), gridspec_kw={'height_ratios': [3, 1]})
        ax[0].scatter(wavelength, flux, label="Data", color='k', s=10)
        ax[0].plot(wavelength, model_flux, 'orangered', label="Model")
        ax[0].fill_between(wavelength, flux-err, flux+err, color='gray', alpha=0.5)
        ax[0].set_xlabel("Wavelength (Å)")
        ax[0].set_ylabel("Flux")
        ax[0].legend()
        ax[1].scatter(wavelength, residuals, label="Residuals", color='k', s=10)
        
        err = cov_obj.err
        ax[1].fill_between(wavelength, -err, err, color='gray', alpha=0.5)
        ax[1].set_xlabel("Wavelength (Å)")
        ax[1].set_ylabel("Residuals")
        [ax[i].axvspan(region[0], region[1], color='gray', alpha=0.2, zorder=-1) for i in range(2)]
        
        plt.tight_layout()
        plt.show()
        
        # Global kernel hyperparameters and local kernel width.
        global_params = {'a_G': 0.03, 'l': 20.0}  # global kernel amplitude and correlation length (km/s)
        sigma_local = 5.0  # local kernel width in km/s
        b = 1.05         # scaling factor for the diagonal noise
        threshold = 3.0  # threshold for robust outlier detection
        
        # Build the composite covariance matrix (if needed for further analysis)
        C_total, components = cov_obj.build_composite_covariance(residuals, b, global_params, sigma_local, threshold)
        
        # Plot the decomposition of the covariance matrix and the residuals.
        cov_obj.plot_decomposition(residuals, b, global_params, sigma_local, region=region, threshold=threshold, n_draws=int(1e4))
