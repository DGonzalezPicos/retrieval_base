import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded
from scipy.linalg import eig_banded
def get_Covariance_class(err, mode=None, **kwargs):

    if mode == 'GP':
        # Use a GaussianProcesses instance
        return GaussianProcesses(err, **kwargs)
    
    # Use a Covariance instance instead
    return Covariance(err, **kwargs)

class Covariance:
     
    def __init__(self, err, **kwargs):

        # Set-up the covariance matrix, manage the case where err is negative or zero
        self.err = np.where(err > 0, err, np.inf)
        # print(f' [Covariance.__init__]: err.shape {self.err.shape}')
        # print(f' [Covariance.__init__]: err.min {self.err.min():.2e} err.max {self.err.max():.2e}')
        # print(f' [Covariance.__init__]: err.mean {self.err.mean():.2e}, err.std {self.err.std():.2e}')
        self.cov_reset()

        # Set to None initially
        self.cov_cholesky = None
        self.cholesky_failed = False

    def __call__(self, params, grating, order, det, **kwargs):

        # Reset the covariance matrix
        self.cov_reset()
        # check there's no zeros in cov
        assert not np.any(self.cov == 0), f'Covariance matrix has {np.sum(self.cov == 0)} zeros'
        
        if params[f'beta_{grating}'][order,det] != 1:
            self.add_data_err_scaling(
                params[f'beta_{grating}'][order,det]
                )
        return self

    def cov_reset(self):

        # Create the covariance matrix from the uncertainties
        self.cov = self.err**2
        self.is_matrix = (self.cov.ndim == 2)

        self.cov_shape = self.cov.shape
        return self

    def add_data_err_scaling(self, beta):
        # print(f' self.cov.shape {self.cov.shape}')
        # print(f' beta.shape {beta.shape}')
        # Scale the uncertainty with a (beta) factor
        if not self.is_matrix:
            self.cov *= beta**2
        else:
            self.cov[np.diag_indices_from(self.cov)] *= beta**2
            
        return self

    def add_model_err(self, model_err):

        # Add a model uncertainty term
        if not self.is_matrix:
            self.cov += model_err**2
        else:
            self.cov += np.diag(model_err**2)
        return self

    def get_logdet(self):

        # Calculate the log of the determinant
        self.logdet = np.sum(np.log(self.cov))
        return self
    

    def solve(self, b):
        '''
        Solve the system cov*x = b, for x (x = cov^{-1}*b).

        Input
        -----
        b : np.ndarray
            Righthand-side of cov*x = b.
        
        Returns
        -------
        x : np.ndarray

        '''
        
        if self.is_matrix:
            return np.linalg.solve(self.cov, b)
            
        # Only invert the diagonal
        return (1/self.cov) * b
    
    def get_dense_cov(self):

        if self.is_matrix:
            return self.cov
        
        return np.diag(self.cov)
    
    def get_err(self, mask=None):
        
        if mask is None:
            mask = np.ones(self.cov.shape[-1])
        err = np.nan * np.ones_like(mask)

        if not self.is_matrix:
            err[mask] = np.sqrt(self.cov)
        
        else: # diagonal elements
            err[mask] = np.sqrt(np.diag(self.get_dense_cov()))
        return err

class GaussianProcesses(Covariance):

    def get_banded(cls, array, max_value=None):

        # Make banded covariance matrix
        banded_array = []

        for i in range(len(array)):
            # Retrieve the i-th diagonal
            diag_i = np.diag(array, k=i)

            if (diag_i == 0).all() and (i != 0):
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

    def __init__(self, err, separation, err_eff=None, flux_eff=None, max_separation=None, **kwargs):
        '''
        Create a covariance matrix suited for Gaussian processes. 

        Input
        -----
        err : np.ndarray
            Uncertainty in the flux.
        separation : np.ndarray
            Separation between pixels, can be in units of wavelength, 
            pixels, or velocity.
        err_eff : np.ndarray
            Average squared error between pixels.
        '''
        
        # Pre-computed average error and wavelength separation
        self.separation = np.abs(separation)
        self.err_eff  = err_eff
        self.flux_eff = flux_eff

        # Convert to banded matrices
        self.separation = self.get_banded(
            self.separation, max_value=max_separation
            )
        if isinstance(self.err_eff, np.ndarray):
            self.err_eff = self.get_banded(self.err_eff)
            self.err_eff = self.err_eff[:self.separation.shape[0]]

        if isinstance(self.flux_eff, np.ndarray):
            self.flux_eff = self.get_banded(self.flux_eff)
            self.flux_eff = self.flux_eff[:self.separation.shape[0]]

        # Give arguments to the parent class
        super().__init__(err)

    def __call__(self, params, grating, **kwargs): # remove order, det as arguments

        # Reset the covariance matrix
        self.cov_reset()

        # if params[f'beta_{grating}'][order,det] != 1:
        #     self.add_data_err_scaling(
        #         params[f'beta_{grating}'][order,det]
        #         )
        
        if params.get(f'a_{grating}_G', None) is not None:
            if isinstance(params[f'a_{grating}_G'], float):
                self.add_RBF_kernel(
                    a=params[f'a_{grating}_G'], 
                    l = params['l_G'],
                    **kwargs
                    )
            

    def cov_reset(self):

        # Create the covariance matrix from the uncertainties
        self.cov = np.zeros_like(self.separation)
        self.cov[0] = self.err**2

        self.is_matrix = True
        return self
        
    def add_data_err_scaling(self, beta):
        # Scale the uncertainty with a (beta) factor
        assert len(self.cov.shape) == 2, f'Covariance matrix is not banded: {self.cov.shape}'
        self.cov[0] *= beta**2
        return self
    
    def hanning_window(self, trunc_dist=5, l=1, cosine_taper=True):
        # Hann window function to ensure sparsity
        if cosine_taper:
            ratio = self.separation / (trunc_dist * l)
            w_ij = np.where(
                self.separation < trunc_dist * l,
                0.5 * (1 + np.cos(np.pi * ratio)),
                0
            ).astype(bool)
        else:
            w_ij = self.separation < trunc_dist * l
        return w_ij

    def add_RBF_kernel(self, a, l, trunc_dist=5, scale_GP_amp=False, **kwargs):
        '''
        Add a radial-basis function kernel to the covariance matrix. 
        The amplitude can be scaled by the flux-uncertainties of 
        pixels i and j if scale_GP_amp=True. 

        Input
        -----
        a : float
            Square-root of amplitude of the RBF kernel.
        l : float
            Length-scale of the RBF kernel.
        trunc_dist : float
            Distance at which to truncate the kernel 
            (|wave_i-wave_j| < trunc_dist*l). This ensures
            a relatively sparse covariance matrix. 
        scale_GP_amp : bool
            If True, scale the amplitude at each covariance element, 
            using the flux-uncertainties of the corresponding pixels
            (A = a**2 * (err_i**2 + err_j**2)/2).
        '''

        # Hann window function to ensure sparsity
        w_ij = self.hanning_window(trunc_dist, l, cosine_taper=kwargs.get('cosine_taper', False))
        
        # print(f' w_ij.shape {w_ij.shape}')
        # GP amplitude
        err_eff = 1.0 if not scale_GP_amp else self.err_eff
        GP_amp = (a * err_eff)**2
        if isinstance(GP_amp, np.ndarray):
            GP_amp = GP_amp[w_ij]

        # Gaussian radial-basis function kernel
        self.cov[w_ij] += GP_amp * np.exp(-(self.separation[w_ij])**2/(2*l**2))
        
        return self
    
    def check_cov(self):
        assert np.all(np.diag(self.cov) >= 0), f'Covariance matrix has negative diagonal elements: {self.cov.shape}'
        
        # Compute eigenvalues of the banded covariance matrix
        eigenvalues, eigenvectors = eig_banded(self.cov, lower=True)
        
        if np.any(eigenvalues < 0):
            print(f'Warning: Negative eigenvalues detected: {eigenvalues[eigenvalues < 0]}')
            print(f' shape of cov {self.cov.shape}')
            # Clip negative eigenvalues to a small positive value
            eigenvalues = np.clip(eigenvalues, a_min=1e-6, a_max=None)
            
            # Reconstruct the covariance matrix
            self.cov = (eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T)
            self.cov = self.get_banded(self.cov)[:self.separation.shape[0]]
            
                                                          
            print(f' shape of cov {self.cov.shape} after reconstruction')
        
        # cond_number = np.linalg.cond(self.cov)
        # assert cond_number < 1e10, f'Covariance matrix is ill-conditioned: {cond_number}'
        return self
    
    
    def get_cholesky(self, log_jitter=-4, max_iter=2, debug=False):
        """
        Ensure self.cov is a robust banded covariance matrix that will pass Cholesky decomposition.
        """
        
        k = self.separation.shape[0]
        mask_nonzero_diag = (self.cov != 0).any(axis=1)
        if mask_nonzero_diag.sum() == 1:
            return np.sqrt(self.cov[0])
        
        C = self.cov[mask_nonzero_diag,:]
        
        try:
            self.cov_cholesky = cholesky_banded(C, lower=True, check_finite=False)
            if debug:
                print(f' --> L.shape {self.cov_cholesky.shape}')
            return self
        except Exception as e:
            if debug:
                print(f' --> Cholesky decomposition failed: {e}')
            eigvals, eigvecs = eig_banded(C, lower=True)
            if debug:
                print(f' --> eigvals.shape {eigvals.shape}')
                print(f' --> eigvecs.shape {eigvecs.shape}')
            
            neg_eigvals = eigvals[eigvals < 0]
            if debug:
                print(f' Number of negative eigenvalues: {len(neg_eigvals)} / {len(eigvals)}')
            eigvals[eigvals <= 0] = np.min(eigvals[eigvals > 0])
            
            for _i in range(max_iter):  
                jitter = 10**(log_jitter * (_i + 1))
                eigvals += jitter
                
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                C_banded = self.get_banded(C)[:k,:]
                if debug:
                    print(f' --> C_banded.shape {C_banded.shape}')
                try:
                    self.cov_cholesky = cholesky_banded(C_banded, lower=True, check_finite=False)
                    if debug:
                        print(f' --> L.shape {self.cov_cholesky.shape}')
                    return self
                except Exception as e:
                    if debug:
                        print(f' --> Cholesky decomposition failed: {e}')
                    continue
            
            # Return an all-zero matrix with the same shape as C if Cholesky fails after max_iter
            if debug:
                print("Cholesky decomposition failed after maximum iterations. Returning zero matrix.")
            self.cov_cholesky = np.zeros_like(C)
            return self


    def get_cholesky_old(self, debug=False):
        '''
        Get the Cholesky decomposition. Employs a banded 
        decomposition with scipy. 
        '''        
        self.cholesky_failed = False
        mask_nonzero_diag = (self.cov != 0).any(axis=1)
        if debug:
            # print(f' [GaussianProcesses.get_cholesky]: mask_nonzero_diag.sum() {mask_nonzero_diag.sum()}')
            self.check_cov()
        
        if mask_nonzero_diag.sum() == 1:
            # Only the diagonal is non-zero
            self.cov = self.cov[0]
            self.cov_cholesky = np.sqrt(self.cov)
            
            return 
    
        self.cov = self.cov[mask_nonzero_diag,:]
        
        eps = 1e-4
        for eps_i in range(3):
            try:
                # self.cov += eps_i*eps*np.ones_like(self.cov)
                self.cov_cholesky = cholesky_banded(self.cov,
                                                    lower=True, 
                                                    check_finite=False)
                return self
            except np.linalg.LinAlgError:
                # if debug:
                print(f' !!!!!! Cholesky decomposition failed {eps_i}/3...')
                # self.cov_cholesky = np.sqrt(self.cov)
                
            self.cholesky_failed = True
        
        return self

    def get_logdet(self):
        '''
        Calculate the log of the determinant. Uses diagonal 
        elements of banded Cholesky decomposition.
        '''

        self.logdet = 2*np.sum(np.log(self.cov_cholesky[0]))

    def solve(self, b):
        '''
        Solve the system cov*x = b, for x (x = cov^{-1}*b). 
        Employs a sparse or banded Cholesky decomposition.

        Input
        -----
        b : np.ndarray
            Righthand-side of cov*x = b.
        
        Returns
        -------
        x : np.ndarray

        '''

        return cho_solve_banded((self.cov_cholesky, True), b, check_finite=False, overwrite_b=False)
    
    def get_dense_cov(self):
        
        # Full covariance matrix
        cov_full = np.zeros((self.cov.shape[1], self.cov.shape[1]))
        
        for i, diag_i in enumerate(self.cov):

            if i != 0:
                diag_i = diag_i[:-i]

            # Fill upper diagonals
            cov_full += np.diag(diag_i, k=i)
            if i != 0:
                # Fill lower diagonals
                cov_full += np.diag(diag_i, k=-i)

        return cov_full
    
    
    
if __name__ == '__main__':
    # test GP covariance matrix with off-diagonal elements
    
    import matplotlib.pyplot as plt
    # plot imshow in logscale
    from matplotlib.colors import LogNorm
    from retrieval_base.local_covariance_kernel import LocalCovarianceKernel
    
    np.random.seed(1234)
    wave = np.linspace(2300, 2350, 400)
    p = 20
    flux = 5.0 + np.sin(wave / p)
    snr = 20.0
    
    model = flux.copy()
    
    noise = np.random.normal(0, 1/snr, wave.shape)
    flux += noise
    err = 1 * np.sqrt(flux.mean()) / snr * np.ones(wave.shape)
    
    # add outliers
    def add_random_outlier(flux, amplitude=1.0, width=2, n_outliers=2):
        n_outliers = np.random.randint(1, n_outliers)
        for i in range(n_outliers):
            i = np.random.randint(0, len(flux))
            flux[i:i+width] += np.random.normal(amplitude, amplitude/2)
        return flux
    flux = add_random_outlier(flux, 0.4, 2, 4)
    # flux[1008:1016] += 1.0
    
    res = flux - model
    chi2_pp = res**2 * err**-2
    
    
    lck = LocalCovarianceKernel(wave, flux, err, lck_width=4)
    lck.s = lck(model, sigma_threshold=2.0, n_max_regions=5)
    
    kernel = lck.correlated_kernel()
    
    # print(stop)
    print(f' Scaling factors:')
    print(f' Outliers: {lck.s_regions[lck.mask]}')
    print(f' Rest: {lck.s_rest:.2f}')

    # create a 1D array of wavelength
    separation = np.abs(wave[None,:] - wave[:,None])
    # create a 1D array of uncertainties
    # err = np.ones_like(wave) + 0.1*np.random.randn(len(wave))
    # create a 1D array of flux uncertainties
    err_eff = np.mean(err)
    
    grating = 'NIRSpec'
    order = 0
    det = 0
    
    a_G = 0.4 * np.ones((1,1))
    l_G = 0.5 * np.ones((1,1))
    params = {
        f'a_{grating}_G': a_G,
        f'l_{grating}_G': l_G,
        
        f'a_{grating}_K': np.sqrt(kernel[None,None,...]),
        f'l_{grating}_K': 5 * l_G,
        
        
    }
    # create a GP covariance matrix
    cov = GaussianProcesses(err, separation, err_eff, max_separation=2)
    cov(params, grating, order, det, scale_GP_amp=True)
    # get the Cholesky decomposition
    cov.get_cholesky()
    print(f' cov_cholesky.shape {cov.cov_cholesky.shape}')
    
    cov_full = cov.get_dense_cov()
    cov_err = np.sqrt(np.diag(cov_full))

    # create gridspec with 5 rows and 5 columns
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14,10))
    gs = GridSpec(5, 5)
    ax_spec = fig.add_subplot(gs[0:3,0:3])
    ax_cov = fig.add_subplot(gs[0:3,3:5])
    ax_res = fig.add_subplot(gs[3:5,0:3])
    
    ax_spec.plot(wave, flux, label='flux', color='k')
    ax_spec.fill_between(wave, flux - err, flux + err, alpha=0.2, color='k', lw=0)

    ax_spec.fill_between(wave, flux - cov_err, flux + cov_err, alpha=0.2, color='red', lw=0)
    ax_spec.plot(wave, model, label='model', color='orange')
    ax_spec.legend()
    ax_spec.set_title('Spectrum')
    ax_res.plot(wave, res, '.', label='residuals', color='k')
    ax_res.fill_between(wave, -cov_err, cov_err, alpha=0.2, color='red', lw=0)
    ax_res.fill_between(wave, -err, err, alpha=0.2, color='k', lw=0)
    ax_res.legend()
    
    extent = [wave[0], wave[-1], wave[0], wave[-1]]
    ax_cov.imshow(cov_full, origin='lower', aspect='auto', norm=LogNorm(), extent=extent)
    ax_cov.set_title('Covariance')
    
    for region in lck.regions:
        ax_res.axvspan(region[0], region[1], alpha=0.1, color='k', lw=0, zorder=-1)
        # draw a vertical arrow at the bottom of the plot pointing down from the top of the plot
        # ax_res.arrow(region[0], np.min(res), 0, np.max(res) - np.min(res), head_width=0.1, head_length=0.1, fc='r', ec='r')
    plt.show()
