import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded

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
        # print(f' [Covariance.__init__]: err.min {self.err.min():.2e} err.max {self.err.max():.2e} err.mean {self.err.mean():.2e}')
        self.cov_reset()

        # Set to None initially
        self.cov_cholesky = None

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
                    # l=params[f'l_{grating}_G'], 
                    l = params['l_G'],
                    array=self.err_eff, 
                    **kwargs
                    )
        # if params.get(f'a_{grating}_K', None) is not None:
        #     # a = self.get_banded(np.diag(params[f'a_{grating}'][order,det]))[:self.separation.shape[0]]
        #     # a = np.tile(params[f'a_{grating}'][order,det], (self.separation.shape[0], 1))
        #     a = self.get_banded(params[f'a_{grating}_K'])[:self.separation.shape[0]]
            
        #     if a.shape[0] < self.separation.shape[0]:
        #         # fill with zeros along axis 0
        #         a = np.concatenate((a, np.zeros((self.separation.shape[0] - a.shape[0], a.shape[1]))), axis=0)
            
        #     assert a.shape[0] == self.separation.shape[0], f'a.shape {a.shape} != self.separation.shape {self.separation.shape}'
        #     self.add_RBF_kernel(
        #         a=a, 
        #         # l=params[f'l_{grating}_K'], 
        #         l = params.get('l_K', params['l_G']),
        #         array=self.err_eff, # FIXME: check whether we use the mean error or what
        #         **kwargs
        #     )


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

    def add_RBF_kernel(self, a, l, array, trunc_dist=5, scale_GP_amp=False, **kwargs):
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
        w_ij = (self.separation < trunc_dist*l)
        # print(f' w_ij.shape {w_ij.shape}')
        # GP amplitude
        GP_amp = a**2
        if scale_GP_amp:
            # Use amplitude as fraction of flux uncertainty
            if isinstance(array, float):
                GP_amp *= array**2
            else:
                GP_amp *= array[w_ij]**2
                
        if GP_amp.shape == self.separation.shape:
            GP_amp = GP_amp[w_ij]

        # Gaussian radial-basis function kernel
        self.cov[w_ij] += GP_amp * np.exp(-(self.separation[w_ij])**2/(2*l**2))
        
        return self
        
    def get_cholesky(self, max_attempts=10, epsilon=1e-2, debug=False):
        '''
        Get the Cholesky decomposition. Employs a banded 
        decomposition with scipy. 
        '''        
        
        mask_nonzero_diag = (self.cov != 0).any(axis=1)
        self.cov = self.cov[mask_nonzero_diag,:]
        
        # print(f' [GaussianProcesses.get_cholesky]: mask_nonzero_diag.sum() {mask_nonzero_diag.sum()}'
        #       )
        # print(f' [GaussianProcesses.get_cholesky]: self.cov.shape {self.cov.shape}')
        # print(f' [GaussianProcesses.get_cholesky]: self.cov.min() {self.cov.min():.2e} self.cov.max() {self.cov.max():.2e} self.cov.mean() {self.cov.mean():.2e}')
        
        if mask_nonzero_diag.sum() == 1:
            # Only the diagonal is non-zero
            self.cov = self.cov[0]
            self.cov_cholesky = np.sqrt(self.cov)
            
            return 
        
        # mean_cov = np.nanmean(self.cov)
        # self.cov /= mean_cov
        self.cov = self.cov[mask_nonzero_diag,:]
        try:
            self.cov_cholesky = cholesky_banded(self.cov, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            # print(f' !!!!!! Cholesky decomposition failed...')
            self.cov_cholesky = np.sqrt(self.cov)
        '''
        # Compute banded Cholesky decomposition
        for _ in range(max_attempts):
            try:
                self.cov_cholesky = cholesky_banded(
                    self.cov, lower=True, check_finite=False,
                    )
                if debug:
                    print(f' self.cov_chol.shape {self.cov_cholesky.shape}')

                return self
            except np.linalg.LinAlgError:
                # Add a small number to the diagonal
                if debug:
                    print(f' Cholesky decomposition failed...')
                    print(f' epsilon={epsilon}')
                    print(f' self.cov.shape {self.cov.shape}')
                    print(f' Min: {np.min(self.cov)} Max: {np.max(self.cov)} Median: {np.median(self.cov)}')
                # self.cov[0] *= (1 + epsilon)
                self.cov[0] += epsilon
                epsilon *= 10
        '''
        # self.cov_cholesky *= mean_cov
        # delattr(self, 'cov')
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
    
    def get_err(self, mask=None):
        if mask is None:
            mask = np.ones(self.cov.shape[-1])
        err = np.nan * np.ones_like(mask)
        err[mask] = np.sqrt(np.diag(self.get_dense_cov()))
        return err
    
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
